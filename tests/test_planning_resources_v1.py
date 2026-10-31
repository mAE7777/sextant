"""Resource freshness, blind context and ambiguous execution-source regressions."""

import json
from copy import deepcopy

import pytest
from test_application_v1 import OWNER, ControlledGenerator, call

from sextant.application import Application
from sextant.domain import DomainError, digest
from sextant.planning_resources import POOL_NAMES, planning_material
from sextant.storage import Store


def snapshot(used=0, phase=None):
    return {"schema_version": 1, "phase": phase, "deadline_monotonic": 100.0,
            "observed_monotonic": 20.0, "remaining_seconds": 80.0,
            "pools": {key: {"capacity": 20, "used": used, "reserved": 0,
                            "remaining": 20 - used,
                            "unit": "allowance_seconds" if key == "native_seconds" else "opportunities"}
                      for key in POOL_NAMES}}


def state():
    return {"mission_id": "m", "revision": "r", "options": {"environment": {"phase": "p0"}},
            "research_queue": [{"branch": "secret-branch", "signature": "s", "experiment": {"key": "private-design"},
                                "origin_proposal_ref": None, "protocol_fingerprint": "f"}],
            "pending_experiments": []}


def test_blind_queue_preserves_occurrences_without_design_or_branch_leak():
    original = state()
    original["pending_experiments"] = deepcopy(original["research_queue"])
    blind = planning_material("challenger", original, snapshot(phase="p0"))
    assert blind["queue_occurrences"] == 2 and blind["unique_designs"] == 1
    assert "private-design" not in json.dumps(blind) and "secret-branch" not in json.dumps(blind)
    ordinary = planning_material("primary", original, snapshot(phase="p0"))
    assert ordinary["queue"][0]["experiment"]["key"] == "private-design"
    ordinary["queue"][0]["experiment"]["key"] = "changed"
    assert original["pending_experiments"][0]["experiment"]["key"] == "private-design"


@pytest.mark.parametrize("mutation", ["bool", "arithmetic", "phase", "nan", "deadline", "lost_pool", "duplicate"])
def test_bad_host_counters_or_queue_are_rejected(mutation):
    original, resources = state(), snapshot(phase="p0")
    if mutation == "bool":
        resources["pools"]["native"]["used"] = True
    elif mutation == "arithmetic":
        resources["pools"]["native"]["remaining"] = 19
    elif mutation == "phase":
        resources["phase"] = "p1"
    elif mutation == "nan":
        resources["pools"]["native_seconds"]["capacity"] = float("nan")
    elif mutation == "deadline":
        resources["remaining_seconds"] = 90
    elif mutation == "lost_pool":
        resources["pools"].pop("typed")
    else:
        original["pending_experiments"] = deepcopy(original["research_queue"])
        original["pending_experiments"][0]["experiment"]["key"] = "different"
    with pytest.raises(DomainError):
        planning_material("primary", original, resources)


def test_each_role_observes_fresh_actual_counts_and_callback_cannot_edit_state(tmp_path):
    generator = ControlledGenerator()
    seen = []

    def callback(stage, captured):
        seen.append((stage, captured["mission_id"]))
        captured["mission_id"] = "MUST-NOT-BECOME-AUTHORITY"
        return snapshot(used=len(generator.requests))

    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator}, planning_callback=callback)
    started = call(app, "start", mission={"goal": "Compare competing mechanisms"}, options={"jev_density": "none"},
                   authorization={"tools": ["claude-code"], "send_to": {"claude-code": ["*"]}})
    done = call(app, "run", started, steps=1)
    assert not done.errors
    material = [json.loads(r["prompt"])["planning_resources"] for r in generator.requests]
    assert [m["pools"]["native"]["used"] for m in material] == [0, 1, 2]
    assert [m["stage"] for m in material] == ["primary", "challenger", "synthesis"]
    assert all(m["mission_id"] == started.mission_id for m in material)
    assert all(mid == started.mission_id for _, mid in seen)


class ConflictingGenerator(ControlledGenerator):
    def invoke(self, request, context):
        result = super().invoke(request, context)
        experiments = result.output["payload"]["candidates"][1]["experiments"]
        conflicting = deepcopy(experiments[0])
        conflicting["inputs"] = {"assignments": {"a": 0, "b": 1}}
        result.output["payload"]["policy"]["next_experiments"] = [conflicting]
        result.receipt.output_hash = digest(result.output)
        return result


def test_same_source_name_cannot_dispatch_incompatible_design(tmp_path):
    generator = ConflictingGenerator()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator})
    started = call(app, "start", mission={"goal": "Compare mechanisms"}, options={"jev_density": "none"},
                   authorization={"tools": ["claude-code"], "send_to": {"claude-code": ["*"]}})
    done = call(app, "run", started, steps=1)
    assert done.status == "experiment_identity"
    viewed = call(app, "inspect", done).outputs
    assert viewed["state"]["pending_experiments"] == []
    assert viewed["state"]["proposal_ref"] is None
    assert all(r["receipt"]["executor"] == "claude-code" for r in viewed["saved_receipts"])
