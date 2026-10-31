"""Interaction regressions found by manipulating saved state, imports, and schemas.

All executors are controlled.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_runtime_integrity_v1 import Capture, call, experiment, fresh, projection, proposal, start

from sextant.domain import Command, DomainError, ObjectRef, canonical, digest
from sextant.representation_control import SubmissionProjection, submitted_shape


@pytest.mark.parametrize("damage", ["none", "after_observe", "after_recompile"])
def test_retained_research_keeps_historical_evidence_obligation(tmp_path: Path, damage: str) -> None:
    env = Capture("environment", {"observations": []})
    app, generator = fresh(tmp_path, proposal([experiment(marker="retained-design")]), env)
    first = call(app, "run", start(app))
    inspected = call(app, "inspect", first, events=True)
    original = inspected.outputs["state"]["proposal_ref"]
    received = next(e for e in inspected.events if e["kind"] == "effect.received")
    evidence = app.store.artifact_path(received["payload"]["artifact"]["sha256"])
    observed = call(app, "observe", first, content="independent new evidence", source="fixture")
    generator.output = {"payload": proposal()}
    if damage == "after_observe":
        evidence.write_bytes(b"corrupt original receipt")
    second = call(app, "run", observed)
    assert not second.errors, second
    snapshot = call(app, "inspect", second, events=True)
    queue = snapshot.outputs["state"]["pending_experiments"]
    if damage == "after_observe":
        assert queue == []
        assert any(e["kind"] == "research.discarded" for e in snapshot.events)
        assert any("originating evidence" in s for o in snapshot.outputs["objects"]
                   if o["revision"]["ref"]["object_kind"] == "policy" for s in o["revision"]["payload"]["unresolved"])
    else:
        assert queue[0]["origin_proposal_ref"] == original
        policy_ref = snapshot.outputs["state"]["policy_ref"]
        assert next(o["current_validity"]["valid"] for o in snapshot.outputs["objects"] if o["revision"]["ref"] == policy_ref)
    if damage == "after_recompile":
        evidence.unlink()
    third = call(app, "run", second)
    assert not third.errors, third
    assert len(env.requests) == (1 if damage == "none" else 0)
    if damage == "none":
        assert env.requests[0]["inputs"]["marker"] == "retained-design"


def test_protocol_change_cannot_dispatch_saved_queue(tmp_path: Path) -> None:
    env = Capture("environment", {"observations": []})
    app, generator = fresh(tmp_path, proposal([experiment()]), env)
    first = call(app, "run", start(app))
    app.protocol_fingerprint = "changed-host-protocol"
    generator.output = {"payload": proposal()}
    second = call(app, "run", first)
    assert not second.errors and second.outputs["steps"][0]["kind"] == "compile"
    assert env.requests == []


@pytest.mark.parametrize("cross_operation", [False, True])
def test_missing_request_is_not_reconstructed_on_reuse(tmp_path: Path, cross_operation: bool) -> None:
    app, _ = fresh(tmp_path)
    created = start(app)
    executor = Capture("environment", {"observations": []})
    with app.store.transaction() as db:
        state = app.store.mission(db, created.mission_id)
        inputs = [ObjectRef.model_validate(state["mission_ref"])]
        auth = ObjectRef.model_validate(state["authorization_ref"])
        token = app.runtime.begin(db, created.mission_id, "original", state["revision"], inputs, auth)
    app.runtime.dispatch(token, executor, {"action": "measure"}, effect_key="design")
    if cross_operation:
        with app.store.transaction() as db:
            app.runtime.finish(db, token)
            token = app.runtime.begin(db, created.mission_id, "second", state["revision"], inputs, auth)
    sent = next(e for e in call(app, "inspect", created, events=True).events if e["kind"] == "effect.sent")
    path = app.store.artifact_path(sent["payload"]["request"]["sha256"])
    path.unlink()
    with pytest.raises(DomainError, match=sent["payload"]["request"]["sha256"]) as error:
        app.runtime.dispatch(token, executor, {"action": "measure"}, effect_key="design", reuse_completed=cross_operation)
    assert error.value.code == "missing_artifact" and not path.exists()
    assert len(executor.requests) == 1


def malformed_package(tmp_path: Path, mutation: str) -> dict[str, Any]:
    app, _ = fresh(tmp_path / "sender")
    compiled = call(app, "run", start(app))
    observed = call(app, "observe", compiled, content="new evidence", source="fixture")
    package = call(app, "export", observed).outputs
    if mutation == "duplicate_invalidation":
        package["invalidations"].append(copy.deepcopy(package["invalidations"][0]))
    elif mutation == "duplicate_effect":
        row = copy.deepcopy(package["attempts"][0])
        row["attempt_id"] += "_another"
        package["attempts"].append(row)
    elif mutation.startswith("state_"):
        package["mission"][mutation.removeprefix("state_")] = []
    elif mutation.startswith("operation_"):
        field = mutation.removeprefix("operation_")
        package["operations"][0][field] = "{}" if field == "inputs" else "[]"
    elif mutation.startswith("attempt_"):
        field = mutation.removeprefix("attempt_")
        package["attempts"][0][field] = "{not-json" if field == "receipt" else "{}"
    elif mutation == "receipt_executor":
        row = package["attempts"][0]
        receipt = json.loads(row["receipt"])
        receipt["executor"] = "wrong-executor"
        row["receipt"] = canonical(receipt)
    elif mutation == "fence_overflow":
        package["operations"][0]["fence"] = 2**100
    elif mutation == "event_reference":
        package["events"][0]["operation_id"] = "missing-operation"
    else:
        raise AssertionError(mutation)
    package["content_hash"] = digest({k: v for k, v in package.items() if k != "content_hash"})
    return package


@pytest.mark.parametrize("mutation", ["duplicate_invalidation", "duplicate_effect", "state_generation", "state_branch_visits",
    "state_revision", "operation_inputs", "operation_authorization", "operation_failure", "attempt_receipt", "attempt_request",
    "attempt_output", "receipt_executor", "fence_overflow", "event_reference"])
def test_malformed_import_is_durable_and_atomic(tmp_path: Path, mutation: str) -> None:
    package = malformed_package(tmp_path, mutation)
    app, _ = fresh(tmp_path / "receiver")
    command = Command(kind="import", command_id="bad-import", payload={"package": package})
    result = app.execute(command)
    assert result.errors and result == app.execute(command)
    assert result.status.startswith("import_") or result.status == "invalid_input"
    with app.store.connection(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM missions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM commands WHERE result IS NOT NULL").fetchone()[0] == 1


@pytest.mark.parametrize("mutation", ["duplicate_invalidation", "duplicate_effect", "state_generation", "attempt_receipt"])
def test_malformed_import_actual_cli_keeps_single_json_error(tmp_path: Path, mutation: str) -> None:
    package = malformed_package(tmp_path, mutation)
    raw = canonical({"kind": "import", "command_id": "malformed", "payload": {"package": package}})
    argv = [sys.executable, "-m", "sextant", "--root", str(tmp_path / "cli"), "agent"]
    first = subprocess.run(argv, input=raw, text=True, capture_output=True, timeout=10, check=False)
    second = subprocess.run(argv, input=raw, text=True, capture_output=True, timeout=10, check=False)
    assert first.returncode == 1 and not first.stderr and first.stdout == second.stdout
    assert json.loads(first.stdout)["errors"]


def test_import_database_identity_collision_is_structured(tmp_path: Path) -> None:
    sender, _ = fresh(tmp_path / "sender")
    package = call(sender, "export", start(sender)).outputs
    receiver, _ = fresh(tmp_path / "receiver")
    existing = start(receiver)
    package["events"][0]["event_id"] = existing.events[0]["event_id"]
    package["content_hash"] = digest({k: v for k, v in package.items() if k != "content_hash"})
    command = Command(kind="import", command_id="collision", payload={"package": package})
    result = receiver.execute(command)
    assert result.status == "import_structure" and result == receiver.execute(command)
    assert len(call(receiver, "inspect").outputs["missions"]) == 1


@pytest.mark.parametrize("field", ["mission_ref", "authorization_ref", "context_ref", "proposal_ref", "policy_ref"])
def test_import_missing_receiving_reference_is_rejected_atomically(tmp_path: Path, field: str) -> None:
    sender, _ = fresh(tmp_path / "sender")
    package = call(sender, "export", start(sender, mission={"goal": "calculate 2 + 3"})).outputs
    package["mission"].pop(field)
    package["content_hash"] = digest({k: v for k, v in package.items() if k != "content_hash"})
    receiver, _ = fresh(tmp_path / "receiver")
    command = Command(kind="import", command_id="missing-reference", payload={"package": package})
    result = receiver.execute(command)
    assert result.status == "import_state" and field in result.errors[0]["message"]
    assert receiver.execute(command) == result
    with receiver.store.connection(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM missions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM commands WHERE result IS NOT NULL").fetchone()[0] == 1


def test_import_explicit_null_references_can_authorize_and_run(tmp_path: Path) -> None:
    sender, _ = fresh(tmp_path / "sender")
    package = call(sender, "export", start(sender, mission={"goal": "calculate 2 + 3"})).outputs
    assert all(package["mission"][key] is None for key in ("policy_ref", "proposal_ref", "context_ref"))
    receiver, _ = fresh(tmp_path / "receiver")
    imported = call(receiver, "import", package=package)
    assert imported.status == "needs_authority"
    active = call(receiver, "authorize", imported, action="update", authorization={"status": "active"})
    done = call(receiver, "run", active)
    assert done.status == "complete" and done.outputs["steps"][0]["direct_result"]["value"] == 5
    assert call(receiver, "replay", done).outputs["matches_current"]


def test_blind_challenger_does_not_receive_prior_frozen_source(tmp_path: Path) -> None:
    generated = proposal()
    canary = "prior-program-only-canary-7ac9"
    generated["candidates"][0]["representation"] = {"kind": "program", "body": {"program": f"# {canary}\nprint(1)"}, "composition_rule": "program"}
    app, generator = fresh(tmp_path, generated)
    first = call(app, "run", start(app, options={"challenger": True, "rerepresentation": False}))
    assert not first.errors
    observed = call(app, "observe", first, content="new task evidence", source="fixture")
    second = call(app, "run", observed)
    assert not second.errors
    primary, challenger, synthesis = generator.requests[-3:]
    assert canary in primary["prompt"] and canary in synthesis["prompt"]
    assert canary not in challenger["prompt"]
    assert "frozen_structure" not in json.loads(challenger["prompt"])["representation_treatment"]


def strict_projection() -> SubmissionProjection:
    spec = projection()
    expr = spec.model_schema["$defs"]["expr"]["oneOf"]
    operation = expr[-1]
    # Alternative arities reproduce canonical args-before-op recursion blowup.
    unary = copy.deepcopy(operation)
    unary["properties"]["op"] = {"const": "not"}
    unary["properties"]["args"].update(minItems=1, maxItems=1)
    ternary = copy.deepcopy(operation)
    ternary["properties"]["op"] = {"const": "if"}
    ternary["properties"]["args"].update(minItems=3, maxItems=3)
    expr.extend([unary, ternary])
    table = spec.model_schema["oneOf"][1]["properties"]["payload"]["properties"]
    table["rows"]["items"]["properties"]["inputs"] = {"type": "object", "additionalProperties": {"type": "integer"}}
    return SubmissionProjection.model_validate(json.loads(canonical(spec)))


@pytest.mark.parametrize("variant", ["expression", "table_input", "table_output", "duplicate_numeric"])
def test_integral_float_cannot_establish_representation_freeze(tmp_path: Path, variant: str) -> None:
    generated = proposal()
    model = {"kind": "table", "payload": {"variables": ["x"], "rows": [{"inputs": {"x": 0}, "output": 1}]}}
    if variant == "expression":
        model = {"kind": "expression", "payload": 1.0}
    elif variant == "table_input":
        model["payload"]["rows"][0]["inputs"]["x"] = 0.0
    elif variant == "table_output":
        model["payload"]["rows"][0]["output"] = 1.0
    else:
        model["payload"]["rows"].append({"inputs": {"x": 0.0}, "output": 0})
    generated["direct_result"] = {"prediction_model": model}
    app, _ = fresh(tmp_path, generated)
    result = call(app, "run", start(app, options={"rerepresentation": False, "submission_projection": strict_projection().model_dump()}))
    inspected = call(app, "inspect", result, events=True)
    assert not result.errors and inspected.outputs["state"]["representation_state"]["external"] is None
    assert any(e["kind"] == "representation.checked" and e["payload"]["external"]["status"] == "invalid_unfrozen" for e in inspected.events)


def test_arithmetic_route_obeys_frozen_treatment_and_explicit_reset(tmp_path: Path) -> None:
    app, _ = fresh(tmp_path)
    initial = call(app, "run", start(app, options={"rerepresentation": False}))
    state = call(app, "inspect", initial).outputs["state"]["representation_state"]
    revised = call(app, "revise", initial, mission_input={"goal": "calculate 2 + 3"})
    rejected = call(app, "run", revised)
    assert rejected.status == "representation_change"
    assert call(app, "inspect", rejected).outputs["state"]["representation_state"] == state
    reset = call(app, "revise", rejected, options={"rerepresentation": True})
    done = call(app, "run", reset)
    assert done.status == "complete" and done.outputs["steps"][0]["direct_result"]["value"] == 5
    events = call(app, "inspect", done, events=True).events
    assert any(e["kind"] == "representation.treatment_changed" for e in events)
    assert sum(e["kind"] == "representation.checked" for e in events) == 2


@pytest.mark.parametrize("depth", [12, 32, 33])
def test_canonical_recursive_schema_application_is_bounded(tmp_path: Path, depth: int) -> None:
    node: Any = True
    for _ in range(depth):
        node = {"op": "not", "args": [node]}
    generated = proposal()
    generated["direct_result"] = {"prediction_model": {"kind": "expression", "payload": node}}
    # A child deadline prevents a regression from hanging the suite. It executes
    # actual Application storage/canonicalization and command persistence.
    program = """
import json, sys
from pathlib import Path
from test_runtime_integrity_v1 import fresh, start, call
data = json.loads(sys.stdin.read())
app, _ = fresh(Path(data['root']), data['proposal'])
created = start(app, options={'rerepresentation': False, 'submission_projection': data['projection']})
result = call(app, 'run', created)
snapshot = call(app, 'inspect', result, events=True)
print(json.dumps({'result':result.model_dump(), 'state':snapshot.outputs['state'], 'operations':snapshot.outputs['operations']}))
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    completed = subprocess.run([sys.executable, "-c", program], input=canonical({"root": str(tmp_path), "proposal": generated,
                                "projection": strict_projection().model_dump()}), text=True, capture_output=True, timeout=5, env=env, check=False)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert not result["result"]["errors"]
    assert (result["state"]["representation_state"]["external"] is None) == (depth > 32)
    assert all(o["state"] != "running" for o in result["operations"])


@pytest.mark.parametrize("schema,valid", [({"oneOf": [True, False]}, True), ({"oneOf": [True, True]}, False),
                                       ({"oneOf": [False, False]}, False), ({"$ref": "#"}, False)])
def test_bounded_schema_retains_exact_one_semantics(schema: dict[str, Any], valid: bool) -> None:
    spec = SubmissionProjection(path=["model"], model_schema=schema)
    value = {"model": {"kind": "expression", "payload": 1}}
    if valid:
        assert submitted_shape(value, spec) is not None
    else:
        with pytest.raises(DomainError):
            submitted_shape(value, spec)


def test_schema_step_budget_fails_closed_and_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    import sextant.representation_control as module
    monkeypatch.setattr(module, "MAX_SCHEMA_STEPS", 10)
    spec = SubmissionProjection(path=["model"], model_schema={"oneOf": [False] * 20 + [True]})
    value = {"model": {"kind": "expression", "payload": 1}}
    with pytest.raises(DomainError, match="validation steps"):
        submitted_shape(value, spec)
    assert submitted_shape(value, SubmissionProjection(path=["model"], model_schema={})) is not None
