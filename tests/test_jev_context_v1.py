"""Bounded context, counterevidence/withdrawal and actual consumer controls."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest
from test_application_v1 import OWNER, call
from test_jev_compact_state import response
from test_runtime_integrity_v1 import Capture, proposal

from sextant import wire
from sextant.application import Application
from sextant.domain import (
    ObjectRef,
    Observation,
    Receipt,
    Revision,
    Scope,
    canonical,
    digest,
    utcnow,
)
from sextant.evaluators import RuntimeJevClient
from sextant.jev_client import JevClientError
from sextant.jev_context import select_context
from sextant.runtime import Invocation
from sextant.storage import Store


def observation(name, topic, content, *, source=None, dependencies=(), status="active", supersedes=()):
    payload = Observation(content=content, source=source or name, source_group=source or name,
                          supports_scope=topic, observed_at="2026-10-17T00:00:00+00:00",
                          measurement_protocol="public-trace/1", status=status,
                          supersedes=list(supersedes)).model_dump()
    ref = ObjectRef(object_id=name, revision_id=digest(payload), object_kind="observation")
    return Revision(ref=ref, mission_id="public_mission", scope=Scope(), payload=payload, dependencies=list(dependencies))


def pack(candidate=None):
    return {"task": {"goal": "Find the supported current branch; preserve other component history", "hard_constraints": []},
            "evidence": [], "proposal": candidate or {"goal": "public diagnostic"},
            "evaluation": {"authority": "advisory"}, "access": {"can_verify_world": False}}


def selected(view):
    return {r.object_id for r in view.refs}


def test_structural_closure_preserves_contrary_joint_and_shared_source_inputs():
    left = observation("left", "a-transaction", {"bit": 0})
    right = observation("right", "z-input", {"bit": 1}, dependencies=[left.ref])
    contrary = observation("contrary", "z-counter", {"result": False}, dependencies=[right.ref], source="shared")
    source_peer = observation("source_peer", "z-source", {"interpretation": "same source"}, source="shared")
    unrelated = observation("unrelated", "z-load", {"requests": list(range(2000))})
    rows = [left, right, contrary, source_peer, unrelated]
    original = deepcopy(rows)
    view = select_context("R3", pack({"evidence": ["object:left"]}), rows, bounded=True)
    assert selected(view) == {"left", "right", "contrary", "source_peer"}
    assert view.manifest["omitted"][0]["ref"] == unrelated.ref.model_dump()
    assert rows == original and "Only this declared diagnostic component" in view.manifest["loss"][0]
    assert not view.manifest["judgment_scope"]["whole_mission_completeness"]
    reordered = select_context("R3", pack({"evidence": ["object:left"]}), list(reversed(rows)), bounded=True)
    assert canonical(view.pack) == canonical(reordered.pack) and view.manifest == reordered.manifest


def test_negative_evidence_in_same_scope_and_unscoped_evidence_are_never_rank_pruned():
    support = observation("support", "a-policy", True)
    negative = observation("negative", "a-policy", False)
    unknown = observation("uncategorized", "unknown", {"unexpected": "adverse"})
    other = observation("other", "z-other", True)
    view = select_context("R1", pack(), [other, unknown, negative, support], bounded=True)
    assert selected(view) == {"support", "negative", "uncategorized"}


def test_withdrawn_source_is_a_status_header_not_current_factual_support():
    old = observation("old", "a-policy", {"claim": "retracted"})
    withdrawn = observation("withdrawn", "a-policy", old.payload["content"], status="withdrawn", supersedes=[old.ref])
    current = observation("counter", "a-policy", {"result": "opposite"})
    view = select_context("R7", pack({"evidence": [old.ref.model_dump()]}), [current, withdrawn], bounded=True)
    assert not view.failure and selected(view) == {"withdrawn", "counter"}
    header = next(r["observation"] for r in view.pack["evidence"] if r["ref"] == withdrawn.ref.model_dump())
    assert header["status"] == "withdrawn" and "content" not in header and header["supersedes"] == [old.ref.model_dump()]
    assert next(r for r in view.manifest["selected"] if r["ref"] == withdrawn.ref.model_dump())["payload_sha256"] == digest(withdrawn.payload)


def test_missing_or_ambiguous_explicit_dependencies_withhold_diagnostic():
    row = observation("one", "a-policy", {})
    assert select_context("R3", pack({"evidence": ["object:missing"]}), [row], bounded=True).failure
    second = observation("two", "b-policy", {})
    for item in (row, second):
        item.payload["external_observation_id"] = "same-id"
    assert select_context("R3", pack({"evidence": ["external:same-id"]}), [row, second], bounded=True).failure


def test_full_and_bounded_views_check_required_joint_inputs_and_task_roots():
    required = observation("guard", "z-safety", {"permitted": False})
    target = observation("target", "a-policy", {"value": 1})
    task = pack({"evidence": ["object:target"]})
    task["task"]["hard_constraints"] = [{"required_evidence": "object:guard"}]
    view = select_context("R7", task, [target, required], bounded=True)
    assert not view.failure and selected(view) == {"guard", "target"}
    target.payload["content"]["required_joint_input"] = "object:missing"
    for bounded in (False, True):
        view = select_context("R7", task, [target, required], bounded=bounded)
        assert view.failure and "object:missing" in view.manifest["unresolved_references"]
    target.payload["content"].pop("required_joint_input")
    invalid = select_context("R7", task, [target, required], bounded=False,
                             validity={"target": {"valid": False}})
    assert invalid.failure and "invalid-current-observation:target" in invalid.manifest["unresolved_references"]


def test_historical_aliases_are_unambiguous_withdrawal_not_active_rebinding():
    old = observation("old", "a-policy", {"bit": 1})
    current = observation("old", "a-policy", {"bit": 0})
    current.predecessor = old.ref
    assert select_context("R3", pack({"evidence": [old.ref.model_dump()]}), [current], bounded=False).failure
    one = observation("one", "a-policy", {}, status="withdrawn", supersedes=[old.ref])
    two = observation("two", "a-policy", {}, status="withdrawn", supersedes=[old.ref])
    assert select_context("R3", pack({"evidence": [old.ref.model_dump()]}), [one, two], bounded=True).failure
    view = select_context("R3", pack({"evidence": [old.ref.model_dump()]}), [one], bounded=False)
    assert not view.failure
    item = view.manifest["selected"][0]
    assert canonical(old.ref) in item["reference_aliases"] and "withdrawal attribution only" in item["historical_alias_scope"]


def test_required_joint_closure_over_capacity_is_not_truncated_or_dispatched():
    row = observation("joint", "a-policy", {"actual_public_measurements": list(range(60000))})
    view = select_context("R3", pack({"evidence": ["object:joint"]}), [row], bounded=True)
    client = RuntimeJevClient(None, None, [])
    card = client.cards["R3"]
    questions = client.render(card, view.pack)
    assert not client.capacity(view.pack, {r.key: r.sdk for r in questions})["fits"]
    with pytest.raises(JevClientError, match="exceed provider capacity"):
        client._call(card, view.pack, questions, variant="base", role="primary", run_id="public", input_language="en")
    assert not client.rows


def test_large_history_local_view_binds_exact_question_scope_and_omission_manifest():
    target = observation("target", "a-policy", {"predicate": False})
    archive = [observation("trace_" + str(n), "z-component-" + str(n),
                           {"requests": [{"request": i, "key": i % 17, "latency_ms": (i * (n + 1)) % 91}
                                         for i in range(140)]}) for n in range(30)]
    rows = [target, *archive]
    client = RuntimeJevClient(None, None, [])
    original = select_context("R2", pack({"context": {"selected": [r.ref.model_dump() for r in rows]}}), rows, bounded=False)
    assert not client.capacity(original.pack, {r.key: r.sdk for r in client.render(client.cards["R2"], original.pack)})["fits"]
    view = select_context("R2", pack({"context": {"selected": [r.ref.model_dump() for r in rows]}}), rows, bounded=True)
    questions = client.render(client.cards["R2"], view.pack)
    assert client.capacity(view.pack, {r.key: r.sdk for r in questions})["fits"]
    assert selected(view) == {"target"} and len(view.manifest["omitted"]) == 30
    assert all("Omitted material has NOT been assessed" in r.sdk["instructions"] for r in questions)
    assert all(r.rendered_hash == wire.rendered_hash("R2", r.key, r.primitive.value, r.sdk) for r in questions)
    assert len(view.pack["proposal"]["context"]["selected"]) == 1


class Advisory(Capture):
    def __init__(self):
        super().__init__("typesafe", None)

    def invoke(self, request, context):
        self.requests.append(deepcopy(request))
        result = response(request["questions"])
        output = {"model_id": result.model_id, "answers": result.answers, "usage": result.usage,
                  "latency_ms": result.latency_ms, "request_id": result.request_id, "transport": "controlled"}
        return Invocation(Receipt(executor=self.name, request_hash=digest(request), output_hash=digest(output),
                                  started_at=utcnow(), ended_at=utcnow(), status="completed"), output)


def test_real_application_preserves_loss_manifest_in_export_and_downstream_consumer(tmp_path):
    generated = proposal()
    generated["policy"]["basis"] = "Controlled consumer rejects advisory authority; observed contradictory evidence takes precedence."
    generator, advisory = Capture("claude-code", {"payload": generated}), Advisory()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator, "typesafe": advisory})
    started = call(app, "start", mission={"goal": "Select current predicate within public component scopes"},
                   options={"jev_density": "context_local", "challenger": False},
                   authorization={"tools": ["claude-code", "typesafe"], "send_to": {"claude-code": ["*"], "typesafe": ["*"]}},
                   observations=[{"content": {"bit": 1}, "source": "public-receiver", "supports_scope": "predicate"}])
    done = call(app, "run", started, steps=1)
    assert not done.errors, done
    assert advisory.requests
    synthesis = next(json.loads(r["prompt"]) for r in generator.requests if r["purpose"] == "synthesis")
    diagnostics = synthesis["advisory_diagnostics"]
    assert len(diagnostics) == 2 and all(d["context_selection"]["loss"] for d in diagnostics)
    exported = call(app, "export", done).outputs
    for diagnostic in diagnostics:
        manifest = diagnostic["context_selection"]["manifest_artifact"]
        assert manifest["sha256"] in exported["artifacts"]
        # The same immutable artifact is owned by the saved judgment/context.
        rows = [r for r in exported["revisions"] if r["ref"]["object_kind"] == "judgment"]
        assert any(manifest in row["artifacts"] for row in rows)
    assert all("context_selection" in request["state"]["evaluation"] for request in advisory.requests)


def test_wrong_sufficiency_advice_cannot_replace_missing_measured_effect_qualification(tmp_path):
    class WrongAdvice(Advisory):
        def invoke(self, request, context):
            result = super().invoke(request, context)
            output = deepcopy(result.output)
            for answer in output["answers"].values():
                if answer["type"] == "noul":
                    answer["noul"] = 1.0
            result.receipt.output_hash = digest(output)
            return Invocation(result.receipt, output)

    proposed = proposal()
    proposed["policy"]["next_experiments"] = [{"key": "unqualified-apply", "adapter": "environment",
        "action": "apply", "inputs": {"candidate": "bad-advice"}, "purpose": "advisory claims ready"}]
    generator, advisory = Capture("claude-code", {"payload": proposed}), WrongAdvice()
    environment = Capture("environment", {"observations": []})
    app = Application(Store(tmp_path / "runtime"), OWNER,
                      executors={"claude-code": generator, "typesafe": advisory, "environment": environment})
    started = call(app, "start", mission={"goal": "Apply only after measured qualification"},
                   options={"jev_density": "final_only", "challenger": False,
                            "environment": {"actions": {"apply": {}}, "effect_gate": {
                                "measurement_protocol": "public-qualified/1", "required_scenarios": ["recovery"],
                                "scenario_path": ["scenario"], "constraints": []}}},
                   authorization={"tools": ["claude-code", "typesafe", "environment", "environment.apply"],
                                  "send_to": {name: ["*"] for name in ("claude-code", "typesafe", "environment")}})
    done = call(app, "run", started, steps=2)
    assert not environment.requests
    inspected = call(app, "inspect", done).outputs
    judgments = [r["revision"]["payload"] for r in inspected["objects"]
                 if r["revision"]["ref"]["object_kind"] == "judgment"]
    assert any(j["card_id"] == "R7" and j["derived"] for j in judgments)
    assert any(record.get("reason", "").find("effect_evidence_required") >= 0
               or record.get("code") == "effect_evidence_required"
               for record in inspected["state"].get("refused_experiment_records", [])), inspected["state"]
