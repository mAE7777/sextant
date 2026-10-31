"""Regression counterexamples for decision analysis and failure reporting; no live providers.

Each test reproduces an observed defect of an earlier build: hard constraints ignored by
decision analysis, damaged input evidence silently dropped, inability to run recorded as an
unknown outcome, a misstated native blocker, withdrawal invalidating independent branches,
and "exact" arithmetic computed in binary floating point.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from test_runtime_integrity_v1 import OWNER, Capture, call, experiment, fresh, proposal, start

from sextant.application import Application
from sextant.decision import compare, measured_violations
from sextant.domain import Receipt, digest, utcnow
from sextant.evaluators import TypeSafeExecutor
from sextant.providers.native import ClaudeHeadless
from sextant.render import chinese
from sextant.representations import arithmetic
from sextant.runtime import AttemptContext, Invocation, NotSent
from sextant.storage import Store

CORRECT_PRICES = {"id": "correct-prices", "predicate": {"op": "eq", "args": [{"var": "mismatches"}, 0]}}
FEWER_CALLS = [{"metric": "origin_calls", "direction": "min", "unit": "calls"}]
ALL_TOOLS = ["claude-code", "environment", "environment.apply", "source.file"]


def option(calls: int, mismatches: int = 0) -> dict[str, Any]:
    return {"mixed": {"origin_calls": {"value": calls, "unit": "calls"}, "mismatches": {"value": mismatches, "unit": "count"}}}


def branch(key: str, calls: int, *, experiments: list[Any] | None = None, evidence: list[str] | None = None) -> dict[str, Any]:
    return {"key": key, "mechanism": f"mechanism {key}", "assumptions": [], "evidence": evidence or [],
            "representation": {"kind": "table", "body": {"inputs": [], "rows": [{"input": {}, "output": key}]},
                               "composition_rule": "constant"},
            "experiments": experiments or [], "scenario_metrics": option(calls)}


def objects(app: Application, prior: Any) -> dict[str, Any]:
    return call(app, "inspect", prior).outputs


# ------------------------------------------------------------------ hard constraints first


def test_option_violating_a_hard_constraint_is_infeasible_before_dominance() -> None:
    result = compare({"a": option(2, mismatches=2), "b": option(4)}, FEWER_CALLS, hard_constraints=[CORRECT_PRICES])
    mixed = result["scenarios"]["mixed"]
    # Before the repair "a" (fewer calls) was the only nondominated option.
    assert mixed["nondominated"] == ["b"]
    assert mixed["infeasible"] == {"a": ["correct-prices (stated values)"]} and "a" not in mixed["dominated_by"]


def test_unevaluable_hard_constraint_is_reported_never_passed() -> None:
    result = compare({"a": option(2)}, FEWER_CALLS, hard_constraints=[{"id": "legal", "text": "must be legal"}])
    assert result["scenarios"]["mixed"]["constraints_unverified"] == {"a": ["legal"]}
    assert any("could not be evaluated" in loss for loss in result["loss"])


def test_infeasible_option_gets_no_cardinal_score() -> None:
    mapping = {"kind": "affine-cardinal/1", "metrics": {"origin_calls": {"unit": "calls", "weight": 1, "scale": 1, "offset": 0}}}
    result = compare({"a": option(2, mismatches=1), "b": option(4)}, FEWER_CALLS, utility_mapping=mapping,
                     hard_constraints=[CORRECT_PRICES])
    assert result["scores_by_scenario"]["a"]["mixed"] is None and result["scores_by_scenario"]["b"]["mixed"] == 4


def test_measured_violation_comes_only_from_the_branch_own_receipt_backed_result() -> None:
    records = [{"branch": "a", "experiment": {"key": "run-sku"}, "observations": [{"object_id": "o1", "revision_id": "r1"}]},
               {"branch": "b", "experiment": {"key": "run-region"}, "observations": [{"object_id": "o2", "revision_id": "r2"}]},
               {"branch": "policy", "experiment": {"key": "shared"}, "observations": [{"object_id": "o3", "revision_id": "r3"}]}]
    current = {("o1", "r1"): {"status": "active", "provenance": "environment", "content": {"mismatches": 2}},
               ("o2", "r2"): {"status": "active", "provenance": "declared", "content": {"mismatches": 5}},
               ("o3", "r3"): {"status": "active", "provenance": "constructed", "content": {"mismatches": 9}}}
    measured = measured_violations([CORRECT_PRICES], records, current)
    assert list(measured) == ["a"] and measured["a"][0]["experiment"] == "run-sku"
    result = compare({"a": option(2), "b": option(4)}, FEWER_CALLS, hard_constraints=[CORRECT_PRICES], measured_violations=measured)
    assert result["scenarios"]["mixed"]["nondominated"] == ["b"]
    assert "measured by experiment run-sku" in result["scenarios"]["mixed"]["infeasible"]["a"][0]


def test_run_excludes_candidate_whose_own_experiment_measured_a_violation(tmp_path: Path) -> None:
    env = Capture("environment", {"observations": [{"observation_id": "sku-run", "source_id": "service",
        "content": {"mismatches": 2}, "observed_at": utcnow(), "measurement_protocol": "fixture/1", "supports_scope": {"phase": 0}}]})
    generated = proposal()
    generated["candidates"] = [branch("a", 2, experiments=[experiment(mode="sku")]), branch("b", 4)]
    app, _ = fresh(tmp_path, generated, env)
    created = start(app, mission={"hard_constraints": [CORRECT_PRICES], "objective_metrics": FEWER_CALLS})
    first = call(app, "run", created, steps=1)
    analysis = objects(app, first)["objects"]
    policy = next(o["revision"]["payload"] for o in analysis if o["revision"]["ref"] == objects(app, first)["state"]["policy_ref"])
    assert policy["decision_analysis"]["scenarios"]["mixed"]["nondominated"] == ["a"]  # stated values only so far
    done = call(app, "run", first, steps=2)  # the experiment, then a recompilation
    assert not done.errors, done
    snapshot = objects(app, done)
    policy = next(o["revision"]["payload"] for o in snapshot["objects"] if o["revision"]["ref"] == snapshot["state"]["policy_ref"])
    mixed = policy["decision_analysis"]["scenarios"]["mixed"]
    assert mixed["nondominated"] == ["b"] and "measured by experiment measure" in mixed["infeasible"]["a"][0]
    assert any("Candidate a is excluded as infeasible" in reason for reason in policy["unresolved"])


# ------------------------------------------------------------------ damaged input evidence


def measured_mission(tmp_path: Path) -> tuple[Application, Capture, Any, Path]:
    env = Capture("environment", {"observations": [{"observation_id": "m1", "source_id": "sensor", "content": {"reading": 4},
        "observed_at": utcnow(), "measurement_protocol": "fixture/1", "supports_scope": {"phase": 0}}]})
    app, generator = fresh(tmp_path, proposal([experiment()]), env)
    measured = call(app, "run", start(app), steps=2)
    assert not measured.errors, measured
    events = call(app, "inspect", measured, events=True).events
    sent = next(e for e in events if e["kind"] == "effect.sent" and e["payload"]["executor"] == "environment")
    received = next(e for e in events if e["kind"] == "effect.received" and e["attempt_id"] == sent["attempt_id"])
    return app, generator, measured, app.store.artifact_path(received["payload"]["artifact"]["sha256"])


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_damaged_input_evidence_blocks_run_instead_of_being_dropped(tmp_path: Path, damage: str) -> None:
    app, generator, measured, path = measured_mission(tmp_path)
    original = path.read_bytes()
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"damaged receipt")
    sent_before = len(generator.requests)
    blocked = call(app, "run", measured, steps=1)
    assert blocked.status == "damaged_evidence" and len(generator.requests) == sent_before
    assert "restore the original artifact bytes" in blocked.errors[0]["message"].lower()
    state = objects(app, blocked)["state"]
    assert state["status"] == "blocked" and state["blocker"]["code"] == "damaged_evidence"
    assert blocked.current_revision == state["revision"]
    path.write_bytes(original)  # restoring the original bytes is a repair
    resumed = call(app, "run", blocked, steps=1)
    assert not resumed.errors and resumed.outputs["steps"][0]["kind"] == "compile"
    assert "blocker" not in objects(app, resumed)["state"]


def test_restoring_evidence_of_a_settled_mission_lifts_the_block(tmp_path: Path) -> None:
    app, generator, measured, path = measured_mission(tmp_path)
    settled = call(app, "run", measured, steps=1)
    assert not settled.errors and settled.outputs["steps"][0]["kind"] == "compile"
    idle = call(app, "run", settled, steps=1)
    assert idle.status == "waiting" and idle.outputs["steps"] == []
    status_before = objects(app, idle)["state"]["status"]
    original = path.read_bytes()
    path.write_bytes(b"damaged receipt")
    blocked = call(app, "run", idle, steps=1)
    assert blocked.status == "damaged_evidence"
    assert objects(app, blocked)["state"]["status"] == "blocked"
    path.write_bytes(original)
    sent_before = len(generator.requests)
    restored = call(app, "run", blocked, steps=1)
    assert not restored.errors and restored.outputs["steps"] == [] and len(generator.requests) == sent_before
    state = objects(app, restored)["state"]
    assert "blocker" not in state and state["status"] == status_before
    assert restored.current_revision == state["revision"]
    assert "受阻" not in chinese(call(app, "inspect", restored))
    again = call(app, "run", restored, steps=1)
    assert again.status == "waiting" and again.current_revision == restored.current_revision


def test_damaged_observation_can_be_withdrawn_to_continue_without_it(tmp_path: Path) -> None:
    app, _, measured, path = measured_mission(tmp_path)
    path.write_bytes(b"damaged receipt")
    blocked = call(app, "run", measured, steps=1)
    snapshot = objects(app, blocked)
    damaged = next(o["revision"]["ref"] for o in snapshot["objects"] if o["revision"]["ref"]["object_kind"] == "observation"
                   and any(r.startswith("corrupt_artifact") for r in o["current_validity"]["reasons"]))
    withdrawn = call(app, "withdraw", blocked, observation_ref=damaged, reason="receipt bytes were damaged")
    assert not withdrawn.errors, withdrawn
    revision = next(o["revision"] for o in objects(app, withdrawn)["objects"] if o["revision"]["ref"]["object_id"] == damaged["object_id"])
    assert revision["payload"]["status"] == "withdrawn" and "original artifacts damaged" in revision["payload"]["correction_reason"]
    resumed = call(app, "run", withdrawn, steps=1)
    assert not resumed.errors and resumed.outputs["steps"][0]["kind"] == "compile"


# ------------------------------------------------------------------ inability to run is not an unknown outcome


def test_missing_typesafe_key_marks_diagnostics_unavailable_and_sends_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    app, _ = fresh(tmp_path)
    app.executors["typesafe"] = TypeSafeExecutor()
    created = start(app, authorization={"tools": [*ALL_TOOLS, "typesafe"], "send_to": {"claude-code": ["*"], "typesafe": ["*"]}},
                    options={"jev_density": "final_only"})
    done = call(app, "run", created, steps=1)
    assert not done.errors and done.status != "reconciliation_required", done
    snapshot = objects(app, done)
    typed = [r for r in snapshot["saved_receipts"] if r["executor"] == "typesafe"]
    assert len(typed) == 1 and typed[0]["state"] == "failed"  # later cards were not sent
    assert "TYPESAFE_API_KEY" in typed[0]["receipt"]["errors"][0] and "no external request was sent" in typed[0]["receipt"]["errors"][0]
    assert {o["state"] for o in snapshot["operations"]} == {"completed"}
    policy = next(o["revision"]["payload"] for o in snapshot["objects"] if o["revision"]["ref"] == snapshot["state"]["policy_ref"])
    assert sum("TYPESAFE_API_KEY" in reason for reason in policy["unresolved"]) == 4  # R6, R7, RM1, RM3


def test_out_of_scope_file_read_is_refused_with_its_cause(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("private")
    read = {"key": "read-outside", "adapter": "source.file", "action": "read", "inputs": {"path": str(outside)},
            "purpose": "read a file the grant does not cover"}
    app, _ = fresh(tmp_path / "runtime", proposal([read]))
    done = call(app, "run", start(app), steps=2)
    assert not done.errors and done.outputs["steps"][-1] == {"kind": "action_refused", "reason": "unauthorized_read", "effects_sent": 0}
    snapshot = objects(app, done)
    receipt = next(r for r in snapshot["saved_receipts"] if r["executor"] == "source.file")
    assert receipt["state"] == "failed" and "outside the authorized read scope" in receipt["receipt"]["errors"][0]
    assert "reconciliation_required" not in {o["state"] for o in snapshot["operations"]}


def test_sandbox_request_without_program_is_refused_not_unknown(tmp_path: Path) -> None:
    empty = {"key": "no-program", "adapter": "sandbox.python", "action": "execute", "inputs": {}, "purpose": "missing program"}
    app, _ = fresh(tmp_path, proposal([empty]))
    created = start(app, authorization={"tools": [*ALL_TOOLS, "sandbox.python"],
                                        "send_to": {"claude-code": ["*"], "sandbox.python": ["*"]}})
    done = call(app, "run", created, steps=2)
    assert not done.errors and done.outputs["steps"][-1]["reason"] == "program"


def test_unknown_outcome_keeps_its_local_cause(tmp_path: Path) -> None:
    class Disconnected(Capture):
        def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
            raise ConnectionResetError("connection reset by peer after the request was written")

    app, _ = fresh(tmp_path, proposal([experiment()]), Disconnected("environment", None))
    result = call(app, "run", start(app), steps=2)
    assert result.status == "reconciliation_required"
    receipt = next(r for r in objects(app, result)["saved_receipts"] if r["executor"] == "environment")
    assert receipt["state"] == "unknown" and "connection reset by peer" in receipt["receipt"]["errors"][0]
    assert "inspect local diagnostics" not in receipt["receipt"]["errors"][0]


# ------------------------------------------------------------------ the native blocker is named truthfully


def test_missing_native_cli_is_named_and_blocks_the_mission(tmp_path: Path) -> None:
    native = ClaudeHeadless()
    native.binary = None
    app = Application(Store(tmp_path), OWNER, executors={"claude-code": native})
    result = call(app, "run", start(app), steps=1)
    assert result.status == "runtime_unavailable"
    message = result.errors[0]["message"]
    assert "not found" in message and "isolation flags" not in message
    snapshot = objects(app, result)
    assert snapshot["state"]["status"] == "blocked" and snapshot["state"]["blocker"]["code"] == "runtime_unavailable"
    assert result.current_revision == snapshot["state"]["revision"]
    assert [o["state"] for o in snapshot["operations"]] == ["failed"]
    assert [(r["executor"], r["state"]) for r in snapshot["saved_receipts"]] == [("claude-code", "failed")]
    resumed = call(app, "resume", result)
    assert resumed.status == "runtime_unavailable" and "blocker" in objects(app, resumed)["state"]


def test_native_preflight_reports_every_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    native = ClaudeHeadless(binary="/usr/bin/true")
    monkeypatch.setattr(native, "capabilities", lambda: {"available": False, "missing_flags": ["--restricted"], "help_exit_code": 0})
    monkeypatch.setattr(native, "auth_status", lambda: {"loggedIn": False})
    with pytest.raises(NotSent) as error:
        native.invoke({"schema": {}, "system": "", "prompt": ""}, AttemptContext("o", "a", "m", 1, "e"))
    assert error.value.kind == "unavailable" and error.value.code == "runtime_unavailable"
    assert "--restricted" in str(error.value) and "login unavailable" in str(error.value)


# ------------------------------------------------------------------ withdrawal changes only what cited it


def cited_branches(request: dict[str, Any]) -> dict[str, Any]:
    """A generator whose branches and claims cite the observations currently supplied."""
    content = json.loads(request["prompt"])
    ids = {item["observation"]["source"]: item["ref"]["object_id"] for item in content["evidence"]}
    body = proposal()
    body["candidates"] = [branch("stale-cache", 2, evidence=[f"object:{ids['trace']}"])]
    body["evidence_links"] = {"the cache served a stale entry": [f"object:{ids['trace']}"]}
    if "ticket" in ids:
        body["candidates"].append(branch("regional-key", 4, evidence=[f"object:{ids['ticket']}"]))
        body["evidence_links"]["a regional price differs"] = [f"object:{ids['ticket']}"]
    return {"payload": copy.deepcopy(body)}


def test_withdrawing_one_source_changes_only_what_cited_it(tmp_path: Path) -> None:
    app, generator = fresh(tmp_path)
    generator.output = cited_branches
    created = start(app, observations=[{"content": "a regional price mismatch was reported", "source": "ticket"},
                                       {"content": "the cache served a stale entry", "source": "trace"}])
    compiled = call(app, "run", created, steps=1)
    first = objects(app, compiled)
    candidates = dict(first["state"]["candidates"])
    ticket = next(o["revision"]["ref"] for o in first["objects"] if o["revision"]["ref"]["object_kind"] == "observation"
                  and o["revision"]["payload"]["source"] == "ticket")
    withdrawn = call(app, "withdraw", compiled, observation_ref=ticket, reason="the ticket was mistaken")
    after = objects(app, withdrawn)
    valid = {o["revision"]["ref"]["object_id"]: o["current_validity"]["valid"] for o in after["objects"]}
    assert valid[candidates["stale-cache"]["object_id"]] and not valid[candidates["regional-key"]["object_id"]]
    claims = {o["revision"]["payload"]["proposition"]: o["current_validity"]["valid"]
              for o in after["objects"] if o["revision"]["ref"]["object_kind"] == "claim"}
    assert claims == {"the cache served a stale entry": True, "a regional price differs": False}
    assert not valid[after["state"]["policy_ref"]["object_id"]]  # the synthesized policy must be recompiled
    recompiled = call(app, "run", withdrawn, steps=1)
    assert not recompiled.errors, recompiled
    final = objects(app, recompiled)
    assert final["state"]["candidates"]["stale-cache"] == candidates["stale-cache"]  # same revision kept
    valid = {o["revision"]["ref"]["object_id"]: o["current_validity"] for o in final["objects"]}
    assert valid[candidates["stale-cache"]["object_id"]]["valid"]
    assert not valid[candidates["regional-key"]["object_id"]]["valid"]


def test_branch_without_cited_evidence_still_depends_on_the_whole_context(tmp_path: Path) -> None:
    app, _ = fresh(tmp_path)
    created = start(app, observations=[{"content": "one", "source": "first"}, {"content": "two", "source": "second"}])
    compiled = call(app, "run", created, steps=1)
    first = objects(app, compiled)
    candidate = first["state"]["candidates"]["candidate"]
    observation = next(o["revision"]["ref"] for o in first["objects"] if o["revision"]["ref"]["object_kind"] == "observation")
    withdrawn = call(app, "withdraw", compiled, observation_ref=observation, reason="either one")
    valid = {o["revision"]["ref"]["object_id"]: o["current_validity"]["valid"] for o in objects(app, withdrawn)["objects"]}
    assert not valid[candidate["object_id"]]


# ------------------------------------------------------------------ exact arithmetic


@pytest.mark.parametrize("goal,value,exact", [
    ("compute 0.1 + 0.2", 0.3, (3, 10)),
    ("compute 1 / 3", "1/3", (1, 3)),
    ("计算 7 * (3 + 2)", 35, (35, 1)),
    ("compute 1e-3 * 1000", 1, (1, 1)),
    ("compute 7 // 2 + 7 % 2", 4, (4, 1)),
    ("compute 3.5 / 2", 1.75, (7, 4)),
    ("compute 1_000.5 - 0.5", 1000, (1000, 1)),
    ("compute -0.25", -0.25, (-1, 4)),
])
def test_arithmetic_result_is_exact(goal: str, value: Any, exact: tuple[int, int]) -> None:
    result = arithmetic(goal)
    assert result is not None and result["value"] == value and type(result["value"]) is type(value)
    assert (result["exact"]["numerator"], result["exact"]["denominator"]) == exact
    assert result["protocol"] == "bounded-arithmetic/2"


@pytest.mark.parametrize("goal", ["compute 1 / 0", "compute 10**3", "compute True + 1", f"compute {'9' * 102}"])
def test_arithmetic_outside_its_language_is_not_solved_directly(goal: str) -> None:
    assert arithmetic(goal) is None


def test_arithmetic_certificate_states_the_exact_result(tmp_path: Path) -> None:
    app, generator = fresh(tmp_path)
    created = call(app, "start", mission={"goal": "compute 0.1 + 0.2"}, authorization={"tools": []})
    done = call(app, "run", created, steps=1)
    assert done.status == "complete" and generator.requests == []
    certificate = next(o["revision"]["payload"] for o in objects(app, done)["objects"] if o["revision"]["ref"]["object_kind"] == "certificate")
    assert certificate["protocol"] == "bounded-arithmetic/2"
    assert certificate["result"]["value"] == 0.3 and certificate["result"]["exact"] == {"numerator": 3, "denominator": 10}


def test_receipt_contract_accepts_not_sent_fields() -> None:
    # The not-sent path builds its receipt from executor-supplied fields; they must be real Receipt fields.
    fields = {"requested_model": "opus", "requested_effort": "high", "auth_route": "native-claude.ai-subscription",
              "auth_evidence": "{}"}
    Receipt(executor="claude-code", started_at=utcnow(), ended_at=utcnow(), request_hash=digest({}), output_hash=digest(None),
            status="failed", **fields)
