"""Executable user-path counterexamples for the October 2 V1 completion work.

Providers are controlled fixtures. No native model, Jev, or external service is called.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
from typing import Any

import pytest
from test_runtime_integrity_v1 import Capture, call, experiment, fresh, proposal, start

from sextant.domain import Command, ObjectRef, Prediction, digest, identity, utcnow


def snapshot(app: Any, result: Any) -> dict[str, Any]:
    return call(app, "inspect", result).outputs


@pytest.mark.parametrize("stop", ["pause", "cancel"])
@pytest.mark.parametrize("change", ["observe", "revise", "authorize"])
def test_input_changes_do_not_resume_stopped_mission(tmp_path: Path, stop: str, change: str) -> None:
    app, generator = fresh(tmp_path)
    stopped = call(app, stop, start(app))
    payload = {"observe": {"content": {"new": 1}}, "revise": {"mission_input": {"goal": "Updated goal"}},
               "authorize": {"action": "update", "authorization": {"status": "active"}}}[change]
    changed = call(app, change, stopped, **payload)
    assert not changed.errors
    expected = "paused" if stop == "pause" else "cancelled"
    assert changed.status == snapshot(app, changed)["state"]["status"] == expected
    rejected = call(app, "run", changed)
    assert rejected.status == expected and generator.requests == []
    resumed = call(app, "resume", rejected, steps=1)
    assert not resumed.errors and len(generator.requests) == 2


@pytest.mark.parametrize("stop", ["pause", "cancel"])
def test_stopped_completed_mission_cannot_take_idle_shortcut(tmp_path: Path, stop: str) -> None:
    app, generator = fresh(tmp_path)
    completed = call(app, "run", start(app, mission={"goal": "Compute 1 + 1"}), steps=1)
    assert snapshot(app, completed)["state"]["status"] == "complete"
    stopped = call(app, stop, completed)
    result = call(app, "run", stopped, steps=1)
    assert result.status == ("paused" if stop == "pause" else "cancelled")
    assert result.errors and result.outputs["steps"] == [] and generator.requests == []


def test_compile_then_idle_reports_calls_for_whole_command(tmp_path: Path) -> None:
    app, generator = fresh(tmp_path)
    result = call(app, "run", start(app, options={"challenger": True}), steps=3)
    assert not result.errors and len(generator.requests) == 3
    assert result.outputs["external_calls"] == 3
    idle = call(app, "run", result, steps=1)
    assert idle.outputs["external_calls"] == 0 and len(generator.requests) == 3


def test_challenger_receives_measurement_without_generator_experiment_label(tmp_path: Path) -> None:
    canary = "PRIVATE_PRIMARY_DESIGN_CANARY_19"
    exp = {"key": canary, "adapter": "sandbox.python", "action": "execute", "inputs": {"program": "unused"},
           "purpose": "private primary design"}
    sandbox = Capture("sandbox.python", {"value": {"measurement": 7}, "measurement_protocol": "controlled/1",
                                          "supports_scope": "this controlled measurement"})
    app, generator = fresh(tmp_path, proposal([exp]))
    app.executors["sandbox.python"] = sandbox
    created = start(app, options={"challenger": True}, authorization={"tools": ["claude-code", "sandbox.python"],
        "send_to": {"claude-code": ["*"], "sandbox.python": ["*"]}})
    planned = call(app, "run", created, steps=1)
    measured = call(app, "run", planned, steps=1)
    assert not measured.errors
    result = call(app, "run", measured, steps=1)
    assert not result.errors
    primary, challenger, _ = generator.requests[-3:]
    assert canary in primary["prompt"] and '"measurement":7' in challenger["prompt"]
    assert canary not in challenger["prompt"]
    # Canonical provenance remains inspectable; only the blind request is projected.
    assert any(canary in o["revision"]["payload"].get("source", "") for o in snapshot(app, result)["objects"])


@pytest.mark.parametrize("refusal", ["missing_effect_grant", "hard_constraint", "file_read"])
def test_challenger_does_not_receive_refused_primary_design(tmp_path: Path, refusal: str) -> None:
    canary = "PRIVATE_REFUSED_DESIGN_CANARY_31"
    exp = experiment("apply", confidential_design=canary)
    arguments: dict[str, Any] = {}
    if refusal == "missing_effect_grant":
        arguments["authorization"] = {"tools": ["claude-code", "environment"]}
    elif refusal == "hard_constraint":
        arguments["mission"] = {"hard_constraints": [{"predicate": {"op": "eq", "args": [{"var": "changes_world"}, False]}}]}
    else:
        outside = tmp_path / "outside.txt"
        outside.write_text("controlled source content")
        exp = {"key": "read-refused", "adapter": "source.file", "action": "read",
               "inputs": {"path": str(outside), "confidential_design": canary}, "purpose": "read without a grant"}
    environment = Capture("environment", {"observations": []})
    app, generator = fresh(tmp_path, proposal([exp]), environment)
    planned = call(app, "run", start(app, options={"challenger": True}, **arguments), steps=1)
    refused = call(app, "run", planned, steps=1)
    assert not refused.errors and refused.outputs["steps"][0]["kind"] == "action_refused"
    result = call(app, "run", refused, steps=1)
    assert not result.errors
    primary, challenger, _ = generator.requests[-3:]
    assert canary in primary["prompt"] and canary not in challenger["prompt"]
    blind = json.loads(challenger["prompt"])
    assert any("refused" in loss for loss in blind["context"]["loss"])
    assert environment.requests == []


def test_pending_resolution_preserves_current_policy_and_revision(tmp_path: Path) -> None:
    app, generator = fresh(tmp_path)
    planned = call(app, "run", start(app), steps=1)
    observed = call(app, "observe", planned, content={"y": 1})
    stable = call(app, "run", observed, steps=1)
    state = snapshot(app, stable)["state"]
    prediction = Prediction(question="Will later measured y equal one?", outcome_domain=[True, False], horizon="later",
                            resolver="declared/1", resolver_spec={}, sampling_rule="declared",
                            censoring_rule="keep pending", frozen_at=utcnow())
    with app.store.transaction() as db:
        ref = app.store.put(db, stable.mission_id, "prediction", prediction.model_dump(),
                            dependencies=[ObjectRef.model_validate(state["mission_ref"])])
        state["predictions"].append(ref.model_dump())
        revision = app.store.save_mission(db, state, expected=state["revision"])
    before = len(generator.requests)
    result = app.execute(Command(command_id=identity("cmd"), kind="resolve", mission_id=stable.mission_id,
                                 expected_revision=revision,
                                 payload={"prediction_ref": ref.model_dump(), "observation_ref": state["observations"][0]}))
    assert result.outputs["status"] == "pending" and not result.errors
    assert result.current_revision == revision and len(generator.requests) == before
    policy = next(o for o in snapshot(app, result)["objects"] if o["revision"]["ref"] == state["policy_ref"])
    assert policy["current_validity"]["valid"]


def test_model_cannot_grant_its_policy_deterministic_authority(tmp_path: Path) -> None:
    generated = proposal()
    generated["policy"]["ordering"] = "deterministic"
    app, _ = fresh(tmp_path, generated)
    result = call(app, "run", start(app), steps=1)
    policy = next(o["revision"]["payload"] for o in snapshot(app, result)["objects"]
                  if o["revision"]["ref"]["object_kind"] == "policy")
    assert policy["ordering"] == "conditional-partial-order"
    assert any("deterministic" in note for note in policy["unresolved"])
    assert not any(o["revision"]["ref"]["object_kind"] == "certificate" for o in snapshot(app, result)["objects"])


def test_independent_alternative_is_retained_when_synthesis_already_has_two(tmp_path: Path) -> None:
    app, generator = fresh(tmp_path)

    def generate(request: dict[str, Any]) -> dict[str, Any]:
        result = proposal()
        if request["purpose"] == "challenger":
            result["candidates"][0].update(key="blind-competitor", mechanism="independent competing mechanism")
        elif request["purpose"] == "synthesis":
            second = copy.deepcopy(result["candidates"][0])
            second.update(key="second", mechanism="second synthesis mechanism")
            result["candidates"].append(second)
        return {"payload": result}

    generator.output = generate
    result = call(app, "run", start(app, options={"challenger": True}), steps=1)
    assert not result.errors
    candidate_payloads = [o["revision"]["payload"] for o in snapshot(app, result)["objects"]
                          if o["revision"]["ref"]["object_kind"] == "candidate"]
    assert "independent competing mechanism" in {c["mechanism"] for c in candidate_payloads}


def test_missing_revision_is_distinguished_from_concurrent_change(tmp_path: Path) -> None:
    app, generator = fresh(tmp_path)
    created = start(app)
    result = app.execute(Command(command_id=identity("cmd"), kind="run", mission_id=created.mission_id))
    assert result.status == "revision_required" and generator.requests == []
    assert "changed" not in result.errors[0]["message"].lower()


def test_local_read_permission_does_not_force_private_material_into_model_request(tmp_path: Path) -> None:
    secret = "READABLE_NOT_SENDABLE_CANARY_47"
    app, generator = fresh(tmp_path)
    created = start(app, observations=[{"content": secret, "source": "local private material"}])
    state = snapshot(app, created)["state"]
    changed = call(app, "authorize", created, action="update",
                   authorization={"tools": ["claude-code"], "send_to": {"claude-code": [state["mission_ref"]["object_id"]]}})
    result = call(app, "run", changed, steps=1)
    assert not result.errors and len(generator.requests) == 2
    assert all(secret not in r["prompt"] for r in generator.requests)
    view = snapshot(app, result)
    assert any(o["revision"]["payload"].get("content") == secret for o in view["objects"])
    context = next(o["revision"]["payload"] for o in view["objects"] if o["revision"]["ref"]["object_kind"] == "context")
    assert state["observations"][0]["object_id"] in context["omissions"]
    assert any("send" in loss for loss in context["loss"])


def test_new_codex_route_uses_selected_port_without_claude_fallback(tmp_path: Path) -> None:
    app, claude = fresh(tmp_path)
    codex = Capture("codex", {"payload": proposal()})
    app.executors["codex"] = codex
    result = call(app, "run", start(app, options={"generator": "codex", "model": "gpt-6-astra", "effort": "max"},
                                   authorization={"tools": ["codex"], "send_to": {"codex": ["*"]}}), steps=1)
    assert not result.errors and len(codex.requests) == 2 and claude.requests == []
    assert all(r["model"] == "gpt-6-astra" and r["effort"] == "max" for r in codex.requests)


def test_new_mission_host_default_is_codex_and_direct_math_needs_no_client(tmp_path: Path) -> None:
    app, claude = fresh(tmp_path)
    app.default_generator = "codex"
    created = start(app, mission={"goal": "Compute 0.1 + 0.2"})
    options = snapshot(app, created)["state"]["options"]
    assert (options["generator"], options["model"], options["effort"]) == ("codex", "gpt-6.1-sol", "xhigh")
    result = call(app, "run", created, steps=1)
    assert not result.errors and result.status == "complete" and claude.requests == []


def test_explicit_astra_without_effort_keeps_authorized_max(tmp_path: Path) -> None:
    app, _ = fresh(tmp_path)
    created = start(app, options={"generator": "codex", "model": "gpt-6-astra"})
    options = snapshot(app, created)["state"]["options"]
    assert (options["model"], options["effort"]) == ("gpt-6-astra", "max")


def test_normal_run_returns_finite_completion_before_idle_step(tmp_path: Path) -> None:
    app, generator = fresh(tmp_path)
    created = start(app, mission={"goal": "Compute 7 * (3 + 2)"}, options={"steps": 7})
    result = call(app, "run", created)
    assert result.status == "complete" and not result.errors and generator.requests == []
    assert len(result.outputs["steps"]) == 1 and result.outputs["external_calls"] == 0
    assert result.next_actions == ["inspect", "export"]
    assert call(app, "run", result).status == "waiting"


def test_bad_import_artifact_is_rejected_before_any_blob_is_published(tmp_path: Path) -> None:
    source, _ = fresh(tmp_path / "source")
    result = call(source, "run", start(source, mission={"goal": "Compute 2 + 3"}), steps=1)
    package = call(source, "export", result).outputs
    package = copy.deepcopy(package)
    entry = next(iter(package["artifacts"].values()))
    entry["base64"] = base64.b64encode(b"rejected corrupted import bytes").decode()
    package["content_hash"] = digest({k: v for k, v in package.items() if k != "content_hash"})
    target, _ = fresh(tmp_path / "target")
    before = sorted(target.store.artifacts.iterdir())
    imported = call(target, "import", package=package)
    assert imported.status == "import_artifact"
    assert sorted(target.store.artifacts.iterdir()) == before
    assert call(target, "inspect").outputs["missions"] == []


def test_unknown_authorization_tool_is_rejected_without_starting_a_mission(tmp_path: Path) -> None:
    app, generator = fresh(tmp_path)
    result = start(app, authorization={"tools": ["invented-provider"]})
    assert result.status == "authorization_tool" and generator.requests == []
    assert call(app, "inspect").outputs["missions"] == []


def test_unchanged_frozen_question_is_not_duplicated_on_recompile(tmp_path: Path) -> None:
    generated = proposal()
    generated["candidates"][0]["predictions"] = [{"question": "Will the declared outcome occur?",
        "outcome_domain": [True, False], "horizon": "future", "resolver": "declared/1", "resolver_spec": {},
        "sampling_rule": "declared", "censoring_rule": "keep pending"}]
    app, _ = fresh(tmp_path, generated)
    first = call(app, "run", start(app), steps=1)
    prior = snapshot(app, first)["state"]["predictions"]
    assert len(prior) == 1
    changed = call(app, "observe", first, content={"unrelated": 1})
    revised = call(app, "run", changed, steps=1)
    assert not revised.errors and snapshot(app, revised)["state"]["predictions"] == prior


def test_resolved_question_is_not_refrozen_after_new_evidence(tmp_path: Path) -> None:
    generated = proposal([experiment()])
    generated["candidates"][0]["predictions"] = [{"question": "Does the controlled output equal seven?",
        "outcome_domain": [True, False], "horizon": "next controlled measurement", "resolver": "field-equals/1",
        "resolver_spec": {"measurement_protocol": "controlled/1", "source": "controlled-executable",
            "supports_scope": "this controlled measurement", "field": ["y"], "equals": 7},
        "sampling_rule": "one measurement", "censoring_rule": "keep pending"}]
    env = Capture("environment", {"observations": [{"content": {"y": 7}, "source": "controlled-executable",
        "measurement_protocol": "controlled/1", "supports_scope": "this controlled measurement"}]})
    app, _ = fresh(tmp_path, generated, env)
    planned = call(app, "run", start(app), steps=1)
    measured = call(app, "run", planned, steps=1)
    settled = snapshot(app, measured)["state"]["predictions"]
    assert len(settled) == 1
    new_input = call(app, "observe", measured, content={"unrelated": 2})
    revised = call(app, "run", new_input, steps=1)
    assert not revised.errors and snapshot(app, revised)["state"]["predictions"] == settled
    prediction = next(o["revision"]["payload"] for o in snapshot(app, revised)["objects"]
                      if o["revision"]["ref"] == settled[0])
    assert prediction["status"] == "resolved" and prediction["outcome"] is True


@pytest.mark.parametrize("change", ["authorization", "observation", "mission"])
def test_identical_local_refusal_waits_for_new_basis_then_reopens(tmp_path: Path, change: str) -> None:
    env = Capture("environment", {"observations": []})
    app, generator = fresh(tmp_path, proposal([experiment("apply")]), env)
    created = start(app, authorization={"tools": ["claude-code", "environment"]})
    result = call(app, "run", created, steps=12)
    view = snapshot(app, result)
    assert not result.errors and result.status == "waiting" and env.requests == []
    assert [s["kind"] for s in result.outputs["steps"]] == ["compile", "action_refused", "compile"]
    assert len(generator.requests) == 4 and view["state"]["completed_experiments"] == []
    assert view["state"]["pending_experiments"] == []
    policy = next(o["revision"]["payload"] for o in view["objects"]
                  if o["revision"]["ref"] == view["state"]["policy_ref"])
    assert any("locally refused" in reason for reason in policy["unresolved"])
    idle = call(app, "run", result, steps=10)
    assert idle.outputs["external_calls"] == 0 and len(generator.requests) == 4
    if change == "authorization":
        updated = call(app, "authorize", idle, action="update",
                       authorization={"tools": ["claude-code", "environment", "environment.apply"]})
    elif change == "observation":
        updated = call(app, "observe", idle, content={"new": "relevant input supplied by owner"})
    else:
        updated = call(app, "revise", idle, mission_input={"goal": "Reconsider the controlled system"})
    reopened = call(app, "run", updated, steps=2)
    assert not reopened.errors
    assert reopened.outputs["steps"][-1]["kind"] == ("experiment" if change == "authorization" else "action_refused")
    assert len(env.requests) == (1 if change == "authorization" else 0)
