"""Controlled workflow counterexamples, not live model or task-quality evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sextant.application import Application
from sextant.domain import (
    AuthorityContext,
    Candidate,
    Command,
    DomainError,
    Experiment,
    Mission,
    ObjectRef,
    Observation,
    Policy,
    Prediction,
    PredictionSpec,
    Proposal,
    Receipt,
    Representation,
    canonical,
    digest,
    identity,
    utcnow,
)
from sextant.grounding import normalize_policy, settle
from sextant.representations import arithmetic, evaluate
from sextant.runtime import AttemptContext, Invocation
from sextant.storage import Store

OWNER = AuthorityContext(actor="test-host", source="constructed test", can_authorize=True)


def call(app: Application, kind: str, prior: Any = None, **payload: Any) -> Any:
    command = Command(command_id=identity("test"), kind=kind, mission_id=prior.mission_id if prior else None,
                      expected_revision=prior.current_revision if prior else None, payload=payload)
    return app.execute(command)


def fresh(tmp_path: Path) -> Application:
    return Application(Store(tmp_path / "runtime"), OWNER)


def test_direct_complete_query_replay_no_model(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    start = call(app, "start", mission={"goal": "计算 7 * (3 + 2)", "scope": {"language": "zh"}})
    result = call(app, "run", start)
    assert not result.errors, result
    assert result.outputs["steps"][0]["direct_result"]["value"] == 35
    inspection = call(app, "inspect", result)
    assert inspection.outputs["saved_receipts"] == []
    assert call(app, "replay", result).outputs["matches_current"]
    assert call(app, "run", result).status == "waiting"


def test_command_id_and_stale_write(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    cmd = Command(command_id="start-once", kind="start", payload={"mission": {"goal": "calculate 1 + 1"}})
    first = app.execute(cmd)
    assert first == app.execute(cmd)
    collision = cmd.model_copy(update={"payload": {"mission": {"goal": "changed"}}})
    assert app.execute(collision).status == "command_conflict"
    assert first == app.execute(cmd)
    updated = call(app, "observe", first, content="source claim", source="test")
    assert not updated.errors
    assert call(app, "pause", first).status == "conflict"


@pytest.mark.parametrize("boundary", ["before_commit", "after_commit"])
@pytest.mark.parametrize("kind", ["start", "observe", "import"])
def test_local_command_crash_recovery_has_one_mutation_and_exact_saved_result(tmp_path: Path, boundary: str, kind: str) -> None:
    app = fresh(tmp_path)
    prior = call(app, "start", mission={"goal": "calculate 2+3"}) if kind == "observe" else None
    payload = ({"mission": {"goal": "calculate 2+3"}} if kind == "start" else
               {"content": "one source record", "source": "host"} if kind == "observe" else
               {"package": {"format": "sextant-legacy-import/1", "graph": {"note": "historical"}, "ledger_lines": []}})
    command = Command(command_id="recover-the-same-request", kind=kind, payload=payload,
                      mission_id=prior.mission_id if prior else None, expected_revision=prior.current_revision if prior else None)

    class ProcessLost(BaseException):
        pass

    def crash(point: str) -> None:
        if point == boundary:
            raise ProcessLost()

    app.store.fault = crash
    with pytest.raises(ProcessLost):
        app.execute(command)
    recovered = Application(Store(app.store.root), OWNER)
    result = recovered.execute(command)
    assert not result.errors and result == recovered.execute(command)
    assert result.status != "in_progress"
    with recovered.store.connection(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM missions").fetchone()[0] == 1
        saved = db.execute("SELECT result FROM commands WHERE command_id=?", (command.command_id,)).fetchone()[0]
        assert json.loads(saved) == result.model_dump(mode="json")
        if kind == "observe":
            assert len(recovered.store.mission(db, result.mission_id)["observations"]) == 1
        if kind == "import":
            assert db.execute("SELECT COUNT(*) FROM imports").fetchone()[0] == 1


def test_invalid_local_command_rolls_back_work_and_preserves_error_receipt(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    command = Command(command_id="invalid-start-once", kind="start", payload={"mission": {"goal": "calculate 1+1"},
                     "observations": [{"content": "first"}, {"unsupported": "no content"}]})
    result = app.execute(command)
    assert result.errors and result == app.execute(command)
    with app.store.connection(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM missions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0


def test_withdraw_and_changed_goal_invalidate_only_dependencies(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    first = call(app, "start", mission={"goal": "calculate 2 + 2"}, observations=[{"content": "claim", "source": "speaker"}])
    done = call(app, "run", first, steps=1)
    snapshot = call(app, "inspect", done).outputs
    old_policy = ObjectRef.model_validate(snapshot["state"]["policy_ref"])
    observation = snapshot["state"]["observations"][0]
    changed = call(app, "withdraw", done, observation_ref=observation, reason="counterexample")
    with app.store.connection(readonly=True) as db:
        assert not app.store.validity(db, old_policy)["valid"]
        assert app.store.validity(db, ObjectRef.model_validate(snapshot["state"]["mission_ref"]))["valid"]
    revised = call(app, "revise", changed, mission_input={"goal": "calculate 2 + 3"})
    rerun = call(app, "run", revised, steps=1)
    assert rerun.outputs["steps"][0]["direct_result"]["value"] == 5
    assert len(call(app, "inspect", rerun).outputs["state"]["history"]) == 1


def test_local_recomputation_cannot_promote_stale_policy_after_counterevidence(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    start = call(app, "start", mission={"goal": "calculate 2+3"})
    done = call(app, "run", start, steps=1)
    old = call(app, "inspect", done).outputs["state"]["policy_ref"]
    changed = call(app, "observe", done, content="new constraint not semantically evaluated", source="host")
    recomputed = call(app, "run", changed, mode="recompute")
    assert not recomputed.errors and recomputed.outputs["steps"][0]["policy_promoted"] is False
    inspected = call(app, "inspect", recomputed).outputs
    assert inspected["state"]["policy_ref"] == old and inspected["state"]["last_fingerprint"] is None
    assert not next(o["current_validity"]["valid"] for o in inspected["objects"] if o["revision"]["ref"] == old)


def test_returning_to_old_goal_creates_fresh_revision_without_reviving_old_policy(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    first = call(app, "start", mission={"goal": "calculate 2+3"})
    done = call(app, "run", first, steps=1)
    old = call(app, "inspect", done).outputs["state"]
    changed = call(app, "revise", done, mission_input={"goal": "calculate 4+5"})
    returned = call(app, "revise", changed, mission_input={"goal": "calculate 2+3"})
    ready = call(app, "inspect", returned).outputs["state"]
    assert ready["mission_ref"]["revision_id"] != old["mission_ref"]["revision_id"]
    with app.store.connection(readonly=True) as db:
        assert not app.store.validity(db, ObjectRef.model_validate(old["policy_ref"]))["valid"]
        assert app.store.validity(db, ObjectRef.model_validate(ready["mission_ref"]))["valid"]
    rerun = call(app, "run", returned, steps=1)
    assert not rerun.errors and rerun.outputs["steps"][0]["direct_result"]["value"] == 5


def test_external_observation_identity_duplicate_and_exact_withdrawal(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    start = call(app, "start", mission={"goal": "calculate 2+3"})
    experiment = Experiment(key="read-environment", purpose="Read exact measurement identities", adapter="environment", action="observe", inputs={})

    def measurement(key: str, value: int) -> dict[str, Any]:
        return {"observation_id": key, "source_id": "shared-lab", "source_revision": "phase-0",
                "content": {"value": value}, "observed_at": "2026-09-27T00:00:00+00:00",
                "measurement_protocol": "constructed-test/1", "supports_scope": {"episode_id": "one", "phase": 0}}

    def ingest(*items: dict[str, Any]) -> list[ObjectRef]:
        with app.store.transaction() as db:
            state = app.store.mission(db, start.mission_id)
            refs = app._ingest_effect(db, state, experiment, {"observations": list(items)})
            app.store.save_mission(db, state, expected=state["revision"])
            return [ObjectRef.model_validate(r) for r in refs]

    first, second = measurement("measurement-a", 1), measurement("measurement-b", 2)
    a, b = ingest(first, second)
    current = call(app, "inspect", start)
    done = call(app, "run", current, steps=1)
    policy = ObjectRef.model_validate(call(app, "inspect", done).outputs["state"]["policy_ref"])
    assert ingest(second) == [b]
    with app.store.connection(readonly=True) as db:
        assert len(app.store.mission(db, start.mission_id)["observations"]) == 2
        assert app.store.validity(db, policy)["valid"]
    correction = {**measurement("measurement-c", 3), "withdraws": ["measurement-a"]}
    c = ingest(correction)[0]
    with app.store.connection(readonly=True) as db:
        assert not app.store.validity(db, a)["valid"]
        assert app.store.validity(db, b)["valid"] and app.store.validity(db, c)["valid"]
        assert not app.store.validity(db, policy)["valid"]
    with pytest.raises(DomainError, match="reused with changed content"):
        ingest({**second, "content": {"value": 999}})


def test_manual_provenance_and_host_authority_cannot_be_forged(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    start = call(app, "start", mission={"goal": "calculate 1+2", "goal_source": "model invented owner"})
    observed = call(app, "observe", start, content=True, source="claimed-lab", provenance="constructed", measurement_protocol="environment/1")
    ref = ObjectRef.model_validate(observed.outputs["observation_ref"])
    with app.store.connection(readonly=True) as db:
        assert app.store.get(db, ref).payload["provenance"] == "declared"
    delegated = Application(app.store, AuthorityContext(actor="agent", source="untrusted", can_authorize=False))
    denied = call(delegated, "authorize", observed, action="update", authorization={"tools": ["claude-code"], "actor": "owner"})
    assert denied.status == "authority_required"
    revoked = call(app, "authorize", observed, action="revoke")
    assert call(app, "resume", revoked).status == "revoked"


def test_portable_export_import_inert_idempotent_and_corruption(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    started = call(app, "start", mission={"goal": "calculate 4*5"})
    done = call(app, "run", started, steps=1)
    package = call(app, "export", done).outputs
    second = Application(Store(tmp_path / "imported"), OWNER)
    imported = call(second, "import", package=package)
    assert not imported.errors, imported
    assert imported.status == "needs_authority"
    assert call(second, "import", package=package).status == "already_imported"
    assert call(second, "run", imported).status == "revoked"
    assert call(second, "replay", imported).outputs["matches_current"]
    inspected = call(second, "inspect", imported).outputs
    policies = [v for v in inspected["objects"] if v["revision"]["ref"]["object_kind"] == "policy"]
    assert policies and not policies[0]["current_validity"]["valid"]
    bad = json.loads(canonical(package))
    bad["mission"]["status"] = "complete-forged"
    assert call(second, "import", package=bad).status == "import_hash"


def test_user_dictionary_that_resembles_artifact_is_preserved_as_data(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    fake = {"sha256": "a" * 64, "size": 42, "media_type": "text/plain", "source": "untrusted statement"}
    start = call(app, "start", mission={"goal": "calculate 1+2"}, options={"environment": {"notes": fake}})
    exported = call(app, "export", start)
    assert not exported.errors
    assert exported.outputs["mission"]["options"]["environment"]["notes"] == fake
    assert fake["sha256"] not in exported.outputs["artifacts"]


def test_backup_restore_preserves_receipts_replay_and_rejects_interruption(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    start = call(app, "start", mission={"goal": "calculate 3*7"}, observations=[{"content": "source", "source": "test"}])
    done = call(app, "run", start, steps=1)
    destination = tmp_path / "backup"
    report = app.store.backup_store(destination)
    assert report["status"] == "complete" and report["artifacts"]
    restored = Application(Store(destination, create=False), OWNER)
    assert call(restored, "replay", done).outputs["matches_current"]
    assert call(restored, "export", done).outputs == call(app, "export", done).outputs
    (destination / "backup.incomplete").write_text("interrupted")
    with pytest.raises(DomainError, match="interrupted"):
        Store(destination, create=False)
    app.store.fault = lambda point: (_ for _ in ()).throw(OSError("interrupted backup")) if point == "backup_before_complete" else None
    interrupted = tmp_path / "interrupted-backup"
    with pytest.raises(OSError, match="interrupted backup"):
        app.store.backup_store(interrupted)
    with pytest.raises(DomainError, match="interrupted"):
        Store(interrupted, create=False)
    assert call(app, "replay", done).outputs["matches_current"]


def test_legacy_loss_and_torn_line_preserved(tmp_path: Path) -> None:
    app = fresh(tmp_path)
    package = {"format": "sextant-legacy-import/1", "goal": "inspect history", "graph": {"nodes": [{"id": "old", "gate": "PASS"}]},
               "ledger_lines": ['{"call":"old"}', '{"torn":']}
    imported = call(app, "import", package=package)
    assert imported.outputs["failures"][0]["line"] == 2
    assert not imported.outputs["authorization_executable"]
    assert call(app, "import", package=package).status == "already_imported"


def test_executable_representations_preserve_joint_xor_and_unknown() -> None:
    table = Representation(kind="table", body={"inputs": ["a", "b"], "rows": [
        {"input": {"a": a, "b": b}, "output": bool(a ^ b)} for a in range(2) for b in range(2)]}, composition_rule="finite lookup")
    assert evaluate(table, {"a": 1, "b": 1})["value"] is False
    assert evaluate(table, {"a": 2, "b": 1})["status"] == "unknown"
    graph = Representation(kind="argument_graph", body={"nodes": [{"id": "joint", "expression": {"op": "xor", "args": [{"var": "a"}, {"var": "b"}]}}],
                                                            "outputs": {"y": {"var": "joint"}}}, composition_rule="boolean XOR")
    assert evaluate(graph, {"a": True, "b": False})["value"] == {"y": True}
    assert evaluate(graph, {"a": True})["status"] == "unknown"
    cyclic = graph.model_copy(deep=True)
    cyclic.body["nodes"][0]["expression"] = {"var": "joint"}
    assert evaluate(cyclic, {})["status"] == "unknown"
    policy = Representation(kind="conditional_policy", body={"branches": [{"when": {"op": "gt", "args": [{"var": "cost"}, 5]}, "then": "wait"}]}, composition_rule="all matches")
    assert evaluate(policy, {"cost": 6})["value"] == ["wait"]
    assert evaluate(policy, {})["status"] == "partial"
    program = Representation(kind="program", body={"program": "raise RuntimeError('MUST NOT EXECUTE HERE')"}, composition_rule="sandbox only")
    assert evaluate(program, {})["status"] == "requires_sandbox"
    assert arithmetic("calculate __import__('os').system('touch /tmp/sextant-escape')") is None


def test_resolver_refuses_source_statements_wrong_protocol_and_unearned_utility() -> None:
    prediction = Prediction(question="Does controlled y equal 3?", outcome_domain=[True, False], horizon="controlled run",
                            resolver="field-equals/1", resolver_spec={"field": ["y"], "equals": 3, "measurement_protocol": "executable/1", "source": "fixture", "supports_scope": "this run"},
                            sampling_rule="one controlled run", censoring_rule="keep pending", frozen_at=utcnow())
    observation = Observation(content={"y": 3}, source="fixture", source_group="fixture", observed_at=utcnow(),
                              measurement_protocol="executable/1", supports_scope="this run", provenance="declared")
    assert settle(prediction, observation, now=utcnow())["status"] == "pending"
    observation.provenance = "constructed"
    assert settle(prediction, observation, now=utcnow())["outcome"] is True
    observation.measurement_protocol = "wrong/1"
    with pytest.raises(Exception, match="measurement"):
        settle(prediction, observation, now=utcnow())
    mission = Mission(goal="choose", goal_source="test", hard_constraints=[{"max": 3}], preference_order=["correctness", "speed"])
    policy = Policy(summary="model ranking", branches=[], hard_constraints=[], unresolved=[], reversal_triggers=[], ordering="authorized-cardinal", basis="invented weights")
    normalized = normalize_policy(policy, mission)
    assert normalized.ordering == "conditional-partial-order" and normalized.hard_constraints == mission.hard_constraints


class ControlledGenerator:
    name = "claude-code"

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def capabilities(self) -> dict[str, Any]:
        return {"evidence_class": "controlled fixture"}

    def cancel(self, context: AttemptContext) -> dict[str, Any]:
        return {"status": "unsupported"}

    def reconcile(self, context: AttemptContext) -> None:
        return None

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        self.requests.append(request)
        content = json.loads(request["prompt"])
        observed = any(e["observation"]["provenance"] == "environment" for e in content["evidence"])
        experiment = Experiment(key="joint-input", adapter="environment", action="experiment", inputs={"assignments": {"a": 1, "b": 1}}, purpose="distinguish additive from XOR")
        candidates = [Candidate(key="additive", mechanism="additive", assumptions=[], representation=Representation(kind="unsupported", body={}, composition_rule="unmeasured sum", loss=["pending experiment"])),
                      Candidate(key="joint", mechanism="XOR interaction", assumptions=[], representation=Representation(kind="table", body={"inputs": ["a", "b"], "rows": [{"input": {"a": 1, "b": 1}, "output": False}]}, composition_rule="finite observed row"),
                                experiments=[] if observed else [experiment], temporary_disadvantage="components alone uninformative", combination_dependencies=["a", "b"])]
        proposal = Proposal(understanding="controlled fixture", candidates=candidates, unknowns=[],
                            policy=Policy(summary="joint-zero observed; additive rejected" if observed else "additive remains plausible; measure joint input",
                                          branches=[], hard_constraints=[], unresolved=[] if observed else ["joint outcome"], reversal_triggers=["counterevidence"], ordering="conditional-partial-order", basis="controlled fixture"))
        output = {"payload": proposal.model_dump(mode="json")}
        return Invocation(Receipt(executor=self.name, started_at=utcnow(), ended_at=utcnow(), request_hash=digest(request), output_hash=digest(output), status="completed", auth_route="controlled fixture"), output)


class ControlledEnvironment(ControlledGenerator):
    name = "environment"

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        self.requests.append(request)
        a, b = request["inputs"]["assignments"].values()
        output = {"observations": [{"source_id": "controlled-executable", "content": {"y": bool(a ^ b)}, "measurement_protocol": "controlled-xor/1", "supports_scope": "this controlled input"}]}
        return Invocation(Receipt(executor=self.name, started_at=utcnow(), ended_at=utcnow(), request_hash=digest(request), output_hash=digest(output), status="completed", auth_route="controlled fixture"), output)


def test_narrowed_grants_do_not_resend_withdrawn_material_through_history(tmp_path: Path) -> None:
    sentinel = "retired-private-input-328e8ba9"

    class EchoingGenerator(ControlledGenerator):
        def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
            invocation = super().invoke(request, context)
            if sentinel in request["prompt"]:
                proposal = invocation.output["payload"]
                proposal["understanding"] = sentinel
                for candidate in proposal["candidates"]:
                    for experiment in candidate["experiments"]:
                        experiment["inputs"]["retained_material"] = sentinel
                invocation.receipt.output_hash = digest(invocation.output)
            return invocation

    generator, environment = EchoingGenerator(), ControlledEnvironment()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator, "environment": environment})
    start = call(app, "start", mission={"goal": "Discover the controlled mechanism"}, options={"jev_density": "none"},
                 observations=[{"content": sentinel, "source": "private input"}],
                 authorization={"tools": ["claude-code", "environment"], "send_to": {"claude-code": ["*"], "environment": ["*"]}})
    first = call(app, "run", start, steps=2)
    assert not first.errors
    state = call(app, "inspect", first).outputs["state"]
    assert sentinel in canonical(state["completed_experiment_records"])
    allowed = [state["mission_ref"]["object_id"], *[r["object_id"] for r in state["observations"][1:]]]
    narrowed = call(app, "authorize", first, action="update", authorization={"tools": ["claude-code"], "send_to": {"claude-code": allowed}})
    assert not call(app, "inspect", narrowed).outputs["state"]["pending_experiments"]
    withdrawn = call(app, "withdraw", narrowed, observation_ref=state["observations"][0], reason="retired input")
    count = len(generator.requests)
    revised = call(app, "run", withdrawn, steps=1)
    assert not revised.errors, revised
    assert len(generator.requests) > count
    assert all(sentinel not in canonical(r) for r in generator.requests[count:])
    inspection = call(app, "inspect", revised, events=True)
    assert any(e["kind"] == "compiler.history_withheld" for e in inspection.events)


def test_complete_feedback_blind_challenger_and_low_valley_opportunity(tmp_path: Path) -> None:
    generator, environment = ControlledGenerator(), ControlledEnvironment()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator, "environment": environment})
    start = call(app, "start", mission={"goal": "Discover the controlled mechanism"}, options={"jev_density": "none", "steps": 3},
                 authorization={"tools": ["claude-code", "environment"], "send_to": {"claude-code": ["*"], "environment": ["*"]}})
    done = call(app, "run", start)
    assert not done.errors, done
    assert len(environment.requests) == 1
    assert [s["kind"] for s in done.outputs["steps"]] == ["compile", "experiment", "compile"]
    inspected = call(app, "inspect", done).outputs
    assert inspected["state"]["branch_visits"]["joint"] == 1
    assert inspected["state"]["history"][0]["changed"] is True
    policies = [o for o in inspected["objects"] if o["revision"]["ref"]["object_kind"] == "policy" and o["current_validity"]["valid"]]
    assert policies[0]["revision"]["payload"]["summary"].startswith("joint-zero")
    challengers = [json.loads(r["prompt"]) for r in generator.requests if r["purpose"] == "challenger"]
    assert all("alternatives" not in c and "advisory_diagnostics" not in c and "previous_policy_for_revision" not in c for c in challengers)
    assert call(app, "run", done).status == "waiting"
    # Export includes every actual request/receipt, even those not individually
    # referenced by a proposal object; no second model call occurs on export.
    package = call(app, "export", done).outputs
    assert len(package["attempts"]) == 7
    for attempt in package["attempts"]:
        assert json.loads(attempt["request"])["artifact"]["sha256"] in package["artifacts"]


def test_failed_advisory_is_visible_and_reaches_revision_without_claiming_a_judgment(tmp_path: Path) -> None:
    class FailedEvaluator(ControlledGenerator):
        name = "typesafe"

        def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
            self.requests.append(request)
            return Invocation(Receipt(executor=self.name, started_at=utcnow(), ended_at=utcnow(),
                                      request_hash=digest(request), output_hash=digest(None), status="failed",
                                      errors=["controlled HTTP 500 response; usage unknown"]), None)

    generator, evaluator = ControlledGenerator(), FailedEvaluator()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator, "typesafe": evaluator})
    start = call(app, "start", mission={"goal": "Discover the controlled mechanism"}, options={"jev_density": "final_only"},
                 authorization={"tools": ["claude-code", "typesafe"], "send_to": {"claude-code": ["*"], "typesafe": ["*"]}})
    run = call(app, "run", start, steps=1)
    assert not run.errors
    revisions = [json.loads(r["prompt"]) for r in generator.requests if r["purpose"] == "diagnostic_revision"]
    assert len(revisions) == 1 and len(evaluator.requests) == 4
    assert all(d["status"] == "unavailable" and d["derived"] == {} for d in revisions[0]["advisory_diagnostics"])
    inspection = call(app, "inspect", run).outputs
    assert len(inspection["saved_compiler_stages"]) == 4
    assert all(o["revision"]["payload"]["automatic_decisions"] == 0 for o in inspection["objects"] if o["revision"]["ref"]["object_kind"] == "judgment")


def test_partial_trajectory_and_unknown_effect_are_recoverable_without_resend(tmp_path: Path) -> None:
    class LostEnvironment(ControlledEnvironment):
        def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
            self.requests.append(request)
            raise TimeoutError("response lost after send")

    generator, environment = ControlledGenerator(), LostEnvironment()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator, "environment": environment})
    start = call(app, "start", mission={"goal": "Discover mechanism"}, options={"jev_density": "none"},
                 authorization={"tools": ["claude-code", "environment"], "send_to": {"claude-code": ["*"], "environment": ["*"]}})
    run = call(app, "run", start, steps=3)
    assert run.status == "reconciliation_required" and run.current_revision
    assert [step["kind"] for step in run.outputs["steps"]] == ["compile"]
    assert run.outputs["failed_operation"]
    assert call(app, "run", run).status == "reconciliation_required"
    assert len(environment.requests) == 1 and len(generator.requests) == 3
    assert len(call(app, "inspect", run).outputs["saved_compiler_stages"]) == 3


def test_received_effect_survives_failed_observation_commit_without_duplicate_send(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generator, environment = ControlledGenerator(), ControlledEnvironment()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator, "environment": environment})
    start = call(app, "start", mission={"goal": "Discover mechanism"}, options={"jev_density": "none"},
                 authorization={"tools": ["claude-code", "environment"], "send_to": {"claude-code": ["*"], "environment": ["*"]}})
    planned = call(app, "run", start, steps=1)
    original = app._ingest_effect

    def crash(*args: Any, **kwargs: Any) -> Any:
        original(*args, **kwargs)
        raise DomainError("injected_commit_failure", "effect succeeded, local observation transaction rolls back")

    monkeypatch.setattr(app, "_ingest_effect", crash)
    failed = call(app, "run", planned, steps=1)
    assert failed.status == "injected_commit_failure" and len(environment.requests) == 1
    monkeypatch.setattr(app, "_ingest_effect", original)
    recovered = call(app, "run", failed, steps=1)
    assert not recovered.errors and len(environment.requests) == 1
    inspection = call(app, "inspect", recovered, events=True)
    assert len(inspection.outputs["state"]["observations"]) == 1
    assert any(e["kind"] == "effect.reused" for e in inspection.events)


def test_legacy_import_rollback_does_not_leave_an_unregistered_mission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = fresh(tmp_path)
    original = app.store.event

    def fail_import(db: Any, mid: str, kind: str, *args: Any, **kwargs: Any) -> Any:
        if kind == "import.completed":
            raise DomainError("injected_import_failure", "crash before import commit")
        return original(db, mid, kind, *args, **kwargs)

    monkeypatch.setattr(app.store, "event", fail_import)
    result = call(app, "import", package={"format": "sextant-legacy-import/1", "graph": {}, "ledger_lines": []})
    assert result.status == "injected_import_failure"
    with app.store.connection(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM missions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM imports").fetchone()[0] == 0


def test_same_design_is_measured_again_in_new_environment_phase(tmp_path: Path) -> None:
    class RepeatingGenerator(ControlledGenerator):
        def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
            response = super().invoke(request, context)
            proposal = response.output["payload"]
            proposal["candidates"][1]["experiments"] = [Experiment(key="joint-input", adapter="environment", action="experiment",
                        inputs={"assignments": {"a": 1, "b": 1}}, purpose="repeat only in new scope").model_dump()]
            return Invocation(response.receipt.model_copy(update={"output_hash": digest(response.output)}), response.output)

    generator, environment = RepeatingGenerator(), ControlledEnvironment()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator, "environment": environment})
    start = call(app, "start", mission={"goal": "Discover mechanism"}, options={"jev_density": "none", "environment": {"phase": 0}},
                 authorization={"tools": ["claude-code", "environment"], "send_to": {"claude-code": ["*"], "environment": ["*"]}})
    first = call(app, "run", start, steps=3)
    assert len(environment.requests) == 1 and not first.errors
    assert call(app, "run", first).status == "waiting"
    revised = call(app, "revise", first, options={"environment": {"phase": 1}})
    second = call(app, "run", revised, steps=3)
    assert not second.errors and len(environment.requests) == 2
    assert [r["phase"] for r in environment.requests] == [0, 1]


def test_independent_information_batch_runs_before_recompilation(tmp_path: Path) -> None:
    class BatchGenerator(ControlledGenerator):
        def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
            response = super().invoke(request, context)
            proposal = response.output["payload"]
            if proposal["candidates"][1]["experiments"]:
                proposal["candidates"][1]["experiments"] = [Experiment(key=f"row-{a}-{b}", adapter="environment", action="experiment",
                    inputs={"assignments": {"a": a, "b": b}}, purpose="finite domain coverage", independent_group="matrix").model_dump()
                    for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
            return Invocation(response.receipt.model_copy(update={"output_hash": digest(response.output)}), response.output)

    generator, environment = BatchGenerator(), ControlledEnvironment()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": generator, "environment": environment})
    start = call(app, "start", mission={"goal": "Discover mechanism"}, options={"jev_density": "none"},
                 authorization={"tools": ["claude-code", "environment"], "send_to": {"claude-code": ["*"], "environment": ["*"]}})
    result = call(app, "run", start, steps=6)
    assert not result.errors, result
    assert [s["kind"] for s in result.outputs["steps"]] == ["compile", "experiment", "experiment", "experiment", "experiment", "compile"]
    assert len(environment.requests) == 4 and len(generator.requests) == 6


def test_model_cannot_label_an_unrecognized_environment_mutation_readonly() -> None:
    from sextant.application import world_effect
    proposed = Experiment(key="delete", adapter="environment", action="delete", inputs={}, purpose="pretend read", changes_world=False)
    assert world_effect(proposed)
    assert not world_effect(proposed.model_copy(update={"action": "experiment"}))


def test_frozen_prediction_settles_from_later_real_adapter_receipt(tmp_path: Path) -> None:
    class PredictingGenerator(ControlledGenerator):
        def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
            response = super().invoke(request, context)
            proposal = response.output["payload"]
            if proposal["candidates"][1]["experiments"]:
                proposal["candidates"][1]["predictions"] = [PredictionSpec(question="Does joint intervention return false?",
                    outcome_domain=[True, False], horizon="next controlled intervention", resolver="field-equals/1",
                    resolver_spec={"measurement_protocol": "controlled-xor/1", "source": "controlled-executable",
                                   "supports_scope": "this controlled input", "field": ["y"], "equals": False},
                    sampling_rule="one fixed intervention", censoring_rule="keep pending").model_dump()]
            return Invocation(response.receipt.model_copy(update={"output_hash": digest(response.output)}), response.output)

    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"claude-code": PredictingGenerator(), "environment": ControlledEnvironment()})
    start = call(app, "start", mission={"goal": "Discover mechanism"}, options={"jev_density": "none"},
                 authorization={"tools": ["claude-code", "environment"], "send_to": {"claude-code": ["*"], "environment": ["*"]}})
    planned = call(app, "run", start, steps=1)
    before = call(app, "inspect", planned).outputs
    prediction = next(o for o in before["objects"] if o["revision"]["ref"]["object_kind"] == "prediction")
    assert prediction["revision"]["payload"]["status"] == "pending"
    measured = call(app, "run", planned, steps=1)
    after = call(app, "inspect", measured).outputs
    settled = next(o for o in after["objects"] if o["revision"]["ref"]["object_kind"] == "prediction")
    assert settled["revision"]["payload"]["status"] == "resolved"
    assert settled["revision"]["payload"]["outcome"] is True
    ref = after["state"]["observations"][0]
    withdrawn = call(app, "withdraw", measured, observation_ref=ref, reason="measurement revoked")
    final = call(app, "inspect", withdrawn).outputs
    historical = next(o for o in final["objects"] if o["revision"]["ref"]["object_kind"] == "prediction")
    assert not historical["current_validity"]["valid"]
