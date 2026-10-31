"""Local lifecycle counterexamples; no fake measurements or provider effects."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_runtime_integrity_v1 import call, fresh, start

from sextant.domain import AuthorityContext, Command, DomainError, ObjectRef, Prediction, digest
from sextant.render import chinese


def pending(root: Path, **changes: Any) -> tuple[Any, Any, ObjectRef, dict[str, Any]]:
    app, _ = fresh(root)
    created = start(app)
    payload = Prediction(question="Does the declared finite measurement satisfy this frozen predicate?",
        outcome_domain=[True, False], horizon="one future measurement", resolver="field-equals/1",
        resolver_spec={"field_path": "checks.ok", "equals": True, "measurement_protocol": "public/1",
                       "source": "public fixture", "supports_scope": "one finite computation"},
        sampling_rule="one measurement", censoring_rule="keep pending",
        frozen_at="2020-01-01T00:00:00+00:00").model_dump()
    payload.pop("resolution_correction")  # Preserve absence, not only null values.
    payload.update(changes)
    with app.store.transaction() as db:
        state = app.store.mission(db, created.mission_id)
        artifact = app.store.put_artifact(b"public original frozen contract")
        pref = app.store.put(db, created.mission_id, "prediction", payload,
            dependencies=[ObjectRef.model_validate(state["mission_ref"])], artifacts=[artifact])
        dependent = app.store.put(db, created.mission_id, "context", {"public": True}, dependencies=[pref])
        state.update(predictions=[pref.model_dump()], prediction_keys={"frozen-spec": pref.model_dump()},
                     context_ref=dependent.model_dump(), research_queue=[{"stale": True}], pending_experiments=[])
        revision = app.store.save_mission(db, state, expected=state["revision"])
    return app, created.model_copy(update={"current_revision": revision}), pref, payload


def invalidate(app: Any, before: Any, pref: ObjectRef, **extra: Any) -> Any:
    return call(app, "resolve", before, mode="invalidate-contract", prediction_ref=pref.model_dump(),
                reason="Host found an unexecutable frozen selector", **extra)


@pytest.mark.parametrize("selector", ["checks.ok", "truth.rows[0]", "candidate.errors", "checks.all", "intervention.output"])
def test_invalid_contract_appends_exact_history_without_a_measurement(tmp_path: Path, selector: str) -> None:
    app, before, pref, payload = pending(tmp_path, resolver_spec={"field_path": selector, "equals": True})
    command = Command(command_id="local-quarantine", kind="resolve", mission_id=before.mission_id,
        expected_revision=before.current_revision,
        payload={"mode": "invalidate-contract", "prediction_ref": pref.model_dump(), "reason": "Malformed JSON selector"})
    result = app.execute(command)
    assert not result.errors and result.outputs["status"] == "invalid" and result.outputs["outcome"] is None
    new = ObjectRef.model_validate(result.outputs["prediction_ref"])
    with app.store.connection(readonly=True) as db:
        old, current = app.store.get(db, pref), app.store.get(db, new)
        assert old.payload == payload and current.payload == {**payload, "status": "invalid"}
        assert current.predecessor == pref and current.scope == old.scope
        assert current.dependencies == old.dependencies and current.artifacts == old.artifacts
        assert current.payload["outcome"] is None and current.payload["resolution_refs"] == []
        assert "resolution_correction" not in current.payload
        assert not app.store.validity(db, pref)["valid"] and app.store.validity(db, new)["valid"]
        events = app.store.events(db, before.mission_id)
        audit = [e["payload"] for e in events if e["kind"] == "prediction.contract_invalidated"]
        assert len(audit) == 1 and audit[0] == result.outputs["contract_invalidation"]
        assert audit[0]["original_payload_sha256"] == digest(payload)
        assert audit[0]["old_ref"] == pref.model_dump() and audit[0]["new_ref"] == new.model_dump()
        assert audit[0]["confirming_sample"] is False and audit[0]["measurement_ref"] is None
        assert audit[0]["validator_code"] == "resolver_field"
    assert app.execute(command) == result  # A lost reply cannot append another transition.
    inspection = call(app, "inspect", result)
    state = inspection.outputs["state"]
    assert state["predictions"] == [new.model_dump()] and state["prediction_keys"]["frozen-spec"] == new.model_dump()
    assert state["research_queue"] == state["pending_experiments"] == []
    context = next(row for row in inspection.outputs["objects"] if row["revision"]["ref"] == state["context_ref"])
    assert not context["current_validity"]["valid"]
    assert "prediction contract invalidated" in context["current_validity"]["reasons"]
    assert inspection.outputs["saved_receipts"] == [] and inspection.outputs["effects_executed"] == 0
    assert "未获得测量" in chinese(inspection) and "不是新增确认样本" in chinese(result)
    assert call(app, "replay", result).outputs["matches_current"] is True
    exported = call(app, "export", result).outputs
    assert any(row["ref"] == pref.model_dump() and row["payload"] == payload for row in exported["revisions"])
    assert any(e["kind"] == "prediction.contract_invalidated" for e in exported["events"])


@pytest.mark.parametrize("change", [
    {"resolver": "declared/1"}, {"status": "resolved", "outcome": False}, {"status": "invalid"},
    {"status": "censored"}, {"outcome": False},
    {"resolver_spec": {"field": [], "equals": True, "measurement_protocol": "public/1",
                       "source": "public fixture", "supports_scope": "one finite computation"}},
])
def test_no_quarantine_for_valid_waiting_or_terminal_questions(tmp_path: Path, change: dict[str, Any]) -> None:
    app, before, pref, payload = pending(tmp_path, **change)
    result = invalidate(app, before, pref)
    assert result.errors
    with app.store.connection(readonly=True) as db:
        assert app.store.head(db, pref.object_id).payload == payload
        assert app.store.mission(db, before.mission_id)["revision"] == before.current_revision
        assert not any(e["kind"] == "prediction.contract_invalidated" for e in app.store.events(db, before.mission_id))


@pytest.mark.parametrize("extra", [{"observation_ref": {}}, {"outcome": False}, {"recompute": True},
    {"correction_reason": "rewrite"}, {"resolver_spec": {"field": []}}, {"confirming_sample": True}])
def test_quarantine_rejects_measurements_labels_and_rewrites(tmp_path: Path, extra: dict[str, Any]) -> None:
    app, before, pref, _ = pending(tmp_path)
    result = invalidate(app, before, pref, **extra)
    assert result.status == "resolver_input"
    assert call(app, "inspect", before).current_revision == before.current_revision


@pytest.mark.parametrize("guard", ["non-owner", "mission-cas", "object-cas", "dependency-invalid", "foreign", "unknown-mode", "empty-reason"])
def test_quarantine_requires_owner_and_current_intact_local_contract(tmp_path: Path, guard: str) -> None:
    app, before, pref, payload = pending(tmp_path)
    expected, ref, mode, reason = before.current_revision, pref.model_dump(), "invalidate-contract", "invalid selector"
    if guard == "non-owner":
        app.authority = AuthorityContext(actor="reader", source="public reader", can_authorize=False)
    elif guard == "mission-cas":
        expected = "stale"
    elif guard == "object-cas":
        with app.store.transaction() as db:
            app.store.put(db, before.mission_id, "prediction", {**payload, "horizon": "different future"},
                          object_id=pref.object_id, expected=pref)
    elif guard == "dependency-invalid":
        with app.store.transaction() as db:
            old = app.store.get(db, pref)
            app.store.invalidate(db, old.dependencies[0].object_id, old.dependencies[0].revision_id, "withdrawn input")
    elif guard == "foreign":
        _, other, foreign, _ = pending(tmp_path / "other")
        assert other.mission_id != before.mission_id
        ref = foreign.model_dump()
    elif guard == "unknown-mode":
        mode = "force-invalid"
    else:
        reason = "  "
    result = app.execute(Command(kind="resolve", command_id="guard-test", mission_id=before.mission_id,
        expected_revision=expected, payload={"mode": mode, "prediction_ref": ref, "reason": reason}))
    assert result.errors
    with app.store.connection(readonly=True) as db:
        assert app.store.mission(db, before.mission_id)["revision"] == before.current_revision
        assert not any(e["kind"] == "prediction.contract_invalidated" for e in app.store.events(db, before.mission_id))


@pytest.mark.parametrize("status", ["paused", "cancelled"])
def test_quarantine_preserves_paused_or_cancelled_mission(tmp_path: Path, status: str) -> None:
    app, before, pref, _ = pending(tmp_path)
    stopped = call(app, "pause" if status == "paused" else "cancel", before)
    result = invalidate(app, stopped, pref)
    assert not result.errors and result.status == status
    assert call(app, "inspect", result).outputs["state"]["status"] == status


def test_mission_conflict_rolls_back_prediction_head_and_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app, before, pref, payload = pending(tmp_path)
    with app.store.connection(readonly=True) as db:
        events_before = app.store.events(db, before.mission_id)

    def conflict(*args: Any, **kwargs: Any) -> str:
        raise DomainError("conflict", "injected lost mission CAS after prediction append")

    monkeypatch.setattr(app.store, "save_mission", conflict)
    result = invalidate(app, before, pref)
    assert result.status == "conflict"
    with app.store.connection(readonly=True) as db:
        assert app.store.head(db, pref.object_id).ref == pref and app.store.head(db, pref.object_id).payload == payload
        assert app.store.events(db, before.mission_id) == events_before
        assert app.store.validity(db, pref)["valid"]

