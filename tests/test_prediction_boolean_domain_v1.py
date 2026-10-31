"""Finite local contracts and injected public Application fixtures; no live calls."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from test_pending_contract_invalidation_v1 import pending
from test_resolver_contract_v1 import BASE, MEASURED, pair, seeded
from test_runtime_integrity_v1 import call, fresh, proposal, start

from sextant.domain import Command, DomainError, ObjectRef, Prediction, PredictionSpec, canonical
from sextant.grounding import settle

IMPOSSIBLE_DOMAINS = [[], [0, 1], [0.0, 1.0], [None]]


def spec(resolver: str) -> dict[str, Any]:
    if resolver == "field-equals/1":
        return {**BASE, "field": ["value"], "equals": 1}
    return {**BASE, "field": ["value"], "threshold": 3, "comparison": "le",
            "unit": "calls", "unit_field": ["unit"]}


@pytest.mark.parametrize("resolver", ["field-equals/1", "numeric-threshold/1"])
@pytest.mark.parametrize("domain", IMPOSSIBLE_DOMAINS)
def test_rejects_incompatible_domain_before_measurement_context(resolver: str, domain: list[Any]) -> None:
    prediction, observation = pair(spec(resolver), {"value": 1, "unit": "calls"}, resolver)
    prediction.outcome_domain = domain
    observation.observed_at = "2019-01-01T00:00:00+00:00"
    with pytest.raises(DomainError) as error:
        settle(prediction, observation, now=MEASURED)
    assert error.value.code == "resolver_outcome_domain"
    assert canonical(prediction.outcome_domain) == canonical(domain)
    assert prediction.status == "pending" and prediction.outcome is None


@pytest.mark.parametrize("resolver", ["field-equals/1", "numeric-threshold/1"])
@pytest.mark.parametrize("domain", [[False, True], [True, False]])
def test_both_boolean_orders_settle_true_and_false(resolver: str, domain: list[bool]) -> None:
    for value, expected in [(1, True), (4, False)]:
        prediction, observation = pair(spec(resolver), {"value": value, "unit": "calls"}, resolver)
        prediction.outcome_domain = domain
        assert settle(prediction, observation, now=MEASURED)["outcome"] is expected
        data = {key: value for key, value in prediction.model_dump().items()
                if key not in {"frozen_at", "status", "outcome", "resolution_refs", "resolution_correction"}}
        assert Draft202012Validator(PredictionSpec.model_json_schema()).is_valid(data)


@pytest.mark.parametrize("domain", [[0, 1], ["pending", "done"], [None, {"result": 2}], [False, True, "other"]])
def test_declared_arbitrary_domains_keep_their_types_and_authority(domain: list[Any]) -> None:
    from sextant.grounding import validate_prediction_contract

    prediction, observation = pair({}, {"value": "no truth authority"}, "declared/1")
    prediction.outcome_domain = domain
    validate_prediction_contract(prediction)
    assert settle(prediction, observation, now=MEASURED)["status"] == "pending"
    assert canonical(prediction.outcome_domain) == canonical(domain)
    data = prediction.model_dump(exclude={"frozen_at", "status", "outcome", "resolution_refs", "resolution_correction"})
    assert Draft202012Validator(PredictionSpec.model_json_schema()).is_valid(data)


@pytest.mark.parametrize("resolver", ["field-equals/1", "numeric-threshold/1"])
@pytest.mark.parametrize("domain", [[0, 1], [True], [False, False], [False, True, None]])
def test_new_prediction_admission_rejects_without_rewriting_model_contract(tmp_path: Path, resolver: str,
                                                                        domain: list[Any]) -> None:
    generated = proposal()
    prediction, _ = pair(spec(resolver), {"value": 1, "unit": "calls"}, resolver)
    data = PredictionSpec.model_validate(prediction.model_dump(exclude={
        "frozen_at", "status", "outcome", "resolution_refs", "resolution_correction"})).model_dump()
    data["outcome_domain"] = domain
    generated["candidates"][0]["predictions"] = [data]
    app, generator = fresh(tmp_path, generated)
    created = start(app)
    done = call(app, "run", created, steps=1)
    assert not done.errors and generator.requests
    inspection = call(app, "inspect", done)
    assert inspection.outputs["state"]["predictions"] == []
    rejected = [event for event in inspection.events if event["kind"] == "prediction.rejected"]
    if not rejected:
        with app.store.connection(readonly=True) as db:
            rejected = [e for e in app.store.events(db, done.mission_id) if e["kind"] == "prediction.rejected"]
    assert rejected
    with app.store.connection(readonly=True) as db:
        candidate = app.store.get(db, ObjectRef.model_validate(inspection.outputs["state"]["candidates"]["candidate"]))
        assert canonical(candidate.payload["predictions"][0]["outcome_domain"]) == canonical(domain)
    assert canonical(generated["candidates"][0]["predictions"][0]["outcome_domain"]) == canonical(domain)


@pytest.mark.parametrize("resolver", ["field-equals/1", "numeric-threshold/1"])
def test_pending_domain_invalidation_is_append_only_and_idempotent(tmp_path: Path, resolver: str) -> None:
    app, before, pref, original = pending(tmp_path, resolver=resolver, resolver_spec=spec(resolver), outcome_domain=[0, 1])
    command = Command(command_id="domain-invalidation-once", kind="resolve", mission_id=before.mission_id,
        expected_revision=before.current_revision, payload={"mode": "invalidate-contract",
        "prediction_ref": pref.model_dump(), "reason": "Frozen domain cannot represent Boolean settlement"})
    result = app.execute(command)
    assert not result.errors and result.outputs["status"] == "invalid" and result.outputs["outcome"] is None
    assert app.execute(command) == result
    new = ObjectRef.model_validate(result.outputs["prediction_ref"])
    with app.store.connection(readonly=True) as db:
        old, current = app.store.get(db, pref), app.store.get(db, new)
        assert old.payload == original and current.payload == {**original, "status": "invalid"}
        # Published storage schemas must still represent both unchanged legacy
        # questions and their explicit invalid versions; new proposals are stricter.
        stored_schema = Draft202012Validator(Prediction.model_json_schema())
        stored_schema.validate(original)
        stored_schema.validate(current.payload)
        assert current.predecessor == pref and current.artifacts == old.artifacts and current.dependencies == old.dependencies
        events = [e for e in app.store.events(db, before.mission_id) if e["kind"] == "prediction.contract_invalidated"]
        assert len(events) == 1
    audit = result.outputs["contract_invalidation"]
    assert audit["validator_code"] == "resolver_outcome_domain"
    assert audit["measurement_ref"] is None and audit["confirming_sample"] is False
    assert call(app, "replay", result).outputs["matches_current"] is True
    assert any(r["ref"] == pref.model_dump() and r["payload"] == original
               for r in call(app, "export", result).outputs["revisions"])


@pytest.mark.parametrize("resolver", ["field-equals/1", "numeric-threshold/1"])
@pytest.mark.parametrize("failure", ["mission-cas", "object-cas", "post-append-rollback"])
def test_domain_invalidation_guards_rollback_without_rewriting(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                           resolver: str, failure: str) -> None:
    app, before, pref, original = pending(tmp_path, resolver=resolver, resolver_spec=spec(resolver), outcome_domain=[0, 1])
    expected = before.current_revision
    if failure == "mission-cas":
        expected = "stale"
    elif failure == "object-cas":
        with app.store.transaction() as db:
            app.store.put(db, before.mission_id, "prediction", {**original, "horizon": "new question"},
                          object_id=pref.object_id, expected=pref)
    else:
        def conflict(*args: Any, **kwargs: Any) -> str:
            raise DomainError("conflict", "fixture lost mission CAS after append")
        monkeypatch.setattr(app.store, "save_mission", conflict)
    with app.store.connection(readonly=True) as db:
        saved_head = app.store.head(db, pref.object_id)
        saved_events = app.store.events(db, before.mission_id)
    result = app.execute(Command(command_id="domain-guard", kind="resolve", mission_id=before.mission_id,
        expected_revision=expected, payload={"mode": "invalidate-contract", "prediction_ref": pref.model_dump(),
                                           "reason": "Unresolvable domain"}))
    assert result.errors
    with app.store.connection(readonly=True) as db:
        assert app.store.head(db, pref.object_id) == saved_head
        assert app.store.events(db, before.mission_id) == saved_events
        assert app.store.mission(db, before.mission_id)["revision"] == before.current_revision


@pytest.mark.parametrize("value,equals,expected", [(1, True, False), (0, False, False), (False, False, True)])
def test_ordinary_resolution_keeps_measured_value_distinct_from_boolean_outcome(tmp_path: Path, value: Any,
                                                                             equals: Any, expected: bool) -> None:
    app, before, pref, _ = pending(tmp_path, resolver_spec={**BASE, "field": ["value"], "equals": equals})
    _, observation = pair({"field": ["value"], "equals": equals}, {"value": value})
    with app.store.transaction() as db:
        state = app.store.mission(db, before.mission_id)
        oref = app.store.put(db, before.mission_id, "observation", observation.model_dump())
        state["observations"].append(oref.model_dump())
        state["research_queue"] = []  # pending() seeds a deliberately stale invalidation-only queue
        revision = app.store.save_mission(db, state, expected=state["revision"])
    before = before.model_copy(update={"current_revision": revision})
    resolved = call(app, "resolve", before, prediction_ref=pref.model_dump(), observation_ref=oref.model_dump())
    assert not resolved.errors and resolved.outputs["outcome"] is expected
    assert call(app, "inspect", resolved).outputs["saved_receipts"] == []


def test_historical_resolved_numeric_domain_recompute_marks_invalid_not_confirmed(tmp_path: Path) -> None:
    app, before, pref, oref, original = seeded(tmp_path)
    with app.store.transaction() as db:
        state = app.store.mission(db, before.mission_id)
        legacy = {**original, "outcome_domain": [0, 1], "outcome": 0}
        new = app.store.put(db, before.mission_id, "prediction", legacy, object_id=pref.object_id, expected=pref)
        state["predictions"] = [new.model_dump() if r == pref.model_dump() else r for r in state["predictions"]]
        revision = app.store.save_mission(db, state, expected=state["revision"])
    before = before.model_copy(update={"current_revision": revision})
    corrected = call(app, "resolve", before, prediction_ref=new.model_dump(), observation_ref=oref.model_dump(),
        recompute=True, correction_reason="Legacy numeric result domain is incompatible with Boolean resolver")
    assert not corrected.errors and corrected.outputs["status"] == "invalid" and corrected.outputs["outcome"] is None
    metadata = corrected.outputs["recomputation"]
    assert metadata["contract_error_code"] == "resolver_outcome_domain" and metadata["confirming_sample"] is False
    assert metadata["original_outcome"] == 0 and type(metadata["original_outcome"]) is int
    with app.store.connection(readonly=True) as db:
        assert app.store.get(db, new).payload == legacy


@pytest.mark.parametrize("resolver", ["field-equals/1", "numeric-threshold/1"])
def test_ordinary_invalid_domain_rejects_atomically(tmp_path: Path, resolver: str) -> None:
    app, before, pref, original = pending(tmp_path, resolver=resolver, resolver_spec=spec(resolver), outcome_domain=[0, 1])
    _, observation = pair(spec(resolver), {"value": 1, "unit": "calls"}, resolver)
    with app.store.transaction() as db:
        state = app.store.mission(db, before.mission_id)
        oref = app.store.put(db, before.mission_id, "observation", observation.model_dump())
        state["observations"].append(oref.model_dump())
        state["research_queue"] = []
        revision = app.store.save_mission(db, state, expected=state["revision"])
    before = before.model_copy(update={"current_revision": revision})
    result = call(app, "resolve", before, prediction_ref=pref.model_dump(), observation_ref=oref.model_dump())
    assert result.status == "resolver_outcome_domain" and result.outputs == {}
    with app.store.connection(readonly=True) as db:
        assert app.store.head(db, pref.object_id).payload == original
        assert app.store.mission(db, before.mission_id)["revision"] == revision


@pytest.mark.parametrize("value,expected", [(1, True), (4, False)])
def test_ordinary_numeric_resolution_has_boolean_outcome(tmp_path: Path, value: int, expected: bool) -> None:
    resolver = "numeric-threshold/1"
    app, before, pref, _ = pending(tmp_path, resolver=resolver, resolver_spec=spec(resolver))
    _, observation = pair(spec(resolver), {"value": value, "unit": "calls"}, resolver)
    with app.store.transaction() as db:
        state = app.store.mission(db, before.mission_id)
        oref = app.store.put(db, before.mission_id, "observation", observation.model_dump())
        state["observations"].append(oref.model_dump())
        state["research_queue"] = []
        revision = app.store.save_mission(db, state, expected=state["revision"])
    result = call(app, "resolve", before.model_copy(update={"current_revision": revision}),
                  prediction_ref=pref.model_dump(), observation_ref=oref.model_dump())
    assert not result.errors and result.outputs["outcome"] is expected


@pytest.mark.parametrize("resolver", ["field-equals/1", "numeric-threshold/1"])
@pytest.mark.parametrize("domain", [[True], [False, False], [False, 1], [False, True, None], [False, True, True]])
def test_legacy_boolean_members_retain_canonical_resolution(resolver: str, domain: list[Any]) -> None:
    for value, expected in [(1, True), (4, False)]:
        prediction, observation = pair(spec(resolver), {"value": value, "unit": "calls"}, resolver)
        prediction.outcome_domain = domain
        if canonical(expected) in {canonical(member) for member in domain}:
            assert settle(prediction, observation, now=MEASURED)["outcome"] is expected
        else:
            with pytest.raises(DomainError) as error:
                settle(prediction, observation, now=MEASURED)
            assert error.value.code == "resolver_domain"
        assert canonical(prediction.outcome_domain) == canonical(domain)


@pytest.mark.parametrize("resolver", ["field-equals/1", "numeric-threshold/1"])
@pytest.mark.parametrize("domain", [[True], [False, True, None], [False, True, True]])
def test_legacy_settleable_domain_is_not_invalidated_for_admission_preference(tmp_path: Path, resolver: str,
                                                                          domain: list[Any]) -> None:
    app, before, pref, original = pending(tmp_path, resolver=resolver, resolver_spec=spec(resolver), outcome_domain=domain)
    with app.store.connection(readonly=True) as db:
        old_events = app.store.events(db, before.mission_id)
    result = call(app, "resolve", before, mode="invalidate-contract", prediction_ref=pref.model_dump(),
                  reason="Fixture checks new admission preference is not retrospective invalidity")
    assert result.status == "resolver_contract" and result.outputs == {}
    with app.store.connection(readonly=True) as db:
        assert app.store.head(db, pref.object_id).payload == original
        assert app.store.events(db, before.mission_id) == old_events
        assert app.store.mission(db, before.mission_id)["revision"] == before.current_revision


def test_compiler_explains_boolean_outcome_and_exact_source() -> None:
    from sextant.compiler import SYSTEM

    assert "[false,true]" in SYSTEM and "executor:<adapter>:<key>" in SYSTEM
    assert "equals target remains the measured JSON value" in SYSTEM and "Do not rewrite observation sources" in SYSTEM


@pytest.mark.parametrize("resolver", ["field-equals/1", "numeric-threshold/1"])
@pytest.mark.parametrize("context,expected", [
    ("predates", "pending"), ("declared", "pending"), ("withdrawn", "withdrawn"),
    ("source", "resolver_source"), ("resolved", "prediction_state"),
])
def test_legacy_context_and_status_precede_field_grammar(resolver: str, context: str, expected: str) -> None:
    malformed = {**spec(resolver), "field": "value"}
    prediction, observation = pair(malformed, {"value": 1, "unit": "calls"}, resolver)
    if context == "predates":
        observation.observed_at = "2019-01-01T00:00:00+00:00"
    elif context == "declared":
        observation.provenance = "declared"
    elif context == "withdrawn":
        observation.status = "withdrawn"
    elif context == "source":
        observation.source = "another source"
    else:
        prediction.status = "resolved"
        prediction.outcome = True
    if expected == "pending":
        assert settle(prediction, observation, now=MEASURED)["status"] == expected
    else:
        with pytest.raises(DomainError) as error:
            settle(prediction, observation, now=MEASURED)
        assert error.value.code == expected
    assert prediction.resolver_spec["field"] == "value"
