"""Invalid frozen contracts and head-visible retrospective provenance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_application_v1 import call
from test_resolver_contract_v1 import seeded

from sextant.domain import ObjectRef


def replace_frozen_spec(app: Any, before: Any, pref: ObjectRef, spec: dict[str, Any]) -> tuple[Any, ObjectRef]:
    # Construct a separate PUBLIC historical record with an unsupported contract.
    # This never rewrites the original revision or a real mission.
    with app.store.transaction() as db:
        previous = app.store.get(db, pref)
        payload = {**previous.payload, "resolver_spec": spec}
        ref = app.store.put(db, before.mission_id, "prediction", payload, scope=previous.scope,
            dependencies=previous.dependencies, artifacts=previous.artifacts,
            object_id=pref.object_id, expected=pref)
        state = app.store.mission(db, before.mission_id)
        state["predictions"] = [ref.model_dump()]
        state["prediction_keys"]["public-frozen-spec"] = ref.model_dump()
        revision = app.store.save_mission(db, state, expected=state["revision"])
    return before.model_copy(update={"current_revision": revision}), ref


@pytest.mark.parametrize("bad", ["dotted-string", "missing-selector", "conflicting-selectors",
                                "bool-index", "float-index", "negative-index"])
def test_unsupported_frozen_contract_is_versioned_invalid_not_a_false_fact(tmp_path: Path, bad: str) -> None:
    app, before, pref, oref, old = seeded(tmp_path)
    spec = dict(old["resolver_spec"])
    if bad == "dotted-string": spec["field_path"] = "checks.ok"
    elif bad == "missing-selector": spec.pop("field_path")
    elif bad == "conflicting-selectors": spec["field"] = []
    else: spec["field_path"] = ["checks", {"bool-index": False, "float-index": 0.0, "negative-index": -1}[bad]]
    before, pref = replace_frozen_spec(app, before, pref, spec)
    with app.store.connection(readonly=True) as db: original = app.store.get(db, pref)
    result = call(app, "resolve", before, prediction_ref=pref.model_dump(), observation_ref=oref.model_dump(),
        recompute=True, correction_reason="unsupported frozen selector contract")
    assert not result.errors and result.outputs["status"] == "invalid" and result.outputs["outcome"] is None
    new = ObjectRef.model_validate(result.outputs["prediction_ref"])
    with app.store.connection(readonly=True) as db:
        current = app.store.get(db, new)
        assert app.store.get(db, pref).payload == original.payload
        assert current.predecessor == pref and current.payload["status"] == "invalid"
        assert current.payload["outcome"] is None and current.payload["resolver_spec"] == spec
        assert current.payload["frozen_at"] == original.payload["frozen_at"]
        assert current.payload["resolution_refs"] == [oref.model_dump()]
        assert current.dependencies == original.dependencies and current.artifacts == original.artifacts
        assert not app.store.validity(db, pref)["valid"]
        metadata = current.payload["resolution_correction"]
        assert metadata == result.outputs["recomputation"]
        assert metadata["original_outcome"] is False and metadata["confirming_sample"] is False
        assert metadata["disposition"] == "invalid-contract" and metadata["contract_error_code"] == "resolver_field"
    inspection = call(app, "inspect", result).outputs
    assert inspection["state"]["prediction_keys"]["public-frozen-spec"] == new.model_dump()
    assert inspection["saved_receipts"] == []
    assert call(app, "replay", result).outputs["matches_current"] is True
    exported = call(app, "export", result).outputs
    package = exported.get("package", exported)
    assert any(r["ref"] == pref.model_dump() and r["payload"] == original.payload for r in package["revisions"])
    assert any(r["ref"] == new.model_dump() and r["payload"]["resolution_correction"] == metadata
               for r in package["revisions"])


def test_corrected_head_itself_exposes_retrospective_nonconfirming_provenance(tmp_path: Path) -> None:
    app, before, pref, oref, old = seeded(tmp_path)
    result = call(app, "resolve", before, prediction_ref=pref.model_dump(), observation_ref=oref.model_dump(),
        recompute=True, correction_reason="repair explicit selector interpretation")
    assert not result.errors and result.outputs["outcome"] is True
    inspection = call(app, "inspect", result).outputs
    ref = inspection["state"]["predictions"][0]
    current = next(r["revision"]["payload"] for r in inspection["objects"] if r["revision"]["ref"] == ref)
    metadata = current["resolution_correction"]
    assert metadata["confirming_sample"] is False and metadata["disposition"] == "recomputed"
    assert metadata["supersedes_prediction_ref"] == pref.model_dump() and metadata["original_outcome"] is False
    assert metadata["original_frozen_at"] == old["frozen_at"] and metadata["observation_ref"] == oref.model_dump()
    assert metadata == result.outputs["recomputation"]


def test_invalid_contract_cannot_be_recomputed_as_a_new_observation(tmp_path: Path) -> None:
    app, before, pref, oref, old = seeded(tmp_path)
    spec = {**old["resolver_spec"], "field_path": "checks.ok"}
    before, pref = replace_frozen_spec(app, before, pref, spec)
    invalid = call(app, "resolve", before, prediction_ref=pref.model_dump(), observation_ref=oref.model_dump(),
        recompute=True, correction_reason="quarantine malformed contract")
    assert not invalid.errors
    newref = invalid.outputs["prediction_ref"]
    again = call(app, "resolve", invalid, prediction_ref=newref, observation_ref=oref.model_dump(),
        recompute=True, correction_reason="must not create a confirming sample")
    assert again.errors
    assert call(app, "inspect", again).outputs["state"]["predictions"] == [newref]


@pytest.mark.parametrize("guard", ["source", "protocol", "scope", "before-freeze", "declared"])
def test_contract_quarantine_does_not_override_measurement_eligibility(tmp_path: Path, guard: str) -> None:
    app, before, pref, oref, old = seeded(tmp_path)
    with app.store.transaction() as db:
        original = app.store.get(db, oref)
        payload = dict(original.payload)
        if guard == "source": payload["source"] = "different-source"
        elif guard == "protocol": payload["measurement_protocol"] = "different/1"
        elif guard == "scope": payload["supports_scope"] = "different scope"
        elif guard == "before-freeze": payload["observed_at"] = "2019-01-01T00:00:00+00:00"
        else: payload["provenance"] = "declared"
        alternate = app.store.put(db, before.mission_id, "observation", payload, artifacts=original.artifacts)
        state = app.store.mission(db, before.mission_id)
        state["observations"].append(alternate.model_dump())
        prediction = app.store.get(db, pref)
        bad = {**prediction.payload, "resolution_refs": [alternate.model_dump()]}
        bad["resolver_spec"] = {**old["resolver_spec"], "field_path": "checks.ok"}
        replacement = app.store.put(db, before.mission_id, "prediction", bad,
            dependencies=[ObjectRef.model_validate(state["mission_ref"]), alternate], artifacts=prediction.artifacts,
            object_id=pref.object_id, expected=pref)
        state["predictions"] = [replacement.model_dump()]
        state["prediction_keys"]["public-frozen-spec"] = replacement.model_dump()
        revision = app.store.save_mission(db, state, expected=state["revision"])
    before = before.model_copy(update={"current_revision": revision})
    result = call(app, "resolve", before, prediction_ref=replacement.model_dump(),
        observation_ref=alternate.model_dump(), recompute=True, correction_reason="must keep measurement guards")
    assert result.errors
    with app.store.connection(readonly=True) as db:
        assert app.store.head(db, pref.object_id).payload == bad
