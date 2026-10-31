"""Constructed resolver and immutable retrospective-correction counterexamples."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from test_application_v1 import OWNER, ControlledEnvironment, ControlledGenerator, call

from sextant.application import Application
from sextant.domain import (
    Command,
    DomainError,
    ObjectRef,
    Observation,
    Prediction,
    canonical,
)
from sextant.grounding import settle
from sextant.storage import Store

FROZEN = "2020-01-01T00:00:00+00:00"
MEASURED = "2020-01-01T00:01:00+00:00"
BASE = {"measurement_protocol": "public-constructed/1", "source": "public-fixture",
        "supports_scope": "one constructed measurement"}


def pair(spec: dict[str, Any], content: Any = None, resolver: str = "field-equals/1") -> tuple[Prediction, Observation]:
    prediction = Prediction(question="Does this exact measurement satisfy the frozen comparison?",
        outcome_domain=[True, False], horizon="one later measurement", resolver=resolver,
        resolver_spec={**BASE, **spec}, sampling_rule="one fixed measurement", censoring_rule="keep pending",
        frozen_at=FROZEN)
    observation = Observation(content={"checks": {"ok": True}} if content is None else content,
        source=BASE["source"], source_group="one fixture", observed_at=MEASURED,
        provenance="constructed", measurement_protocol=BASE["measurement_protocol"],
        supports_scope=BASE["supports_scope"])
    return prediction, observation


@pytest.mark.parametrize("selector", ["field", "field_path", "both"])
@pytest.mark.parametrize("value", [True, False])
def test_explicit_leaf_selectors_preserve_true_and_false(selector: str, value: bool) -> None:
    names = ["field", "field_path"] if selector == "both" else [selector]
    spec = {name: ["checks", "ok"] for name in names}
    prediction, observation = pair({**spec, "equals": True}, {"checks": {"ok": value}})
    assert settle(prediction, observation, now=MEASURED)["outcome"] is value
    assert prediction.resolver_spec == {**BASE, **spec, "equals": True}


@pytest.mark.parametrize("spec", [
    {"equals": True}, {"field": "checks.ok", "equals": True},
    {"field_path": None, "equals": True}, {"field": {}, "equals": True},
    {"field": ["checks", True], "equals": True}, {"field": ["checks", 0.0], "equals": True},
    {"field": ["checks", -1], "equals": True},
    {"field": ["checks", "ok"], "field_path": [], "equals": True},
    {"field": ["checks", "ok"]},
])
def test_missing_malformed_or_conflicting_selector_never_settles_false(spec: dict[str, Any]) -> None:
    prediction, observation = pair(spec)
    with pytest.raises(DomainError): settle(prediction, observation, now=MEASURED)
    assert prediction.status == "pending" and prediction.outcome is None


@pytest.mark.parametrize("path", [["missing"], ["checks", 0], ["checks", "ok", "nested"], ["values", 1]])
def test_absent_or_wrong_container_path_does_not_become_a_false_outcome(path: list[Any]) -> None:
    prediction, observation = pair({"field_path": path, "equals": True}, {"checks": {"ok": True}, "values": [True]})
    with pytest.raises(DomainError, match="field|container"): settle(prediction, observation, now=MEASURED)


@pytest.mark.parametrize("content,expected,path", [(True, True, []), (False, False, []),
    ({"values": [False, True]}, True, ["values", 1]), ({"nullable": None}, None, ["nullable"])])
def test_explicit_root_array_and_null_are_supported(content: Any, expected: Any, path: list[Any]) -> None:
    prediction, observation = pair({"field": path, "equals": expected}, content)
    assert settle(prediction, observation, now=MEASURED)["outcome"] is True


@pytest.mark.parametrize("value", [0, 0.0, "false"])
def test_json_value_types_remain_distinct(value: Any) -> None:
    prediction, observation = pair({"field": ["y"], "equals": False}, {"y": value})
    assert settle(prediction, observation, now=MEASURED)["outcome"] is False


@pytest.mark.parametrize("selector", ["field", "field_path"])
def test_numeric_selector_uses_declared_unit(selector: str) -> None:
    prediction, observation = pair({selector: ["value"], "threshold": 3, "comparison": "le",
        "unit": "calls", "unit_field": ["unit"]}, {"value": 2, "unit": "calls"}, "numeric-threshold/1")
    assert settle(prediction, observation, now=MEASURED)["outcome"] is True
    observation.content["unit"] = "seconds"
    with pytest.raises(DomainError, match="unit"): settle(prediction, observation, now=MEASURED)


def seeded(tmp_path: Path, *, leaf: bool = True) -> tuple[Application, Any, ObjectRef, ObjectRef, dict[str, Any]]:
    app = Application(Store(tmp_path / "runtime"), OWNER)
    start = call(app, "start", mission={"goal": "calculate 2+2"})
    ready = call(app, "run", start)
    state = call(app, "inspect", ready).outputs["state"]
    prediction, observation = pair({"field_path": ["checks", "ok"], "equals": True}, {"checks": {"ok": leaf}})
    # Exact historical defect: missing `field` compared the whole JSON with true.
    prediction.status = "resolved"
    prediction.outcome = canonical(observation.content) == canonical(True)
    with app.store.transaction() as db:
        artifact = app.store.put_artifact(canonical(observation.content).encode(), source="constructed public fixture")
        oref = app.store.put(db, ready.mission_id, "observation", observation.model_dump(), artifacts=[artifact])
        prediction.resolution_refs = [oref]
        pref = app.store.put(db, ready.mission_id, "prediction", prediction.model_dump(),
            dependencies=[ObjectRef.model_validate(state["mission_ref"]), oref], artifacts=[artifact])
        state["observations"].append(oref.model_dump())
        state["predictions"].append(pref.model_dump())
        state["prediction_keys"] = {"public-frozen-spec": pref.model_dump()}
        revision = app.store.save_mission(db, state, expected=state["revision"])
    ready = ready.model_copy(update={"current_revision": revision})
    return app, ready, pref, oref, prediction.model_dump()


def test_versioned_recomputation_preserves_freeze_history_and_receipts(tmp_path: Path) -> None:
    app, before, pref, oref, old = seeded(tmp_path)
    command = Command(command_id="public-correction-once", kind="resolve", mission_id=before.mission_id,
        expected_revision=before.current_revision, payload={"prediction_ref": pref.model_dump(),
        "observation_ref": oref.model_dump(), "recompute": True, "correction_reason": "explicit selector bug"})
    result = app.execute(command)
    assert not result.errors and result.outputs["outcome"] is True
    assert result == app.execute(command)
    corrected = ObjectRef.model_validate(result.outputs["prediction_ref"])
    with app.store.connection(readonly=True) as db:
        original = app.store.get(db, pref)
        current = app.store.get(db, corrected)
        assert original.payload == old and current.predecessor == pref
        assert current.payload["frozen_at"] == old["frozen_at"]
        assert current.payload["resolver_spec"] == old["resolver_spec"]
        assert current.payload["resolution_refs"] == [oref.model_dump()]
        assert current.dependencies == original.dependencies and current.artifacts == original.artifacts
        assert not app.store.validity(db, pref)["valid"] and app.store.validity(db, corrected)["valid"]
    metadata = result.outputs["recomputation"]
    assert metadata["supersedes_prediction_ref"] == pref.model_dump()
    assert metadata["original_outcome"] is False and metadata["confirming_sample"] is False
    snapshot = call(app, "inspect", result).outputs
    assert snapshot["saved_receipts"] == []
    assert snapshot["state"]["prediction_keys"]["public-frozen-spec"] == corrected.model_dump()
    assert call(app, "replay", result).outputs["matches_current"] is True
    exported = call(app, "export", result).outputs
    package = exported.get("package", exported)
    assert any(r["ref"] == pref.model_dump() and r["payload"] == old for r in package["revisions"])
    assert any(e["kind"] == "command.resolve" and e["payload"]["outputs"].get("recomputation") == metadata
               for e in package["events"])


def test_recomputation_that_agrees_is_not_a_new_confirmation(tmp_path: Path) -> None:
    app, before, pref, oref, _ = seeded(tmp_path, leaf=False)
    result = call(app, "resolve", before, prediction_ref=pref.model_dump(), observation_ref=oref.model_dump(),
        recompute=True, correction_reason="check frozen selector")
    assert not result.errors and result.outputs["status"] == "unchanged"
    assert result.current_revision == before.current_revision
    assert call(app, "inspect", result).outputs["state"]["predictions"] == [pref.model_dump()]


@pytest.mark.parametrize("extra", [{}, {"recompute": 1}, {"recompute": True},
    {"recompute": True, "correction_reason": ""}, {"outcome": True},
    {"recompute": False, "correction_reason": "not explicit"}])
def test_no_implicit_correction_or_owner_outcome_label(tmp_path: Path, extra: dict[str, Any]) -> None:
    app, before, pref, oref, old = seeded(tmp_path)
    result = call(app, "resolve", before, prediction_ref=pref.model_dump(), observation_ref=oref.model_dump(), **extra)
    assert result.errors
    with app.store.connection(readonly=True) as db:
        assert app.store.head(db, pref.object_id).payload == old
    assert call(app, "inspect", result).outputs["state"]["predictions"] == [pref.model_dump()]


@pytest.mark.parametrize("invalid", ["new-observation", "withdrawn", "damaged"])
def test_correction_cannot_rebind_or_ignore_invalid_evidence(tmp_path: Path, invalid: str) -> None:
    app, before, pref, oref, old = seeded(tmp_path)
    if invalid == "new-observation":
        before = call(app, "observe", before, content={"checks": {"ok": True}}, source=BASE["source"])
        oref = ObjectRef.model_validate(before.outputs["observation_ref"])
    elif invalid == "withdrawn":
        before = call(app, "withdraw", before, observation_ref=oref.model_dump(), reason="revoked")
    else:
        with app.store.connection(readonly=True) as db:
            artifact = app.store.get(db, oref).artifacts[0]
        app.store.artifact_path(artifact.sha256).write_text("damaged public fixture")
    result = call(app, "resolve", before, prediction_ref=pref.model_dump(), observation_ref=oref.model_dump(),
        recompute=True, correction_reason="check original observation")
    assert result.errors
    with app.store.connection(readonly=True) as db:
        assert app.store.get(db, pref).payload == old


def test_new_freeze_cannot_reuse_old_measurement() -> None:
    prediction, observation = pair({"field_path": ["checks", "ok"], "equals": True})
    prediction.frozen_at = "2020-01-01T00:02:00+00:00"
    assert settle(prediction, observation, now=prediction.frozen_at)["status"] == "pending"


def test_malformed_generated_prediction_is_visible_and_not_published(tmp_path: Path) -> None:
    from sextant.domain import digest
    from sextant.runtime import Invocation

    class BadSelector(ControlledGenerator):
        def invoke(self, request: dict[str, Any], context: Any) -> Invocation:
            response = super().invoke(request, context)
            response.output["payload"]["candidates"][1]["predictions"] = [pair({"equals": True})[0].model_dump(
                exclude={"frozen_at", "status", "outcome", "resolution_refs", "resolution_correction"})]
            return Invocation(response.receipt.model_copy(update={"output_hash": digest(response.output)}), response.output)

    app = Application(Store(tmp_path / "runtime"), OWNER,
        executors={"claude-code": BadSelector(), "environment": ControlledEnvironment()})
    start = call(app, "start", mission={"goal": "Discover mechanism"}, options={"jev_density": "none"},
        authorization={"tools": ["claude-code", "environment"], "send_to": {"claude-code": ["*"], "environment": ["*"]}})
    result = call(app, "run", start, steps=1)
    assert not result.errors
    inspection = call(app, "inspect", result).outputs
    assert inspection["state"]["predictions"] == []
    with app.store.connection(readonly=True) as db:
        assert any(e["kind"] == "prediction.rejected" for e in app.store.events(db, start.mission_id))
