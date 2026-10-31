"""Public scope/recovery counterexamples; all provider entries are controlled."""

import json

import pytest
from test_application_v1 import OWNER, ControlledGenerator, call

from sextant.application import Application
from sextant.domain import ObjectRef, digest
from sextant.storage import Store


class SourceGenerator(ControlledGenerator):
    name = "codex"

    def invoke(self, request, context):
        result = super().invoke(request, context)
        proposal = result.output["payload"]
        material = json.loads(request["prompt"])["evidence"]
        refs = ["object:" + item["ref"]["object_id"] for item in material]
        for candidate in proposal["candidates"]:
            candidate["experiments"] = []
            candidate["evidence"] = refs
        proposal["policy"]["next_experiments"] = []
        proposal["evidence_links"] = {"Public source-dependent hypothesis": refs}
        result.receipt.output_hash = digest(result.output)
        return result


def start(app):
    return call(app, "start", mission={"goal": "Compare public source-dependent mechanisms",
                "scope": {"subject": "public alpha", "population": "public group A"}},
                observations=[{"content": "Public observation", "source": "public source"}],
                options={"generator": "codex", "jev_density": "none"},
                authorization={"tools": ["codex"], "send_to": {"codex": ["*"]}})


def view(app, result):
    return call(app, "inspect", result).outputs


def claim(snapshot):
    return next(item["revision"] for item in snapshot["objects"]
                if item["revision"]["ref"]["object_kind"] == "claim")


def mission_scope(snapshot):
    return next(item["revision"]["payload"]["scope"] for item in snapshot["objects"]
                if item["revision"]["ref"]["object_kind"] == "mission")


@pytest.mark.parametrize(("key", "value"), [
    ("subject", "public beta"), ("population", "public group B"),
    ("units", {"public metric": "seconds"}), ("representation_version", "public v2"),
])
def test_scope_change_versions_same_supported_claim_without_rewriting_history(tmp_path, key, value):
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"codex": SourceGenerator()})
    first = call(app, "run", start(app), steps=1)
    assert not first.errors
    before = view(app, first)
    old = claim(before)
    scope = {**mission_scope(before), key: value}
    revised = call(app, "revise", first, mission_input={"scope": scope})
    second = call(app, "run", revised, steps=1)
    assert not second.errors
    after = view(app, second)
    current = claim(after)
    assert current["ref"]["object_id"] == old["ref"]["object_id"]
    assert current["ref"]["revision_id"] != old["ref"]["revision_id"]
    assert current["scope"] == mission_scope(after) == scope
    assert current["payload"] == old["payload"]
    assert current["dependencies"] == old["dependencies"]
    with app.store.connection(readonly=True) as db:
        historical = app.store.get(db, ObjectRef.model_validate(old["ref"]))
    assert historical.scope.model_dump(mode="json") == old["scope"]
    assert historical.payload == old["payload"]


def test_unchanged_scope_preserves_same_source_claim_reference(tmp_path):
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"codex": SourceGenerator()})
    first = call(app, "run", start(app), steps=1)
    before = view(app, first)
    revised = call(app, "revise", first, mission_input={"scope": mission_scope(before)})
    second = call(app, "run", revised, steps=1)
    assert not second.errors
    assert claim(view(app, second))["ref"] == claim(before)["ref"]


class BadReceiptGenerator(SourceGenerator):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def invoke(self, request, context):
        result = super().invoke(request, context)
        result.receipt.isolation = {"public-invalid-metadata": self.value}
        return result


@pytest.mark.parametrize("value", [{1, 2}, object(), json.loads('"\\ud800"')],
                         ids=["set", "object", "json-surrogate"])
def test_postentry_receipt_encoding_failure_records_unknown_and_first_reconcile_hint(tmp_path, value):
    generator = BadReceiptGenerator(value)
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"codex": generator})
    first = call(app, "run", start(app), steps=1)
    assert first.status == "reconciliation_required"
    assert "reconcile" in first.next_actions
    assert "revise or run with a new command" not in first.next_actions
    with app.store.connection(readonly=True) as db:
        attempt = db.execute("SELECT state,receipt FROM attempts").fetchone()
        operation = db.execute("SELECT state FROM operations").fetchone()
    assert attempt["state"] == "unknown"
    assert json.loads(attempt["receipt"])["status"] == "unknown"
    assert operation["state"] == "reconciliation_required"
    retry = call(app, "run", first, steps=1)
    assert retry.status == "reconciliation_required"
    assert len(generator.requests) == 1


def test_postentry_artifact_failure_uses_durable_reconciliation_state_in_first_reply(tmp_path, monkeypatch):
    generator = SourceGenerator()
    app = Application(Store(tmp_path / "runtime"), OWNER, executors={"codex": generator})
    created = start(app)
    original = app.store.put_artifact

    def fail_receipt(content, **kwargs):
        if kwargs.get("source") == "executor:codex":
            raise OSError("Public injected receipt write failure")
        return original(content, **kwargs)

    monkeypatch.setattr(app.store, "put_artifact", fail_receipt)
    first = call(app, "run", created, steps=1)
    assert first.status == "reconciliation_required"
    assert "reconcile" in first.next_actions
    with app.store.connection(readonly=True) as db:
        attempt = db.execute("SELECT state,receipt FROM attempts").fetchone()
        operation = db.execute("SELECT state,failure FROM operations").fetchone()
    assert attempt["state"] == "in_flight" and attempt["receipt"] is None
    assert operation["state"] == "reconciliation_required"
    assert json.loads(operation["failure"])["code"] == "internal_error"
    assert "internal_error" in first.errors[0]["message"]
    retry = call(app, "run", first, steps=1)
    assert retry.status == "reconciliation_required"
    assert len(generator.requests) == 1
