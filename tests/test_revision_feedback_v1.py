"""Actual outgoing revision provenance and granular grant controls; no live calls."""
from __future__ import annotations

import json

import pytest
from test_application_v1 import OWNER, call
from test_jev_context_v1 import Advisory
from test_runtime_integrity_v1 import Capture, proposal

from sextant.application import Application
from sextant.domain import ObjectRef, digest, utcnow
from sextant.storage import Store


def trajectory(tmp_path, density):
    generated = proposal([{"key": "opposing", "adapter": "environment", "action": "experiment", "inputs": {}, "purpose": "one actual fixture measurement"}])
    generated["candidates"][0]["predictions"] = [{
        "question": "Does the next controlled measurement support the note?", "outcome_domain": [False, True],
        "horizon": "next measurement", "resolver": "field-equals/1",
        "resolver_spec": {"field": ["agrees"], "equals": True, "source": "feedback-measurement",
                          "measurement_protocol": "feedback/1", "supports_scope": "delivery"},
        "sampling_rule": "one controlled measurement", "censoring_rule": "remain pending if unavailable"}]
    generator = Capture("codex", {"payload": generated})
    env = Capture("environment", {"observations": [{"content": {"agrees": False}, "source": "feedback-measurement",
        "observed_at": utcnow(), "measurement_protocol": "feedback/1", "supports_scope": "delivery"}]})
    app = Application(Store(tmp_path / "runtime"), OWNER,
                      executors={"codex": generator, "typesafe": Advisory(), "environment": env}, default_generator="codex")
    tools = ["codex", "environment"] + (["typesafe"] if density != "none" else [])
    started = call(app, "start", mission={"goal": "Revise from the retained observation and actual probe"},
                   observations=[{"content": {"private_old_note": "WITHDRAWN_RAW_CONTENT"}, "source": "old-note", "supports_scope": "delivery"}],
                   options={"generator": "codex", "jev_density": density, "challenger": False},
                   authorization={"tools": tools, "send_to": {tool: ["*"] for tool in tools}})
    state = call(app, "inspect", started).outputs["state"]
    old = state["observations"][0]
    generated["evidence_links"] = {"candidate": ["object:" + old["object_id"]]}
    generator.output = {"payload": generated}
    initial = call(app, "run", started, steps=1)
    assert not initial.errors
    pending = call(app, "inspect", initial).outputs["state"]["predictions"][0]
    # Measurement timing is set only after the prediction actually froze.
    env.output["observations"][0]["observed_at"] = utcnow()
    measured = call(app, "run", initial, steps=1)
    assert not measured.errors and len(env.requests) == 1
    withdrawn = call(app, "withdraw", measured, observation_ref=old, reason="ACTUAL_WITHDRAWAL_REASON")
    assert not withdrawn.errors
    generator.output = {"payload": proposal()}
    generator.requests.clear()
    return app, generator, env, withdrawn, pending


@pytest.mark.parametrize("density", ["none", "context_local"])
def test_actual_primary_and_synthesis_receive_exact_withdrawal_and_settlement(tmp_path, density):
    app, generator, env, withdrawn, pending = trajectory(tmp_path, density)
    before = call(app, "inspect", withdrawn).outputs["state"]
    result = call(app, "run", withdrawn, steps=1)
    assert not result.errors and len(env.requests) == 1
    assert [r["purpose"] for r in generator.requests] == ["primary", "synthesis"]
    for request in generator.requests:
        packet = json.loads(request["prompt"])
        feedback = packet["revision_feedback"]
        assert feedback["history_available"] and not feedback["loss"]
        change, = feedback["source_changes"]
        assert change["header"]["status"] == "withdrawn" and not change["active_factual_support"]
        assert change["header"]["correction_reason"] == "ACTUAL_WITHDRAWAL_REASON"
        assert "content" not in change["header"] and "WITHDRAWN_RAW_CONTENT" not in request["prompt"]
        pred, = feedback["predictions"]
        assert pred["ref"]["object_id"] == pending["object_id"]
        assert pred["record"]["status"] == "resolved" and pred["record"]["outcome"] is False
        assert pred["record"]["frozen_at"] and pred["record"]["resolution_refs"]
        assert not pred["current_validity"]["valid"]  # old support withdrawn, outcome record preserved
        assert pred["resolution_sources"][0]["current_validity"]["valid"]
        assert feedback["settlements"][0]["payload"]["outcome"] is False
        assert not feedback["previous_policy"]["current_validity"]["valid"]
        with app.store.connection(readonly=True) as db:
            for row in (change, pred):
                exact = app.store.get(db, ObjectRef.model_validate(row["ref"]))
                assert row["payload_sha256"] == digest(exact.payload)
    after = call(app, "inspect", result).outputs["state"]
    assert after["policy_ref"] != before["policy_ref"]
    with app.store.connection(readonly=True) as db:
        assert app.store.validity(db, ObjectRef.model_validate(after["policy_ref"]))["valid"]


def test_granular_grants_withhold_historical_packet_and_explicitly_report_loss(tmp_path):
    app, generator, env, withdrawn, _ = trajectory(tmp_path, "none")
    state = call(app, "inspect", withdrawn).outputs["state"]
    with app.store.connection(readonly=True) as db:
        current = [raw for raw in state["observations"] if app.store.get(db, ObjectRef.model_validate(raw)).payload["status"] == "active"]
    allowed = [state["mission_ref"]["object_id"], *[raw["object_id"] for raw in current]]
    changed = call(app, "authorize", withdrawn, action="update", authorization={"tools": ["codex"], "send_to": {"codex": allowed}})
    assert not changed.errors
    result = call(app, "run", changed, steps=1)
    assert not result.errors and len(env.requests) == 1
    assert len(generator.requests) == 2
    for request in generator.requests:
        packet = json.loads(request["prompt"])
        feedback = packet["revision_feedback"]
        assert not feedback["history_available"] and feedback["loss"]
        assert feedback["source_changes"] == feedback["predictions"] == feedback["settlements"] == []
        assert "ACTUAL_WITHDRAWAL_REASON" not in request["prompt"] and "WITHDRAWN_RAW_CONTENT" not in request["prompt"]
