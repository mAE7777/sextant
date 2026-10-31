"""Constructed integrity cases; these are not semantic-model certification."""

from pathlib import Path
from typing import Any

import pytest

from sextant.domain import Authorization, DomainError, ObjectRef, Receipt, digest, utcnow
from sextant.runtime import AttemptContext, Invocation, OperationToken, Runtime
from sextant.storage import Store


def mission(store: Store) -> tuple[str, str, ObjectRef, ObjectRef]:
    with store.transaction() as db:
        inp = store.put(db, "m_test", "mission", {"goal": "diagnose", "goal_source": "test"})
        auth = store.put(db, "m_test", "authorization", Authorization(
            actor="test-owner", source="constructed-test", mission_id="m_test",
            tools=["controlled:counter"], send_to={"controlled:counter": [inp.object_id]},
        ).model_dump())
        rev = store.save_mission(db, {"mission_id": "m_test", "status": "ready",
                                     "input_ref": inp.model_dump(), "authorization_ref": auth.model_dump()}, expected=None)
    return "m_test", rev, inp, auth


def begin(store: Store, runtime: Runtime) -> tuple[OperationToken, ObjectRef, ObjectRef]:
    mid, rev, inp, auth = mission(store)
    with store.transaction() as db:
        token = runtime.begin(db, mid, "test-command", rev, [inp], auth)
    return token, inp, auth


class Counter:
    name = "controlled:counter"

    def __init__(self) -> None:
        self.calls = 0
        self.before_result: Any = None

    def capabilities(self) -> dict[str, Any]:
        return {"evidence": "controlled", "idempotent": False}

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        self.calls += 1
        if self.before_result:
            self.before_result()
        output = {"count": self.calls, "request": request}
        return Invocation(Receipt(executor=self.name, started_at=utcnow(), ended_at=utcnow(),
                                  request_hash=digest(request), output_hash=digest(output),
                                  status="completed"), output)

    def cancel(self, context: AttemptContext) -> dict[str, Any]:
        return {"status": "unsupported"}

    def reconcile(self, context: AttemptContext) -> Invocation | None:
        return None


def test_artifact_publish_failure_leaves_no_committed_reference(tmp_path: Path) -> None:
    def fault(boundary: str) -> None:
        if boundary == "artifact_after_publish":
            raise RuntimeError("constructed crash")
    store = Store(tmp_path / "store", fault=fault)
    with pytest.raises(RuntimeError):
        store.put_artifact(b"durable but not referenced")
    with store.connection(readonly=True) as db:
        assert db.execute("SELECT COUNT(*) FROM revisions").fetchone()[0] == 0
    assert len(list(store.artifacts.iterdir())) == 1


def test_transaction_fault_rolls_back_heads_and_events_together(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    def fail(boundary: str) -> None:
        if boundary == "before_commit":
            raise RuntimeError("constructed crash")
    store.fault = fail
    with pytest.raises(RuntimeError):
        mission(store)
    with store.connection(readonly=True) as db:
        for table in ("revisions", "heads", "events", "missions"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_stable_identity_immutable_revision_and_multihop_invalidation(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    with store.transaction() as db:
        a = store.put(db, "m", "observation", {"value": 1})
        b = store.put(db, "m", "claim", {"rule": "identity"}, dependencies=[a])
        c = store.put(db, "m", "policy", {"action": "retain"}, dependencies=[b])
        unrelated = store.put(db, "m", "claim", {"separate": True})
        new = store.put(db, "m", "observation", {"value": 2}, object_id=a.object_id, expected=a)
        assert a.object_id == new.object_id and a.revision_id != new.revision_id
        assert store.get(db, a).payload == {"value": 1}
        assert not store.validity(db, c)["valid"]
        assert store.validity(db, unrelated)["valid"]
        with pytest.raises(DomainError, match="head changed"):
            store.put(db, "m", "observation", {"value": 3}, object_id=a.object_id, expected=a)


def test_missing_and_wrong_type_references_are_rejected(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    with store.transaction() as db:
        ref = store.put(db, "m", "observation", {})
        with pytest.raises(DomainError, match="kind/schema"):
            store.get(db, ref.model_copy(update={"object_kind": "authorization"}))
        with pytest.raises(DomainError, match="cross|observation"):
            store.put(db, "other", "claim", {}, dependencies=[ref])


def test_revoke_before_send_and_stale_fence_prevent_effect(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    runtime, executor = Runtime(store), Counter()
    token, inp, auth = begin(store, runtime)
    with store.transaction() as db:
        r = store.get(db, auth)
        store.put(db, token.mission_id, "authorization", {**r.payload, "status": "revoked"},
                  object_id=auth.object_id, expected=auth)
    with pytest.raises(DomainError, match="authorization revision"):
        runtime.dispatch(token, executor, {}, effect_key="e", send_refs=[inp])
    assert executor.calls == 0


def test_cancellation_fences_worker_before_send(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    runtime, executor = Runtime(store), Counter()
    token, inp, _ = begin(store, runtime)
    with store.transaction() as db:
        runtime.cancel(db, token.mission_id)
    with pytest.raises(DomainError, match="no longer owns"):
        runtime.dispatch(token, executor, {}, effect_key="e", send_refs=[inp])
    assert executor.calls == 0


def test_revoke_after_send_keeps_receipt_but_cannot_publish_current(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    runtime, executor = Runtime(store), Counter()
    token, inp, auth = begin(store, runtime)
    def revoke() -> None:
        with store.transaction() as db:
            r = store.get(db, auth)
            store.put(db, token.mission_id, "authorization", {**r.payload, "status": "revoked"},
                      object_id=auth.object_id, expected=auth)
    executor.before_result = revoke
    runtime.dispatch(token, executor, {}, effect_key="e", send_refs=[inp])
    assert executor.calls == 1
    with store.transaction() as db:
        assert db.execute("SELECT state FROM attempts").fetchone()[0] == "completed"
        with pytest.raises(DomainError):
            runtime.finish(db, token)


def test_effect_key_idempotence_and_content_conflict(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    runtime, executor = Runtime(store), Counter()
    token, inp, _ = begin(store, runtime)
    a = runtime.dispatch(token, executor, {"x": 1}, effect_key="e", send_refs=[inp])
    b = runtime.dispatch(token, executor, {"x": 1}, effect_key="e", send_refs=[inp])
    assert a == b and executor.calls == 1
    with pytest.raises(DomainError, match="another request"):
        runtime.dispatch(token, executor, {"x": 2}, effect_key="e", send_refs=[inp])


def test_unknown_effect_cannot_retry_and_replay_never_executes(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    runtime, executor = Runtime(store), Counter()
    token, inp, _ = begin(store, runtime)
    def disconnect() -> None:
        raise ConnectionError("effect may already have happened")
    executor.before_result = disconnect
    with pytest.raises(DomainError) as error:
        runtime.dispatch(token, executor, {}, effect_key="e", send_refs=[inp])
    assert error.value.code == "reconciliation_required"
    with pytest.raises(DomainError):
        runtime.dispatch(token, executor, {}, effect_key="e", send_refs=[inp])
    with store.connection(readonly=True) as db:
        replay = store.replay(db, token.mission_id)
        assert replay["effects_executed"] == 0 and replay["matches_current"]
    result = runtime.reconcile(token.operation_id, {executor.name: executor})
    assert result["status"] == "reconciliation_required"
    assert executor.calls == 1


def test_second_worker_cannot_take_active_mission(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    runtime = Runtime(store)
    token, inp, auth = begin(store, runtime)
    other = Runtime(Store(store.root))
    with store.transaction() as db:
        state = store.mission(db, token.mission_id)
        with pytest.raises(DomainError, match="finish or reconcile"):
            other.begin(db, token.mission_id, "other", state["revision"], [inp], auth)


def test_corrupt_artifact_invalidates_dependent_policy(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    artifact = store.put_artifact(b"original")
    with store.transaction() as db:
        a = store.put(db, "m", "observation", {}, artifacts=[artifact])
        policy = store.put(db, "m", "policy", {}, dependencies=[a])
    store.artifact_path(artifact.sha256).write_bytes(b"changed!")
    with store.connection(readonly=True) as db:
        assert not store.validity(db, policy)["valid"]


def test_read_permission_is_not_send_permission(tmp_path: Path) -> None:
    store = Store(tmp_path / "store")
    runtime, executor = Runtime(store), Counter()
    token, _, _ = begin(store, runtime)
    with store.transaction() as db:
        private = store.put(db, token.mission_id, "observation", {"private": True})
    with pytest.raises(DomainError) as error:
        runtime.dispatch(token, executor, {}, effect_key="e", send_refs=[private])
    assert error.value.code == "unauthorized_send" and executor.calls == 0
