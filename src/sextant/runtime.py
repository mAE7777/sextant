"""Fenced operations and a single, authorization-checked effect sending boundary."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast

from sextant.domain import (
    ArtifactRef,
    Authorization,
    DomainError,
    ObjectRef,
    Receipt,
    canonical,
    digest,
    identity,
    utcnow,
)
from sextant.storage import Store


class NotSent(DomainError):
    """An executor established, before any external request, that it could not run.

    ``kind="unavailable"``: the host lacks a required capability (a native client,
    a credential, an OS sandbox). ``kind="refused"``: this request is outside the
    executor's local permission or input contract. Only the executor can know that
    nothing was sent, so only it raises this; the attempt then fails with its real
    cause instead of becoming an unknown outcome that needs reconciliation.
    """

    def __init__(self, code: str, message: str, *, kind: str = "unavailable",
                 receipt_fields: dict[str, Any] | None = None):
        if kind not in ("unavailable", "refused"):
            raise ValueError("NotSent kind must be unavailable or refused")
        super().__init__(code, message)
        self.kind = kind
        self.receipt_fields = receipt_fields or {}


@dataclass(frozen=True)
class AttemptContext:
    operation_id: str
    attempt_id: str
    mission_id: str
    fence: int
    effect_key: str


@dataclass(frozen=True)
class Invocation:
    receipt: Receipt
    output: Any


class Executor(Protocol):
    name: str

    def capabilities(self) -> dict[str, Any]: ...
    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation: ...
    def cancel(self, context: AttemptContext) -> dict[str, Any]: ...
    def reconcile(self, context: AttemptContext) -> Invocation | None: ...


@dataclass(frozen=True)
class OperationToken:
    operation_id: str
    mission_id: str
    fence: int


class Runtime:
    def __init__(self, store: Store):
        self.store = store

    def begin(self, db: sqlite3.Connection, mission_id: str, command_id: str,
              expected_revision: str, inputs: list[ObjectRef], authorization: ObjectRef,
              *, lease_seconds: float = 1800) -> OperationToken:
        state = self.store.mission(db, mission_id)
        if state["revision"] != expected_revision:
            raise DomainError("conflict", "mission changed before operation start")
        if state["status"] in ("paused", "cancelled"):
            raise DomainError(state["status"], "resume the mission before running")
        existing = db.execute("SELECT operation_id,state FROM operations WHERE mission_id=? AND "
                              "state IN ('running','reconciliation_required')", (mission_id,)).fetchone()
        if existing:
            raise DomainError(existing[1], f"operation {existing[0]} must finish or reconcile first")
        for ref in inputs:
            revision = self.store.get(db, ref)
            if revision.mission_id != mission_id or not self.store.validity(db, ref)["valid"]:
                raise DomainError("stale", "operation input is no longer valid")
        auth = self.store.get(db, authorization)
        if auth.mission_id != mission_id or auth.ref.object_kind != "authorization":
            raise DomainError("authorization", "authorization belongs to another mission or kind")
        self._active_authorization(db, authorization)
        fence = db.execute("SELECT COALESCE(MAX(fence),0)+1 FROM operations WHERE mission_id=?",
                           (mission_id,)).fetchone()[0]
        op = identity("op")
        db.execute("INSERT INTO operations(operation_id,mission_id,command_id,input_revision,inputs,"
                   "authorization,state,fence,lease_until) VALUES(?,?,?,?,?,?,?,?,?)",
                   (op, mission_id, command_id, expected_revision,
                    canonical([r.model_dump() for r in inputs]), canonical(authorization),
                    "running", fence, time.time() + lease_seconds))
        self.store.event(db, mission_id, "operation.started", {"inputs": [r.model_dump() for r in inputs],
                         "fence": fence, "lease_seconds": lease_seconds,
                         "lease_basis": "recoverable process lifetime; adjustable, not a spending cap"},
                         operation_id=op)
        return OperationToken(op, mission_id, fence)

    def _active_authorization(self, db: sqlite3.Connection, ref: ObjectRef) -> Authorization:
        r = self.store.get(db, ref)
        if r.ref.object_kind != "authorization" or not self.store.validity(db, ref)["valid"]:
            raise DomainError("authorization", "authorization revision is no longer current")
        auth = Authorization.model_validate(r.payload)
        if auth.status != "active":
            raise DomainError("revoked", "authorization is revoked")
        if auth.valid_until:
            expiry = datetime.fromisoformat(auth.valid_until)
            if expiry.tzinfo is None or expiry.timestamp() <= time.time():
                raise DomainError("expired_authorization", "authorization is expired or has no timezone")
        return auth

    def guard(self, db: sqlite3.Connection, token: OperationToken) -> sqlite3.Row:
        op = db.execute("SELECT * FROM operations WHERE operation_id=?", (token.operation_id,)).fetchone()
        if op is None or op["mission_id"] != token.mission_id or op["fence"] != token.fence:
            raise DomainError("stale_fence", "worker no longer owns this operation")
        if op["state"] != "running" or op["cancel_requested"]:
            raise DomainError("cancelled", "operation is no longer runnable")
        if op["lease_until"] <= time.time():
            raise DomainError("expired_lease", "worker lease has expired")
        state = self.store.mission(db, token.mission_id)
        if state["status"] in ("paused", "cancelled"):
            raise DomainError(state["status"], "mission does not permit advancement")
        if state["revision"] != op["input_revision"]:
            raise DomainError("stale", "mission changed during the operation")
        self._active_authorization(db, ObjectRef.model_validate_json(op["authorization"]))
        for raw in json.loads(op["inputs"]):
            if not self.store.validity(db, ObjectRef.model_validate(raw))["valid"]:
                raise DomainError("stale", "operation input was superseded or withdrawn")
        return cast(sqlite3.Row, op)

    def renew(self, token: OperationToken, seconds: float = 1800) -> None:
        with self.store.transaction() as db:
            self.guard(db, token)
            db.execute("UPDATE operations SET lease_until=? WHERE operation_id=?",
                       (time.time() + seconds, token.operation_id))

    def _attempt_artifacts(self, db: sqlite3.Connection, attempt: sqlite3.Row,
                           events: list[dict[str, Any]]) -> list[ArtifactRef]:
        """Verify the committed send/receive chain, not just redundant SQL copies."""
        request = json.loads(attempt["request"])
        request_ref = ArtifactRef.model_validate(request["artifact"])
        if digest(json.loads(self.store.read_artifact(request_ref))) != request["hash"]:
            raise DomainError("receipt_mismatch", "request artifact differs from saved request hash")
        own = [e for e in events if e["attempt_id"] == attempt["attempt_id"]]
        sent = [e for e in own if e["kind"] == "effect.sent"]
        received = [e for e in own if e["kind"] in ("effect.received", "effect.reconciled")]
        if len(sent) != 1 or not received or sent[0]["payload"]["request"] != request_ref.model_dump():
            raise DomainError("missing_execution_evidence", "attempt lacks its exact committed send/receive chain")
        artifacts = [request_ref]
        latest: dict[str, Any] = {}
        for event in received:
            ref = ArtifactRef.model_validate(event["payload"]["artifact"])
            saved = json.loads(self.store.read_artifact(ref))
            receipt = Receipt.model_validate(saved["receipt"])
            if (receipt.executor != attempt["executor"] or receipt.request_hash != request["hash"]
                    or receipt.output_hash != digest(saved["output"])):
                raise DomainError("receipt_mismatch", "committed receipt does not bind executor/request/output")
            artifacts.append(ref)
            latest = saved
        if (latest["receipt"] != json.loads(attempt["receipt"]) or latest["output"] != json.loads(attempt["output"])
                or latest["receipt"]["status"] != attempt["state"]):
            raise DomainError("receipt_mismatch", "attempt copies differ from committed receipt artifact")
        return artifacts

    def operation_artifacts(self, db: sqlite3.Connection, token: OperationToken) -> list[ArtifactRef]:
        """Exact evidence of this operation, including any explicitly reused attempt."""
        events = self.store.events(db, token.mission_id)
        attempts = {a["attempt_id"]: a for a in db.execute("SELECT * FROM attempts WHERE operation_id=?", (token.operation_id,))}
        for event in events:
            if event["operation_id"] == token.operation_id and event["kind"] == "effect.reused":
                original = event["payload"]["original_attempt"]
                row = db.execute("SELECT a.* FROM attempts a JOIN operations o ON o.operation_id=a.operation_id "
                                 "WHERE a.attempt_id=? AND o.mission_id=? AND o.state!='imported'", (original, token.mission_id)).fetchone()
                if row is None:
                    raise DomainError("missing_execution_evidence", "reused attempt is absent from this mission")
                attempts[original] = row
        artifacts = [a for attempt in attempts.values() for a in self._attempt_artifacts(db, attempt, events)]
        for event in events:
            if event["operation_id"] == token.operation_id and event["kind"] == "compiler.stage":
                ref = ArtifactRef.model_validate(event["payload"]["proposal"])
                self.store.read_artifact(ref)
                artifacts.append(ref)
        return list({canonical(a): a for a in artifacts}.values())

    def dispatch(self, token: OperationToken, executor: Executor, request: dict[str, Any],
                 *, effect_key: str, send_refs: list[ObjectRef] | None = None,
                 reuse_completed: bool = False) -> Invocation:
        """Claim one send, then invoke outside the transaction. Unknown effects never auto-retry."""
        with self.store.transaction() as db:
            op = self.guard(db, token)
            auth = self._active_authorization(db, ObjectRef.model_validate_json(op["authorization"]))
            if executor.name not in auth.tools:
                raise DomainError("unauthorized_tool", f"{executor.name} is not authorized")
            allowed = auth.send_to.get(executor.name, [])
            for ref in send_refs or []:
                stored = self.store.get(db, ref)
                if stored.mission_id != token.mission_id:
                    raise DomainError("cross_mission_send", ref.object_id)
                if "*" not in allowed and ref.object_id not in allowed:
                    raise DomainError("unauthorized_send", f"{ref.object_id} cannot be sent to {executor.name}")
                if not self.store.validity(db, ref)["valid"]:
                    raise DomainError("stale", "send input is no longer valid")
            if reuse_completed:
                # Recover a received experiment whose subsequent state commit
                # failed. Its stable intent key crosses worker operations; a new
                # explicit execution epoch is required to send the same design again.
                prior = db.execute("SELECT a.* FROM attempts a JOIN operations o ON o.operation_id=a.operation_id "
                                   "WHERE o.mission_id=? AND o.state!='imported' AND a.effect_key=? AND a.executor=? "
                                   "AND a.state='completed' ORDER BY a.sent_at DESC LIMIT 1",
                                   (token.mission_id, effect_key, executor.name)).fetchone()
                if prior is not None:
                    self._attempt_artifacts(db, prior, self.store.events(db, token.mission_id))
                    if json.loads(prior["request"])["hash"] != digest(request):
                        raise DomainError("effect_conflict", "saved effect intent has a different request")
                    receipt = Receipt.model_validate_json(prior["receipt"])
                    output = json.loads(prior["output"])
                    if receipt.request_hash != digest(request) or receipt.output_hash != digest(output):
                        raise DomainError("receipt_mismatch", "saved effect receipt is corrupt")
                    self.store.event(db, token.mission_id, "effect.reused", {"original_attempt": prior["attempt_id"],
                                     "effect_key": effect_key, "effects_sent": 0}, operation_id=token.operation_id)
                    return Invocation(receipt, output)
            existing = db.execute("SELECT * FROM attempts WHERE operation_id=? AND effect_key=?",
                                  (token.operation_id, effect_key)).fetchone()
            if existing:
                if json.loads(existing["request"])["hash"] != digest(request):
                    raise DomainError("effect_conflict", "effect key reused with another request")
                if existing["state"] == "completed":
                    self._attempt_artifacts(db, existing, self.store.events(db, token.mission_id))
                    return Invocation(Receipt.model_validate_json(existing["receipt"]),
                                      json.loads(existing["output"]))
                raise DomainError("reconciliation_required", "prior effect outcome must be reconciled")
            if auth.max_uses is not None:
                used = db.execute("SELECT COUNT(*) FROM attempts a JOIN operations o ON "
                                  "o.operation_id=a.operation_id WHERE o.authorization=?",
                                  (op["authorization"],)).fetchone()[0]
                if used >= auth.max_uses:
                    raise DomainError("authorization_limit", "explicit authorization use limit reached")
            # Existing evidence must be checked before any same-hash write can
            # silently reconstruct its missing request. Only a new send creates
            # a new artifact; recovery is never an implicit side effect of reuse.
            request_artifact = self.store.put_artifact(canonical(request).encode(), source="effect-request")
            attempt = identity("attempt")
            db.execute("INSERT INTO attempts(attempt_id,operation_id,effect_key,executor,state,fence,"
                       "request,sent_at) VALUES(?,?,?,?,?,?,?,?)",
                       (attempt, token.operation_id, effect_key, executor.name, "in_flight", token.fence,
                        canonical({"hash": digest(request), "artifact": request_artifact.model_dump()}), utcnow()))
            self.store.event(db, token.mission_id, "effect.sent", {"effect_key": effect_key,
                             "executor": executor.name, "request": request_artifact.model_dump(),
                             "authorization": json.loads(op["authorization"]), "fence": token.fence},
                             operation_id=token.operation_id, attempt_id=attempt)
        context = AttemptContext(token.operation_id, attempt, token.mission_id, token.fence, effect_key)
        started = utcnow()
        not_sent: NotSent | None = None
        encoded: bytes | None = None
        try:
            invocation = executor.invoke(request, context)
            if invocation.receipt.executor != executor.name:
                raise DomainError("receipt_mismatch", "receipt executor differs from dispatched executor")
            if invocation.receipt.request_hash != digest(request):
                raise DomainError("receipt_mismatch", "receipt is not bound to this request")
            if invocation.receipt.output_hash != digest(invocation.output):
                raise DomainError("receipt_mismatch", "receipt is not bound to its output")
            # Receipt metadata must also be encodable before admitting output.
            # A serialization failure after entry cannot prove nothing was sent.
            encoded = canonical({"receipt": invocation.receipt.model_dump(),
                                 "output": invocation.output}).encode()
        except NotSent as exc:
            # The executor established that no external request left this host.
            not_sent = exc
            receipt = Receipt(executor=executor.name, started_at=started, ended_at=utcnow(),
                              request_hash=digest(request), output_hash=digest(None), status="failed",
                              errors=[f"{exc.code}: {exc} (no external request was sent)"],
                              tool_state=f"not sent: {exc.kind}", residual_processes="none", **exc.receipt_fields)
            invocation = Invocation(receipt, None)
        # An unknown external outcome must be recorded, whatever the exception type.
        except Exception as exc:  # noqa: BLE001
            # Once sent, an exception cannot establish that the external effect did not happen.
            # The local cause is kept (secret-looking environment values removed) so the
            # stop can be diagnosed from the saved receipt.
            from sextant.jev_transport import redact

            receipt = Receipt(executor=executor.name, started_at=started, ended_at=utcnow(),
                              request_hash=digest(request), output_hash=digest(None), status="unknown",
                              errors=[f"{type(exc).__name__}: {redact(str(exc))[:1000]}; the external outcome is unknown"])
            invocation = Invocation(receipt, None)
        if encoded is None:
            encoded = canonical({"receipt": invocation.receipt.model_dump(),
                                 "output": invocation.output}).encode()
        artifact = self.store.put_artifact(encoded, source=f"executor:{executor.name}")
        with self.store.transaction() as db:
            status = invocation.receipt.status
            db.execute("UPDATE attempts SET state=?,receipt=?,output=? WHERE attempt_id=?",
                       (status, canonical(invocation.receipt), canonical(invocation.output), attempt))
            self.store.event(db, token.mission_id, "effect.received", {"artifact": artifact.model_dump(),
                             "status": status}, operation_id=token.operation_id, attempt_id=attempt)
            if status == "unknown":
                db.execute("UPDATE operations SET state='reconciliation_required' WHERE operation_id=? AND state IN ('running','cancelled','reconciliation_required')",
                           (token.operation_id,))
        if not_sent is not None:
            raise not_sent
        if invocation.receipt.status != "completed":
            raise DomainError("reconciliation_required" if invocation.receipt.status == "unknown" else "effect_failed",
                              "; ".join(invocation.receipt.errors) or "executor did not complete")
        return invocation

    def finish(self, db: sqlite3.Connection, token: OperationToken, *, status: str = "completed") -> None:
        self.guard(db, token)
        db.execute("UPDATE operations SET state=? WHERE operation_id=?", (status, token.operation_id))
        self.store.event(db, token.mission_id, "operation.finished", {"status": status},
                         operation_id=token.operation_id)

    def fail(self, token: OperationToken, error: DomainError) -> None:
        with self.store.transaction() as db:
            unknown = db.execute("SELECT COUNT(*) FROM attempts WHERE operation_id=? AND "
                                 "state IN ('in_flight','unknown')", (token.operation_id,)).fetchone()[0]
            status = "reconciliation_required" if unknown else "failed"
            db.execute("UPDATE operations SET state=?,failure=? WHERE operation_id=? AND state!='abandoned'",
                       (status, canonical({"code": error.code, "message": str(error)}), token.operation_id))
            self.store.event(db, token.mission_id, "operation.failed", {"status": status, "code": error.code,
                             "message": str(error)}, operation_id=token.operation_id)

    def cancel(self, db: sqlite3.Connection, mission_id: str, operation_id: str | None = None) -> None:
        rows = db.execute("SELECT operation_id FROM operations WHERE mission_id=? AND "
                          "state IN ('running','reconciliation_required')", (mission_id,)).fetchall()
        for row in rows:
            if operation_id and row[0] != operation_id:
                continue
            pending = db.execute("SELECT COUNT(*) FROM attempts WHERE operation_id=? AND "
                                 "state IN ('in_flight','unknown')", (row[0],)).fetchone()[0]
            state = "reconciliation_required" if pending else "cancelled"
            db.execute("UPDATE operations SET cancel_requested=1,state=?,fence=fence+1 WHERE operation_id=?",
                       (state, row[0]))
            self.store.event(db, mission_id, "operation.cancel_requested", {"state": state,
                             "in_flight_effects": pending}, operation_id=row[0])

    def reconcile(self, token_id: str, executors: dict[str, Executor]) -> dict[str, Any]:
        with self.store.connection(readonly=True) as db:
            op = db.execute("SELECT * FROM operations WHERE operation_id=?", (token_id,)).fetchone()
            if op is None:
                raise DomainError("not_found", token_id)
            attempts = db.execute("SELECT * FROM attempts WHERE operation_id=? AND state IN ('in_flight','unknown')",
                                  (token_id,)).fetchall()
        unresolved = []
        for a in attempts:
            executor = executors.get(a["executor"])
            context = AttemptContext(token_id, a["attempt_id"], op["mission_id"], a["fence"], a["effect_key"])
            result = executor.reconcile(context) if executor else None
            if result is None:
                unresolved.append(a["attempt_id"])
                continue
            if result.receipt.request_hash != json.loads(a["request"])["hash"]:
                raise DomainError("receipt_mismatch", "reconciliation response mismatches request")
            if result.receipt.executor != a["executor"]:
                raise DomainError("receipt_mismatch", "reconciliation response mismatches executor")
            if result.receipt.output_hash != digest(result.output):
                raise DomainError("receipt_mismatch", "reconciliation response mismatches output")
            artifact = self.store.put_artifact(canonical({"receipt": result.receipt.model_dump(),
                                                         "output": result.output}).encode(), source="reconciliation")
            with self.store.transaction() as db:
                db.execute("UPDATE attempts SET state=?,receipt=?,output=? WHERE attempt_id=?",
                           (result.receipt.status, canonical(result.receipt), canonical(result.output), a["attempt_id"]))
                self.store.event(db, op["mission_id"], "effect.reconciled", {"artifact": artifact.model_dump()},
                                 operation_id=token_id, attempt_id=a["attempt_id"])
            if result.receipt.status == "unknown":
                unresolved.append(a["attempt_id"])
        with self.store.transaction() as db:
            if not unresolved:
                db.execute("UPDATE operations SET state='cancelled',cancel_requested=1,fence=fence+1 "
                           "WHERE operation_id=?", (token_id,))
                self.store.event(db, op["mission_id"], "operation.reconciled", {"outputs_current": False},
                                 operation_id=token_id)
        return {"operation_id": token_id, "unresolved": unresolved,
                "status": "reconciliation_required" if unresolved else "reconciled",
                "effects_reexecuted": 0}
