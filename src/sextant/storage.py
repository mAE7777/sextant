"""SQLite is the V1 authority; immutable, verified artifacts precede transactions.

Legacy JSONL remains an import/diagnostic format. A store is a cooperative local
runtime, not an authenticity guarantee against a host with database write access.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sextant.domain import (
    ArtifactRef,
    DomainError,
    ObjectRef,
    Revision,
    Scope,
    canonical,
    digest,
    identity,
    utcnow,
)

DDL = """
CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS revisions(
  object_id TEXT NOT NULL, revision_id TEXT NOT NULL, mission_id TEXT NOT NULL,
  kind TEXT NOT NULL, document TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY(object_id,revision_id));
CREATE TABLE IF NOT EXISTS heads(
  object_id TEXT PRIMARY KEY, revision_id TEXT NOT NULL,
  FOREIGN KEY(object_id,revision_id) REFERENCES revisions(object_id,revision_id));
CREATE TABLE IF NOT EXISTS dependencies(
  object_id TEXT NOT NULL, revision_id TEXT NOT NULL,
  input_id TEXT NOT NULL, input_revision TEXT NOT NULL,
  PRIMARY KEY(object_id,revision_id,input_id,input_revision),
  FOREIGN KEY(object_id,revision_id) REFERENCES revisions(object_id,revision_id),
  FOREIGN KEY(input_id,input_revision) REFERENCES revisions(object_id,revision_id));
CREATE INDEX IF NOT EXISTS reverse_dependencies ON dependencies(input_id,input_revision);
CREATE TABLE IF NOT EXISTS invalidations(
  object_id TEXT NOT NULL, revision_id TEXT NOT NULL, reason TEXT NOT NULL,
  at TEXT NOT NULL, PRIMARY KEY(object_id,revision_id,reason));
CREATE TABLE IF NOT EXISTS missions(
  mission_id TEXT PRIMARY KEY, revision TEXT NOT NULL, state TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS events(
  event_id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, seq INTEGER NOT NULL,
  kind TEXT NOT NULL, document TEXT NOT NULL, UNIQUE(mission_id,seq));
CREATE TABLE IF NOT EXISTS commands(
  command_id TEXT PRIMARY KEY, request_hash TEXT NOT NULL, actor TEXT NOT NULL,
  status TEXT NOT NULL, result TEXT, mission_id TEXT, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS operations(
  operation_id TEXT PRIMARY KEY, mission_id TEXT NOT NULL, command_id TEXT NOT NULL,
  input_revision TEXT NOT NULL, inputs TEXT NOT NULL, authorization TEXT NOT NULL,
  state TEXT NOT NULL, fence INTEGER NOT NULL, lease_until REAL NOT NULL,
  cancel_requested INTEGER NOT NULL DEFAULT 0, failure TEXT);
CREATE UNIQUE INDEX IF NOT EXISTS one_active_operation ON operations(mission_id)
 WHERE state IN ('running','reconciliation_required');
CREATE TABLE IF NOT EXISTS attempts(
  attempt_id TEXT PRIMARY KEY, operation_id TEXT NOT NULL, effect_key TEXT NOT NULL,
  executor TEXT NOT NULL, state TEXT NOT NULL, fence INTEGER NOT NULL,
  request TEXT NOT NULL, receipt TEXT, output TEXT, sent_at TEXT,
  UNIQUE(operation_id,effect_key),
  FOREIGN KEY(operation_id) REFERENCES operations(operation_id));
CREATE TABLE IF NOT EXISTS imports(digest TEXT PRIMARY KEY, mission_id TEXT NOT NULL, report TEXT NOT NULL);
"""


class Store:
    def __init__(self, root: Path, *, create: bool = True,
                 fault: Callable[[str], None] | None = None):
        self.root = root.absolute()
        self.database = self.root / "state.sqlite3"
        self.artifacts = self.root / "artifacts"
        self.fault = fault or (lambda _: None)
        if (self.root / "backup.incomplete").exists():
            raise DomainError("backup_incomplete", "backup was interrupted; keep the original store and create a new backup")
        for part in [self.root, *self.root.parents]:
            if part.is_symlink():
                raise DomainError("unsafe_path", f"runtime path contains a symlink: {part}")
        if self.database.is_symlink() or self.artifacts.is_symlink():
            raise DomainError("unsafe_path", "runtime database/artifact root cannot be a symlink")
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
            self.artifacts.mkdir(exist_ok=True)
            with self.connection() as db:
                db.execute("PRAGMA journal_mode=WAL")
                db.executescript(DDL)
                db.execute("INSERT OR IGNORE INTO metadata VALUES('schema_version','1')")
        elif not self.database.is_file():
            raise DomainError("not_found", "runtime database does not exist; query created no files")
        if self.database.exists():
            with self.connection(readonly=True) as db:
                version = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
                if version is None or version[0] != "1":
                    raise DomainError("schema_version", "unsupported runtime schema; restore or migrate a copy")

    @contextmanager
    def connection(self, *, readonly: bool = False) -> Iterator[sqlite3.Connection]:
        if readonly:
            db = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=30)
        else:
            db = sqlite3.connect(self.database, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        if not readonly:
            db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                self.fault("before_commit")
                db.commit()
            except BaseException:
                db.rollback()
                raise
            self.fault("after_commit")

    def artifact_path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise DomainError("artifact_hash", "invalid artifact hash")
        p = self.artifacts / sha256
        if self.artifacts.is_symlink() or p.is_symlink():
            raise DomainError("unsafe_path", "artifact symlinks are forbidden")
        return p

    def put_artifact(self, content: bytes, *, media_type: str = "application/json",
                     source: str = "runtime", locator: str = "whole",
                     access_class: str = "mission-private") -> ArtifactRef:
        sha = hashlib.sha256(content).hexdigest()
        dest = self.artifact_path(sha)
        if not dest.exists():
            fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.artifacts)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
                self.fault("artifact_before_publish")
                # link is an exclusive publish; concurrent identical writes cannot replace bytes.
                try:
                    os.link(temporary, dest)
                except FileExistsError:
                    pass
                directory = os.open(self.artifacts, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
                self.fault("artifact_after_publish")
            finally:
                Path(temporary).unlink(missing_ok=True)
        ref = ArtifactRef(sha256=sha, size=len(content), media_type=media_type,
                          source=source, locator=locator, access_class=access_class)
        self.read_artifact(ref)
        return ref

    def read_artifact(self, ref: ArtifactRef) -> bytes:
        try:
            data = self.artifact_path(ref.sha256).read_bytes()
        except FileNotFoundError as exc:
            raise DomainError("missing_artifact", ref.sha256) from exc
        if len(data) != ref.size or hashlib.sha256(data).hexdigest() != ref.sha256:
            raise DomainError("corrupt_artifact", ref.sha256)
        return data

    def event(self, db: sqlite3.Connection, mission_id: str, kind: str,
              payload: dict[str, Any], *, operation_id: str | None = None,
              attempt_id: str | None = None) -> dict[str, Any]:
        seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE mission_id=?",
                         (mission_id,)).fetchone()[0]
        event = {"schema_version": 1, "event_id": identity("ev"), "mission_id": mission_id,
                 "seq": seq, "kind": kind, "occurred_at": utcnow(), "recorded_at": utcnow(),
                 "operation_id": operation_id, "attempt_id": attempt_id, "payload": payload}
        db.execute("INSERT INTO events VALUES(?,?,?,?,?)",
                   (event["event_id"], mission_id, seq, kind, canonical(event)))
        return event

    def get(self, db: sqlite3.Connection, ref: ObjectRef) -> Revision:
        row = db.execute("SELECT document FROM revisions WHERE object_id=? AND revision_id=?",
                         (ref.object_id, ref.revision_id)).fetchone()
        if row is None:
            raise DomainError("missing_reference", f"{ref.object_id}@{ref.revision_id}")
        revision = Revision.model_validate_json(row[0])
        if revision.ref != ref:
            raise DomainError("reference_type", "reference kind/schema does not match stored object")
        expected = self.revision_hash(revision.ref.object_kind, revision.scope, revision.payload,
                                      revision.dependencies, revision.artifacts, revision.predecessor)
        if expected != ref.revision_id:
            raise DomainError("corrupt_revision", ref.object_id)
        return revision

    @staticmethod
    def revision_hash(kind: str, scope: Scope, payload: dict[str, Any],
                      dependencies: list[ObjectRef], artifacts: list[ArtifactRef],
                      predecessor: ObjectRef | None = None) -> str:
        content = {"object_kind": kind, "schema_version": 1, "scope": scope.model_dump(),
                   "payload": payload, "dependencies": [r.model_dump() for r in dependencies],
                   "artifacts": [a.model_dump() for a in artifacts]}
        if predecessor is not None:
            content["predecessor"] = predecessor.model_dump()
        return digest(content)

    def head(self, db: sqlite3.Connection, object_id: str) -> Revision:
        row = db.execute("SELECT r.document FROM heads h JOIN revisions r "
                         "ON r.object_id=h.object_id AND r.revision_id=h.revision_id "
                         "WHERE h.object_id=?", (object_id,)).fetchone()
        if row is None:
            raise DomainError("not_found", f"object {object_id} does not exist")
        revision = Revision.model_validate_json(row[0])
        return self.get(db, revision.ref)

    def put(self, db: sqlite3.Connection, mission_id: str, kind: str, payload: dict[str, Any],
            *, scope: Scope | None = None, dependencies: list[ObjectRef] | None = None,
            artifacts: list[ArtifactRef] | None = None, object_id: str | None = None,
            expected: ObjectRef | None = None, make_current: bool = True) -> ObjectRef:
        obj = object_id or identity(kind)
        scope, deps, artifacts = scope or Scope(), dependencies or [], artifacts or []
        for dep in deps:
            r = self.get(db, dep)
            if r.mission_id != mission_id:
                raise DomainError("cross_mission_reference", dep.object_id)
        for a in artifacts:
            self.read_artifact(a)
        old = db.execute("SELECT revision_id FROM heads WHERE object_id=?", (obj,)).fetchone()
        predecessor = None
        if old is not None:
            previous = self.head(db, obj)
            predecessor = previous.ref
            if previous.mission_id != mission_id or previous.ref.object_kind != kind:
                raise DomainError("object_identity", "an object cannot change mission or type")
            if expected is not None and previous.ref != expected:
                raise DomainError("conflict", "object head changed; expected reference differs from the current typed head")
        if make_current and old is not None and (expected is None or expected.object_id != obj
                                                or old[0] != expected.revision_id):
            raise DomainError("conflict", "object head changed; supply the current revision")
        if expected is not None and old is None:
            raise DomainError("conflict", "expected object does not exist")
        ref = ObjectRef(object_id=obj, revision_id=self.revision_hash(kind, scope, payload, deps, artifacts, predecessor),
                        object_kind=kind)
        revision = Revision(ref=ref, mission_id=mission_id, scope=scope, payload=payload,
                            dependencies=deps, artifacts=artifacts, predecessor=predecessor)
        inserted = db.execute("INSERT OR IGNORE INTO revisions VALUES(?,?,?,?,?,?)",
                              (obj, ref.revision_id, mission_id, kind, canonical(revision), utcnow()))
        if inserted.rowcount:
            for dep in deps:
                db.execute("INSERT INTO dependencies VALUES(?,?,?,?)",
                           (obj, ref.revision_id, dep.object_id, dep.revision_id))
            self.event(db, mission_id, "revision.created", {"revision": revision.model_dump(mode="json")})
        if make_current:
            if old is not None and old[0] != ref.revision_id:
                self.invalidate(db, obj, old[0], f"superseded by {ref.revision_id}")
            db.execute("INSERT INTO heads VALUES(?,?) ON CONFLICT(object_id) "
                       "DO UPDATE SET revision_id=excluded.revision_id", (obj, ref.revision_id))
            self.event(db, mission_id, "head.changed", {"ref": ref.model_dump()})
        return ref

    def invalidate(self, db: sqlite3.Connection, object_id: str, revision_id: str,
                   reason: str) -> list[str]:
        affected: list[str] = []
        queue = [(object_id, revision_id)]
        seen: set[tuple[str, str]] = set()
        while queue:
            oid, rid = queue.pop()
            if (oid, rid) in seen:
                continue
            seen.add((oid, rid))
            row = db.execute("SELECT mission_id FROM revisions WHERE object_id=? AND revision_id=?",
                             (oid, rid)).fetchone()
            if row is None:
                raise DomainError("missing_reference", oid)
            db.execute("INSERT OR IGNORE INTO invalidations VALUES(?,?,?,?)",
                       (oid, rid, reason, utcnow()))
            self.event(db, row[0], "revision.invalidated", {"object_id": oid, "revision_id": rid,
                                                           "reason": reason})
            affected.append(oid)
            queue.extend((r[0], r[1]) for r in db.execute(
                "SELECT object_id,revision_id FROM dependencies WHERE input_id=? AND input_revision=?",
                (oid, rid)))
        return affected

    def historical_artifacts(self, db: sqlite3.Connection, ref: ObjectRef) -> list[ArtifactRef]:
        """Verify immutable dependency bytes, without asserting current validity.

        Historical research designs can survive expected context supersession.
        Their original revisions and artifacts still must exist and match. This
        deliberately grants no truth, freshness, or execution authority.
        """
        pending, seen, artifacts = [ref], set(), {}
        while pending:
            current = pending.pop()
            key = (current.object_id, current.revision_id)
            if key in seen:
                continue
            seen.add(key)
            revision = self.get(db, current)
            for artifact in revision.artifacts:
                self.read_artifact(artifact)
                artifacts[canonical(artifact)] = artifact
            pending.extend(revision.dependencies)
        return list(artifacts.values())

    def validity(self, db: sqlite3.Connection, ref: ObjectRef) -> dict[str, Any]:
        try:
            pending = [ref]
            seen: set[tuple[str, str]] = set()
            reasons: list[str] = []
            while pending:
                current = pending.pop()
                key = (current.object_id, current.revision_id)
                if key in seen:
                    continue
                seen.add(key)
                r = self.get(db, current)
                now = datetime.now(UTC)
                for boundary, value in (("valid_from", r.scope.valid_from), ("valid_until", r.scope.valid_until)):
                    if value is not None:
                        try:
                            instant = datetime.fromisoformat(value)
                            if instant.tzinfo is None:
                                reasons.append(f"scope {boundary} has no timezone")
                            elif (boundary == "valid_from" and now < instant) or (boundary == "valid_until" and now >= instant):
                                reasons.append(f"scope {boundary} does not include current time")
                        except ValueError:
                            reasons.append(f"scope {boundary} is not an ISO timestamp")
                for a in r.artifacts:
                    self.read_artifact(a)
                reasons.extend(str(row[0]) for row in db.execute(
                    "SELECT reason FROM invalidations WHERE object_id=? AND revision_id=?", key))
                head = db.execute("SELECT revision_id FROM heads WHERE object_id=?", (key[0],)).fetchone()
                if head is None or head[0] != key[1]:
                    reasons.append(f"noncurrent input {key[0]}")
                pending.extend(r.dependencies)
            return {"valid": not reasons, "reasons": sorted(set(reasons))}
        except DomainError as exc:
            return {"valid": False, "reasons": [f"{exc.code}: {exc}"]}

    def mission(self, db: sqlite3.Connection, mission_id: str) -> dict[str, Any]:
        row = db.execute("SELECT state FROM missions WHERE mission_id=?", (mission_id,)).fetchone()
        if row is None:
            raise DomainError("not_found", f"mission {mission_id} does not exist")
        return dict(json.loads(row[0]))

    def save_mission(self, db: sqlite3.Connection, state: dict[str, Any], *, expected: str | None) -> str:
        mid = str(state["mission_id"])
        old = db.execute("SELECT revision FROM missions WHERE mission_id=?", (mid,)).fetchone()
        if (old is None and expected is not None) or (old is not None and old[0] != expected):
            raise DomainError("conflict", "mission changed; inspect before retrying with a new command ID")
        state = dict(state)
        state.pop("revision", None)
        state["generation"] = int(state.get("generation", 0)) + 1
        state["revision"] = digest({"schema_version": 1, "mission_state": state})
        db.execute("INSERT INTO missions VALUES(?,?,?) ON CONFLICT(mission_id) DO UPDATE SET "
                   "revision=excluded.revision,state=excluded.state", (mid, state["revision"], canonical(state)))
        self.event(db, mid, "mission.changed", {"state": state})
        return str(state["revision"])

    def events(self, db: sqlite3.Connection, mission_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        return [json.loads(row[0]) for row in db.execute(
            "SELECT document FROM events WHERE mission_id=? AND seq>? ORDER BY seq", (mission_id, after))]

    def replay(self, db: sqlite3.Connection, mission_id: str) -> dict[str, Any]:
        """Pure reconstruction. Never calls providers, experiments, or dispatchers."""
        state: dict[str, Any] = {}
        heads: dict[str, Any] = {}
        events = self.events(db, mission_id)
        for expected_seq, event in enumerate(events, 1):
            if event["seq"] != expected_seq:
                raise DomainError("event_gap", "mission event sequence is incomplete")
            if event["kind"] == "mission.changed":
                state = event["payload"]["state"]
            elif event["kind"] == "head.changed":
                ref = event["payload"]["ref"]
                heads[ref["object_id"]] = ref
        actual_heads = {r[0]: json.loads(r[1])["ref"] for r in db.execute(
            "SELECT h.object_id,r.document FROM heads h JOIN revisions r ON h.object_id=r.object_id AND "
            "h.revision_id=r.revision_id WHERE r.mission_id=?", (mission_id,))}
        same_state, same_heads = state == self.mission(db, mission_id), heads == actual_heads
        return {"state": state, "heads": heads, "events": len(events), "matches_state": same_state,
                "matches_heads": same_heads, "matches_current": same_state and same_heads, "effects_executed": 0}

    def export(self, db: sqlite3.Connection, mission_id: str) -> dict[str, Any]:
        rows = [Revision.model_validate_json(r[0]) for r in db.execute(
            "SELECT document FROM revisions WHERE mission_id=? ORDER BY created_at,object_id", (mission_id,))]
        artifacts: dict[str, dict[str, Any]] = {}
        for r in rows:
            for a in r.artifacts:
                artifacts[a.sha256] = {"ref": a.model_dump(),
                                       "base64": base64.b64encode(self.read_artifact(a)).decode()}
        events = self.events(db, mission_id)
        # Attempts and interrupted operations can own artifacts before any semantic
        # revision is published. Include every referenced event/request artifact.
        def collect(value: Any) -> None:
            a = ArtifactRef.model_validate(value)
            artifacts[a.sha256] = {"ref": a.model_dump(),
                                   "base64": base64.b64encode(self.read_artifact(a)).decode()}
        artifact_fields = {"effect.sent": "request", "effect.received": "artifact", "effect.reconciled": "artifact",
                           "compiler.stage": "proposal", "proposal.recomputed": "artifact", "import.completed": "source_artifact"}
        for event in events:
            field = artifact_fields.get(event["kind"])
            if field:
                collect(event["payload"][field])
        operations = [dict(r) for r in db.execute("SELECT * FROM operations WHERE mission_id=?", (mission_id,))]
        attempts = [dict(r) for r in db.execute("SELECT a.* FROM attempts a JOIN operations o ON "
                                               "o.operation_id=a.operation_id WHERE o.mission_id=?", (mission_id,))]
        for attempt in attempts:
            collect(json.loads(attempt["request"])["artifact"])
        heads = [dict(r) for r in db.execute("SELECT h.* FROM heads h JOIN revisions r ON "
                                           "h.object_id=r.object_id AND h.revision_id=r.revision_id "
                                           "WHERE r.mission_id=?", (mission_id,))]
        invalid = [dict(r) for r in db.execute("SELECT i.* FROM invalidations i JOIN revisions r ON "
                                             "i.object_id=r.object_id AND i.revision_id=r.revision_id "
                                             "WHERE r.mission_id=?", (mission_id,))]
        package = {"schema_version": 1, "format": "sextant-export/1", "mission": self.mission(db, mission_id),
                   "revisions": [r.model_dump(mode="json") for r in rows], "heads": heads,
                   "invalidations": invalid, "artifacts": artifacts, "events": events,
                   "operations": operations, "attempts": attempts,
                   "authorization_executable": False}
        return {**package, "content_hash": digest(package)}

    def backup(self, destination: Path) -> None:
        if destination.exists():
            raise DomainError("exists", "backup destination already exists")
        # Caller exports artifacts alongside the SQLite snapshot, or uses export for portability.
        with self.connection(readonly=True) as db, sqlite3.connect(destination) as backup:
            db.backup(backup)

    def backup_store(self, destination: Path) -> dict[str, Any]:
        """A verifiable complete snapshot; an interrupted directory is never a restore target."""
        destination = destination.absolute()
        destination.mkdir(parents=False, exist_ok=False)
        marker = destination / "backup.incomplete"
        with marker.open("x") as stream:
            stream.write("in progress\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._sync_directory(destination)
        self._sync_directory(destination.parent)
        self.backup(destination / "state.sqlite3")
        target = destination / "artifacts"
        target.mkdir()
        copied: dict[str, int] = {}
        for path in self.artifacts.iterdir():
            if len(path.name) != 64:
                continue  # staging files are not committed content addresses
            ref = ArtifactRef(sha256=path.name, size=path.stat(follow_symlinks=False).st_size,
                              media_type="application/octet-stream", source="backup verification")
            data = self.read_artifact(ref)
            with (target / path.name).open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            copied[path.name] = len(data)
        with sqlite3.connect(destination / "state.sqlite3") as snapshot:
            snapshot.row_factory = sqlite3.Row
            if snapshot.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or snapshot.execute("PRAGMA foreign_key_check").fetchall():
                raise DomainError("backup_integrity", "backup database failed integrity checks")
            for row in snapshot.execute("SELECT mission_id FROM missions"):
                package = self.export(snapshot, row[0])
                if not set(package["artifacts"]) <= set(copied):
                    raise DomainError("backup_artifact", "backup is missing a referenced artifact")
        report = {"schema_version": 1, "format": "sextant-backup/1", "status": "complete",
                  "created_at": utcnow(), "database_sha256": hashlib.sha256((destination / "state.sqlite3").read_bytes()).hexdigest(),
                  "artifacts": copied, "effects_executed": 0}
        with (destination / "backup-manifest.json").open("x") as stream:
            stream.write(canonical(report) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        with (destination / "state.sqlite3").open("rb") as database:
            os.fsync(database.fileno())
        self._sync_directory(target)
        self._sync_directory(destination)
        self.fault("backup_before_complete")
        marker.unlink()
        self._sync_directory(destination)
        return report

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def checkpoint(self) -> None:
        with self.connection() as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
