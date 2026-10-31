"""Portable, transactional imports. Imported assertions never bring authority."""

from __future__ import annotations

import base64
import hashlib
import json
import sqlite3
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

from sextant.compiler import CompileOptions
from sextant.domain import (
    ArtifactRef,
    Authorization,
    Command,
    CommandResult,
    DomainError,
    Mission,
    ObjectRef,
    Observation,
    Receipt,
    Revision,
    canonical,
    digest,
    utcnow,
)

if TYPE_CHECKING:
    from sextant.application import Application


def import_command(app: Application, command: Command, *, transaction: sqlite3.Connection | None = None) -> CommandResult:
    app._owner()
    package = command.payload["package"]
    if not isinstance(package, dict):
        raise DomainError("invalid_input", "import package must be an object")
    if package.get("format") == "sextant-export/1":
        try:
            return _portable(app, command, package, transaction=transaction)
        except sqlite3.IntegrityError as exc:
            # Still inside the application's command savepoint: preserve the
            # reservation/error receipt while rolling back every imported row.
            raise DomainError("import_structure", "import violates a receiving database identity or reference constraint") from exc
    if package.get("format") == "sextant-legacy-import/1":
        return _legacy(app, command, package, transaction=transaction)
    raise DomainError("import_format", "supported formats: sextant-export/1, sextant-legacy-import/1")


def _state_shape(state: dict[str, Any]) -> None:
    for field in ("generation", "execution_epoch"):
        value = state.get(field, 0)
        if type(value) is not int or value < 0:
            raise DomainError("import_state", f"state.{field} must be a nonnegative integer")
    if type(state.get("schema_version")) is not int or state["schema_version"] != 1 or not isinstance(state.get("status"), str):
        raise DomainError("import_state", "invalid state schema/status")
    for field in ("revision", "last_fingerprint", "protocol_fingerprint"):
        if state.get(field) is not None and not isinstance(state[field], str):
            raise DomainError("import_state", f"state.{field} must be a string or null")
    for field in ("research_queue", "pending_experiments", "completed_experiment_records"):
        if not isinstance(state.get(field, []), list) or any(not isinstance(v, dict) for v in state.get(field, [])):
            raise DomainError("import_state", f"state.{field} must be a list of objects")
    if not isinstance(state.get("completed_experiments", []), list) or any(not isinstance(v, str) for v in state.get("completed_experiments", [])):
        raise DomainError("import_state", "completed experiments must be string identities")
    visits = state.get("branch_visits", {})
    if not isinstance(visits, dict) or any(type(v) is not int or v < 0 for v in visits.values()):
        raise DomainError("import_state", "branch visits must be nonnegative integer counts")


def _serialized(row: dict[str, Any], field: str, expected: type | None = None) -> Any:
    value = json.loads(row[field])
    canonical(value)  # Reject non-JSON finite values accepted by Python's parser.
    if expected is not None and not isinstance(value, expected):
        raise DomainError("import_structure", f"serialized {field} must be {expected.__name__}")
    return value


def _portable(app: Application, command: Command, package: dict[str, Any], *, transaction: sqlite3.Connection | None = None) -> CommandResult:
    for field in ("mission", "artifacts"):
        if not isinstance(package.get(field), dict):
            raise DomainError("invalid_input", f"package.{field} must be an object")
    for field in ("revisions", "heads", "invalidations", "events", "operations", "attempts"):
        values = package.get(field, [] if field in ("operations", "attempts") else None)
        if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
            raise DomainError("invalid_input", f"package.{field} must be a list of objects")
    if any(not isinstance(v, dict) or not isinstance(v.get("base64"), str) for v in package["artifacts"].values()):
        raise DomainError("invalid_input", "artifact entries must have ref and base64 fields")
    shapes: dict[str, dict[str, tuple[type, ...]]] = {
        "heads": {"object_id": (str,), "revision_id": (str,)},
        "invalidations": {"object_id": (str,), "revision_id": (str,), "reason": (str,), "at": (str,)},
        "events": {"event_id": (str,), "seq": (int,), "mission_id": (str,), "kind": (str,), "payload": (dict,)},
        "operations": {"operation_id": (str,), "mission_id": (str,), "command_id": (str,), "input_revision": (str,),
                       "inputs": (str,), "authorization": (str,), "fence": (int,)},
        "attempts": {"attempt_id": (str,), "operation_id": (str,), "effect_key": (str,), "executor": (str,), "state": (str,),
                     "fence": (int,), "request": (str,), "receipt": (str, type(None)), "output": (str, type(None)), "sent_at": (str,)},
    }
    for section, fields in shapes.items():
        for row in package.get(section, []):
            if any(key not in row or type(row[key]) not in types for key, types in fields.items()):
                raise DomainError("invalid_input", f"package.{section} has a malformed record")
    for row in package.get("operations", []):
        if row.get("failure") is not None and not isinstance(row["failure"], str):
            raise DomainError("invalid_input", "operation failure must be a serialized object or null")
    for section, key in (("events", "event_id"), ("operations", "operation_id"), ("attempts", "attempt_id")):
        values = [r[key] for r in package.get(section, [])]
        if len(set(values)) != len(values):
            raise DomainError("import_duplicate", f"duplicate {key}")
    for section, keys in (("invalidations", ("object_id", "revision_id", "reason")),
                          ("attempts", ("operation_id", "effect_key"))):
        values = [tuple(row[key] for key in keys) for row in package.get(section, [])]
        if len(set(values)) != len(values):
            raise DomainError("import_duplicate", f"duplicate {section} identity")
    operation_ids = {r["operation_id"] for r in package.get("operations", [])}
    if any(a["operation_id"] not in operation_ids for a in package.get("attempts", [])):
        raise DomainError("import_operation", "attempt has no imported operation")
    content_hash = package.get("content_hash")
    if package.get("schema_version") != 1 or content_hash != digest({k: v for k, v in package.items() if k != "content_hash"}):
        raise DomainError("import_hash", "unsupported or damaged export")
    with (app.store.connection(readonly=True) if transaction is None else nullcontext(transaction)) as db:
        prior = db.execute("SELECT report FROM imports WHERE digest=?", (content_hash,)).fetchone()
        if prior:
            report = json.loads(prior[0])
            return CommandResult(command_id=command.command_id, mission_id=report["mission_id"], status="already_imported", outputs=report)
    revisions = [Revision.model_validate(r) for r in package["revisions"]]
    state = dict(package["mission"])
    _state_shape(state)
    mid = state["mission_id"]
    if not isinstance(mid, str):
        raise DomainError("import_state", "mission_id must be a string")
    if any(r.mission_id != mid for r in revisions):
        raise DomainError("import_mission", "package contains cross-mission revisions")
    refs = {(r.ref.object_id, r.ref.revision_id): r for r in revisions}
    if len(refs) != len(revisions):
        raise DomainError("import_duplicate", "package contains duplicate revisions")
    heads = {h["object_id"]: h["revision_id"] for h in package["heads"]}
    if len(heads) != len(package["heads"]):
        raise DomainError("import_head", "duplicate heads")

    def state_ref(raw: Any, kind: str | None, *, current: bool = True) -> None:
        ref = ObjectRef.model_validate(raw)
        stored = refs.get((ref.object_id, ref.revision_id))
        if stored is None or stored.ref != ref or (kind is not None and ref.object_kind != kind):
            raise DomainError("import_state", f"mission state requires an in-package {kind} reference")
        if current and heads.get(ref.object_id) != ref.revision_id:
            raise DomainError("import_state", "current mission reference differs from the exported head")

    for field, kind in (("mission_ref", "mission"), ("authorization_ref", "authorization"),
                        ("context_ref", "context"), ("proposal_ref", "proposal"), ("policy_ref", "policy")):
        if field not in state:
            raise DomainError("import_state", f"missing state.{field}; nullable references must be explicitly null")
        if state.get(field) is not None:
            state_ref(state[field], kind)
        elif field in ("mission_ref", "authorization_ref"):
            raise DomainError("import_state", f"missing {field}")
    if state["mission_ref"]["object_id"] != mid:
        raise DomainError("import_state", "mission object identity differs from mission_id")
    for field, kind in (("observations", "observation"), ("claims", "claim"), ("judgments", "judgment"), ("predictions", "prediction")):
        if not isinstance(state.get(field), list):
            raise DomainError("import_state", f"state.{field} must be a list")
        for raw in state[field]:
            state_ref(raw, kind)
    if not isinstance(state.get("candidates"), dict) or not isinstance(state.get("history"), list):
        raise DomainError("import_state", "candidates/history have invalid container types")
    for raw in state["candidates"].values():
        state_ref(raw, "candidate")
    for entry in state["history"]:
        if not isinstance(entry, dict):
            raise DomainError("import_state", "history must contain objects")
        for field, kind in (("policy_ref", "policy"), ("proposal_ref", "proposal"), ("replaced_by", "policy")):
            if entry.get(field) is not None:
                state_ref(entry[field], kind, current=False)
    for revision in revisions:
        if len({canonical(d) for d in revision.dependencies}) != len(revision.dependencies):
            raise DomainError("import_duplicate", "duplicate revision dependency")
        calculated = app.store.revision_hash(revision.ref.object_kind, revision.scope, revision.payload,
                                             revision.dependencies, revision.artifacts, revision.predecessor)
        if calculated != revision.ref.revision_id:
            raise DomainError("import_revision", "revision hash mismatch")
        for dep in revision.dependencies:
            if (dep.object_id, dep.revision_id) not in refs or refs[(dep.object_id, dep.revision_id)].ref != dep:
                raise DomainError("import_reference", "missing or incorrectly typed dependency")
        predecessor = revision.predecessor
        if predecessor is not None and (
            (predecessor.object_id, predecessor.revision_id) not in refs
            or refs[(predecessor.object_id, predecessor.revision_id)].ref != predecessor
            or predecessor.object_id != revision.ref.object_id
            or predecessor.object_kind != revision.ref.object_kind
        ):
            raise DomainError("import_lineage", "missing or mismatched predecessor revision")
    # Validate the entire submitted artifact set before publishing any blob.
    # A malformed import must not leave its rejected bytes in the local store.
    decoded: dict[str, tuple[ArtifactRef, bytes]] = {}
    for sha, entry in package["artifacts"].items():
        ref = ArtifactRef.model_validate(entry["ref"])
        data = base64.b64decode(entry["base64"], validate=True)
        if sha != ref.sha256 or hashlib.sha256(data).hexdigest() != ref.sha256 or len(data) != ref.size:
            raise DomainError("import_artifact", "artifact bytes mismatch declared reference")
        decoded[sha] = (ref, data)

    def artifact_bytes(raw: Any) -> bytes:
        artifact = ArtifactRef.model_validate(raw)
        if artifact.sha256 not in decoded:
            raise DomainError("import_artifact", "referenced artifact is absent from the package")
        data = decoded[artifact.sha256][1]
        if len(data) != artifact.size:
            raise DomainError("import_artifact", "referenced artifact size does not match its bytes")
        return data

    for revision in revisions:
        for artifact in revision.artifacts:
            artifact_bytes(artifact.model_dump())

    for operation in package.get("operations", []):
        if operation["mission_id"] != mid or not 0 <= operation["fence"] < 2**63 - 1:
            raise DomainError("import_operation", "invalid operation mission/fence")
        for raw in _serialized(operation, "inputs", list):
            state_ref(raw, None, current=False)
        state_ref(_serialized(operation, "authorization", dict), "authorization", current=False)
        if operation.get("failure") is not None:
            _serialized(operation, "failure", dict)
    for attempt in package.get("attempts", []):
        if not 0 <= attempt["fence"] < 2**63 - 1 or attempt["state"] not in ("in_flight", "completed", "failed", "unknown", "cancelled"):
            raise DomainError("import_attempt", "invalid attempt state/fence")
        request = _serialized(attempt, "request", dict)
        body = json.loads(artifact_bytes(request["artifact"]))
        if not isinstance(body, dict) or request["hash"] != digest(body):
            raise DomainError("import_attempt", "request artifact/hash mismatch")
        if attempt["receipt"] is None:
            if attempt["output"] is not None or attempt["state"] != "in_flight":
                raise DomainError("import_attempt", "terminal attempt requires a receipt and serialized output")
        else:
            receipt = Receipt.model_validate(_serialized(attempt, "receipt", dict))
            if attempt["output"] is None:
                raise DomainError("import_attempt", "receipt requires a serialized output")
            output = _serialized(attempt, "output")
            if (receipt.executor != attempt["executor"] or receipt.status != attempt["state"]
                    or receipt.request_hash != request["hash"] or receipt.output_hash != digest(output)):
                raise DomainError("import_attempt", "receipt does not match request/executor/output/state")
    options = CompileOptions.model_validate(state["options"])
    events = package["events"]
    if any(e["seq"] != i or e["mission_id"] != mid for i, e in enumerate(events, 1)):
        raise DomainError("import_events", "events are not a complete ordered mission stream")
    attempts = {a["attempt_id"]: a for a in package.get("attempts", [])}
    artifact_fields = {"effect.sent": "request", "effect.received": "artifact", "effect.reconciled": "artifact",
                       "compiler.stage": "proposal", "proposal.recomputed": "artifact", "import.completed": "source_artifact"}
    for event in events:
        op, attempt_id, payload = event.get("operation_id"), event.get("attempt_id"), event["payload"]
        if op is not None and (not isinstance(op, str) or op not in operation_ids):
            raise DomainError("import_events", "event operation is absent")
        if attempt_id is not None and (not isinstance(attempt_id, str) or attempt_id not in attempts or attempts[attempt_id]["operation_id"] != op):
            raise DomainError("import_events", "event attempt is absent or belongs to another operation")
        if event["kind"] == "head.changed":
            state_ref(payload["ref"], None, current=False)
        if event["kind"] == "mission.changed" and not isinstance(payload.get("state"), dict):
            raise DomainError("import_events", "historical mission state must be an object")
        if event["kind"] in artifact_fields:
            content = artifact_bytes(payload[artifact_fields[event["kind"]]])
            if event["kind"] in ("effect.received", "effect.reconciled"):
                envelope = json.loads(content)
                if not isinstance(envelope, dict) or "output" not in envelope:
                    raise DomainError("import_events", "effect evidence must contain receipt and output")
                Receipt.model_validate(envelope["receipt"])
    with (app.store.transaction() if transaction is None else nullcontext(transaction)) as db:
        if db.execute("SELECT 1 FROM missions WHERE mission_id=?", (mid,)).fetchone():
            raise DomainError("import_collision", "mission exists; use a fresh store for restore/import")
        for revision in revisions:
            if db.execute("SELECT 1 FROM revisions WHERE object_id=?", (revision.ref.object_id,)).fetchone():
                existing = db.execute("SELECT mission_id FROM revisions WHERE object_id=? LIMIT 1", (revision.ref.object_id,)).fetchone()
                if existing[0] != mid:
                    raise DomainError("import_collision", "object identity belongs to another mission")
        for ref, data in decoded.values():
            app.store.put_artifact(data, media_type=ref.media_type, source=ref.source,
                                   locator=ref.locator, access_class=ref.access_class)
        original = app.store.put_artifact(canonical(package).encode(), source="portable-import")
        for revision in revisions:
            db.execute("INSERT INTO revisions VALUES(?,?,?,?,?,?)",
                       (revision.ref.object_id, revision.ref.revision_id, mid, revision.ref.object_kind, canonical(revision), utcnow()))
        for revision in revisions:
            for dep in revision.dependencies:
                db.execute("INSERT INTO dependencies VALUES(?,?,?,?)", (revision.ref.object_id, revision.ref.revision_id, dep.object_id, dep.revision_id))
        for head in package["heads"]:
            if (head["object_id"], head["revision_id"]) not in refs:
                raise DomainError("import_head", "head is absent from revisions")
            db.execute("INSERT INTO heads VALUES(?,?)", (head["object_id"], head["revision_id"]))
        for row in package["invalidations"]:
            if (row["object_id"], row["revision_id"]) not in refs:
                raise DomainError("import_invalidation", "invalidation is absent from revisions")
            db.execute("INSERT INTO invalidations VALUES(?,?,?,?)", (row["object_id"], row["revision_id"], row["reason"], row["at"]))
        for event in events:
            db.execute("INSERT INTO events VALUES(?,?,?,?,?)", (event["event_id"], mid, event["seq"], event["kind"], canonical(event)))
        for operation in package.get("operations", []):
            if operation["mission_id"] != mid:
                raise DomainError("import_operation", "operation belongs to another mission")
            db.execute("INSERT INTO operations VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (operation["operation_id"], mid, operation["command_id"], operation["input_revision"], operation["inputs"],
                        operation["authorization"], "imported", operation["fence"] + 1, 0, 1, operation.get("failure")))
        for attempt in package.get("attempts", []):
            db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?,?)",
                       tuple(attempt[k] for k in ("attempt_id", "operation_id", "effect_key", "executor", "state", "fence", "request", "receipt", "output", "sent_at")))
        # A portable hash is not a receiving-host attestation. Invalidate every
        # imported revision, including detached assertions with invented lineage.
        # Receiving mission/auth/declared observation heads are created below.
        reason = "imported assertion is historical; receiving host has not verified it"
        for revision in revisions:
            imported_ref = revision.ref
            if not db.execute("SELECT 1 FROM invalidations WHERE object_id=? AND revision_id=? AND reason=?",
                              (imported_ref.object_id, imported_ref.revision_id, reason)).fetchone():
                app.store.invalidate(db, imported_ref.object_id, imported_ref.revision_id, reason)
        # All prior provenance is retained, but imported authority and empirical
        # assertions cannot silently become the receiving host's current authority.
        mission_ref = ObjectRef.model_validate(state["mission_ref"])
        mission_data = app.store.get(db, mission_ref).payload.copy()
        mission_data["goal_source"] = f"{app.authority.source}; imported {content_hash}"
        mission = Mission.model_validate(mission_data)
        new_mission = app.store.put(db, mid, "mission", mission.model_dump(), scope=mission.scope,
                                    object_id=mission_ref.object_id, expected=mission_ref, artifacts=[original])
        auth_ref = ObjectRef.model_validate(state["authorization_ref"])
        auth = Authorization(actor=app.authority.actor, source="import grants no authority", mission_id=mid, status="revoked")
        new_auth = app.store.put(db, mid, "authorization", auth.model_dump(), object_id=auth_ref.object_id, expected=auth_ref)
        current_observations = []
        for raw_ref in state["observations"]:
            old_ref = ObjectRef.model_validate(raw_ref)
            old = app.store.get(db, old_ref)
            observation = Observation.model_validate(old.payload)
            observation.provenance = "declared"
            observation.measurement_protocol = "imported-assertion/1"
            observation.supports_scope = "imported record states this; receiving host has not independently verified it"
            observation.supersedes = [old_ref]
            new_ref = app.store.put(db, mid, "observation", observation.model_dump(), scope=old.scope,
                                     object_id=old_ref.object_id, expected=old_ref, artifacts=old.artifacts)
            current_observations.append(new_ref.model_dump())
        state.update(mission_ref=new_mission.model_dump(), authorization_ref=new_auth.model_dump(), observations=current_observations,
                     status="needs_authority", pending_experiments=[], research_queue=[], last_fingerprint=None, options=options.model_dump(),
                     completed_experiments=[], completed_experiment_records=[], branch_visits={}, execution_epoch=0)
        state.pop("batch_progress", None)
        state["predictions"] = []
        state["prediction_keys"] = {}
        state["refused_experiment_records"] = []
        state["representation_state"] = {}
        current_revision = app.store.save_mission(db, state, expected=None)
        report = {"mission_id": mid, "source_hash": content_hash, "source_artifact": original.model_dump(),
                  "identity_mapping": "original IDs retained; mission/observation/authorization heads superseded",
                  "loss": ["all imported assertions invalidated regardless of dependency shape",
                           "historical experiment completion/visits retained in source artifact only; receiving scheduler starts fresh",
                           "imported observations demoted to declared", "old predictions retained as historical only"],
                  "authorization_executable": False, "effects_reexecuted": 0}
        app.store.event(db, mid, "import.completed", report)
        db.execute("INSERT INTO imports VALUES(?,?,?)", (content_hash, mid, canonical(report)))
    return CommandResult(command_id=command.command_id, mission_id=mid, status="needs_authority", current_revision=current_revision,
                         outputs=report, next_actions=["inspect", "authorize update"])


def _legacy(app: Application, command: Command, package: dict[str, Any], *, transaction: sqlite3.Connection | None = None) -> CommandResult:
    content_hash = digest(package)
    with (app.store.connection(readonly=True) if transaction is None else nullcontext(transaction)) as db:
        prior = db.execute("SELECT report FROM imports WHERE digest=?", (content_hash,)).fetchone()
        if prior:
            report = json.loads(prior[0])
            return CommandResult(command_id=command.command_id, mission_id=report["mission_id"], status="already_imported", outputs=report)
    graph = package.get("graph", {})
    ledger = package.get("ledger_lines", [])
    if not isinstance(graph, dict) or not isinstance(ledger, list):
        raise DomainError("legacy_format", "graph must be an object, ledger_lines an array of original strings")
    # Preserve exact original payload. Parse diagnostics without interpreting old
    # stamps/labels as current certificates, commands, or authorizations.
    failures = []
    for i, line in enumerate(ledger):
        try:
            json.loads(line)
        except (ValueError, TypeError):
            failures.append({"line": i + 1, "error": "unparseable historical record retained"})
    start = Command(command_id=command.command_id, kind="start", payload={
        "mission": {"goal": package.get("goal", "Review imported historical decision"), "scope": {"language": "unknown"}},
        "authorization": {"status": "revoked"}, "options": {"jev_density": "none"},
        "observations": [{"content": graph, "source": "legacy-graph", "source_group": "legacy-import", "completeness": "unknown"},
                         {"content": ledger, "source": "legacy-ledger", "source_group": "legacy-import", "completeness": "unknown"}]})
    artifact = app.store.put_artifact(canonical(package).encode(), source="legacy-import")
    with (app.store.transaction() if transaction is None else nullcontext(transaction)) as db:
        result = app._start(start, transaction=db)
        assert result.mission_id is not None
        state = app.store.mission(db, result.mission_id)
        report = {"mission_id": result.mission_id, "source_hash": content_hash, "source_artifact": artifact.model_dump(),
                  "identity_mapping": {"legacy_graph": state["observations"][0], "legacy_ledger": state["observations"][1]},
                  "loss": ["legacy graph semantics retained as declared content; no V1 execution interface inferred",
                           "all legacy thresholds/gate stamps are historical diagnostics only"],
                  "failures": failures, "authorization_executable": False, "effects_reexecuted": 0}
        app.store.put(db, result.mission_id, "migration", report, artifacts=[artifact])
        state["status"] = "needs_authority"
        result.current_revision = app.store.save_mission(db, state, expected=state["revision"])
        app.store.event(db, result.mission_id, "import.completed", report)
        db.execute("INSERT INTO imports VALUES(?,?,?)", (content_hash, result.mission_id, canonical(report)))
    result.status, result.outputs = "needs_authority", report
    result.next_actions = ["inspect", "authorize update"]
    return result
