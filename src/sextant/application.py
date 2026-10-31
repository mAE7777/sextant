"""One durable application service for humans and agents.

Host authority is a constructor argument. No request, observation, import or
model proposal can create that authority. Queries never execute effects.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, cast

from pydantic import ValidationError

from sextant.compiler import Compilation, CompileOptions, Compiler
from sextant.decision import constraints, observed_scenario_gate
from sextant.domain import (
    ArtifactRef,
    AuthorityContext,
    Authorization,
    Claim,
    Command,
    CommandResult,
    DomainError,
    Experiment,
    Mission,
    ObjectRef,
    Observation,
    Prediction,
    Proposal,
    ResolutionCorrection,
    canonical,
    digest,
    identity,
    utcnow,
)
from sextant.environments import FileSource, PythonSandbox
from sextant.grounding import (
    normalize_policy,
    resolution_context,
    settle,
    validate_prediction_contract,
)
from sextant.jev_client import JevClientError
from sextant.planning_resources import PlanningCallback, planning_material
from sextant.provenance import protocol_fingerprint
from sextant.providers.codex_native import CodexHeadless
from sextant.providers.native import ClaudeHeadless
from sextant.runtime import Executor, NotSent, OperationToken, Runtime
from sextant.storage import Store

QUERY_COMMANDS = {"inspect", "export", "replay"}
LOCAL_COMMANDS = {"start", "import", "revise", "pause", "cancel", "observe", "withdraw", "authorize", "resolve"}
COMMAND_ERRORS = (DomainError, ValidationError, ValueError, KeyError, JevClientError)
# Stored bytes behind an object are missing or differ from their recorded hash.
DAMAGE_REASONS = ("corrupt_artifact", "missing_artifact", "corrupt_revision", "missing_reference")


def world_effect(experiment: Experiment) -> bool:
    return experiment.changes_world or (experiment.adapter == "environment" and
        experiment.action not in ("status", "read_artifact", "observe", "experiment"))


def experiment_signature(state: dict[str, Any], experiment: Experiment) -> str:
    """A measured design is complete only in its recorded task/environment scope."""
    env = state["options"].get("environment", {})
    return digest({"adapter": experiment.adapter, "action": experiment.action, "inputs": experiment.inputs,
                   "mission_revision": state["mission_ref"]["revision_id"],
                   "execution_epoch": state.get("execution_epoch", 0),
                   "environment_scope": {k: env.get(k) for k in ("episode_id", "phase", "revision")}})


class Application:
    def __init__(self, store: Store, authority: AuthorityContext, *,
                 executors: dict[str, Executor] | None = None, default_generator: str | None = None,
                 planning_callback: PlanningCallback | None = None):
        self.store, self.authority = store, authority
        self.runtime = Runtime(store)
        self.planning_callback = planning_callback
        self.protocol_fingerprint = protocol_fingerprint()
        self.executors: dict[str, Executor] = {"claude-code": ClaudeHeadless(),
                                             "codex": CodexHeadless(evidence_root=store.root / "native-evidence"),
                                             "sandbox.python": PythonSandbox()}
        self.executors.update(executors or {})
        # Explicit dependency injection keeps existing integrations/controlled
        # fixtures on their selected port. Normal new missions use the current
        # owner's Codex route; this never retries through another provider.
        self.default_generator = default_generator or ("claude-code" if executors and "claude-code" in executors and "codex" not in executors else "codex")

    def execute(self, command: Command) -> CommandResult:
        request_hash = digest(command)
        try:
            if command.kind in QUERY_COMMANDS:
                self._validate_payload(command)
                return self._query(command)
            with self.store.transaction() as db:
                prior = db.execute("SELECT * FROM commands WHERE command_id=?", (command.command_id,)).fetchone()
                if prior:
                    if prior["request_hash"] != request_hash or prior["actor"] != self.authority.actor:
                        raise DomainError("command_conflict", "command ID already belongs to another request or actor")
                    if prior["result"]:
                        return CommandResult.model_validate_json(prior["result"])
                    return CommandResult(command_id=command.command_id, mission_id=prior["mission_id"],
                                         status="in_progress", next_actions=["inspect", "reconcile if the prior worker stopped"])
                db.execute("INSERT INTO commands VALUES(?,?,?,?,?,?,?)",
                           (command.command_id, request_hash, self.authority.actor, "running", None, command.mission_id, utcnow()))
                if command.kind in LOCAL_COMMANDS:
                    # Reservation, local changes and the returned receipt share
                    # one commit. A lost reply cannot create a second mission or
                    # leave a committed local change permanently in_progress.
                    db.execute("SAVEPOINT command_work")
                    try:
                        self._validate_payload(command)
                        if command.kind == "start":
                            result = self._start(command, transaction=db)
                        elif command.kind == "import":
                            from sextant.migration import import_command
                            result = import_command(self, command, transaction=db)
                        elif command.kind == "revise":
                            result = self._revise(command, transaction=db)
                        else:
                            result = self._mutate(command, transaction=db)
                    except COMMAND_ERRORS as exc:
                        db.execute("ROLLBACK TO command_work")
                        result = self._command_error(command, exc)
                    db.execute("RELEASE command_work")
                    self._save_command(db, command, result)
                    return result
            self._validate_payload(command)
            if command.kind in ("run", "resume"):
                expected = self._resume(command) if command.kind == "resume" else command.expected_revision
                result = self._advance(command, expected_revision=expected)
            elif command.kind == "reconcile":
                result = self._reconcile(command)
            else:
                raise DomainError("command", "unsupported command")
        except COMMAND_ERRORS as exc:
            result = self._command_error(command, exc)
        if command.kind not in QUERY_COMMANDS:
            with self.store.transaction() as db:
                self._save_command(db, command, result)
        return result

    @staticmethod
    def _validate_payload(command: Command) -> None:
        """Check containers before Python coercion, inside the durable command boundary."""
        maps = {"start": ("mission", "authorization", "options"), "revise": ("mission_input", "options"),
                "authorize": ("authorization",), "import": ("package",)}
        for key in maps.get(command.kind, ()):
            if key in command.payload and not isinstance(command.payload[key], dict):
                raise DomainError("invalid_input", f"payload.{key} must be an object")
        if command.kind == "start" and "observations" in command.payload:
            items = command.payload["observations"]
            if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
                raise DomainError("invalid_input", "payload.observations must be a list of objects")
        if command.kind in ("run", "resume") and "steps" in command.payload and type(command.payload["steps"]) is not int:
            raise DomainError("invalid_input", "payload.steps must be an integer")
        for key in ("mode", "operation_id", "action", "disposition", "reason"):
            if key in command.payload and command.kind not in ("observe", "start", "import") and not isinstance(command.payload[key], str):
                raise DomainError("invalid_input", f"payload.{key} must be a string")
        if command.kind in QUERY_COMMANDS and "events_after" in command.payload:
            value = command.payload["events_after"]
            if type(value) is not int or value < 0:
                raise DomainError("invalid_input", "payload.events_after must be a nonnegative integer")

    @staticmethod
    def _command_error(command: Command, exc: Exception) -> CommandResult:
        code = exc.code if isinstance(exc, DomainError) else "invalid_input"
        # User text remains in this local result, never a provider diagnostic.
        return CommandResult(command_id=command.command_id, mission_id=command.mission_id,
                             status=code, errors=[{"code": code, "message": str(exc)}], next_actions=["inspect"])

    def _save_command(self, db: sqlite3.Connection, command: Command, result: CommandResult) -> None:
        # A conflicting request must never replace the first command's result.
        db.execute("UPDATE commands SET status=?,result=?,mission_id=? WHERE command_id=? AND request_hash=? AND actor=?",
                   (result.status, canonical(result), result.mission_id, command.command_id, digest(command), self.authority.actor))

    def _state(self, db: sqlite3.Connection, command: Command, *, check_revision: bool = True) -> dict[str, Any]:
        if command.mission_id is None:
            raise DomainError("mission_required", "this command requires mission_id")
        state = self.store.mission(db, command.mission_id)
        if check_revision:
            if command.expected_revision is None:
                raise DomainError("revision_required", "supply expected_revision from inspect before changing the mission")
            if command.expected_revision != state["revision"]:
                raise DomainError("conflict", "supply the current mission revision from inspect")
        if command.authorization_ref is not None and command.authorization_ref.model_dump() != state["authorization_ref"]:
            raise DomainError("authorization", "command authorization reference is no longer current")
        return state

    def _owner(self) -> None:
        if not self.authority.can_authorize:
            raise DomainError("authority_required", "this operation needs authority supplied by the local owner/host")

    def _check_tools(self, authorization: Authorization) -> None:
        available = {"claude-code", "codex", "typesafe", "sandbox.python", "source.file", "environment", "environment.apply", *self.executors}
        unknown = set(authorization.tools).union(authorization.send_to).difference(available)
        if unknown:
            raise DomainError("authorization_tool", "unknown effect ports in authorization: " + ", ".join(sorted(unknown)))

    def _start(self, command: Command, *, transaction: sqlite3.Connection | None = None) -> CommandResult:
        self._owner()
        payload = command.payload
        mission_data = dict(payload.get("mission", {}))
        mission_data["goal_source"] = self.authority.source
        mission = Mission.model_validate(mission_data)
        options_data = dict(payload.get("options", {}))
        options_data.setdefault("generator", self.default_generator)
        if options_data["generator"] == "codex":
            options_data.setdefault("model", "gpt-6.1-sol")
            options_data.setdefault("effort", "max" if options_data["model"] == "gpt-6-astra" else "xhigh")
        options = CompileOptions.model_validate(options_data)
        mid = identity("mission")
        auth_data = dict(payload.get("authorization", {}))
        auth_data.update(actor=self.authority.actor, source=self.authority.source, mission_id=mid)
        authorization = Authorization.model_validate(auth_data)
        self._check_tools(authorization)
        initial = payload.get("observations", [])
        with (self.store.transaction() if transaction is None else nullcontext(transaction)) as db:
            mref = self.store.put(db, mid, "mission", mission.model_dump(), scope=mission.scope, object_id=mid)
            aref = self.store.put(db, mid, "authorization", authorization.model_dump(), scope=mission.scope)
            state: dict[str, Any] = {"schema_version": 1, "mission_id": mid, "status": "ready", "generation": 0,
                                     "mission_ref": mref.model_dump(), "authorization_ref": aref.model_dump(),
                                     "observations": [], "context_ref": None, "policy_ref": None, "proposal_ref": None,
                                     "candidates": {}, "claims": [], "predictions": [], "judgments": [], "history": [],
                                     "research_queue": [],
                                     "representation_state": {},
                                     "pending_experiments": [], "completed_experiments": [], "completed_experiment_records": [], "branch_visits": {},
                                     "options": options.model_dump(), "last_fingerprint": None, "protocol_fingerprint": None}
            for item in initial:
                ref = self._declared_observation(db, state, item)
                state["observations"].append(ref.model_dump())
            revision = self.store.save_mission(db, state, expected=None)
            return CommandResult(command_id=command.command_id, mission_id=mid, status="ready", current_revision=revision,
                                 outputs={"mission_ref": mref.model_dump(), "authorization_ref": aref.model_dump()},
                                 events=self.store.events(db, mid), next_actions=["run", "observe", "inspect"])

    def _declared_observation(self, db: sqlite3.Connection, state: dict[str, Any], item: dict[str, Any]) -> ObjectRef:
        if not isinstance(item, dict):
            raise DomainError("invalid_input", "observation must be an object")
        # Explicitly ignore attempted provenance promotion in untrusted/manual input.
        allowed = {k: item[k] for k in ("content", "source", "locator", "source_group", "occurred_at", "completeness", "supports_scope") if k in item}
        allowed.setdefault("source", self.authority.source)
        allowed.setdefault("source_group", allowed["source"])
        allowed.setdefault("supports_scope", "the supplied source declares this content; real-world truth unresolved")
        observation = Observation(**allowed, observed_at=utcnow(), provenance="declared", measurement_protocol="owner-declaration/1")
        raw = self.store.put_artifact(canonical(item).encode(), source=str(allowed["source"]))
        return self.store.put(db, state["mission_id"], "observation", observation.model_dump(), artifacts=[raw])

    def _invalidate_view(self, db: sqlite3.Connection, state: dict[str, Any], reason: str) -> list[str]:
        affected = []
        if state.get("context_ref"):
            ref = ObjectRef.model_validate(state["context_ref"])
            affected = self.store.invalidate(db, ref.object_id, ref.revision_id, reason)
        state["last_fingerprint"] = None
        if reason in ("new observation", "experiment observation received", "prediction outcome received"):
            # Preserve untried information experiments from temporarily weaker
            # branches. Never carry a world-changing action into a revised view.
            state["research_queue"] = [e for e in [*state.get("research_queue", []), *state["pending_experiments"]]
                                       if not world_effect(Experiment.model_validate(e["experiment"]))]
        else:
            state["research_queue"] = []
        state["pending_experiments"] = []
        state.pop("batch_progress", None)
        # The host changed the inputs; the next run re-checks and re-records any cause still present.
        state.pop("blocker", None)
        if state["status"] not in ("paused", "cancelled"):
            state["status"] = "ready"
        return affected

    def _research_evidence(self, db: sqlite3.Connection, state: dict[str, Any], entry: dict[str, Any]) -> list[ArtifactRef]:
        """Bind a retained design to its original, intact proposal and protocol."""
        if not entry.get("origin_proposal_ref") or entry.get("protocol_fingerprint") != self.protocol_fingerprint:
            raise DomainError("research_provenance", "research design lacks current-protocol originating evidence")
        origin = ObjectRef.model_validate(entry["origin_proposal_ref"])
        revision = self.store.get(db, origin)
        if origin.object_kind != "proposal" or revision.mission_id != state["mission_id"]:
            raise DomainError("research_provenance", "origin must be a proposal in this mission")
        proposal = Proposal.model_validate(revision.payload)
        designs = (proposal.policy.next_experiments if entry["branch"] == "policy" else
                   [e for c in proposal.candidates if c.key == entry["branch"] for e in c.experiments])
        if not any(canonical(e) == canonical(entry["experiment"]) for e in designs):
            raise DomainError("research_provenance", "queued design differs from its originating proposal")
        return self.store.historical_artifacts(db, origin)

    def _mutate(self, command: Command, *, transaction: sqlite3.Connection | None = None) -> CommandResult:
        with (self.store.transaction() if transaction is None else nullcontext(transaction)) as db:
            state = self._state(db, command)
            before = state["revision"]
            mid = state["mission_id"]
            outputs: dict[str, Any] = {}
            if command.kind in ("pause", "cancel"):
                self.runtime.cancel(db, mid, command.payload.get("operation_id"))
                state["status"] = "paused" if command.kind == "pause" else "cancelled"
                outputs["meaning"] = "further advancement stopped; already sent effects remain recorded for reconciliation"
            elif command.kind == "observe":
                ref = self._declared_observation(db, state, command.payload)
                state["observations"].append(ref.model_dump())
                outputs = {"observation_ref": ref.model_dump(), "invalidated": self._invalidate_view(db, state, "new observation")}
            elif command.kind == "withdraw":
                ref = ObjectRef.model_validate(command.payload["observation_ref"])
                if ref.model_dump() not in state["observations"]:
                    raise DomainError("observation", "withdrawal requires a current mission observation")
                old = self.store.get(db, ref)
                observation = Observation.model_validate(old.payload)
                observation.status = "withdrawn"
                observation.supersedes = [ref]
                observation.correction_reason = str(command.payload.get("reason", "withdrawn by host"))
                artifacts = old.artifacts
                damaged = []
                for artifact in old.artifacts:
                    try:
                        self.store.read_artifact(artifact)
                    except DomainError as exc:
                        damaged.append(f"{exc.code}: {artifact.sha256}")
                if damaged:
                    # Withdrawing is how the host continues without damaged evidence; the
                    # withdrawn revision cannot re-assert bytes that no longer verify.
                    artifacts = []
                    observation.correction_reason += "; original artifacts damaged (" + ", ".join(damaged) + "), kept only on the superseded revision"
                new = self.store.put(db, mid, "observation", observation.model_dump(), scope=old.scope,
                                     artifacts=artifacts, object_id=ref.object_id, expected=ref)
                state["observations"] = [new.model_dump() if r == ref.model_dump() else r for r in state["observations"]]
                outputs = {"observation_ref": new.model_dump(), "invalidated": self._invalidate_view(db, state, "observation withdrawn")}
            elif command.kind == "authorize":
                self._owner()
                old_ref = ObjectRef.model_validate(state["authorization_ref"])
                authorization = Authorization.model_validate(self.store.get(db, old_ref).payload)
                if command.payload.get("action") == "revoke":
                    authorization.status = "revoked"
                elif command.payload.get("action") == "update":
                    data = authorization.model_dump()
                    data.update(command.payload["authorization"])
                    data.update(actor=self.authority.actor, source=self.authority.source, mission_id=mid)
                    authorization = Authorization.model_validate(data)
                else:
                    raise DomainError("authorization_action", "use update or revoke")
                self._check_tools(authorization)
                new_ref = self.store.put(db, mid, "authorization", authorization.model_dump(), object_id=old_ref.object_id, expected=old_ref)
                state["authorization_ref"] = new_ref.model_dump()
                self.runtime.cancel(db, mid)
                # Queued designs may contain data from the prior broader grant.
                # Recompile from inputs permitted by the new grant before sending.
                self._invalidate_view(db, state, "authorization changed")
                outputs["authorization_ref"] = new_ref.model_dump()
                if state["status"] not in ("paused", "cancelled"):
                    state["status"] = "needs_authority" if authorization.status == "revoked" else "ready"
            elif command.kind == "resolve":
                if "mode" in command.payload:
                    outputs = self._invalidate_prediction_contract(db, state, command)
                else:
                    outputs = self._resolve(db, state, command.payload)
                if outputs["status"] in ("resolved", "censored", "invalid"):
                    reason = "prediction contract invalidated" if "contract_invalidation" in outputs else "prediction outcome received"
                    outputs["invalidated"] = self._invalidate_view(db, state, reason)
                else:
                    # A pending outcome carries no new evidence. Log the command
                    # without changing the mission head or invalidating its policy.
                    self.store.event(db, mid, "command.resolve", {"command_id": command.command_id, "outputs": outputs})
                    return CommandResult(command_id=command.command_id, mission_id=mid, status=state["status"],
                                         current_revision=before, outputs=outputs, next_actions=["inspect", "observe"])
            else:
                raise DomainError("command", "unsupported mutation")
            revision = self.store.save_mission(db, state, expected=before)
            self.store.event(db, mid, f"command.{command.kind}", {"command_id": command.command_id, "outputs": outputs})
            return CommandResult(command_id=command.command_id, mission_id=mid, status=state["status"],
                                 current_revision=revision, outputs=outputs,
                                 next_actions=["inspect", "resume" if state["status"] in ("paused", "cancelled") else
                                               "authorize update" if state["status"] == "needs_authority" else "run"])

    def _invalidate_prediction_contract(self, db: sqlite3.Connection, state: dict[str, Any],
                                        command: Command) -> dict[str, Any]:
        """Quarantine an unexecutable frozen question without inventing a measurement."""
        self._owner()
        payload = command.payload
        if set(payload) != {"mode", "prediction_ref", "reason"} or payload["mode"] != "invalidate-contract":
            raise DomainError("resolver_input", "invalidate-contract accepts only mode, prediction_ref and reason")
        if type(payload["reason"]) is not str or not payload["reason"].strip():
            raise DomainError("resolver_input", "contract invalidation requires an explicit host reason")
        pref = ObjectRef.model_validate(payload["prediction_ref"])
        if pref.object_kind != "prediction" or pref.model_dump() not in state["predictions"]:
            raise DomainError("resolver_reference", "prediction must belong to this mission's current view")
        old = self.store.get(db, pref)
        if old.mission_id != state["mission_id"] or self.store.head(db, pref.object_id).ref != pref:
            raise DomainError("resolver_reference", "contract invalidation requires this mission's current prediction head")
        if not self.store.validity(db, pref)["valid"]:
            raise DomainError("stale", "prediction inputs have been invalidated")
        prediction = Prediction.model_validate(old.payload)
        if (prediction.status != "pending" or prediction.outcome is not None or prediction.resolution_refs
                or prediction.resolution_correction is not None):
            raise DomainError("prediction_state", "contract invalidation requires a pending question with no outcome or measurement")
        if prediction.resolver not in ("field-equals/1", "numeric-threshold/1"):
            raise DomainError("resolver_contract", "only malformed supported deterministic contracts can be invalidated")
        try:
            validate_prediction_contract(prediction)
        except DomainError as exc:
            code, message = exc.code, str(exc)
        else:
            raise DomainError("resolver_contract", "this frozen contract is well formed; waiting for a measurement is not invalidity")
        # Validate without dumping defaults or coercions back into saved history.
        revised = {**old.payload, "status": "invalid"}
        Prediction.model_validate(revised)
        affected = self.store.invalidate(db, pref.object_id, pref.revision_id, "prediction contract invalidated")
        new = self.store.put(db, state["mission_id"], "prediction", revised, scope=old.scope,
                             dependencies=old.dependencies, artifacts=old.artifacts,
                             object_id=pref.object_id, expected=pref)
        state["predictions"] = [new.model_dump() if ref == pref.model_dump() else ref for ref in state["predictions"]]
        state["prediction_keys"] = {key: new.model_dump() if ref == pref.model_dump() else ref
                                    for key, ref in state.get("prediction_keys", {}).items()}
        audit = {"command_id": command.command_id, "actor": self.authority.actor, "source": self.authority.source,
                 "old_ref": pref.model_dump(), "new_ref": new.model_dump(), "original_status": "pending",
                 "original_payload_sha256": digest(old.payload),
                 "frozen_contract_sha256": digest({key: value for key, value in old.payload.items()
                     if key not in {"status", "outcome", "resolution_refs", "resolution_correction"}}),
                 "original_frozen_at": prediction.frozen_at, "validator_code": code, "validator_message": message,
                 "reason": payload["reason"], "computed_at": utcnow(), "protocol_fingerprint": self.protocol_fingerprint,
                 "confirming_sample": False, "measurement_ref": None}
        self.store.event(db, state["mission_id"], "prediction.contract_invalidated", audit)
        return {"status": "invalid", "outcome": None, "prediction_ref": new.model_dump(),
                "contract_invalidation": audit, "invalidated_prediction_dependents": affected}

    def _resolve(self, db: sqlite3.Connection, state: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if set(payload) - {"prediction_ref", "observation_ref", "recompute", "correction_reason"}:
            raise DomainError("resolver_input", "resolve accepts references and explicit recomputation metadata, not outcome labels")
        if "recompute" in payload and type(payload["recompute"]) is not bool:
            raise DomainError("resolver_input", "recompute must be an explicit boolean")
        recompute = payload.get("recompute", False)
        if "correction_reason" in payload and not recompute:
            raise DomainError("resolver_input", "correction_reason requires explicit recompute:true")
        pref = ObjectRef.model_validate(payload["prediction_ref"])
        oref = ObjectRef.model_validate(payload["observation_ref"])
        if pref.model_dump() not in state["predictions"] or oref.model_dump() not in state["observations"]:
            raise DomainError("resolver_reference", "prediction and observation must belong to this mission's current view")
        prediction_revision = self.store.get(db, pref)
        prediction = Prediction.model_validate(prediction_revision.payload)
        if not self.store.validity(db, pref)["valid"] or not self.store.validity(db, oref)["valid"]:
            raise DomainError("stale", "resolver inputs have been invalidated")
        observation = Observation.model_validate(self.store.get(db, oref).payload)
        original = prediction.model_copy(deep=True)
        if recompute:
            reason = payload.get("correction_reason")
            if type(reason) is not str or not reason.strip():
                raise DomainError("resolver_input", "retrospective recomputation requires an explicit correction reason")
            if prediction.status != "resolved" or prediction.resolution_refs != [oref]:
                raise DomainError("resolver_correction", "recompute requires a resolved prediction and its exact original measurement")
            # Preserve the original specification/freeze; recompute the same outcome
            # contract, never refreeze a question after seeing its measurement.
            prediction.status = "pending"
            prediction.outcome = None
            prediction.resolution_refs = []
        at = utcnow()
        resolution: dict[str, Any]
        contract_error = None
        if recompute:
            if resolution_context(prediction, observation, now=at) is not None:
                raise DomainError("resolver_correction", "original measurement is not currently eligible for recomputation")
            try:
                validate_prediction_contract(prediction)
            except DomainError as exc:
                contract_error = exc
        if contract_error is not None:
            resolution = {"status": "invalid", "outcome": None, "protocol": prediction.resolver,
                          "resolution_basis": "unsupported frozen deterministic resolver contract; outcome unavailable",
                          "contract_error_code": contract_error.code, "contract_error": str(contract_error)}
        else:
            resolution = settle(prediction, observation, now=at)
        if recompute:
            if resolution["status"] not in ("resolved", "invalid"):
                raise DomainError("resolver_correction", "original contract cannot currently be deterministically resolved")
            if resolution["status"] == "resolved" and canonical(resolution["outcome"]) == canonical(original.outcome):
                return {"status": "unchanged", "prediction_ref": pref.model_dump(),
                        "reason": "recomputation agrees with the saved outcome; no new revision or confirming sample"}
            prediction.resolution_correction = ResolutionCorrection(
                supersedes_prediction_ref=pref, original_status="resolved", original_outcome=original.outcome,
                original_frozen_at=original.frozen_at, observation_ref=oref,
                correction_reason=cast(str, reason), computed_at=at, protocol_fingerprint=self.protocol_fingerprint,
                confirming_sample=False, disposition="invalid-contract" if contract_error is not None else "recomputed",
                contract_error_code=contract_error.code if contract_error is not None else None,
                contract_error=str(contract_error) if contract_error is not None else None,
            )
            resolution["recomputation"] = prediction.resolution_correction.model_dump(mode="json")
        if resolution["status"] in ("resolved", "censored", "invalid"):
            prediction.status = resolution["status"]
            prediction.outcome = resolution.get("outcome")
            prediction.resolution_refs = [oref]
            dependencies = list(prediction_revision.dependencies)
            if oref not in dependencies:
                dependencies.append(oref)
            new = self.store.put(db, state["mission_id"], "prediction", prediction.model_dump(),
                                 scope=prediction_revision.scope, dependencies=dependencies,
                                 artifacts=prediction_revision.artifacts,
                                 object_id=pref.object_id, expected=pref)
            state["predictions"] = [new.model_dump() if p == pref.model_dump() else p for p in state["predictions"]]
            state["prediction_keys"] = {key: new.model_dump() if value == pref.model_dump() else value
                                        for key, value in state.get("prediction_keys", {}).items()}
            resolution["prediction_ref"] = new.model_dump()
        return resolution

    def _revise(self, command: Command, *, transaction: sqlite3.Connection | None = None) -> CommandResult:
        with (self.store.transaction() if transaction is None else nullcontext(transaction)) as db:
            state = self._state(db, command)
            before = state["revision"]
            if "mission_input" in command.payload:
                self._owner()
                old_ref = ObjectRef.model_validate(state["mission_ref"])
                old = self.store.get(db, old_ref)
                data = {**old.payload, **command.payload["mission_input"], "goal_source": self.authority.source}
                mission = Mission.model_validate(data)
                ref = self.store.put(db, state["mission_id"], "mission", mission.model_dump(), scope=mission.scope,
                                     object_id=old_ref.object_id, expected=old_ref)
                state["mission_ref"] = ref.model_dump()
            if "options" in command.payload:
                self._owner()
                revised_options = CompileOptions.model_validate({**state["options"], **command.payload["options"]}).model_dump()
                treatment_fields = ("rerepresentation", "submission_projection")
                if any(state["options"].get(k) != revised_options.get(k) for k in treatment_fields):
                    self.store.event(db, state["mission_id"], "representation.treatment_changed", {
                        "previous": {k: state["options"].get(k) for k in treatment_fields},
                        "current": {k: revised_options.get(k) for k in treatment_fields},
                        "previous_state_hash": digest(state.get("representation_state", {})),
                        "meaning": "explicit host treatment change; prior comparisons cannot be silently continued"})
                    state["representation_state"] = {}
                state["options"] = revised_options
            affected = self._invalidate_view(db, state, "explicit revision requested")
            self.runtime.cancel(db, state["mission_id"])
            revision = self.store.save_mission(db, state, expected=before)
            return CommandResult(command_id=command.command_id, mission_id=state["mission_id"], status=state["status"],
                                 current_revision=revision, outputs={"invalidated": affected, "history_preserved": True},
                                 next_actions=["resume" if state["status"] in ("paused", "cancelled") else "run"])

    def _resume(self, command: Command) -> str:
        with self.store.transaction() as db:
            state = self._state(db, command)
            self.runtime._active_authorization(db, ObjectRef.model_validate(state["authorization_ref"]))
            if state["status"] in ("paused", "cancelled", "needs_authority", "blocked"):
                before = state["revision"]
                state["status"] = "ready"
                state.pop("blocker", None)
                return self.store.save_mission(db, state, expected=before)
            return str(state["revision"])

    def _query(self, command: Command) -> CommandResult:
        with self.store.connection(readonly=True) as db:
            if command.kind == "inspect" and command.mission_id is None:
                missions = [json.loads(row[0]) for row in db.execute("SELECT state FROM missions ORDER BY mission_id")]
                return CommandResult(command_id=command.command_id, status="ok", outputs={"missions": missions})
            state = self._state(db, command, check_revision=False)
            mid = state["mission_id"]
            if command.kind == "replay":
                outputs = self.store.replay(db, mid)
            elif command.kind == "export":
                outputs = self.store.export(db, mid)
            else:
                objects: list[dict[str, Any]] = []
                contract_invalidations = {canonical(event["payload"]["new_ref"]): event["payload"]
                    for event in self.store.events(db, mid) if event["kind"] == "prediction.contract_invalidated"}
                for row in db.execute("SELECT r.document FROM heads h JOIN revisions r ON h.object_id=r.object_id AND "
                                      "h.revision_id=r.revision_id WHERE r.mission_id=?", (mid,)):
                    raw = json.loads(row[0])
                    ref = ObjectRef.model_validate(raw["ref"])
                    validity = self.store.validity(db, ref)
                    if ref.object_kind in ("context", "candidate", "proposal", "policy", "judgment", "certificate") and state.get("protocol_fingerprint") != self.protocol_fingerprint:
                        validity = {"valid": False, "reasons": [*validity["reasons"], "runtime/card/representation protocol bytes changed; recompute or reexecute required"]}
                    item = {"revision": raw, "current_validity": validity}
                    if canonical(raw["ref"]) in contract_invalidations:
                        item["contract_invalidation"] = contract_invalidations[canonical(raw["ref"])]
                    objects.append(item)
                receipts = [dict(r) for r in db.execute("SELECT a.attempt_id,a.executor,a.state,a.receipt,a.sent_at FROM attempts a "
                                                        "JOIN operations o ON o.operation_id=a.operation_id WHERE o.mission_id=? ORDER BY a.sent_at", (mid,))]
                for item in receipts:
                    item["receipt"] = json.loads(item["receipt"]) if item["receipt"] else None
                operations = [dict(r) for r in db.execute("SELECT operation_id,state,fence,lease_until,cancel_requested,failure FROM operations WHERE mission_id=?", (mid,))]
                partial = [json.loads(r[0]) for r in db.execute("SELECT document FROM events WHERE mission_id=? AND kind='compiler.stage' ORDER BY seq", (mid,))]
                outputs = {"state": state, "objects": objects, "saved_receipts": receipts,
                           "operations": operations, "saved_compiler_stages": partial,
                           "recomputed_validity_at": utcnow(), "effects_executed": 0}
            return CommandResult(command_id=command.command_id, mission_id=mid, status="ok", current_revision=state["revision"],
                                 outputs=outputs, events=self.store.events(db, mid, after=int(command.payload.get("events_after", 0))) if command.payload.get("events") else [])

    def _damaged_inputs(self, db: sqlite3.Connection, state: dict[str, Any]) -> list[dict[str, Any]]:
        """Active input observations whose stored bytes are missing or no longer match."""
        damaged = []
        for raw in state["observations"]:
            ref = ObjectRef.model_validate(raw)
            try:
                payload = self.store.get(db, ref).payload
            except DomainError as exc:
                damaged.append({"observation": ref.object_id, "reasons": [f"{exc.code}: {exc}"]})
                continue
            if payload.get("status") != "active":
                continue
            reasons = [r for r in self.store.validity(db, ref)["reasons"] if r.startswith(DAMAGE_REASONS)]
            if reasons:
                damaged.append({"observation": ref.object_id, "reasons": reasons})
        return damaged

    def _input(self, db: sqlite3.Connection, state: dict[str, Any]) -> tuple[Mission, list[Any], list[ObjectRef], str]:
        mref = ObjectRef.model_validate(state["mission_ref"])
        mission = Mission.model_validate(self.store.get(db, mref).payload)
        observations = [self.store.get(db, ObjectRef.model_validate(r)) for r in state["observations"]]
        observations = [o for o in observations if o.payload["status"] == "active" and self.store.validity(db, o.ref)["valid"]]
        refs = [mref, *(o.ref for o in observations)]
        fingerprint = digest({"inputs": refs, "options": state["options"], "protocol_fingerprint": self.protocol_fingerprint})
        return mission, observations, refs, fingerprint

    def _run_outputs(self, command: Command, outputs: dict[str, Any]) -> dict[str, Any]:
        """Count this command's saved dispatcher attempts, including earlier steps.

        An executor invocation is not an assistant-message count or a Jev HTTP
        attempt count; those retain their own receipt usage. Controlled fixtures
        exercise the same accounting but never establish live service use.
        """
        with self.store.connection(readonly=True) as db:
            rows = list(db.execute("SELECT a.executor,a.receipt FROM attempts a JOIN operations o "
                                   "ON o.operation_id=a.operation_id WHERE o.command_id=?", (command.command_id,)))
        remote = 0
        for row in rows:
            receipt = json.loads(row["receipt"]) if row["receipt"] else {}
            if row["executor"] not in ("sandbox.python", "source.file") and not receipt.get("tool_state", "").startswith("not sent:"):
                remote += 1
        return {**outputs, "external_calls": remote, "executor_attempts": len(rows),
                "external_calls_basis": "nonlocal executor invocations in this command, excluding established not-sent attempts; see receipts for actual messages and HTTP attempts"}

    def _refusal_basis(self, db: sqlite3.Connection, state: dict[str, Any]) -> str:
        """A refusal is retryable after new inputs, authority, scope or protocol.

        Its own local feedback is not new evidence that the same denied action
        can now run. Keep these refusals distinct from completed experiments.
        """
        refused = {canonical(record["observation_ref"]) for record in state.get("refused_experiment_records", [])}
        observations = []
        for raw in state["observations"]:
            if canonical(raw) in refused:
                continue
            ref = ObjectRef.model_validate(raw)
            if self.store.get(db, ref).payload["status"] == "active" and self.store.validity(db, ref)["valid"]:
                observations.append(ref)
        return digest({"mission_ref": state["mission_ref"], "authorization_ref": state["authorization_ref"],
                       "observations": observations, "options": state["options"],
                       "execution_epoch": state.get("execution_epoch", 0), "protocol": self.protocol_fingerprint})

    def _advance(self, command: Command, *, expected_revision: str | None) -> CommandResult:
        token: OperationToken | None = None
        outputs: dict[str, Any] = {"steps": []}
        mode = command.payload.get("mode", "advance")
        if mode not in ("advance", "recompute", "reexecute"):
            raise DomainError("run_mode", "use advance, recompute, or reexecute; replay is a separate query")
        try:
            with self.store.connection(readonly=True) as db:
                state = self._state(db, command, check_revision=False)
                if expected_revision is None:
                    raise DomainError("revision_required", "supply expected_revision from inspect before running the mission")
                if state["revision"] != expected_revision:
                    raise DomainError("conflict", "mission changed before advancement")
                options = CompileOptions.model_validate(state["options"])
            steps = int(command.payload.get("steps", options.steps))
            if not 1 <= steps <= 100:
                raise DomainError("steps", "steps must be between 1 and 100; resume to continue")
            for _ in range(steps):
                with self.store.transaction() as db:
                    state = self._state(db, command, check_revision=False)
                    if state["revision"] != expected_revision:
                        raise DomainError("conflict", "mission changed between command steps")
                    if state["status"] in ("paused", "cancelled"):
                        raise DomainError(state["status"], "resume the mission before running")
                    options = CompileOptions.model_validate(state["options"])
                    damaged = self._damaged_inputs(db, state)
                    if damaged:
                        # Input evidence is never silently dropped: recompiling without it would
                        # publish a new "valid" policy that lost evidence nobody chose to lose.
                        raise DomainError("damaged_evidence", "input evidence is damaged: " + "; ".join(
                            f"{d['observation']} ({', '.join(d['reasons'])})" for d in damaged) +
                            ". Restore the original artifact bytes from a backup, or withdraw the damaged "
                            "observation to continue without it; nothing was recompiled or executed")
                    mission, observations, refs, fingerprint = self._input(db, state)
                    policy_valid = not state["policy_ref"] or self.store.validity(db, ObjectRef.model_validate(state["policy_ref"]))["valid"]
                    lost_execution_evidence = not policy_valid and state["last_fingerprint"] is not None
                    protocol_changed = state.get("protocol_fingerprint") not in (None, self.protocol_fingerprint)
                    if lost_execution_evidence or protocol_changed:
                        # A saved queue is derived evidence too. It cannot send
                        # an action after its generating receipt becomes damaged.
                        self._invalidate_view(db, state, "runtime protocol changed" if protocol_changed else "execution or policy evidence is no longer valid")
                    if state["last_fingerprint"] == fingerprint and not state["pending_experiments"] and mode == "advance":
                        if (state.get("blocker") or {}).get("code") == "damaged_evidence":
                            # The inputs verified above, so the host restored them: lift the stop
                            # without recompiling, since nothing about the evidence changed.
                            lifted = state.pop("blocker")
                            state["status"] = lifted.get("previous_status") if lifted.get("previous_status") not in (None, "blocked") else "waiting"
                            self.store.event(db, state["mission_id"], "mission.unblocked",
                                             {"code": lifted["code"], "reason": "input evidence verifies again"})
                            expected_revision = self.store.save_mission(db, state, expected=expected_revision)
                            state = self.store.mission(db, state["mission_id"])
                        return CommandResult(command_id=command.command_id, mission_id=state["mission_id"], status="waiting",
                                             current_revision=state["revision"], outputs=self._run_outputs(command, {**outputs, "reason": "no new input or untried experiment"}),
                                             next_actions=["observe", "revise"])
                    token = self.runtime.begin(db, state["mission_id"], command.command_id, state["revision"], refs,
                                               ObjectRef.model_validate(state["authorization_ref"]))
                    previous = self.store.get(db, ObjectRef.model_validate(state["proposal_ref"])).payload if state["proposal_ref"] else None
                    if (lost_execution_evidence or protocol_changed) and mode != "recompute":
                        previous = None
                if mode == "recompute":
                    if previous is None:
                        raise DomainError("no_saved_proposal", "recompute requires a saved proposal")
                    from sextant.compiler import context_bundle
                    from sextant.representations import evaluate
                    proposal = Proposal.model_validate(previous)
                    proposal.policy = normalize_policy(proposal.policy, mission)
                    checks = {"kind": "recompute", "external_calls": 0, "policy_promoted": False,
                              "policy": proposal.policy.model_dump(), "context": context_bundle(observations).model_dump(),
                              "interfaces": [{"candidate": c.key, "interface_check": evaluate(c.representation, {})} for c in proposal.candidates],
                              "scope": "local checks of a saved proposal; no semantic revalidation against new evidence"}
                    artifact = self.store.put_artifact(canonical(checks).encode(), source="local-recomputation")
                    with self.store.transaction() as db:
                        self.runtime.finish(db, token)
                        self.store.event(db, state["mission_id"], "proposal.recomputed", {"artifact": artifact.model_dump(),
                                         "policy_promoted": False}, operation_id=token.operation_id)
                    outputs["steps"].append(checks)
                    token = None
                    break
                if state["pending_experiments"] and mode != "reexecute":
                    chosen = state["pending_experiments"][0]
                    with self.store.transaction() as db:
                        self.runtime.guard(db, token)
                        self._research_evidence(db, state, chosen)
                    experiment = Experiment.model_validate(chosen["experiment"])
                    try:
                        invocation = self._experiment(token, state, experiment, refs)
                    except DomainError as exc:
                        locally_refused = isinstance(exc, NotSent) and exc.kind == "refused"
                        if exc.code not in ("hard_constraint", "effect_evidence_required", "unauthorized_action", "unauthorized_effect") and not locally_refused:
                            raise
                        # This is an executed local refusal, not an external action
                        # result. Feed the concrete missing condition back to the
                        # compiler so it can design the required experiment.
                        with self.store.transaction() as db:
                            self.runtime.guard(db, token)
                            refusal_basis = self._refusal_basis(db, state)
                            observation = Observation(content={"proposed_experiment": experiment.model_dump(), "code": exc.code, "reason": str(exc)},
                                                      source="runtime-effect-guard", source_group="runtime-effect-guard", observed_at=utcnow(),
                                                      measurement_protocol="effect-guard/1", provenance="constructed",
                                                      supports_scope="the proposed action was locally refused; no external effect was sent")
                            ref = self.store.put(db, state["mission_id"], "observation", observation.model_dump(), dependencies=refs)
                            state.setdefault("refused_experiment_records", []).append({"signature": chosen["signature"],
                                "basis": refusal_basis, "reason": exc.code, "observation_ref": ref.model_dump()})
                            state["observations"].append(ref.model_dump())
                            state["pending_experiments"] = state["pending_experiments"][1:]
                            self._invalidate_view(db, state, "new observation")
                            self.runtime.finish(db, token)
                            expected_revision = self.store.save_mission(db, state, expected=expected_revision)
                        outputs["steps"].append({"kind": "action_refused", "reason": exc.code, "effects_sent": 0})
                        token = None
                        continue
                    with self.store.transaction() as db:
                        self.runtime.guard(db, token)
                        batch_before = state.get("batch_progress", {})
                        batch_count = batch_before.get("count", 0) if batch_before.get("group") == experiment.independent_group else 0
                        companions = [e for e in state["pending_experiments"][1:]
                                      if experiment.independent_group and e["experiment"].get("independent_group") == experiment.independent_group
                                      and not world_effect(Experiment.model_validate(e["experiment"]))]
                        state["pending_experiments"] = state["pending_experiments"][1:]
                        produced = self._ingest_effect(db, state, experiment, invocation.output,
                                                       evidence=self.runtime.operation_artifacts(db, token))
                        state["completed_experiments"].append(chosen["signature"])
                        state.setdefault("completed_experiment_records", []).append({"signature": chosen["signature"],
                            "experiment": experiment.model_dump(), "observations": produced, "branch": chosen["branch"],
                            "environment_scope": {k: options.environment.get(k) for k in ("episode_id", "phase", "revision")}})
                        state.pop("blocker", None)
                        branch = chosen["branch"]
                        state["branch_visits"][branch] = state["branch_visits"].get(branch, 0) + 1
                        if companions and not world_effect(experiment) and batch_count + 1 < options.experiment_batch_size:
                            state["pending_experiments"] = companions[:options.experiment_batch_size - batch_count - 1]
                            carried = {e["signature"] for e in state["pending_experiments"]}
                            state["research_queue"] = [e for e in state["research_queue"] if e["signature"] not in carried]
                            state["batch_progress"] = {"group": experiment.independent_group, "count": batch_count + 1}
                        self.runtime.finish(db, token)
                        expected_revision = self.store.save_mission(db, state, expected=expected_revision)
                    outputs["steps"].append({"kind": "experiment", "experiment": experiment.key,
                                              "branch": branch, "observations": produced})
                else:
                    if mode == "reexecute":
                        # An explicit new execution is distinct from replay or
                        # local recomputation. Old receipts remain historical.
                        state["completed_experiments"] = []
                        state["execution_epoch"] = state.get("execution_epoch", 0) + 1
                    def planning_context(stage: str, captured_state: dict[str, Any] = state) -> dict[str, Any]:
                        assert self.planning_callback is not None
                        snapshot = self.planning_callback(stage, deepcopy(captured_state))
                        return planning_material(stage, captured_state, snapshot)

                    compiler = Compiler(self.runtime, self.executors.get(options.generator), self.executors.get("typesafe"),
                                        planning_context=planning_context if self.planning_callback is not None else None)
                    compiled = compiler.compile(token, mission, refs[0], observations, options,
                                                previous=previous, completed_experiments=state.get("completed_experiment_records", []),
                                                representation_state=state.get("representation_state"))
                    expected_revision = self._publish(token, state, mission, refs, fingerprint, compiled)
                    outputs["steps"].append({"kind": "compile", "candidates": len(compiled.proposal.candidates),
                                              "direct_result": compiled.proposal.direct_result})
                token = None
                if state["status"] == "complete":
                    break  # a finite verified result should return before the next idle step
                mode = "advance"
            with self.store.connection(readonly=True) as db:
                state = self.store.mission(db, state["mission_id"])
                if state["revision"] != expected_revision:
                    raise DomainError("conflict", "mission changed before the command returned")
            return CommandResult(command_id=command.command_id, mission_id=state["mission_id"], status=state["status"],
                                 current_revision=state["revision"], outputs=self._run_outputs(command, {**outputs, "policy_ref": state["policy_ref"]}),
                                 next_actions=["inspect", "export"] if state["status"] == "complete" else
                                              ["run" if state["pending_experiments"] else "observe", "inspect"])
        # Release an owned operation even on a programming error.
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, DomainError):
                error = exc
            elif isinstance(exc, (ValidationError, ValueError, KeyError, TypeError, JevClientError)):
                error = DomainError("invalid_proposal", str(exc))
            else:
                error = DomainError("internal_error", f"{type(exc).__name__}: {exc}")
            if token:
                self.runtime.fail(token, error)
                with self.store.connection(readonly=True) as db:
                    operation = db.execute("SELECT state FROM operations WHERE operation_id=?",
                                           (token.operation_id,)).fetchone()
                if operation and operation["state"] == "reconciliation_required" and error.code != "reconciliation_required":
                    error = DomainError("reconciliation_required",
                                        f"An effect remains uncertain after {error.code}: {error}. Inspect and reconcile before another run.")
            blocked = (isinstance(error, NotSent) and error.kind == "unavailable") or error.code == "damaged_evidence"
            current = self._record_blocker(command, error, token) if blocked else None
            if current is None:
                with self.store.connection(readonly=True) as db:
                    current = self.store.mission(db, command.mission_id) if command.mission_id else None
            if blocked:
                next_actions = ["inspect", "remove the blocker named in errors, then run with the current revision"]
            else:
                next_actions = ["inspect", "reconcile" if error.code == "reconciliation_required" else "revise or run with a new command"]
            return CommandResult(command_id=command.command_id, mission_id=command.mission_id,
                                 current_revision=current["revision"] if current else None,
                                 status=error.code, outputs=self._run_outputs(command, {**outputs, "failed_operation": token.operation_id if token else None,
                                                                                      "completed_steps_preserved": True}),
                                 errors=[{"code": error.code, "message": str(error)}],
                                 next_actions=next_actions)

    def _record_blocker(self, command: Command, error: DomainError, token: OperationToken | None) -> dict[str, Any] | None:
        """Make a stop the host must resolve visible on the mission itself, not only in this reply."""
        if command.mission_id is None:
            return None
        try:
            with self.store.transaction() as db:
                state = self.store.mission(db, command.mission_id)
                before = state["revision"]
                previous_status = (state.get("blocker") or {}).get("previous_status", state["status"])
                state["status"] = "blocked"
                state["blocker"] = {"code": error.code, "message": str(error), "at": utcnow(),
                                    "operation_id": token.operation_id if token else None,
                                    "previous_status": previous_status,
                                    "meaning": "nothing further was sent or executed; run again once the named cause is removed"}
                self.store.save_mission(db, state, expected=before)
                self.store.event(db, state["mission_id"], "mission.blocked", state["blocker"])
                return self.store.mission(db, command.mission_id)
        except DomainError:
            return None  # a concurrent change wins; the reply still names the cause

    def _experiment(self, token: OperationToken, state: dict[str, Any], experiment: Experiment, refs: list[ObjectRef]) -> Any:
        with self.store.connection(readonly=True) as db:
            op = self.runtime.guard(db, token)
            auth = self.runtime._active_authorization(db, ObjectRef.model_validate_json(op["authorization"]))
        if world_effect(experiment) and "environment.apply" not in auth.tools:
            raise DomainError("unauthorized_effect", "world-changing experiments require explicit environment.apply authority")
        if experiment.adapter == "source.file":
            executor: Executor = FileSource(auth.read_paths)
            request = experiment.inputs
        elif experiment.adapter == "environment":
            executor = self.executors.get("environment")  # type: ignore[assignment]
            if executor is None:
                raise DomainError("environment_unavailable", "no environment broker was supplied by the host")
            env = state["options"]["environment"]
            actions = env.get("actions", {name: {} for name in ("status", "read_artifact", "experiment", "observe", "apply")})
            if experiment.action == "runtime.invoke" or experiment.action not in actions:
                raise DomainError("unauthorized_action", "environment action is outside the host-supplied public interface")
            request = {"episode_id": env.get("episode_id"), "phase": env.get("phase", 0),
                       "kind": "apply" if world_effect(experiment) else "read" if experiment.action in ("status", "read_artifact", "observe") else "experiment",
                       "action": experiment.action, "inputs": experiment.inputs}
        else:
            executor = self.executors["sandbox.python"]
            request = experiment.inputs
        with self.store.connection(readonly=True) as db:
            mission = Mission.model_validate(self.store.get(db, ObjectRef.model_validate(state["mission_ref"])).payload)
            # Flat input fields remain available for existing predicates, while
            # runtime-owned facts cannot be overwritten by generated inputs.
            checked = constraints(mission.hard_constraints, {**experiment.inputs, **request, "inputs": experiment.inputs,
                                  "action": experiment.action, "adapter": experiment.adapter,
                                  "changes_world": world_effect(experiment)})
            if any(c["status"] == "violated" for c in checked):
                raise DomainError("hard_constraint", "proposed experiment violates an executable hard constraint")
            spec = state["options"]["environment"].get("effect_gate")
            if world_effect(experiment) and any(c["status"] == "unknown" for c in checked) and not spec:
                raise DomainError("effect_evidence_required", "natural-language hard constraints are unresolved; the host must supply a matching observation gate or executable predicates")
            if spec and world_effect(experiment):
                evidence = []
                for raw in state["observations"]:
                    ref = ObjectRef.model_validate(raw)
                    obs = self.store.get(db, ref).payload
                    if self.store.validity(db, ref)["valid"] and obs["status"] == "active" and obs["provenance"] in ("constructed", "environment"):
                        evidence.append({**obs, "ref": ref.object_id})
                gate = observed_scenario_gate(spec, experiment.inputs, evidence, environment=state["options"]["environment"])
                if gate["status"] != "passed":
                    raise DomainError("effect_evidence_required", canonical(gate))
        return self.runtime.dispatch(token, executor, request, effect_key=f"experiment-{experiment_signature(state, experiment)}",
                                     send_refs=refs, reuse_completed=True)

    def _ingest_effect(self, db: sqlite3.Connection, state: dict[str, Any], experiment: Experiment, output: Any,
                       *, evidence: list[ArtifactRef] | None = None) -> list[dict[str, Any]]:
        if not isinstance(output, dict):
            raise DomainError("invalid_proposal", "executor output must be an object")
        raw = self.store.put_artifact(canonical(output).encode(), source=f"executor:{experiment.adapter}")
        produced = []
        changed = False
        items = output.get("observations", []) if experiment.adapter == "environment" else [output]
        if not isinstance(items, list):
            raise DomainError("environment_observations", "environment observations must be a list")
        for item in items:
            if not isinstance(item, dict):
                raise DomainError("environment_observations", "each observation must be an object")
            is_environment = experiment.adapter == "environment"
            is_file = experiment.adapter == "source.file"
            content = item["content"] if is_environment else item.get("value", item.get("content", item))
            source = str(item.get("source_id", item.get("source", f"executor:{experiment.adapter}:{experiment.key}")))
            observation = Observation(content=content, source=source, source_group=source,
                                      locator=f"bytes:{item['read_range'][0]}-{item['read_range'][1]} (end exclusive)" if is_file else item.get("locator", "whole"),
                                      read_range=item.get("read_range"), total_bytes=item.get("total_bytes"),
                                      external_observation_id=item.get("observation_id") if is_environment else None,
                                      source_revision=item.get("source_sha256") if is_file else item.get("source_revision"),
                                      environment_scope=item.get("supports_scope") if is_environment and isinstance(item.get("supports_scope"), dict) else None,
                                      observed_at=str(item.get("observed_at", utcnow())), occurred_at=item.get("occurred_at"),
                                      measurement_protocol=str(item["measurement_protocol"]),
                                      completeness="partial" if item.get("truncated") else item.get("completeness", "complete"),
                                      provenance="environment" if is_environment else "constructed" if experiment.adapter == "sandbox.python" else "local-file",
                                      supports_scope=canonical(item["supports_scope"]) if isinstance(item["supports_scope"], dict) else str(item["supports_scope"]))
            if observation.external_observation_id:
                duplicate = None
                for previous_raw in state["observations"]:
                    previous_ref = ObjectRef.model_validate(previous_raw)
                    previous = Observation.model_validate(self.store.get(db, previous_ref).payload)
                    if previous.external_observation_id == observation.external_observation_id:
                        if previous != observation:
                            raise DomainError("observation_identity", "external observation ID was reused with changed content")
                        duplicate = previous_ref
                        break
                if duplicate is not None:
                    produced.append(duplicate.model_dump())
                    continue
            ref = self.store.put(db, state["mission_id"], "observation", observation.model_dump(), artifacts=[raw, *(evidence or [])])
            changed = True
            state["observations"].append(ref.model_dump())
            produced.append(ref.model_dump())
            claim = Claim(proposition=f"The recorded {experiment.adapter} execution returned {canonical(content)} within {observation.supports_scope}",
                          state="supported", support=[ref], derivation_rule="receipt-bound-output/1; no claim beyond declared measurement scope")
            claim_ref = self.store.put(db, state["mission_id"], "claim", claim.model_dump(), dependencies=[ref])
            state.setdefault("claims", []).append(claim_ref.model_dump())
            # Retract exact observations, never every measurement sharing a source.
            replaced = set(item.get("withdraws", []) + item.get("supersedes", []))
            for prior_raw in state["observations"][:-1]:
                prior_ref = ObjectRef.model_validate(prior_raw)
                prior = self.store.get(db, prior_ref)
                if prior.payload.get("external_observation_id") in replaced:
                    self.store.invalidate(db, prior_ref.object_id, prior_ref.revision_id, "environment corrected source")
        if changed:
            self._invalidate_view(db, state, "experiment observation received")
        for prediction_raw in list(state["predictions"]):
            prediction_ref = ObjectRef.model_validate(prediction_raw)
            prediction = self.store.get(db, prediction_ref).payload
            if prediction["status"] != "pending" or not self.store.validity(db, prediction_ref)["valid"]:
                continue
            for observation_raw in produced:
                try:
                    result = self._resolve(db, state, {"prediction_ref": prediction_raw, "observation_ref": observation_raw})
                except DomainError:
                    continue  # observation has a different frozen source/scope/protocol
                if result["status"] in ("resolved", "censored"):
                    self.store.event(db, state["mission_id"], "prediction.settled", result)
                    break
        return produced

    def _publish(self, token: OperationToken, state: dict[str, Any], mission: Mission, refs: list[ObjectRef],
                 fingerprint: str, compiled: Compilation) -> str:
        artifact = self.store.put_artifact(canonical({"stages": compiled.stages, "interfaces": compiled.interfaces,
                                                     "judgments": compiled.judgments}).encode(), source="compiler-trace")
        with self.store.transaction() as db:
            self.runtime.guard(db, token)
            mid = state["mission_id"]
            evidence = self.runtime.operation_artifacts(db, token)
            evidence.extend(compiled.diagnostic_artifacts)
            retained_queue = []
            for retained in state.get("research_queue", []):
                try:
                    evidence.extend(self._research_evidence(db, state, retained))
                    retained_queue.append(retained)
                except (DomainError, ValidationError, KeyError, ValueError) as exc:
                    reason = f"Untried research design discarded because its originating evidence is unavailable: {exc}"
                    compiled.proposal.policy.unresolved.append(reason)
                    self.store.event(db, mid, "research.discarded", {"signature": retained.get("signature"),
                                     "origin_proposal_ref": retained.get("origin_proposal_ref"), "reason": reason}, operation_id=token.operation_id)
            # A source default contains adapter/key, so incompatible designs
            # sharing that identity cannot satisfy an unambiguous result
            # contract. Refuse the proposal before publishing or dispatching it.
            designs: dict[tuple[str, str], str] = {}
            experiments = [*(e for c in compiled.proposal.candidates for e in c.experiments),
                           *compiled.proposal.policy.next_experiments,
                           *(Experiment.model_validate(e["experiment"]) for e in retained_queue)]
            for experiment in experiments:
                design_key = (experiment.adapter, experiment.key)
                body = canonical({"action": experiment.action, "inputs": experiment.inputs})
                if design_key in designs and designs[design_key] != body:
                    raise DomainError("experiment_identity", "incompatible queued designs share adapter/key "
                                      f"{experiment.adapter}:{experiment.key}; use distinct keys and exact prediction sources")
                designs[design_key] = body
            evidence = list({canonical(a): a for a in evidence}.values())
            cref = self.store.put(db, mid, "context", compiled.context.model_dump(), dependencies=refs,
                                  artifacts=evidence, scope=mission.scope)
            for judgment in compiled.judgments:
                jref = self.store.put(db, mid, "judgment", judgment, dependencies=[cref],
                                      artifacts=[artifact, *compiled.diagnostic_artifacts], scope=mission.scope)
                state["judgments"].append(jref.model_dump())
            # Support canonical internal IDs and unambiguous external aliases.
            # A collision must remain unresolved, never silently select a source.
            internal: dict[str, ObjectRef] = {}
            external: dict[str, list[ObjectRef]] = {}
            for ref in compiled.context.selected:
                if ref.object_kind != "observation":
                    continue
                observation = self.store.get(db, ref).payload
                internal[ref.object_id] = ref
                if observation.get("external_observation_id"):
                    external.setdefault(observation["external_observation_id"], []).append(ref)

            def resolve_alias(value: str) -> list[ObjectRef]:
                if value.startswith("object:"):
                    ref = internal.get(value.removeprefix("object:"))
                    return [ref] if ref else []
                if value.startswith("external:"):
                    return external.get(value.removeprefix("external:"), [])
                matches = external.get(value, [])[:]
                if value in internal and internal[value] not in matches:
                    matches.append(internal[value])
                return matches

            def cited(values: list[str]) -> tuple[list[ObjectRef], list[str]]:
                resolved = {value: resolve_alias(value) for value in values}
                unknown = [value for value in values if len(resolved[value]) != 1]
                found: list[ObjectRef] = []
                for value in values:
                    if value not in unknown and resolved[value][0] not in found:
                        found.append(resolved[value][0])
                return found, unknown

            def unchanged(previous: ObjectRef | None, payload: dict[str, Any], deps: list[ObjectRef]) -> bool:
                """An identical, still-valid current head keeps its revision across compilations."""
                if previous is None or self.store.head(db, previous.object_id).ref != previous:
                    return False
                prior = self.store.get(db, previous)
                return (prior.scope == mission.scope and canonical(prior.payload) == canonical(payload) and prior.dependencies == deps
                        and self.store.validity(db, previous)["valid"])

            def current_head(previous: ObjectRef | None) -> ObjectRef | None:
                return self.store.head(db, previous.object_id).ref if previous else None

            candidate_refs = []
            proposed_keys: set[str] = set()
            for candidate in compiled.proposal.candidates:
                proposed_keys.add(candidate.key)
                previous_raw = state["candidates"].get(candidate.key)
                previous = ObjectRef.model_validate(previous_raw) if previous_raw else None
                cites, unresolved_cites = cited(candidate.evidence)
                if candidate.evidence and not unresolved_cites:
                    # The goal and the evidence it cites: a change to other evidence leaves
                    # this branch, and its revision, intact.
                    deps = [refs[0], *cites]
                else:
                    deps = [cref]
                    if unresolved_cites:
                        compiled.proposal.policy.unresolved.append(
                            f"Candidate {candidate.key} cites unresolved or ambiguous evidence ({', '.join(unresolved_cites)}); "
                            "it is treated as depending on the whole context")
                kept = unchanged(previous, candidate.model_dump(), deps)
                ref = previous if kept and previous else self.store.put(
                    db, mid, "candidate", candidate.model_dump(), scope=mission.scope, dependencies=deps,
                    object_id=previous.object_id if previous else None, expected=current_head(previous))
                state["candidates"][candidate.key] = ref.model_dump()
                candidate_refs.append(ref)
                if kept:
                    continue  # its predictions were frozen when this revision was first published
                for proposed_prediction in candidate.predictions:
                    key = digest({"spec": proposed_prediction, "scope": mission.scope,
                                  "mission_ref": refs[0], "execution_epoch": state.get("execution_epoch", 0)})
                    prior_raw = state.setdefault("prediction_keys", {}).get(key)
                    if prior_raw:
                        prior_ref = ObjectRef.model_validate(prior_raw)
                        if self.store.validity(db, prior_ref)["valid"]:
                            continue  # an unchanged question is not a new prediction
                    try:
                        validate_prediction_contract(proposed_prediction, new_admission=True)
                        prediction = Prediction.model_validate({**proposed_prediction.model_dump(), "frozen_at": utcnow(), "status": "pending", "outcome": None, "resolution_refs": []})
                    except (ValidationError, DomainError) as exc:
                        self.store.event(db, mid, "prediction.rejected", {"candidate": candidate.key, "reason": str(exc)})
                        continue
                    # Frozen predictions depend on their question/mission and the evidence their
                    # branch cites, not a mutable policy context, so later evidence can settle them.
                    pref = self.store.put(db, mid, "prediction", prediction.model_dump(), scope=mission.scope,
                                          dependencies=refs if deps == [cref] else deps, artifacts=evidence)
                    state["predictions"].append(pref.model_dump())
                    state["prediction_keys"][key] = pref.model_dump()
            for key, raw in state["candidates"].items():
                if key in proposed_keys:
                    continue
                dropped = ObjectRef.model_validate(raw)
                if current_head(dropped) == dropped and self.store.validity(db, dropped)["valid"]:
                    self.store.invalidate(db, dropped.object_id, dropped.revision_id, "candidate not in the current proposal")

            prior_claims: dict[str, ObjectRef] = {}
            kept_claims: list[dict[str, Any]] = []
            for raw in state.get("claims", []):
                ref = ObjectRef.model_validate(raw)
                try:
                    payload = self.store.get(db, ref).payload
                except DomainError:
                    kept_claims.append(raw)  # its damage stays visible to inspect; never silently dropped
                    continue
                if str(payload.get("derivation_rule", "")).startswith("model proposed evidence linkage"):
                    prior_claims[payload["proposition"]] = ref
                else:
                    kept_claims.append(raw)  # receipt-bound claims belong to their observations
            new_claims: list[dict[str, Any]] = []
            for proposition, ids in compiled.proposal.evidence_links.items():
                support, unknown = cited(ids)
                claim = Claim(proposition=proposition, state="hypothesis", support=support,
                              unresolved_support=unknown,
                              derivation_rule="model proposed evidence linkage; not independently established")
                claim_deps = list(support) if support else [cref]
                previous = prior_claims.pop(proposition, None)
                if unchanged(previous, claim.model_dump(), claim_deps) and previous:
                    claim_ref = previous
                else:
                    claim_ref = self.store.put(db, mid, "claim", claim.model_dump(), dependencies=claim_deps, scope=mission.scope,
                                               object_id=previous.object_id if previous else None, expected=current_head(previous))
                    if unknown:
                        self.store.event(db, mid, "claim.unresolved_links", {"claim_ref": claim_ref.model_dump(), "unresolved_ids": unknown})
                new_claims.append(claim_ref.model_dump())
                if unknown:
                    compiled.proposal.policy.unresolved.append(f"Unresolved or ambiguous evidence links for {proposition}: {', '.join(unknown)}")
            for dropped in prior_claims.values():
                if current_head(dropped) == dropped and self.store.validity(db, dropped)["valid"]:
                    self.store.invalidate(db, dropped.object_id, dropped.revision_id, "claim not re-proposed by the current compilation")
            state["claims"] = kept_claims + new_claims
            refusal_basis = self._refusal_basis(db, state)
            refused = {record["signature"]: record for record in state.get("refused_experiment_records", [])
                       if record["basis"] == refusal_basis}
            proposed_signatures = {experiment_signature(state, experiment)
                                   for experiment in [*(e for c in compiled.proposal.candidates for e in c.experiments),
                                                      *compiled.proposal.policy.next_experiments]}
            proposed_signatures.update(e["signature"] for e in retained_queue)
            for signature in sorted(proposed_signatures.intersection(refused)):
                compiled.proposal.policy.unresolved.append(
                    f"Experiment {signature[:12]} was locally refused ({refused[signature]['reason']}); "
                    "it was not completed and will wait for changed evidence, authorization, mission or execution scope.")
            pref = self.store.put(db, mid, "proposal", compiled.proposal.model_dump(), dependencies=[cref, *candidate_refs], artifacts=[artifact], scope=mission.scope)
            policy_ref = self.store.put(db, mid, "policy", compiled.proposal.policy.model_dump(), dependencies=[pref], scope=mission.scope)
            if state["policy_ref"]:
                state["history"].append({"policy_ref": state["policy_ref"], "proposal_ref": state["proposal_ref"],
                                         "replaced_by": policy_ref.model_dump(), "reason": "new compilation",
                                         "changed": self.store.get(db, ObjectRef.model_validate(state["policy_ref"])).payload != compiled.proposal.policy.model_dump()})
            state.update(context_ref=cref.model_dump(), proposal_ref=pref.model_dump(), policy_ref=policy_ref.model_dump(), last_fingerprint=fingerprint,
                         protocol_fingerprint=self.protocol_fingerprint, representation_state=compiled.representation_state)
            state.pop("blocker", None)
            # Fair finite opportunity by branch visits. Temporary disadvantage is
            # not a pruning rule; combination branches receive the same queue.
            pending = []
            seen: set[str] = set()
            for candidate in compiled.proposal.candidates:
                for experiment in candidate.experiments:
                    signature = experiment_signature(state, experiment)
                    if signature not in state["completed_experiments"] and signature not in refused and signature not in seen:
                        pending.append({"branch": candidate.key, "experiment": experiment.model_dump(), "signature": signature,
                                        "origin_proposal_ref": pref.model_dump(), "protocol_fingerprint": self.protocol_fingerprint})
                        seen.add(signature)
            for experiment in compiled.proposal.policy.next_experiments:
                signature = experiment_signature(state, experiment)
                if signature not in state["completed_experiments"] and signature not in refused and signature not in seen:
                    pending.append({"branch": "policy", "experiment": experiment.model_dump(), "signature": signature,
                                    "origin_proposal_ref": pref.model_dump(), "protocol_fingerprint": self.protocol_fingerprint})
                    seen.add(signature)
            for retained in retained_queue:
                signature = retained["signature"]
                if signature not in state["completed_experiments"] and signature not in refused and signature not in seen:
                    pending.append({**retained, "retained_reason": "untried branch; original evidence integrity verified; authority and constraints checked at dispatch"})
                    seen.add(signature)
            state["research_queue"] = []
            pending.sort(key=lambda e: (state["branch_visits"].get(e["branch"], 0), e["branch"] == "policy"))
            state["pending_experiments"] = pending
            state["status"] = "ready" if pending else "waiting" if compiled.proposal.policy.unresolved else "proposal_ready"
            if compiled.deterministic:
                state["status"] = "complete"
                self.store.put(db, mid, "certificate", {"protocol": compiled.deterministic["protocol"], "result": compiled.deterministic,
                                                       "domain": "explicit finite arithmetic only", "authority": "deterministic",
                                                       "invalidated_by": "mission revision", "identity_attestation": False},
                               dependencies=[refs[0]], scope=mission.scope)
            self.runtime.finish(db, token)
            return self.store.save_mission(db, state, expected=state["revision"])

    def _reconcile(self, command: Command) -> CommandResult:
        with self.store.connection(readonly=True) as db:
            state = self._state(db, command)
            operation_id = str(command.payload["operation_id"])
            op = db.execute("SELECT mission_id,state FROM operations WHERE operation_id=?", (operation_id,)).fetchone()
            if op is None or op["mission_id"] != state["mission_id"]:
                raise DomainError("operation", "operation is not in this mission")
            if op["state"] == "running":
                raise DomainError("running", "cancel the stopped worker's operation before reconciliation")
        if command.payload.get("disposition") == "abandon":
            self._owner()
            reason = command.payload.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise DomainError("reconciliation_reason", "abandon requires an explicit reason; the unknown result stays unknown")
            with self.store.transaction() as db:
                current = self._state(db, command)
                db.execute("UPDATE operations SET state='abandoned',cancel_requested=1,fence=fence+1 WHERE operation_id=?", (operation_id,))
                output: dict[str, Any] = {"operation_id": operation_id, "status": "abandoned", "reason": reason,
                          "outcome": "unknown; no fabricated reconciliation", "effects_reexecuted": 0,
                          "authority_source": self.authority.source}
                self.store.event(db, current["mission_id"], "operation.abandoned", output, operation_id=operation_id)
                return CommandResult(command_id=command.command_id, mission_id=current["mission_id"], status="abandoned",
                                     current_revision=current["revision"], outputs=output, next_actions=["inspect", "resume with a new command only if a new attempt is intended"])
        output = self.runtime.reconcile(operation_id, self.executors)
        return CommandResult(command_id=command.command_id, mission_id=state["mission_id"], status=output["status"],
                             current_revision=state["revision"], outputs=output, next_actions=["resume"])
