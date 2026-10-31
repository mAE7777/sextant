"""Real typed Jev through per-call dispatch; semantic diagnostics stay advisory."""

from __future__ import annotations

import re
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import Any

from sextant import wire
from sextant.domain import ObjectRef, Receipt, canonical, digest, utcnow
from sextant.jev_client import JevClient, RenderedQuestion
from sextant.jev_transport import JevResult, JevUnavailable, call_jev, redact
from sextant.rolecards import RoleCard, load_all
from sextant.runtime import AttemptContext, Executor, Invocation, NotSent, OperationToken, Runtime
from sextant.schema import CalibrationStatus, ThresholdRef

JEV_STATE_ENCODING = "canonical-json-text/1"


def sdk_state(state: Any) -> str:
    """Preserve a textual state, or losslessly encode a structured JSON state."""
    def json_types(value: Any) -> None:
        if type(value) is dict:
            if not all(type(key) is str for key in value):
                raise ValueError("JSON state object keys must be strings")
            for child in value.values():
                json_types(child)
        elif type(value) is list:
            for child in value:
                json_types(child)
        elif value is not None and type(value) not in (str, int, float, bool):
            raise ValueError("state contains non-JSON values")
    json_types(state)
    value = state if type(state) is str else canonical(state)
    value.encode("utf-8")
    return value


class TypeSafeExecutor:
    name = "typesafe"

    def capabilities(self) -> dict[str, Any]:
        return {"transport": "typesafe", "wire": wire.WIRE_PROTOCOL_VERSION,
                "state_encoding": JEV_STATE_ENCODING,
                "idempotent": False, "cancel": "unsupported", "reconcile": "unsupported"}

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        from typesafe_sdk import TypeSafeAPIError

        started = utcnow()
        try:
            request_hash = digest(request)
            encoded = sdk_state(request["state"])
            if not isinstance(request["questions"], dict) or not request["questions"] or not isinstance(request["model"], str):
                raise ValueError("typed questions and model are required")
            # Local wrapper parameters, not captured HTTPS/backend identity.
            sdk_request = {"state": encoded, "questions": request["questions"], "model": request["model"]}
            encoding = {"state_encoding": JEV_STATE_ENCODING,
                        "logical_state_hash": digest(request["state"]), "sdk_state_hash": digest(encoded),
                        "sdk_request_hash": digest(sdk_request), "sdk_request": sdk_request,
                        "observation_boundary": "local trusted SDK wrapper call parameters; actual HTTPS body not captured"}
        except (ValueError, TypeError, KeyError, UnicodeError, RecursionError) as exc:
            raise NotSent("invalid_typed_state", "TypeSafe state is not valid UTF8/JSON; no request was sent") from exc
        try:
            result = call_jev(encoded, request["questions"], request["model"])
        except JevUnavailable as exc:
            # Raised before any request is built (no key in the host environment).
            raise NotSent("runtime_unavailable", f"TypeSafe Jev was not called: {exc}",
                          receipt_fields={"requested_model": request["model"],
                                          "auth_route": "authorized-typesafe-key",
                                          "auth_evidence": "no key present in the host environment"}) from exc
        except TypeSafeAPIError as exc:
            # An actual HTTP failure establishes that no usable typed answer was
            # returned. It does not establish zero server computation or usage.
            # Connection loss/timeout is deliberately not caught here.
            output: dict[str, Any] = {"http_status": exc.status, "request_id": exc.request_id,
                      "typed_answer": "unavailable", "server_usage": "unknown"}
            detail = exc.body.get("detail") if isinstance(exc.body, dict) else None
            error_type = detail.get("error_type") if isinstance(detail, dict) else None
            if isinstance(error_type, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", error_type):
                output["server_error_type"] = redact(error_type)
            return Invocation(Receipt(executor=self.name, requested_model=request["model"],
                                      auth_route="authorized-typesafe-key", auth_evidence="SDK HTTP error response; no key retained",
                                      started_at=started, ended_at=utcnow(), request_hash=request_hash,
                                      output_hash=digest(output), status="failed", external_id=exc.request_id,
                                      isolation=encoding,
                                      errors=[f"TypeSafe HTTP {exc.status}: no usable typed answer; usage unknown"],
                                      tool_state="HTTP error response received; no automatic retry", residual_processes="none"), output)
        output = {"model_id": result.model_id, "answers": result.answers, "usage": result.usage,
                  "latency_ms": result.latency_ms, "request_id": result.request_id, "transport": "typesafe"}
        return Invocation(Receipt(executor=self.name, requested_model=request["model"],
                                  observed_model=result.model_id, auth_route="authorized-typesafe-key",
                                  auth_evidence="SDK transport; no key retained", started_at=started,
                                  ended_at=utcnow(), request_hash=request_hash, output_hash=digest(output),
                                  status="completed", usage=result.usage, external_id=result.request_id,
                                  isolation=encoding,
                                  tool_state="typed response received", residual_processes="none"), output)

    def cancel(self, context: AttemptContext) -> dict[str, Any]:
        return {"status": "unsupported"}

    def reconcile(self, context: AttemptContext) -> Invocation | None:
        return None


class RuntimeJevClient(JevClient):
    """Reuse wire rendering/whole-run derivation without introducing a writable JSONL truth.

Each raw call is already persisted as a SQLite attempt/receipt. The complete run
and derived output are returned to the compiler for one immutable judgment object.
"""

    def __init__(self, runtime: Runtime, token: OperationToken, refs: list[ObjectRef],
                 *, model: str = "jev-1.13.0", cards: dict[str, RoleCard] | None = None,
                 executor: Executor | None = None):
        self.runtime, self.token, self.refs = runtime, token, refs
        self.rows: list[dict[str, Any]] = []
        self.last_wire: dict[str, Any] | None = None
        self._diagnostic_binding: dict[str, Any] | None = None
        self.sequence = 0
        self.executor: Executor = executor if executor is not None else TypeSafeExecutor()
        config = {"jev": {"transport": "typesafe", "model_id": model, "canonical_language": "en",
                          "requests_per_minute": 1000, "max_state_tokens": 30000, "max_request_tokens": 60000}}
        super().__init__(config, cards or load_all(Path(str(files("sextant") / "resources" / "v1-role-cards"))))
        # Trusted composition of the real TypeSafe port, not an injected test caller.
        self.caller = self._dispatched
        # Only this concrete trusted SDK port uses the production encoding label.
        # An injected port's own capabilities cannot certify live transport.
        if type(self.executor) is TypeSafeExecutor:
            self.transport = "typesafe-canonical-json-text/1"
        else:
            self.transport = "controlled:runtime-executor"

    def render(self, card: RoleCard, pack: dict[str, Any], keys: list[str] | None = None,
               leaf_space: dict[str, Any] | None = None) -> list[RenderedQuestion]:
        questions = super().render(card, pack, keys=keys, leaf_space=leaf_space)
        selection = pack.get("evaluation", {}).get("context_selection")
        if not selection:
            return questions
        prefix = ("Judgment is limited to the selected evidence and stated diagnostic scope in "
                  "evaluation.context_selection. Omitted material has NOT been assessed. Do not infer "
                  "whole-mission sufficiency, absence of counterevidence or execution authority. "
                  "A withdrawn source is not current factual support. Selection manifest: "
                  + str(selection["manifest_sha256"]) + ".\n")
        result = []
        for question in questions:
            sdk = {**question.sdk, "instructions": prefix + question.sdk["instructions"]}
            result.append(replace(question, text=prefix + question.text, sdk=sdk,
                                  rendered_hash=wire.rendered_hash(card.id, question.key, question.primitive.value, sdk)))
        return result

    def capacity(self, pack: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Conservative checks on the exact textual SDK state and question bodies."""
        state = sdk_state(pack)
        state_tokens = wire.estimate_tokens(state)
        longest = max((wire.request_tokens(state, {key: value}) for key, value in questions.items()), default=state_tokens)
        total = wire.request_tokens(state, questions)
        return {"encoding": JEV_STATE_ENCODING, "sdk_state_sha256": digest(state),
                "sdk_questions_sha256": digest(questions), "estimated_state_tokens": state_tokens,
                "estimated_state_plus_longest_question_tokens": longest, "estimated_request_tokens": total,
                "state_limit": self.max_state_tokens, "state_plus_longest_question_limit": 32000,
                "request_limit": self.max_request_tokens,
                "fits": state_tokens <= self.max_state_tokens and longest <= 32000 and total <= self.max_request_tokens,
                "measurement": "conservative local estimate of exact SDK parameters; not server token usage"}

    def _dispatched(self, state: Any, questions: dict[str, dict[str, Any]], model: str) -> JevResult:
        self.sequence += 1
        request = {"state": state, "questions": questions, "model": model,
                   "purpose": "V1 advisory role diagnostic"}
        binding = dict(self._diagnostic_binding) if self._diagnostic_binding is not None else None
        if binding is not None:
            request["diagnostic_binding"] = binding
        response = self.runtime.dispatch(self.token, self.executor, request,
                                         effect_key=f"jev-{self.sequence}", send_refs=self.refs)
        raw = response.output
        self.last_wire = {"request_hash": response.receipt.request_hash,
                          "output_hash": response.receipt.output_hash,
                          "diagnostic_binding": binding,
                          "local_sdk_parameters": response.receipt.isolation}
        return JevResult(model_id=raw["model_id"], answers=raw["answers"], usage=raw["usage"],
                         latency_ms=raw["latency_ms"], request_id=raw["request_id"], transport="typesafe")

    def _call(self, card: RoleCard, pack: dict[str, Any], rqs: list[RenderedQuestion], *,
              variant: str, role: wire.CallRole, run_id: str,
              input_language: str | None) -> tuple[JevResult, dict[str, Any]]:
        capacity = self.capacity(pack, {r.key: r.sdk for r in rqs})
        if not capacity["fits"]:
            from sextant.jev_client import JevClientError
            raise JevClientError("Required Jev context and actual questions exceed provider capacity: " + canonical(capacity))
        self._diagnostic_binding = {"card_id": card.id, "run_id": run_id, "variant": variant, "call_role": role,
                                    "sdk_state_sha256": capacity["sdk_state_sha256"],
                                    "sdk_questions_sha256": capacity["sdk_questions_sha256"]}
        try:
            result, row = super()._call(card, pack, rqs, variant=variant, role=role, run_id=run_id,
                                       input_language=input_language)
        finally:
            self._diagnostic_binding = None
        row["seq"] = len(self.rows) + 1
        row["latency_ms"] = result.latency_ms
        row["runtime_dispatch"] = self.last_wire
        row["request_state_scope"] = "logical role state; local SDK parameters are separately receipt-bound in runtime_dispatch"
        row["capacity"] = capacity
        self.rows.append(row)
        return result, row

    def _calibration(self, card: RoleCard, rq: RenderedQuestion, asked: list[str],
                     input_language: str | None) -> tuple[CalibrationStatus, ThresholdRef | None]:
        return CalibrationStatus(kind="agreement", state="pending", n=0,
                                 reason="V1 advisory deployment; automatic semantic authority disabled"), None

    def diagnose(self, card: str, pack: dict[str, Any], *, language: str = "en") -> dict[str, Any]:
        before = len(self.rows)
        result = self.ask(card, pack, input_language=language)
        return {"schema_version": 1, "card_id": card, "run_id": result.run_id,
                "authority": "advisory", "automatic_decisions": 0,
                "uncertainty_kind": "model judgment distribution and protocol stability; not outcome probability",
                "raw_calls": self.rows[before:],
                "derived": {key: qe.estimate.model_dump(mode="json") for key, qe in result.estimates.items()},
                "purpose": self.cards[card].purpose, "requests": result.requests,
                "usage_input_tokens": result.usage_input_tokens}
