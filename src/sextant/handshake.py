"""Request creation, fulfilment and answer ingestion (the runtime side of the handshake).

Rejection path, in full: a non-conforming or malformed answer file is renamed to
`<id>.answer.rejected<attempt>.json`, recorded as a rejected llm-calls row, and the request
is re-issued once with the errors appended (attempt 2). A second rejection is recorded as a
blocker in state.json and events.jsonl, for the user.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sextant.jev_transport import redact
from sextant.ledger import Ledger, exclusive, now_iso
from sextant.providers import get_provider
from sextant.providers.base import (
    Answer,
    Request,
    answer_path,
    check_answer_schema,
    new_request_id,
    read_request,
    rejected_answer_path,
    validate_answer,
    write_answer,
    write_request,
)
from sextant.providers.base import (
    pending_requests as file_pending,
)
from sextant.state import State, add_pending, clear_pending, load_state, save_state

MAX_ATTEMPTS = 2


class HandshakeError(RuntimeError):
    pass


class BlockerError(HandshakeError):
    pass


class ProviderError(HandshakeError):
    """A provider could not fulfil a request (quota, outage, missing key, unknown name)."""


class SupersededError(HandshakeError):
    """The request was rejected and re-issued; answers for the old id are no longer accepted."""


def _require_decision(decision_dir: Path) -> None:
    """Checked before taking the lock, which would otherwise create the directory it guards."""
    try:
        load_state(decision_dir)
    except FileNotFoundError as e:
        raise HandshakeError(f"no initialized decision at {decision_dir}; run init first") from e


def _with_state(decision_dir: Path, fn: Any) -> None:
    """Load, mutate, and save while holding the directory lock.

    The second writer waits and then loads the state the first writer saved,
    so its pending update is applied to that snapshot rather than to a copy
    that no longer matches disk.
    """
    with exclusive(decision_dir):
        try:
            st = load_state(decision_dir)
        except FileNotFoundError:
            return
        fn(st)
        save_state(st, decision_dir)


def create_request(
    decision_dir: Path,
    *,
    role: str,
    purpose: str,
    prompt: str,
    answer_schema: dict[str, Any],
    provider: str,
    node_ids: list[str] | None = None,
    constraints: dict[str, Any] | None = None,
    attempt: int = 1,
    previous_errors: list[str] | None = None,
) -> Request:
    schema_errors = check_answer_schema(answer_schema)
    if schema_errors:
        raise HandshakeError("refusing to create a request with an invalid answer schema: " + "; ".join(schema_errors))
    _require_decision(decision_dir)
    with exclusive(decision_dir):
        return _create_request_locked(
            decision_dir, role=role, purpose=purpose, prompt=prompt, answer_schema=answer_schema,
            provider=provider, node_ids=node_ids, constraints=constraints, attempt=attempt,
            previous_errors=previous_errors,
        )


def _create_request_locked(
    decision_dir: Path,
    *,
    role: str,
    purpose: str,
    prompt: str,
    answer_schema: dict[str, Any],
    provider: str,
    node_ids: list[str] | None,
    constraints: dict[str, Any] | None,
    attempt: int,
    previous_errors: list[str] | None,
) -> Request:
    try:
        load_state(decision_dir)
    except FileNotFoundError as e:
        raise HandshakeError(f"no initialized decision at {decision_dir}; run init first") from e
    req = Request(
        id=new_request_id(),
        role=role,  # type: ignore[arg-type]
        purpose=purpose,
        node_ids=node_ids or [],
        prompt=prompt,
        answer_schema=answer_schema,
        constraints=constraints or {},
        provider=provider,
        created=now_iso(),
        attempt=attempt,
        previous_errors=previous_errors or [],
    )
    write_request(decision_dir, req)
    _with_state(decision_dir, lambda st: add_pending(st, req.id))
    Ledger(decision_dir).event("request", request_id=req.id, role=role, provider=provider, purpose=purpose, attempt=attempt)
    return req


def fulfil(decision_dir: Path, request_id: str, config: dict[str, Any], provider_override: str | None = None) -> Answer | None:
    """Ask the configured provider to fulfil the request. Session providers print AWAITING and return None."""
    req = read_request(decision_dir, request_id)
    name = provider_override or req.provider
    shown_name = redact(name)
    ledger = Ledger(decision_dir)
    _refuse_if_superseded(decision_dir, req)
    # Historical session-file diagnostics have no effect dispatcher or mission
    # authorization. They cannot invoke generative providers, including overrides.
    if name not in ("claude-code", "claude-code-subagent"):
        reason = "legacy fulfil is diagnostic-only; use V1 start/run for authorized native subscription execution; no fallback was attempted"
        ledger.event("provider_fail", request_id=req.id, provider=shown_name, reason=reason, fallback_attempted=False)
        raise ProviderError(reason)
    try:
        provider = get_provider(name, config)
        answer = provider.fulfil(req, decision_dir)
    except Exception as e:
        reason = redact(str(e))
        ledger.event(
            "provider_fail",
            request_id=req.id,
            provider=shown_name,
            reason=reason[-600:],
            fallback_attempted=False,
        )
        raise ProviderError(
            f"provider {shown_name} failed for {req.id}: {reason[-400:]}. no fallback was attempted"
        ) from e
    if answer is None:
        return None
    write_answer(decision_dir, answer)
    return ingest(decision_dir, request_id)


def _reject(decision_dir: Path, req: Request, ans: Answer | None, errors: list[str], provider: str, family: str, isolation: str) -> Request | None:
    """Record the rejection, move the answer file aside, then re-issue or raise a blocker."""
    ledger = Ledger(decision_dir)
    ledger.record_llm_call(
        request_id=req.id, role=req.role, provider=provider, family=family, isolation=isolation,
        answer=ans.answer if ans is not None else None, status="rejected", errors=errors,
    )
    src = answer_path(decision_dir, req.id)
    if src.exists():
        src.rename(rejected_answer_path(decision_dir, req.id, req.attempt))
    if req.attempt >= MAX_ATTEMPTS:
        msg = f"request {req.id} rejected {req.attempt} times; BLOCKER for the user: " + "; ".join(errors[:3])
        ledger.event("blocker", request_id=req.id, errors=errors[:5])

        def _mark(st: State) -> None:
            clear_pending(st, req.id)
            if msg not in st.blockers:
                st.blockers.append(msg)

        _with_state(decision_dir, _mark)
        raise BlockerError(msg)
    new = reissue(decision_dir, req.id, errors)
    return new


def superseded_by(decision_dir: Path, request_id: str) -> str | None:
    """The id that replaced this request after a rejection, if any (from the events ledger)."""
    for row in Ledger(decision_dir).read("events"):
        if row.get("kind") == "reissue" and row.get("request_id") == request_id:
            return str(row.get("new_request_id"))
    return None


def _refuse_if_superseded(decision_dir: Path, req: Request) -> None:
    ledger = Ledger(decision_dir)
    blocked = any(r.get("kind") == "blocker" and r.get("request_id") == req.id for r in ledger.read("events"))
    if blocked:
        raise SupersededError(f"request {req.id} was rejected twice and is blocked; nothing to answer until the user rules")
    new = superseded_by(decision_dir, req.id)
    rejected = any((decision_dir / "requests").glob(f"{req.id}.answer.rejected*.json"))
    if new or rejected:
        raise SupersededError(
            f"request {req.id} was rejected and re-issued" + (f" as {new}" if new else "") + "; answer the current request instead"
        )
    try:
        st = load_state(decision_dir)
    except FileNotFoundError:
        return
    if req.id not in st.pending_requests:
        raise SupersededError(f"request {req.id} is not pending (already accepted or superseded)")


def ingest(decision_dir: Path, request_id: str) -> Answer:
    """Validate the answer file against the request schema; record it; clear the pending mark.

    On rejection the request is re-issued (attempt 2) and a HandshakeError names the new id;
    a second rejection raises BlockerError. A request that was superseded refuses late answers.
    The whole check-then-record sequence holds the directory lock, so two processes cannot both
    accept one answer or both re-issue one rejected request.
    """
    _require_decision(decision_dir)
    with exclusive(decision_dir):
        return _ingest_locked(decision_dir, request_id)


def _ingest_locked(decision_dir: Path, request_id: str) -> Answer:
    req = read_request(decision_dir, request_id)
    _refuse_if_superseded(decision_dir, req)
    p = answer_path(decision_dir, request_id)
    if not p.exists():
        raise HandshakeError(f"no answer file yet: {p}")
    ans: Answer | None
    try:
        parsed = Answer.model_validate_json(p.read_text())
    except ValidationError as e:
        errors = [f"answer envelope: {'/'.join(str(x) for x in err['loc']) or '$'}: {err['msg']}" for err in e.errors()]
        ans = None
    except ValueError as e:
        errors = [f"answer file is not valid JSON: {e}"]
        ans = None
    else:
        ans = parsed
        errors = validate_answer(req, parsed)
    if errors:
        new = _reject(
            decision_dir, req, ans, errors,
            provider=ans.provider if ans else "unknown", family=ans.family if ans else "unknown", isolation=ans.isolation if ans else "n/a",
        )
        raise HandshakeError(
            f"answer {request_id} rejected: " + "; ".join(errors[:5]) + (f". Re-issued as {new.id} (attempt {new.attempt})." if new else "")
        )
    assert ans is not None
    Ledger(decision_dir).record_llm_call(
        request_id=req.id, role=req.role, provider=ans.provider, family=ans.family, isolation=ans.isolation, answer=ans.answer,
    )
    _with_state(decision_dir, lambda st: clear_pending(st, request_id))
    return ans


def reissue(decision_dir: Path, request_id: str, errors: list[str]) -> Request:
    """Re-emit a rejected request with the validation errors appended (attempt + 1)."""
    with exclusive(decision_dir):
        return _reissue_locked(decision_dir, request_id, errors)


def _reissue_locked(decision_dir: Path, request_id: str, errors: list[str]) -> Request:
    req = read_request(decision_dir, request_id)
    if req.attempt >= MAX_ATTEMPTS:
        raise BlockerError(f"request {request_id} failed {req.attempt} times; BLOCKER for the user")
    new = create_request(
        decision_dir,
        role=req.role,
        purpose=req.purpose,
        prompt=req.prompt + "\n\nYour previous answer was rejected by the validator:\n- " + "\n- ".join(errors),
        answer_schema=req.answer_schema,
        provider=req.provider,
        node_ids=req.node_ids,
        constraints=req.constraints,
        attempt=req.attempt + 1,
        previous_errors=errors,
    )
    _with_state(decision_dir, lambda st: clear_pending(st, request_id))
    Ledger(decision_dir).event("reissue", request_id=request_id, new_request_id=new.id, attempt=new.attempt)
    return new


def pending(decision_dir: Path) -> list[str]:
    """One source of truth: the state's pending list. No state, no decision (FileNotFoundError)."""
    st = load_state(decision_dir)
    return list(st.pending_requests)


def orphaned_request_files(decision_dir: Path) -> list[str]:
    """Request files with no accepted or rejected answer that the state does not list as pending (a consistency check)."""
    return [r for r in file_pending(decision_dir) if r not in pending(decision_dir)]
