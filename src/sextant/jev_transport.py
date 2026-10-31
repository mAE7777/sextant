"""The thinnest possible typed call to Jev (TypeSafe direct, pinned model), used by the setup
smoke test and by the full client that S2 builds on top of it.

Returns the documented answer shapes unchanged:
  noul   -> {"type": "noul", "noul": p}
  choice -> {"type": "choice", "choice", "probabilities", "confidence"}
  score  -> {"type": "score", "score", "legend", "probabilities", "confidence"}
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, cast


class JevUnavailable(RuntimeError):
    pass


SECRET_ENV_SUFFIXES = ("_KEY", "_TOKEN", "_SECRET", "_KEYS")


def redact(text: str) -> str:
    """Replace the value of any secret-looking environment variable that appears in text."""
    out = text
    for name, value in os.environ.items():
        if value and len(value) >= 8 and name.upper().endswith(SECRET_ENV_SUFFIXES):
            out = out.replace(value, f"[{name} redacted]")
    return out


@dataclass
class JevResult:
    model_id: str
    answers: dict[str, dict[str, Any]]
    usage: dict[str, int]
    latency_ms: float
    request_id: str | None
    transport: str = "typesafe"


def key_present() -> bool:
    return bool(os.environ.get("TYPESAFE_API_KEY"))


def _to_sdk_questions(questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from typesafe_sdk import Choice, Noul, NoulCriteria, Score

    from sextant.wire import ordered_criteria

    out: dict[str, Any] = {}
    for key, q in questions.items():
        t = q["type"]
        if t == "noul":
            crit = ordered_criteria(q) if q.get("criteria") else None
            out[key] = Noul(instructions=q["instructions"], criteria=cast(NoulCriteria, crit)) if crit else Noul(instructions=q["instructions"])
        elif t == "choice":
            out[key] = Choice(instructions=q["instructions"], criteria=ordered_criteria(q))
        elif t == "score":
            out[key] = Score(instructions=q["instructions"], criteria=list(q["criteria"]))
        else:
            raise ValueError(f"unknown question type {t}")
    return out


def call_jev(state: Any, questions: dict[str, dict[str, Any]], model_id: str = "jev-1.13.0", timeout_s: float = 60.0) -> JevResult:
    """One typed request. Raises JevUnavailable when the transport cannot be used at all."""
    if not key_present():
        raise JevUnavailable("TYPESAFE_API_KEY is not set in this shell")
    from typesafe_sdk import RetryPolicy, TypeSafeClient

    t0 = time.perf_counter()
    # Every physical attempt belongs in the dispatcher ledger. SDK retries hide
    # requests/usage and can resend a request whose outcome is unknown.
    with TypeSafeClient(timeout=timeout_s, retry=RetryPolicy(max_retries=0, timeout=None)) as client:
        resp = client.system_one(state=state, questions=_to_sdk_questions(questions), model=model_id)
    latency = (time.perf_counter() - t0) * 1000
    answers: dict[str, dict[str, Any]] = {}
    for key, ans in resp.answers.items():
        d = ans.model_dump(mode="json")
        if "probabilities" in d and isinstance(d["probabilities"], dict):
            d["probabilities"] = {str(k): float(v) for k, v in d["probabilities"].items()}
        if "legend" in d and isinstance(d["legend"], dict):
            d["legend"] = {str(k): v for k, v in d["legend"].items()}
        answers[key] = d
    usage = resp.usage.model_dump(mode="json") if resp.usage is not None else {}
    return JevResult(
        model_id=str(resp.model),
        answers=answers,
        usage={k: int(v) for k, v in usage.items() if isinstance(v, (int, float))},
        latency_ms=latency,
        request_id=_request_id(resp),
    )


def _request_id(resp: Any) -> str | None:
    """The provider's request id, or None when the response carries none.

    The SDK raises its own `TypeSafeError` (not `AttributeError`) when the header is missing,
    so `getattr` with a default does not protect the call. A missing id is recorded as None and
    the ledger audit decides what that row can support.
    """
    from typesafe_sdk import TypeSafeError

    try:
        value = resp.request_id
    except (AttributeError, TypeSafeError):
        return None
    return str(value) if value is not None else None


def smoke() -> dict[str, Any]:
    """One Noul on a trivial state; returns what setup records."""
    try:
        r = call_jev(
            {"note": "The meeting is at 3pm on Friday."},
            {"afternoon": {"type": "noul", "instructions": "Is the meeting in the afternoon (12:00 to 18:00)?"}},
        )
    except JevUnavailable as e:
        return {"available": False, "reason": redact(str(e))}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": redact(f"{type(e).__name__}: {str(e)[:160]}")}
    return {
        "available": True,
        "model_id": r.model_id,
        "latency_ms": round(r.latency_ms),
        "usage": r.usage,
        "answer": r.answers.get("afternoon", {}).get("noul"),
    }
