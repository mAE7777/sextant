"""Append-only ledgers per decision. Nothing is edited; corrections are retraction rows.

Files under decisions/<slug>/:
  context-ledger.jsonl   every item of context (source, date, grade, quote, nodes served)
  jev-calls.jsonl        every Jev request (card, protocol, per-question payload hashes, answers)
  llm-calls.jsonl        every fulfilled request file (provider, family, isolation, answer sha)
  human-inputs.jsonl     every answer the user gave through a gate
  events.jsonl           gate stamps, mode runs, degradations, blockers
  compute-ledger.jsonl   tokens and calls per pass
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

LEDGER_FILES = (
    "context-ledger",
    "jev-calls",
    "llm-calls",
    "human-inputs",
    "events",
    "compute-ledger",
)

COST_FIELDS = ("provider", "model", "request_id", "input_tokens", "output_tokens", "purpose")

_registry = threading.Lock()
_thread_locks: dict[str, threading.RLock] = {}
_depth: dict[str, int] = {}
# Per ledger file: (size, mtime_ns, last seq, ids, holds pre-protocol jev rows). Valid only while
# the file's size and mtime still match, so another process's append forces a re-read.
_append_index: dict[str, tuple[int, int, int, set[str], bool]] = {}


class WriterBusy(RuntimeError):
    """A second writer found the single-writer lock already held."""


class IncompleteCostRecord(ValueError):
    """A compute-ledger row that does not name the call and its token use."""


class IncompleteLedger(ValueError):
    """A ledger file ends in a partial line: a write was torn and must be recovered first."""


class LedgerIdentityError(ValueError):
    """A row reuses an id the ledger already holds, or mixes protocols it must not mix."""


def concurrency_status() -> dict[str, str]:
    """Crash-safe multi-file transactions are not built yet. The lock is the current guard."""
    return {
        "mode": "single_writer_lock",
        "transactions": "deferred",
        "deferred_to": "P1",
    }


def _lock_key(directory: Path) -> str:
    return str(directory.resolve())


def _rlock(key: str) -> threading.RLock:
    with _registry:
        lock = _thread_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _thread_locks[key] = lock
        return lock


def writer_lock_held(directory: Path) -> bool:
    """True when this process holds the lock, or another process does."""
    with _registry:
        if _depth.get(_lock_key(directory), 0) > 0:
            return True
    path = directory / ".writer.lock"
    if not path.exists():
        return False
    with path.open("a+") as probe:
        try:
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(probe.fileno(), fcntl.LOCK_UN)
    return False


@contextmanager
def exclusive(directory: Path, *, block: bool = True) -> Iterator[None]:
    """Serialize writers in this process and across processes.

    Re-entering from the same thread is allowed so a state update can append a
    ledger row while it already holds the directory. A different thread or
    process waits. This is not a crash-safe transaction: a kill mid-write can
    still tear one file, and ledger plus state are not one commit.
    """
    directory.mkdir(parents=True, exist_ok=True)
    key = _lock_key(directory)
    lock = _rlock(key)
    if block:
        lock.acquire()
    elif not lock.acquire(blocking=False):
        raise WriterBusy(f"single-writer lock held for {directory}")
    fd: TextIO | None = None
    try:
        with _registry:
            depth = _depth.get(key, 0)
            _depth[key] = depth + 1
            first = depth == 0
        if first:
            fd = (directory / ".writer.lock").open("a+")
            if block:
                fcntl.flock(fd.fileno(), fcntl.LOCK_EX)
            else:
                try:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    fd.close()
                    fd = None
                    raise WriterBusy(f"single-writer lock held by another process for {directory}") from exc
        yield
    finally:
        with _registry:
            _depth[key] = _depth.get(key, 1) - 1
            last = _depth[key] == 0
            if last:
                _depth.pop(key, None)
        if last and fd is not None:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
            fd.close()
        lock.release()


def _require_cost_chain(row: dict[str, Any]) -> None:
    if row.get("type") == "retract":
        return  # a correction names the row it retracts, not a call
    missing = [key for key in COST_FIELDS if row.get(key) in (None, "")]
    if missing:
        raise IncompleteCostRecord(
            "a cost record must name the provider, model, request, purpose, and token counts; "
            "a filename is not a production-cost chain (missing " + ", ".join(missing) + ")"
        )


def now_iso() -> str:
    """Observational timestamp for calls and stamps (a fact about when the call happened)."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha_of(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()[:16]


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Ledger:
    """Reads never touch the disk layout; only `append` creates the decision folder."""

    def __init__(self, decision_dir: Path):
        self.dir = decision_dir

    def path(self, name: str) -> Path:
        if name not in LEDGER_FILES:
            raise ValueError(f"unknown ledger {name}")
        return self.dir / f"{name}.jsonl"

    def read(self, name: str) -> list[dict[str, Any]]:
        """Every row, retracted ones included. A torn final line is refused, not skipped.

        A partial last line means a write stopped half way. Parsing around it would hide
        the loss, and appending after it would bury it under a valid row, so both stop
        here until the file is recovered (transactional storage is not built yet).
        """
        p = self.path(name)
        if not p.exists():
            return []
        text = p.read_text()
        if text and not text.endswith("\n"):
            raise IncompleteLedger(
                f"{p} ends with an incomplete line (a torn write); recover it before reading or appending"
            )
        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def append(self, name: str, row: dict[str, Any], at: str | None = None) -> dict[str, Any]:
        """Append a row. The runtime owns `seq` and `at`; a row may not override them.

        The sequence is chosen while the directory's single-writer lock is held,
        so two writers cannot both observe the same last sequence.
        """
        if name == "compute-ledger":
            _require_cost_chain(row)
        with exclusive(self.dir):
            last_seq, ids, legacy = self._append_state(name)
            row_id = row.get("id")
            if row_id is not None and row_id in ids:
                raise LedgerIdentityError(
                    f"{name} already holds a row with id {row_id}; an id names one row, never two"
                )
            if name == "jev-calls" and row.get("protocol_version") and legacy:
                raise LedgerIdentityError(
                    f"{self.path(name)} holds calls recorded before the wire protocol; start a new "
                    "decision directory instead of mixing them with new evidence"
                )
            seq = last_seq + 1
            body = {k: v for k, v in row.items() if k not in ("seq", "at")}
            full = {"seq": seq, "at": at or now_iso(), **body}
            self.dir.mkdir(parents=True, exist_ok=True)
            path = self.path(name)
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(full, ensure_ascii=False) + "\n")
            if row_id is not None:
                ids.add(str(row_id))
            st = path.stat()
            _append_index[str(path.resolve())] = (st.st_size, st.st_mtime_ns, seq, ids, legacy)
            return full

    def _append_state(self, name: str) -> tuple[int, set[str], bool]:
        """Last seq, known ids and whether pre-protocol Jev rows exist, read once per file change."""
        path = self.path(name)
        if not path.exists():
            return 0, set(), False
        st = path.stat()
        cached = _append_index.get(str(path.resolve()))
        if cached is not None and cached[0] == st.st_size and cached[1] == st.st_mtime_ns:
            return cached[2], set(cached[3]), cached[4]
        rows = self.read(name)
        last_seq = max((int(r["seq"]) for r in rows), default=0)
        ids = {str(r["id"]) for r in rows if "id" in r}
        legacy = name == "jev-calls" and any(r.get("type") != "retract" and not r.get("protocol_version") for r in rows)
        _append_index[str(path.resolve())] = (st.st_size, st.st_mtime_ns, last_seq, ids, legacy)
        return last_seq, set(ids), legacy

    def record_cost(
        self,
        *,
        provider: str,
        model: str,
        request_id: str,
        input_tokens: int,
        output_tokens: int,
        purpose: str,
    ) -> dict[str, Any]:
        """Record token use for one call. `priced` stays false: this is not a currency ledger."""
        return self.append(
            "compute-ledger",
            {
                "provider": provider,
                "model": model,
                "request_id": request_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "purpose": purpose,
                "priced": False,
            },
        )

    def retract(self, name: str, seq: int, reason: str) -> dict[str, Any]:
        """Check and append under one lock, so two writers cannot both retract one row."""
        with exclusive(self.dir):
            rows = self.read(name)
            existing = {r["seq"] for r in rows if r.get("type") != "retract"}
            already = {r["retracts_seq"] for r in rows if r.get("type") == "retract"}
            if seq not in existing:
                raise ValueError(f"cannot retract seq {seq} in {name}: no such row")
            if seq in already:
                raise ValueError(f"seq {seq} in {name} is already retracted")
            return self.append(name, {"type": "retract", "retracts_seq": seq, "reason": reason})

    def active(self, name: str) -> list[dict[str, Any]]:
        """Rows minus retracted ones."""
        rows = self.read(name)
        retracted = {r["retracts_seq"] for r in rows if r.get("type") == "retract"}
        return [r for r in rows if r.get("type") != "retract" and r["seq"] not in retracted]

    # ---- jev calls

    def record_jev_call(
        self,
        *,
        card_id: str,
        protocol_version: str,
        question_keys: list[str],
        question_bindings: dict[str, dict[str, str]],
        questions_digest: str,
        model_id: str,
        transport: str,
        state_sha: str,
        request: dict[str, Any],
        response: dict[str, Any],
        latency_ms: float,
        usage: dict[str, Any] | None,
        run_id: str,
        call_role: str,
        run_protocol: dict[str, Any],
        variant: str = "base",
        call_id: str | None = None,
        input_language: str | None = None,
    ) -> dict[str, Any]:
        """One row per request, with the per-question payload each answer can be traced to.

        A request carries several questions, so a single `question_hash` over all of them
        could never match one judgment's hash. `question_bindings` maps each question
        key to the template and rendered hashes actually sent for it in this call. `run_id`
        names the one `ask` this call belongs to, `call_role` says what the call was for, and
        `run_protocol` records the re-test settings in force, so an audit can re-derive a
        judgment from every call of its run and tell a missing re-run from one never required.
        `input_language` is the caller's declaration of the language the state is written in
        (None when undeclared); it is part of the request digest when declared.
        """
        cid = call_id or new_id("jc")
        return self.append(
            "jev-calls",
            {
                "id": cid,
                "card_id": card_id,
                "protocol_version": protocol_version,
                "run_id": run_id,
                "call_role": call_role,
                "run_protocol": run_protocol,
                "question_keys": list(question_keys),
                "question_bindings": question_bindings,
                "questions_digest": questions_digest,
                "model_id": model_id,
                "transport": transport,
                "state_sha": state_sha,
                "input_language": input_language,
                "variant": variant,
                "request": request,
                "response": response,
                "latency_ms": round(latency_ms, 1),
                "usage": usage or {},
            },
        )

    def call_index(self) -> dict[str, dict[str, Any]]:
        """Every recorded call by id, retracted ones included and marked.

        Retracted rows stay visible because a judgment re-derived without them could hide an
        unfavourable re-run behind a correction. Ids recorded more than once are marked
        ambiguous rather than resolved to whichever row came last.
        """
        rows = self.read("jev-calls")
        retracted = {r["retracts_seq"] for r in rows if r.get("type") == "retract"}
        index: dict[str, dict[str, Any]] = {}
        counts: dict[str, int] = {}
        for r in rows:
            if r.get("type") == "retract" or "id" not in r:
                continue
            counts[r["id"]] = counts.get(r["id"], 0) + 1
            index[r["id"]] = {**r, "_retracted": True} if r["seq"] in retracted else r
        for call_id, n in counts.items():
            if n > 1:
                index[call_id] = {"id": call_id, "_ambiguous": n}
        return index

    def has_call(self, call_id: str) -> bool:
        return call_id in self.call_index()

    # ---- llm calls

    def record_llm_call(
        self,
        *,
        request_id: str,
        role: str,
        provider: str,
        family: str,
        isolation: str,
        answer: Any,
        status: str = "accepted",
        errors: list[str] | None = None,
    ) -> dict[str, Any]:
        return self.append(
            "llm-calls",
            {
                "request_id": request_id,
                "role": role,
                "provider": provider,
                "family": family,
                "isolation": isolation,
                "answer_sha": sha_of(answer),
                "status": status,
                "errors": errors or [],
            },
        )

    # ---- events

    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        return self.append("events", {"kind": kind, **fields})

    # ---- context

    def add_context(
        self,
        *,
        source: str,
        date: str,
        quote: str,
        grade: str,
        kind: str,
        nodes: list[str],
        original: str | None = None,
        language: str = "en",
        directness: str = "secondary",
        added_by: str = "session",
    ) -> dict[str, Any]:
        item_id = new_id("ctx")
        return self.append(
            "context-ledger",
            {
                "id": item_id,
                "kind": kind,
                "source": source,
                "date": date,
                "quote": quote,
                "original": original,
                "language": language,
                "grade": grade,
                "directness": directness,
                "nodes": nodes,
                "added_by": added_by,
            },
        )

    def context_index(self) -> dict[str, dict[str, Any]]:
        return {r["id"]: r for r in self.active("context-ledger") if "id" in r}
