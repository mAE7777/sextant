"""Regression tests for the uninitialized-decision path."""

from __future__ import annotations

from pathlib import Path

import pytest

from sextant.handshake import (
    HandshakeError,
    SupersededError,
    create_request,
    ingest,
    orphaned_request_files,
)
from sextant.ledger import Ledger
from sextant.providers.base import Answer, request_path, write_answer
from sextant.state import init_state

SCHEMA = {"type": "object", "required": ["ok"], "additionalProperties": False, "properties": {"ok": {"type": "boolean"}}}


@pytest.mark.unit
def test_create_request_requires_initialized_decision(tmp_path: Path) -> None:
    ghost = tmp_path / "ghost"
    with pytest.raises(HandshakeError, match="run init first"):
        create_request(ghost, role="generator", purpose="x", prompt="p", answer_schema=SCHEMA, provider="claude-code")
    assert not ghost.exists()


@pytest.mark.unit
def test_ledger_reads_never_create_folders(tmp_path: Path) -> None:
    ghost = tmp_path / "ghost"
    led = Ledger(ghost)
    assert led.read("events") == [] and led.call_index() == {}
    assert not ghost.exists()
    led.event("x")
    assert ghost.exists()


@pytest.mark.unit
def test_path_lookups_never_create_requests_dir(tmp_path: Path) -> None:
    p = request_path(tmp_path, "rq_abc")
    assert not p.parent.exists()


@pytest.mark.fixture
def test_blocked_request_message_and_no_orphans(tmp_path: Path) -> None:
    init_state(tmp_path, "demo")
    req = create_request(tmp_path, role="generator", purpose="x", prompt="p", answer_schema=SCHEMA, provider="claude-code")
    write_answer(tmp_path, Answer(request_id=req.id, answer={"ok": "no"}, provider="claude-code", family="anthropic", isolation="same-session"))
    with pytest.raises(HandshakeError):
        ingest(tmp_path, req.id)
    from sextant.handshake import pending

    second = pending(tmp_path)[0]
    write_answer(tmp_path, Answer(request_id=second, answer={"ok": "still no"}, provider="claude-code", family="anthropic", isolation="same-session"))
    with pytest.raises(HandshakeError, match="BLOCKER"):
        ingest(tmp_path, second)
    write_answer(tmp_path, Answer(request_id=second, answer={"ok": True}, provider="claude-code", family="anthropic", isolation="same-session"))
    with pytest.raises(SupersededError, match="blocked; nothing to answer"):
        ingest(tmp_path, second)
    assert orphaned_request_files(tmp_path) == []


@pytest.mark.fixture
def test_provider_name_is_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sextant.handshake import ProviderError, fulfil

    init_state(tmp_path, "demo")
    monkeypatch.setenv("QA_PROBE_KEY", "hunter2secretvalue")
    req = create_request(tmp_path, role="generator", purpose="x", prompt="p", answer_schema=SCHEMA, provider="claude-code")
    with pytest.raises(ProviderError) as ei:
        fulfil(tmp_path, req.id, {"providers": {"auto_fulfil": []}}, provider_override="hunter2secretvalue")
    assert "hunter2secretvalue" not in str(ei.value)
    events = Ledger(tmp_path).read("events")
    fail = [e for e in events if e["kind"] == "provider_fail"][-1]
    assert "hunter2secretvalue" not in fail["provider"] and "hunter2secretvalue" not in fail["reason"]
