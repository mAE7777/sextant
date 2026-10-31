"""Regression tests for graph saves, pending requests, and retractions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sextant.graph import check_ids, save_graph
from sextant.handshake import HandshakeError, SupersededError, create_request, ingest, pending
from sextant.ledger import Ledger
from sextant.providers.base import Answer, pending_requests, write_answer
from sextant.schema import DecisionGraph
from sextant.selftest import base_graph, edge, node
from sextant.state import init_state

SCHEMA = {"type": "object", "required": ["ok"], "additionalProperties": False, "properties": {"ok": {"type": "boolean"}}}


def _graph(version: int = 1, parent: int | None = None, crit: str = "crit A") -> DecisionGraph:
    q = node("question", "Should I take the offer?")
    c = node("criterion", crit)
    r = node("factor", "other components not listed", residual=True)
    data = base_graph([q, c, r], [edge("decomposes_into", q, c), edge("decomposes_into", q, r)])
    data["version"] = version
    if parent is not None:
        data["parent_version"] = parent
    return DecisionGraph.model_validate(data)


@pytest.mark.unit
def test_save_writes_diff_and_requires_parent(tmp_path: Path) -> None:
    save_graph(_graph(1), tmp_path)
    with pytest.raises(ValueError, match="parent_version"):
        save_graph(_graph(2), tmp_path)
    with pytest.raises(FileNotFoundError):
        save_graph(_graph(3, parent=2), tmp_path)
    save_graph(_graph(2, parent=1, crit="crit B"), tmp_path)
    diff = tmp_path / "diff.v1-v2.md"
    assert diff.exists() and "Diff v1 -> v2" in diff.read_text()
    with pytest.raises(ValueError, match="v1 cannot"):
        save_graph(_graph(1, parent=1), tmp_path / "other")


@pytest.mark.unit
def test_content_addressed_ids_checked() -> None:
    f = node("factor", "salary covers expenses")
    good = DecisionGraph.model_validate(base_graph([f], []))
    assert not check_ids(good)
    f2 = dict(f)
    f2["id"] = "n_000000000000"
    bad = DecisionGraph.model_validate(base_graph([f2], []))
    assert check_ids(bad) and "expected" in check_ids(bad)[0].message


@pytest.mark.fixture
def test_superseded_request_refuses_late_answer(tmp_path: Path) -> None:
    init_state(tmp_path, "demo")
    req = create_request(tmp_path, role="generator", purpose="x", prompt="p", answer_schema=SCHEMA, provider="claude-code")
    write_answer(tmp_path, Answer(request_id=req.id, answer={"ok": "no"}, provider="claude-code", family="anthropic", isolation="same-session"))
    with pytest.raises(HandshakeError, match="Re-issued as"):
        ingest(tmp_path, req.id)
    # a late, conforming answer for the OLD id must be refused, not accepted
    write_answer(tmp_path, Answer(request_id=req.id, answer={"ok": True}, provider="claude-code", family="anthropic", isolation="same-session"))
    with pytest.raises(SupersededError, match="re-issued"):
        ingest(tmp_path, req.id)
    accepted = [r for r in Ledger(tmp_path).read("llm-calls") if r["status"] == "accepted"]
    assert accepted == []
    # and an id that was already accepted is not pending, so a second answer is refused too
    new_id = pending(tmp_path)[0]
    write_answer(tmp_path, Answer(request_id=new_id, answer={"ok": True}, provider="claude-code", family="anthropic", isolation="same-session"))
    ingest(tmp_path, new_id)
    write_answer(tmp_path, Answer(request_id=new_id, answer={"ok": False}, provider="claude-code", family="anthropic", isolation="same-session"))
    with pytest.raises(SupersededError, match="not pending"):
        ingest(tmp_path, new_id)


@pytest.mark.unit
def test_pending_is_read_only(tmp_path: Path) -> None:
    ghost = tmp_path / "ghost"
    assert pending_requests(ghost) == []
    with pytest.raises(FileNotFoundError):
        pending(ghost)
    assert not ghost.exists()


@pytest.mark.unit
def test_retract_twice_refused(tmp_path: Path) -> None:
    led = Ledger(tmp_path)
    led.append("events", {"kind": "a"})
    led.retract("events", 1, "first")
    with pytest.raises(ValueError, match="already retracted"):
        led.retract("events", 1, "again")


@pytest.mark.fixture
def test_historical_parser_plain_lines(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from sextant import __main__ as cli

    # Historical library behavior remains tested in temp fixtures; V1 refuses
    # its write/live front doors in test_v1_legacy_boundary. Data-writing
    # commands require a config file to exist; use an empty one outside tmp_path.

    config_file = tmp_path_factory.mktemp("config") / "config.toml"
    config_file.write_text("")
    monkeypatch.setenv("SEXTANT_CONFIG", str(config_file))
    cfg = {"paths": {"decisions_dir": str(tmp_path), "rolecards_dir": str(tmp_path / "cards")}, "providers": {"auto_fulfil": ["deepseek"]}}
    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    assert cli.legacy_main(["status", "ghost-slug"]) == 1
    assert "NOT FOUND" in capsys.readouterr().out
    assert not (tmp_path / "ghost-slug").exists()
    assert cli.legacy_main(["init", "cli-b"]) == 0
    assert cli.legacy_main(["answer", "cli-b", "rq_doesnotexist"]) == 1
    assert "NOT FOUND" in capsys.readouterr().out
    assert cli.legacy_main(["answer", "cli-b", "rq_../x"]) == 1
    assert "INVALID" in capsys.readouterr().out
    assert cli.legacy_main(["request", "cli-b", "--role", "generator", "--purpose", "x", "--prompt", "p", "--schema", '{"type": "bogus"}']) == 1
    assert "REFUSED" in capsys.readouterr().out
    assert cli.legacy_main(["request", "cli-b", "--role", "generator", "--purpose", "x", "--prompt", "p", "--schema", json.dumps(SCHEMA)]) == 0
    rid = capsys.readouterr().out.split()[1]
    assert cli.legacy_main(["fulfil", "cli-b", rid, "--provider", "nonesuch"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("PROVIDER FAIL") and "no fallback was attempted" in out
    assert "fallbacks" not in out
    assert cli.legacy_main(["selfcheck", "cli-b", "--version", "99"]) == 1
    assert "NOT FOUND" in capsys.readouterr().out
