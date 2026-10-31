from __future__ import annotations

import json
from pathlib import Path

import pytest

from sextant.config import load_config
from sextant.handshake import HandshakeError, create_request, fulfil, ingest, reissue
from sextant.ledger import Ledger
from sextant.providers.base import Answer, pending_requests, read_request, write_answer
from sextant.providers.codex import CodexProvider
from sextant.providers.openai_compat import OpenAICompatProvider
from sextant.state import init_state, load_state

FAKE_CODEX = Path(__file__).parent / "bin" / "fake_codex"
SCHEMA = {
    "type": "object",
    "required": ["ok", "note"],
    "additionalProperties": False,
    "properties": {"ok": {"type": "boolean"}, "note": {"type": "string"}},
}


@pytest.mark.fixture
def test_claude_code_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    init_state(tmp_path, "demo")
    config = load_config()
    req = create_request(tmp_path, role="generator", purpose="demo", prompt="say ok", answer_schema=SCHEMA, provider="claude-code")
    assert pending_requests(tmp_path) == [req.id]
    assert fulfil(tmp_path, req.id, config) is None
    assert f"AWAITING {req.id}" in capsys.readouterr().out
    # the session writes the answer file, then the runtime ingests it
    write_answer(tmp_path, Answer(request_id=req.id, answer={"ok": True, "note": "done"}, provider="claude-code", family="anthropic", isolation="same-session"))
    ans = ingest(tmp_path, req.id)
    assert ans.family == "anthropic"
    assert pending_requests(tmp_path) == []
    assert load_state(tmp_path).pending_requests == []
    rows = Ledger(tmp_path).read("llm-calls")
    assert rows[-1]["status"] == "accepted" and rows[-1]["isolation"] == "same-session"


@pytest.mark.fixture
def test_bad_answer_rejected_and_reissued(tmp_path: Path) -> None:
    init_state(tmp_path, "demo")
    req = create_request(tmp_path, role="generator", purpose="demo", prompt="say ok", answer_schema=SCHEMA, provider="claude-code")
    write_answer(tmp_path, Answer(request_id=req.id, answer={"ok": "yes"}, provider="claude-code", family="anthropic", isolation="same-session"))
    with pytest.raises(HandshakeError, match="rejected"):
        ingest(tmp_path, req.id)
    rows = Ledger(tmp_path).read("llm-calls")
    assert rows[-1]["status"] == "rejected" and rows[-1]["errors"]
    # ingest re-issued the request automatically as attempt 2
    pending_now = load_state(tmp_path).pending_requests
    assert len(pending_now) == 1 and pending_now[0] != req.id
    new = read_request(tmp_path, pending_now[0])
    assert new.attempt == 2 and "rejected by the validator" in new.prompt
    with pytest.raises(HandshakeError, match="BLOCKER"):
        reissue(tmp_path, new.id, ["still wrong"])


@pytest.mark.fixture
def test_codex_provider_with_fake_binary(tmp_path: Path) -> None:
    init_state(tmp_path, "demo")
    req = create_request(tmp_path, role="challenger", purpose="demo", prompt="what is missing?", answer_schema=SCHEMA, provider="codex")
    provider = CodexProvider(binary=str(FAKE_CODEX), timeout_s=30)
    assert provider.probe()["available"] is True
    ans = provider.fulfil(read_request(tmp_path, req.id), tmp_path)
    assert ans is not None and ans.family == "openai" and ans.isolation == "isolated"
    assert ans.answer["ok"] is True
    cmd = provider.command(Path("s.json"), Path("o.json"))
    assert cmd[1:4] == ["exec", "--skip-git-repo-check", "--ephemeral"] and cmd[-1] == "-"
    assert (tmp_path / "requests" / f"{req.id}.schema.json").exists()
    write_answer(tmp_path, ans)
    accepted = ingest(tmp_path, req.id)
    assert accepted.provider == "codex"


@pytest.mark.unit
def test_codex_probe_missing_binary() -> None:
    p = CodexProvider(binary="definitely-not-a-binary-xyz").probe()
    assert p["available"] is False


@pytest.mark.unit
def test_openai_compat_probe_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    p = OpenAICompatProvider("deepseek").probe()
    assert p["available"] is False and "DEEPSEEK_API_KEY" in p["reason"]
    with pytest.raises(ValueError):
        OpenAICompatProvider("nope")


@pytest.mark.unit
def test_crafted_ids_cannot_escape_requests_dir(tmp_path: Path) -> None:
    from sextant.providers.base import answer_path, request_path

    for bad in ("../../x", "rq_../x", "rq_a/b", "", "x" * 60):
        with pytest.raises(ValueError):
            request_path(tmp_path, bad)
        with pytest.raises(ValueError):
            answer_path(tmp_path, bad)
    with pytest.raises(Exception, match="pattern|string_pattern|match"):
        Answer(request_id="../../evil", answer={}, provider="claude-code", family="anthropic", isolation="same-session")
    assert request_path(tmp_path, "rq_ok-1").parent == tmp_path / "requests"


@pytest.mark.unit
def test_request_file_shape(tmp_path: Path) -> None:
    init_state(tmp_path, "demo")
    req = create_request(tmp_path, role="grader", purpose="label", prompt="p", answer_schema=SCHEMA, provider="codex", node_ids=["n_1"])
    data = json.loads((tmp_path / "requests" / f"{req.id}.request.json").read_text())
    assert set(data) >= {"id", "role", "purpose", "node_ids", "prompt", "answer_schema", "provider", "created", "attempt"}
