"""Legacy boundary probes; fixtures live in pytest temp, provider calls are intercepted."""
from __future__ import annotations

import hashlib
import json

import pytest

from sextant import __main__ as old
from sextant import cli


def snapshot(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("entry", ["legacy-wrapper", "module-old-verb", "installed-old-verb"])
@pytest.mark.parametrize("argv,handler", [
    (["init", "blocked"], "cmd_init"),
    (["schema", "--exp", "would-write.json"], "cmd_schema"),
    (["config", "--write-defaults"], "cmd_config"),
    (["setup", "--probe-only"], "cmd_setup"),
    (["doctor", "--live", "--record", "health.json"], "cmd_doctor"),
    (["selftest"], "cmd_selftest"),
    (["stamp", "blocked", "frame", "SKIP", "--token", "old-token"], "cmd_stamp"),
    (["gate", "blocked", "frame"], "cmd_gate"),
    (["request", "blocked", "--role", "generator", "--purpose", "probe", "--prompt", "probe", "--schema", "{}", "--fulfil"], "cmd_request"),
    (["fulfil", "blocked", "request-1"], "cmd_fulfil"),
    (["answer", "blocked", "request-1"], "cmd_answer"),
    (["rolecards", "seal"], "cmd_rolecards"),
    (["jev", "ask", "R1", "--pack", "pack.json"], "cmd_jev_ask"),
    (["meta", "rm1"], "cmd_meta"),
    (["calibrate", "measure", "R1"], "cmd_calibrate"),
])
def test_unsafe_handler_is_never_reached(entry, argv, handler, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "preserved.txt").write_text("historical evidence\n")
    before = snapshot(tmp_path)
    calls = []
    def unsafe(args):
        calls.append(handler)
        (tmp_path / "would-write.txt").write_text("intercepted unsafe handler\n")
        return 0
    monkeypatch.setattr(old, handler, unsafe)
    code = (cli.main(["legacy", *argv]) if entry == "legacy-wrapper" else
            cli.main(argv) if entry == "installed-old-verb" else old.main(argv))
    output = capsys.readouterr()
    assert code == 1
    result = json.loads(output.out)
    assert result["status"] == "legacy_read_only"
    assert result["errors"][0]["code"] == "legacy_read_only"
    assert result["next_actions"] and "import" in output.out and "guide" in output.out
    assert not calls and snapshot(tmp_path) == before and not output.err


@pytest.mark.parametrize("entry", [cli.main, old.main])
def test_real_init_does_not_create_a_decision(entry, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.toml"
    config.write_text('[paths]\ndecisions_dir="decisions"\n')
    monkeypatch.setenv("SEXTANT_CONFIG", str(config))
    before = snapshot(tmp_path)
    argv = ["init", "new-decision", "--question", "must not write"]
    code = entry(["legacy", *argv]) if entry is cli.main else entry(argv)
    output = capsys.readouterr()
    assert code == 1 and "legacy_read_only" in output.out
    assert snapshot(tmp_path) == before and not (tmp_path / "decisions").exists()


def test_jev_client_is_not_constructed_or_asked(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pack.json").write_text("{}")
    calls = []
    class Client:
        def ask(self, *args, **kwargs):
            calls.append("ask")
            raise ValueError("intercepted; no network request")
    def client(*args):
        calls.append("client")
        return Client()
    monkeypatch.setattr(old, "_client", client)
    monkeypatch.setattr(old, "load_config", dict)
    code = old.main(["jev", "ask", "R1", "--pack", "pack.json"])
    out = capsys.readouterr().out
    assert code == 1 and "legacy_read_only" in out and calls == []


@pytest.mark.parametrize("argv", [["invented"], ["calibrate"], ["schema", "--unknown"]])
def test_unsupported_legacy_command_is_a_structured_refusal(argv, capsys):
    code = cli.main(["legacy", *argv])
    output = capsys.readouterr()
    assert code == 1 and json.loads(output.out)["status"] == "legacy_read_only"


@pytest.mark.parametrize("argv", [["--help"], ["jev", "ask", "--help"], ["init", "--help"]])
def test_legacy_help_keeps_old_parser_without_side_effects(argv, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(old, "cmd_jev_ask", lambda args: pytest.fail("help dispatched a handler"))
    before = snapshot(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["legacy", *argv])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out and snapshot(tmp_path) == before


@pytest.mark.parametrize("argv", [["schema"], ["status", "historical"], ["pending", "historical"], ["ledger", "historical", "events"]])
def test_audited_queries_preserve_historical_bytes(argv, tmp_path, monkeypatch, capsys):
    from sextant.state import State
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "decisions" / "historical"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(State(slug="historical", created="constructed snapshot").model_dump_json())
    (directory / "events.jsonl").write_text('{"seq":1,"type":"historical-test"}\n')
    config = tmp_path / "config.toml"
    config.write_text('[paths]\ndecisions_dir="decisions"\n')
    monkeypatch.setenv("SEXTANT_CONFIG", str(config))
    before = snapshot(tmp_path)
    code = cli.main(["legacy", *argv])
    output = capsys.readouterr()
    assert code == 0 and output.out and not output.err and snapshot(tmp_path) == before
