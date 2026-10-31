"""Regression tests for graph checks, the ledger, and request handling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from sextant.graph import check_reified, save_graph
from sextant.handshake import BlockerError, HandshakeError, create_request, ingest, pending
from sextant.ledger import Ledger
from sextant.providers.base import Answer, check_request_id, pending_requests, write_answer
from sextant.schema import DecisionGraph, Node
from sextant.selftest import base_graph, edge, node
from sextant.state import init_state, load_state

SCHEMA = {"type": "object", "required": ["ok"], "additionalProperties": False, "properties": {"ok": {"type": "boolean"}}}


@pytest.mark.unit
def test_conflict_with_identical_members_rejected() -> None:
    f = node("factor", "salary covers expenses")
    with pytest.raises(ValidationError, match="distinct"):
        Node.model_validate(node("conflict", "self conflict", members=[f["id"], f["id"]]))


@pytest.mark.unit
def test_conflict_reified_check_needs_two_distinct_edges() -> None:
    a = node("factor", "rent level")
    b = node("factor", "savings rate")
    c = node("conflict", "rent vs savings", members=[a["id"], b["id"]])
    g = DecisionGraph.model_validate(base_graph([a, b, c], [edge("conflicts", c, a)]))
    assert check_reified(g)
    g2 = DecisionGraph.model_validate(base_graph([a, b, c], [edge("conflicts", c, a), edge("conflicts", c, b)]))
    assert not check_reified(g2)


@pytest.mark.unit
def test_value_node_rejects_sufficiency_and_state_vector() -> None:
    from sextant.selftest import sample_estimate

    with pytest.raises(ValidationError, match="value node never"):
        Node.model_validate(node("value", "autonomy", sufficiency=sample_estimate()))
    with pytest.raises(ValidationError, match="value node never"):
        Node.model_validate(node("value", "autonomy", state_vector={"verifiability": sample_estimate()}))


@pytest.mark.unit
def test_graph_version_must_be_positive() -> None:
    data = base_graph([], [])
    data["version"] = 0
    with pytest.raises(ValidationError):
        DecisionGraph.model_validate(data)


@pytest.mark.unit
def test_objective_view_rerendered_every_save(tmp_path: Path) -> None:
    q = node("question", "Should I take the offer?")
    c1 = node("criterion", "crit A")
    c2 = node("criterion", "crit B")
    r = node("factor", "other components not listed", residual=True)
    g1 = DecisionGraph.model_validate(base_graph([q, c1, r], [edge("decomposes_into", q, c1), edge("decomposes_into", q, r)]))
    save_graph(g1, tmp_path)
    g2 = DecisionGraph.model_validate({**base_graph([q, c2, r], [edge("decomposes_into", q, c2), edge("decomposes_into", q, r)]), "version": 2, "parent_version": 1})
    save_graph(g2, tmp_path)
    current = json.loads((tmp_path / "objective.json").read_text())
    assert [c["text_en"] for c in current["criteria"]] == ["crit B"]
    assert (tmp_path / "objective.v1.json").exists() and (tmp_path / "objective.v2.json").exists()


@pytest.mark.unit
def test_ledger_owns_seq_and_validates_retract(tmp_path: Path) -> None:
    led = Ledger(tmp_path)
    led.append("events", {"kind": "a"})
    row = led.append("events", {"kind": "b", "seq": 1, "at": "1999-01-01T00:00:00Z"})
    assert row["seq"] == 2 and row["at"] != "1999-01-01T00:00:00Z"
    with pytest.raises(ValueError, match="no such row"):
        led.retract("events", 999, "nope")


@pytest.mark.unit
def test_request_id_rejects_trailing_newline() -> None:
    with pytest.raises(ValueError):
        check_request_id("rq_abc\n")
    assert check_request_id("rq_abc") == "rq_abc"


@pytest.mark.unit
def test_invalid_answer_schema_refused_at_creation(tmp_path: Path) -> None:
    with pytest.raises(HandshakeError, match="invalid answer schema"):
        create_request(tmp_path, role="generator", purpose="x", prompt="p", answer_schema={"type": "bogus"}, provider="claude-code")


@pytest.mark.fixture
def test_rejection_reissues_then_blocks_and_pending_agrees(tmp_path: Path) -> None:
    init_state(tmp_path, "demo")
    req = create_request(tmp_path, role="generator", purpose="x", prompt="p", answer_schema=SCHEMA, provider="claude-code")
    # first rejection: file moved aside, request re-issued as attempt 2
    write_answer(tmp_path, Answer(request_id=req.id, answer={"ok": "no"}, provider="claude-code", family="anthropic", isolation="same-session"))
    with pytest.raises(HandshakeError, match="Re-issued as"):
        ingest(tmp_path, req.id)
    st = load_state(tmp_path)
    assert len(st.pending_requests) == 1 and st.pending_requests[0] != req.id
    new_id = st.pending_requests[0]
    assert pending(tmp_path) == pending_requests(tmp_path) == [new_id]
    assert (tmp_path / "requests" / f"{req.id}.answer.rejected1.json").exists()
    kinds = [r["kind"] for r in Ledger(tmp_path).read("events")]
    assert "reissue" in kinds
    # second rejection: blocker recorded, pending cleared
    write_answer(tmp_path, Answer(request_id=new_id, answer={"ok": "still no"}, provider="claude-code", family="anthropic", isolation="same-session"))
    with pytest.raises(BlockerError):
        ingest(tmp_path, new_id)
    st = load_state(tmp_path)
    assert st.blockers and st.pending_requests == []
    assert "blocker" in [r["kind"] for r in Ledger(tmp_path).read("events")]


@pytest.mark.fixture
def test_malformed_envelope_is_a_recorded_rejection(tmp_path: Path) -> None:
    init_state(tmp_path, "demo")
    req = create_request(tmp_path, role="generator", purpose="x", prompt="p", answer_schema=SCHEMA, provider="claude-code")
    (tmp_path / "requests" / f"{req.id}.answer.json").write_text(json.dumps({"request_id": req.id, "answer": {"ok": True}}))
    with pytest.raises(HandshakeError, match="envelope"):
        ingest(tmp_path, req.id)
    rows = Ledger(tmp_path).read("llm-calls")
    assert rows[-1]["status"] == "rejected" and rows[-1]["provider"] == "unknown"


@pytest.mark.unit
def test_historical_selfcheck_and_selftest_parser(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from sextant import __main__ as cli

    # Preserve historical parser/library checks in isolated fixtures. The V1
    # public entry rejects these mutators; test_v1_legacy_boundary covers that.
    # Data-writing commands require a config file to exist; point at an empty
    # one outside the decisions directory so the working directory does not matter.

    config_file = tmp_path_factory.mktemp("config") / "config.toml"
    config_file.write_text("")
    monkeypatch.setenv("SEXTANT_CONFIG", str(config_file))
    monkeypatch.setattr(cli, "load_config", lambda: {"paths": {"decisions_dir": str(tmp_path), "rolecards_dir": str(tmp_path / "cards"), "calibration_dir": str(tmp_path / "calibration-v2")}, "providers": {}})
    assert cli.legacy_main(["selftest", "--disable", "no-such-check"]) == 2
    assert cli.legacy_main(["init", "demo-cli"]) == 0
    assert cli.legacy_main(["selfcheck", "demo-cli"]) == 1  # no graph yet
    q = node("question", "Should I take the offer?")
    k = node("factor", "salary covers expenses")
    r = node("factor", "other components not listed", residual=True)
    g = DecisionGraph.model_validate(base_graph([q, k, r], [edge("decomposes_into", q, k), edge("decomposes_into", q, r)]))
    save_graph(g, tmp_path / "demo-cli")
    assert cli.legacy_main(["selfcheck", "demo-cli"]) == 0
    out = capsys.readouterr().out
    assert "selfcheck graph.v1: PASS" in out
    assert cli.legacy_main(["rolecards", "check"]) == 1
    with pytest.raises(SystemExit):
        cli.legacy_main(["init", "../evil"])
