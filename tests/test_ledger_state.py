from __future__ import annotations

from pathlib import Path

import pytest

from sextant.ledger import Ledger
from sextant.schema import GateAuthority
from sextant.state import (
    GateError,
    assert_gate,
    init_state,
    load_state,
    save_state,
    stamp,
    status_text,
)

pytestmark = pytest.mark.unit


def test_ledger_append_seq_and_retract(tmp_path: Path) -> None:
    led = Ledger(tmp_path)
    r1 = led.append("events", {"kind": "a"})
    r2 = led.append("events", {"kind": "b"})
    assert (r1["seq"], r2["seq"]) == (1, 2)
    led.retract("events", 1, "typo")
    active = led.active("events")
    assert [r["kind"] for r in active] == ["b"]
    assert len(led.read("events")) == 3


def test_jev_call_index(tmp_path: Path) -> None:
    led = Ledger(tmp_path)
    row = led.record_jev_call(
        card_id="R2",
        protocol_version="sextant-jev-wire/2",
        question_keys=["sufficient"],
        question_bindings={"sufficient": {"template_hash": "qh_x", "rendered_hash": "rq_x"}},
        questions_digest="qs_x",
        model_id="jev-1.13.0", transport="typesafe",
        state_sha="s", request={}, response={}, latency_ms=300.0, usage={"input_tokens": 10},
        run_id="run_x", call_role="primary", run_protocol={"permutations": 4, "reruns": 3, "contested_margin": 0.1},
    )
    assert led.has_call(row["id"])
    assert led.call_index()[row["id"]]["card_id"] == "R2"
    assert not led.has_call("jc_nope")


def test_unknown_ledger_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        Ledger(tmp_path).read("not-a-ledger")


def test_state_gates(tmp_path: Path) -> None:
    st = init_state(tmp_path, "demo", created="2026-09-18T00:00:00Z")
    with pytest.raises(GateError):
        assert_gate(st, "generate")
    with pytest.raises(GateError, match="no stage check"):
        assert_gate(st, "frame")
    with pytest.raises(GateError, match="token"):
        stamp(st, "frame", "FAIL", "  ")
    with pytest.raises(GateError, match="unknown gate"):
        stamp(st, "not-a-gate", "FAIL", "x")
    with pytest.raises(GateError, match="checker PASS"):
        stamp(st, "frame", "PASS", "sh:exit0")
    recorded = stamp(st, "generate", "FAIL", "sh:exit1", note="operator saw the draft")
    assert recorded.authority == GateAuthority.user_judgment
    save_state(st, tmp_path)
    again = load_state(tmp_path)
    with pytest.raises(GateError, match="user judgment"):
        assert_gate(again, "generate")
    assert "user_judgment" in status_text(again)


def test_init_twice_refused(tmp_path: Path) -> None:
    init_state(tmp_path, "demo")
    with pytest.raises(FileExistsError):
        init_state(tmp_path, "demo")
