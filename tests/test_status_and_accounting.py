"""Counterexamples for status reporting, cost accounting, and concurrent writers.

These tests name the current entry points. The same calls keep a missing live probe, an
unfitted threshold, a filename, and a second writer from reading as success.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

from sextant.config import DEFAULTS, load_config
from sextant.doctor import Rung, summary
from sextant.handshake import ProviderError, create_request, fulfil
from sextant.ledger import Ledger
from sextant.meta import RM1Row
from sextant.rolecards import RoleCard
from sextant.selftest import base_graph, fx_rolecard, node
from sextant.state import add_pending, init_state, load_state, save_state

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]


def _rm1_row(**over: Any) -> RM1Row:
    data: dict[str, Any] = {
        "card_id": "R3",
        "key": "relevance",
        "scoping_defined": 0.6,
        "asks_computation": 0.4,
        "answerable_from_fields": 0.9,
        "criteria_consistent": 0.9,
        "has_scoping_word": True,
        "contested": False,
    }
    data.update(over)
    return RM1Row(**data)


def test_default_generative_fallback_is_disabled() -> None:
    """config.toml and the defaults still list paid API fallbacks."""
    assert DEFAULTS["providers"]["auto_fulfil"] == []
    loaded = load_config(ROOT / "config.toml")
    assert loaded["providers"]["auto_fulfil"] == []


def test_provider_failure_does_not_hop_to_another_paid_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider failure names other paid APIs instead of stopping."""
    calls: list[str] = []

    def boom(name: str, config: dict[str, Any]) -> Any:
        del config
        calls.append(name)
        raise RuntimeError("provider down")

    monkeypatch.setattr("sextant.handshake.get_provider", boom)
    init_state(tmp_path, "demo", created="2026-09-22T00:00:00Z")
    schema = {"type": "object", "additionalProperties": False, "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
    req = create_request(
        tmp_path,
        role="generator",
        purpose="x",
        prompt="p",
        answer_schema=schema,
        provider="deepseek",
    )
    config = {"providers": {"auto_fulfil": ["gemini", "openai"]}}
    with pytest.raises(ProviderError) as caught:
        fulfil(tmp_path, req.id, config)
    assert calls == []  # V1 refuses legacy paid providers before construction or dispatch.
    text = str(caught.value)
    assert "gemini" not in text and "openai" not in text
    assert "no fallback was attempted" in text


def test_a_filename_is_not_a_production_cost_record(tmp_path: Path) -> None:
    """Compute-ledger accepts a row that is only the ledger's own filename."""
    from sextant.ledger import IncompleteCostRecord

    ledger = Ledger(tmp_path)
    with pytest.raises(IncompleteCostRecord):
        ledger.append("compute-ledger", {"filename": "compute-ledger.jsonl"})


def test_a_cost_row_names_the_call_and_its_tokens(tmp_path: Path) -> None:
    """A complete offline cost row is the call and its token use, not a path."""
    ledger = Ledger(tmp_path)
    row = ledger.record_cost(
        provider="scripted",
        model="jev-1.13.0",
        request_id="rq_cost",
        input_tokens=12,
        output_tokens=3,
        purpose="offline-test",
    )
    assert row["request_id"] == "rq_cost"
    assert row["input_tokens"] == 12 and row["output_tokens"] == 3
    assert row["priced"] is False


def test_error_budget_target_does_not_constrain_a_run() -> None:
    """The recorded target is not an executing total error budget."""
    from sextant.graph import check_budgets
    from sextant.schema import TIER_BUDGETS, DecisionGraph, Tier, error_budget_constrains_run

    assert error_budget_constrains_run(TIER_BUDGETS[Tier.T1]) is False
    question = node("question", "Should I take the offer?")
    graph = DecisionGraph.model_validate(base_graph([question], []))
    graph.budgets = TIER_BUDGETS[Tier.T2].model_copy(update={"error_budget_target": 0.0})
    findings = check_budgets(graph)
    assert not any("error_budget" in f"{item.check} {item.message}" for item in findings)


def test_flags_use_the_fitted_threshold_when_one_exists() -> None:
    """A score between 0.5 and the fitted threshold is not judged at a fixed 0.5.

    Flags follow the deployed decision rule: a value inside the band around the
    threshold is neither a flag nor a pass; it is undecided, which doctor never reports as PASS.
    """
    row = _rm1_row()
    assert row.flags() == []
    assert row.flags({"scoping_defined": 0.8}) == [] and row.undecided({"scoping_defined": 0.8}) == ["scoping_defined"]
    scoped = _rm1_row(scoping_defined=0.1).flags({"scoping_defined": 0.8})
    assert any(item.startswith("scoping_defined") for item in scoped)
    computed = _rm1_row(asks_computation=0.9).flags({"asks_computation": 0.8})
    assert any(item.startswith("asks_computation") for item in computed)
    assert _rm1_row(asks_computation=0.9, contested=True).flags({"asks_computation": 0.8}) == []


def test_a_thresholds_dict_is_not_evidence_of_a_fit() -> None:
    """Fx cards carry a thresholds dict whose certificate was never validated."""
    from sextant.doctor import fitted_flag_thresholds

    card = RoleCard.model_validate(fx_rolecard(False))
    assert card.thresholds
    assert card.thresholds["sufficient"].value is not None
    assert card.thresholds["sufficient"].status != "validated"
    assert fitted_flag_thresholds(card) == {}


def test_rm1_without_a_fitted_threshold_is_not_a_formal_pass() -> None:
    """No fitted threshold, with or without flags, still came out PASS."""
    from sextant.doctor import judge_rm1

    assert judge_rm1(executed=True, fitted=False, flagged=False) == "OBSERVED"
    assert judge_rm1(executed=True, fitted=False, flagged=True) == "OBSERVED"
    assert judge_rm1(executed=False, fitted=False, flagged=False) == "NOT_RUN"
    assert judge_rm1(executed=True, fitted=True, flagged=False) == "PASS"
    assert judge_rm1(executed=True, fitted=True, flagged=True) == "FAIL"


def test_a_missing_live_probe_is_not_run() -> None:
    """No live key was recorded as SKIP, which is not a run and not a pass."""
    from sextant.doctor import live_probe_rung

    rung = live_probe_rung("canary-drift", executed=False, reason="no live Jev key")
    assert rung.status == "NOT_RUN"
    assert rung.lane == "live_health"
    assert rung.status not in {"PASS", "SKIP"}


def test_observed_informational_and_not_applicable_are_not_pass() -> None:
    """Those three words must not collapse into PASS or into each other."""
    rungs = [
        Rung("selftest", "PASS", "ok", "offline_integrity"),
        Rung("RM1-cards", "OBSERVED", "scores without a fitted threshold", "live_health"),
        Rung("rm1-note", "INFORMATIONAL", "a measurement is not a release decision", "live_health"),
        Rung("goldrun-replay", "NOT-APPLICABLE", "not built in this slice", "product_acceptance"),
        Rung("canary-drift", "NOT_RUN", "no live Jev key", "live_health"),
    ]
    counts = summary(rungs)
    assert counts["pass"] == 1
    assert counts["observed"] == 1
    assert counts["informational"] == 1
    assert counts["not_applicable"] == 1
    assert counts["not_run"] == 1
    assert summary([Rung("only", "OBSERVED", "m", "live_health")])["pass"] == 0
    assert summary([Rung("only", "INFORMATIONAL", "m", "live_health")])["pass"] == 0
    assert summary([Rung("only", "NOT-APPLICABLE", "m", "product_acceptance")])["pass"] == 0
    assert summary([Rung("only", "NOT_RUN", "m", "live_health")])["pass"] == 0


def test_release_eligibility_is_an_explicit_checklist() -> None:
    """No FAIL rung is not the same as eligible to release."""
    from sextant.doctor import release_eligibility

    rungs = [
        Rung("selftest", "PASS", "ok", "offline_integrity"),
        Rung("pytest", "PASS", "ok", "offline_integrity"),
        Rung("ruff", "PASS", "ok", "offline_integrity"),
        Rung("mypy", "PASS", "ok", "offline_integrity"),
        Rung("canary-drift", "NOT_RUN", "no key", "live_health"),
        Rung("RM1-cards", "OBSERVED", "no fitted threshold", "live_health"),
        Rung("goldrun-replay", "NOT-APPLICABLE", "not built", "product_acceptance"),
        Rung("render-checks", "NOT-APPLICABLE", "not built", "product_acceptance"),
        Rung("verifier-fixture", "NOT-APPLICABLE", "not built", "product_acceptance"),
        Rung("registry-sweeps", "NOT-APPLICABLE", "not built", "product_acceptance"),
    ]
    decision = release_eligibility(rungs)
    assert decision["eligible"] is False
    assert decision["rule"] == "checklist"
    assert any(item["status"] == "NOT_RUN" and item["met"] is False for item in decision["checklist"])
    assert all(item["status"] != "FAIL" for item in decision["checklist"])


def test_role_cards_are_not_loaded_only_from_the_home_directory() -> None:
    """Defaults, config, and the calibration and lint tests load role cards from the package."""
    assert DEFAULTS["paths"]["rolecards_dir"] == "package"
    text = (ROOT / "config.toml").read_text()
    assert ".claude/skills" not in text
    for name in ("tests/test_calibrate.py", "tests/test_lint_client.py"):
        body = (ROOT / name).read_text()
        assert "Path.home()" not in body
        assert "pytest.skip" not in body
    loaded = load_config(ROOT / "config.toml")
    cards = Path(loaded["paths"]["rolecards_dir"])
    assert cards.is_dir()
    assert ".claude" not in str(cards)
    assert (cards / "R3-edge.yaml").is_file()


def test_package_says_what_is_loadable_and_what_is_partial() -> None:
    """Candidate metadata must not promote installation to release acceptance."""
    from sextant.package_status import describe_package

    described = describe_package()
    assert described["compiled_product"] is False
    assert described["offline_is_live_verification"] is False
    assert described["loadable"]
    assert "role cards are question texts, not a calibrated product" in described["partial"]
    pyproject = (ROOT / "pyproject.toml").read_text()
    # Candidate metadata carries the actual acceptance boundary. The canonical
    # README may reorganize headings without changing install/release semantics.
    assert "formal comparative capability confirmation and independent candidate acceptance" in described["partial"]
    assert "candidate" in pyproject.split("description", 1)[1][:240]
    assert described["release_accepted"] is False


def test_a_stale_state_save_cannot_drop_a_pending_update(tmp_path: Path) -> None:
    """The second writer saves an older snapshot and the first pending id disappears."""
    init_state(tmp_path, "demo", created="2026-09-22T00:00:00Z")
    first = load_state(tmp_path)
    second = load_state(tmp_path)
    add_pending(first, "rq_keep")
    save_state(first, tmp_path)
    second.pending_requests = ["rq_other"]
    with pytest.raises(Exception, match="stale"):
        save_state(second, tmp_path)
    assert load_state(tmp_path).pending_requests == ["rq_keep"]


def test_the_single_writer_lock_is_checkable_and_blocks_a_second_append(tmp_path: Path) -> None:
    """Two appends can both read seq 0 and both write seq 1. Transactions stay deferred."""
    from sextant.ledger import WriterBusy, concurrency_status, exclusive, writer_lock_held

    status = concurrency_status()
    assert status["mode"] == "single_writer_lock"
    assert status["transactions"] == "deferred"
    assert status["deferred_to"] == "P1"
    ledger = Ledger(tmp_path)
    started = threading.Event()
    finished = threading.Event()

    def other() -> None:
        started.set()
        try:
            ledger.append("events", {"kind": "second"})
        except WriterBusy:
            finished.set()
            return
        finished.set()

    with exclusive(tmp_path):
        assert writer_lock_held(tmp_path) is True
        thread = threading.Thread(target=other)
        thread.start()
        assert started.wait(2)
        assert not finished.wait(0.2)
        assert ledger.read("events") == []
    thread.join(2)
    assert finished.is_set()
    assert writer_lock_held(tmp_path) is False
    assert [row["seq"] for row in ledger.read("events")] == [1]
