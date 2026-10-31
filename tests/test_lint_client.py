"""Tests for the question linter and the Jev client (offline, with a scripted caller)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from sextant.config import load_config, packaged_rolecards_dir
from sextant.jev_client import JevClient, JevClientError, RungError
from sextant.jev_transport import JevResult
from sextant.ledger import Ledger
from sextant.lint import lint_card, lint_dir
from sextant.rolecards import RoleCard, load_all, seal, set_rung
from sextant.schema import Rung
from sextant.selftest import fx_rolecard

CARDS_DIR = packaged_rolecards_dir()

pytestmark = pytest.mark.unit


def _card(**over: Any) -> RoleCard:
    data = fx_rolecard(False)
    data.update(over)
    return RoleCard.model_validate(data)


def test_linter_passes_the_shipped_cards() -> None:
    results = lint_dir(CARDS_DIR)
    assert len(results) >= 10
    assert not any(results.values()), {k: [f.message for f in v] for k, v in results.items() if v}


def test_linter_catches_planted_wording_faults() -> None:
    data = fx_rolecard(False)
    q = data["questions"]["sufficient"]
    q["text"] = "Is it not unclear how many items are enough"
    card = RoleCard.model_validate(data)
    codes = {f.code for f in lint_card(card)}
    assert {"Q001", "Q002", "Q004"} <= codes


def test_linter_choice_and_score_rules() -> None:
    data = fx_rolecard(False)
    data["questions"]["kind"] = {"primitive": "choice", "text": "Which kind is it?", "options": {"a": "x", "b": None}, "escape": "zzz"}
    data["questions"]["level"] = {"primitive": "score", "text": "How bad is it?", "levels": ["low", "a bit worse than low"]}
    card = RoleCard.model_validate(data)
    codes = [f.code for f in lint_card(card)]
    assert codes.count("Q005") >= 2 and "Q006" in codes


def _scripted_caller(answers_by_key: dict[str, dict[str, Any]], calls: list[dict[str, Any]]) -> Any:
    def caller(state: Any, questions: dict[str, dict[str, Any]], model: str) -> JevResult:
        calls.append(questions)
        out: dict[str, dict[str, Any]] = {}
        for k, q in questions.items():
            if q["type"] == "choice":
                opts = list(q["criteria"].keys())
                probs = {o: 0.0 for o in opts}
                probs[opts[0]] = 0.55
                probs[opts[1]] = 0.45
                out[k] = {"type": "choice", "choice": opts[0], "probabilities": probs, "confidence": 0.4}
            else:
                out[k] = dict(answers_by_key.get(k, {"type": "noul", "noul": 0.9}))
        return JevResult(model_id=model, answers=out, usage={"input_tokens": 50, "output_tokens": 5}, latency_ms=1.0, request_id="t")

    return caller


@pytest.fixture
def cards() -> dict[str, RoleCard]:
    return load_all(CARDS_DIR)


def test_client_refuses_bad_packs_and_unknown_cards(cards: dict[str, RoleCard]) -> None:
    client = JevClient(load_config(), cards, caller=_scripted_caller({}, []), seed=1)
    with pytest.raises(JevClientError, match="unknown card"):
        client.ask("R99", {})
    with pytest.raises(JevClientError, match="missing fields"):
        client.ask("R3", {"parent_question": "p"})
    with pytest.raises(JevClientError, match="outside its state pack"):
        client.ask("R6", {"decision_question": "d", "options": "a/b", "node_question": "n", "known_about_node": "k", "extra": 1})
    with pytest.raises(JevClientError, match="over the state budget"):
        client.ask("R6", {"decision_question": "d" * 200_000, "options": "a/b", "node_question": "n", "known_about_node": "k"})


def test_client_refuses_drifted_and_degraded_cards(tmp_path: Path, cards: dict[str, RoleCard]) -> None:
    p = tmp_path / "R2.yaml"
    p.write_text(yaml.safe_dump(fx_rolecard(False), sort_keys=False))
    seal(p)
    data = yaml.safe_load(p.read_text())
    data["questions"]["sufficient"]["text"] += " Really?"
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    drifted_cards = load_all(tmp_path)
    client = JevClient(load_config(), drifted_cards, caller=_scripted_caller({}, []), seed=1)
    with pytest.raises(Exception, match="changed since its hash"):
        client.ask("R2", {"question": "q", "answer_space": "yes/no", "context_items": ["a"]})
    seal(p)
    set_rung(p, Rung.llm_isolated, "test", "2026-09-18")
    degraded = load_all(tmp_path)
    client = JevClient(load_config(), degraded, caller=_scripted_caller({}, []), seed=1)
    with pytest.raises(RungError):
        client.ask("R2", {"question": "q", "answer_space": "yes/no", "context_items": ["a"]})
    res = client.ask("R2", {"question": "q", "answer_space": "yes/no", "context_items": ["a", "b"]}, allow_degraded=True)
    assert res.estimates["sufficient"].estimate.calibration_status.state == "degraded"
    assert set(res.estimates) == {"sufficient"}  # the fixture card carries only that one question
    # the installed R2 card expands its item question once per context item
    live = JevClient(load_config(), cards, caller=_scripted_caller({}, []), seed=1)
    res2 = live.ask("R2", {"question": "q", "answer_space": "yes/no", "context_items": ["a", "b"]})
    assert set(res2.estimates) >= {"sufficient", "sufficient_neg", "item_relevance[1]", "item_relevance[2]"}


def test_client_permutations_boundary_twins_and_ledger(tmp_path: Path, cards: dict[str, RoleCard]) -> None:
    calls: list[dict[str, Any]] = []
    answers = {"relevance": {"type": "noul", "noul": 0.52}, "relevance_neg": {"type": "noul", "noul": 0.40}, "more_concrete": {"type": "noul", "noul": 0.95}, "skipped_level": {"type": "noul", "noul": 0.1}}
    led = Ledger(tmp_path)
    client = JevClient(load_config(), cards, ledger=led, caller=_scripted_caller(answers, calls), seed=3)
    pack = {"parent_question": "Should I accept this job offer?", "child_question": "Will the base salary cover my monthly expenses?", "parent_kind": "question", "child_kind": "factor"}
    res = client.ask("R3", pack)
    # base + 3 permutations (choice) + 3 re-runs (every yes/no question is re-run by protocol, and the
    # contested edge_type choice rides along for stability); each of those re-test calls reports 50
    # input tokens, and the total must include them
    assert res.requests == 1 + 3 + 3
    assert res.usage_input_tokens == 50 * res.requests
    assert len(res.call_ids) == 7 and all(led.has_call(c) for c in res.call_ids)
    rel = res.estimates["relevance"].estimate
    assert rel.stability.n_reruns == 3 and rel.stability.negation_gap == pytest.approx(0.08, abs=1e-6)
    et = res.estimates["edge_type"]
    assert et.estimate.stability.n_permutations == 3
    assert et.contested  # the scripted caller always favors whatever option comes first
    assert res.passes("relevance") is None  # no fitted threshold yet
    rows = led.read("jev-calls")
    assert rows[0]["card_id"] == "R3" and rows[0]["model_id"] == "jev-1.13.0" and rows[0]["variant"] == "base"
    assert {r["variant"] for r in rows} >= {"base", "perm1", "rerun1"}
    assert {r["call_role"] for r in rows} == {"primary", "permutation", "rerun"}
    assert len({r["run_id"] for r in rows}) == 1


def test_client_r7_leaf_takes_node_answer_space(cards: dict[str, RoleCard]) -> None:
    calls: list[dict[str, Any]] = []
    client = JevClient(load_config(), cards, caller=_scripted_caller({}, calls), seed=1)
    pack = {"question": "Which department?", "definition": "d", "answer_space": {"primitive": "choice"}, "evidence_items": ["1. billing complaint"]}
    res = client.ask("R7", pack, leaf_space={"primitive": "choice", "options": {"billing": None, "technical": None, "none": "none fits"}})
    assert calls[0]["leaf"]["type"] == "choice" and set(calls[0]["leaf"]["criteria"]) == {"billing", "technical", "none"}
    assert res.estimates["leaf"].estimate.calibration_status.kind == "outcome"
    assert res.estimates["leaf"].estimate.sufficiency_p is not None
