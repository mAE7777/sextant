"""Contract tests: the wire prompt that is actually sent, and per-question provenance.

Each test is a minimal counterexample turned into a regression: the card's shared
instructions reach the wire, each estimate cites only its own calls, and request accounting
and the measurement-cache identity cover everything that was sent.

The cards are built inline rather than loaded from the operator's home directory, so these
tests cannot silently skip.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sextant.config import load_config
from sextant.graph import check_all, load_graph, save_graph
from sextant.jev_client import JevClient, JevClientError
from sextant.jev_transport import JevResult
from sextant.ledger import Ledger
from sextant.rolecards import RoleCard, compute_hash
from sextant.schema import DecisionGraph, Estimate
from sextant.selftest import base_graph
from sextant.selftest import edge as fx_edge
from sextant.selftest import node as fx_node

pytestmark = pytest.mark.unit

# A string that exists nowhere except the card's shared instructions. The defect this file
# guards against is that it never reached the transport.
SENTINEL = "ZZ-CARD-INSTRUCTIONS-SENTINEL-7f3a91"

PACK = {
    "parent_question": "Should I take the offer?",
    "child_question": "Will the base salary cover my monthly expenses?",
}


def sealed(data: dict[str, Any]) -> RoleCard:
    """A card whose question hashes are already recorded, as `rolecards seal` would leave it."""
    draft = RoleCard.model_validate(data)
    data = copy.deepcopy(data)
    data["question_hashes"] = {k: compute_hash(draft, k) for k in draft.questions}
    return RoleCard.model_validate(data)


def edge_card(instructions: str = f"Judge only what the state says. {SENTINEL}") -> RoleCard:
    return sealed(
        {
            "id": "R3",
            "purpose": "edge verification",
            "primitive": "noul",
            "state_pack": ["parent_question", "child_question"],
            "instructions": instructions,
            "questions": {
                "relevance": {
                    "primitive": "noul",
                    "text": "Does answering {child_question} contribute to {parent_question}?",
                    "criteria_true": "It contributes directly.",
                    "criteria_false": "It does not contribute at all.",
                },
                "more_concrete": {
                    "primitive": "noul",
                    "text": "Is {child_question} more concrete than {parent_question}?",
                },
            },
        }
    )


class Recorder:
    """A scripted transport that keeps the exact arguments each call received."""

    def __init__(self, series: dict[str, list[float]]) -> None:
        self.series = {k: list(v) for k, v in series.items()}
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, state: Any, questions: dict[str, dict[str, Any]], model: str
    ) -> JevResult:
        self.calls.append(
            {"state": copy.deepcopy(state), "questions": copy.deepcopy(questions), "model": model}
        )
        answers: dict[str, dict[str, Any]] = {}
        for key in questions:
            pending = self.series.get(key)
            value = pending.pop(0) if pending else 0.5
            answers[key] = {"type": "noul", "noul": value, "confidence": 0.7}
        return JevResult(
            model_id=model,
            answers=answers,
            usage={"input_tokens": 40, "output_tokens": 4},
            latency_ms=1.0,
            request_id="scripted",
        )

    @property
    def sent_instructions(self) -> list[str]:
        return [q["instructions"] for call in self.calls for q in call["questions"].values()]


# Every yes/no question is asked once and re-run three times (wire protocol 2), so each reported
# value is the mean of four answers. `relevance` gets four differing answers; `more_concrete`
# four equal ones.
SERIES = {"relevance": [0.52, 0.56, 0.60, 0.64], "more_concrete": [0.95, 0.95, 0.95, 0.95]}
RELEVANCE_MEAN = pytest.approx((0.52 + 0.56 + 0.60 + 0.64) / 4)


def client_for(
    card: RoleCard, recorder: Recorder, ledger: Ledger | None = None, **jev: Any
) -> JevClient:
    config = copy.deepcopy(load_config())
    config["jev"].update(jev)
    return JevClient(config, {card.id: card}, ledger=ledger, caller=recorder, seed=11)


def asked(card: RoleCard | None = None, ledger: Ledger | None = None) -> tuple[Any, Recorder]:
    card = card or edge_card()
    recorder = Recorder(SERIES)
    result = client_for(card, recorder, ledger).ask(card.id, PACK)
    return result, recorder


# ---------------------------------------------------------------- the wire prompt


def test_card_instructions_reach_every_question_on_the_wire() -> None:
    """The card's shared instructions were hashed and sealed but never sent."""
    _, recorder = asked()
    assert recorder.sent_instructions
    missing = [text for text in recorder.sent_instructions if SENTINEL not in text]
    assert not missing, "card instructions never reached the transport"


def test_wire_payload_carries_the_question_criteria_and_language() -> None:
    _, recorder = asked()
    first = recorder.calls[0]["questions"]
    assert first["relevance"]["criteria"]["true"] == "It contributes directly."
    assert "Will the base salary cover my monthly expenses?" in first["relevance"]["instructions"]
    language = load_config()["jev"]["canonical_language"]
    assert all(f"language: {language}" in text for text in recorder.sent_instructions)


def test_wire_state_is_exactly_the_declared_state_pack() -> None:
    _, recorder = asked()
    assert recorder.calls[0]["state"] == PACK


def test_changing_shared_instructions_changes_the_bound_hashes() -> None:
    """A judgment recorded under different shared instructions must not look identical."""
    plain, _ = asked(edge_card())
    reworded, _ = asked(edge_card(instructions=f"Answer literally. {SENTINEL} Extra clause."))
    a = plain.estimates["relevance"].estimate
    b = reworded.estimates["relevance"].estimate
    assert a.question_hash != b.question_hash
    assert a.rendered_hash != b.rendered_hash


def test_protocol_version_is_recorded_on_the_judgment_and_the_call(tmp_path: Path) -> None:
    from sextant.wire import WIRE_PROTOCOL_VERSION

    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    estimate = result.estimates["relevance"].estimate
    assert estimate.protocol_version == WIRE_PROTOCOL_VERSION
    rows = ledger.read("jev-calls")
    assert rows and all(r["protocol_version"] == WIRE_PROTOCOL_VERSION for r in rows)


# ---------------------------------------------------------------- per-question provenance


def test_each_judgment_binds_only_its_own_contributing_calls(tmp_path: Path) -> None:
    """Every estimate used to receive the whole CardResult's call ids.

    A paraphrase asked for `relevance` alone is a call about that question only; it must never
    be attributed to `more_concrete`, which shared the base call and the re-runs.
    """
    ledger = Ledger(tmp_path)
    card = edge_card()
    recorder = Recorder({**SERIES, "relevance": [*SERIES["relevance"], 0.5]})
    result = client_for(card, recorder, ledger).ask(card.id, PACK, paraphrases={"relevance": ["Is it related?"]})
    assert len(result.call_ids) == 5  # base, three re-runs, one paraphrase of relevance

    relevance = result.estimates["relevance"].estimate
    concrete = result.estimates["more_concrete"].estimate
    assert len(relevance.call_ids) == 5
    paraphrase_call = result.call_ids[-1]
    assert paraphrase_call not in concrete.call_ids, "a call about another question was attributed"
    assert concrete.call_ids == result.call_ids[:4]
    assert relevance.p == RELEVANCE_MEAN, "a paraphrase measures stability; it never moves the value"
    assert concrete.p == pytest.approx(0.95)


def test_each_judgment_names_its_question_key_state_and_aggregation(tmp_path: Path) -> None:
    result, _ = asked(ledger=Ledger(tmp_path))
    relevance = result.estimates["relevance"].estimate
    concrete = result.estimates["more_concrete"].estimate
    assert relevance.question_key == "relevance"
    assert relevance.state_sha and relevance.state_sha == concrete.state_sha
    assert relevance.aggregation == "mean_of_runs"
    assert concrete.aggregation == "mean_of_runs"
    assert relevance.run_id and relevance.run_id == concrete.run_id
    variants = [c.variant for c in relevance.contributions]
    assert variants == ["base", "rerun1", "rerun2", "rerun3"]
    assert [c.value for c in relevance.contributions] == [0.52, 0.56, 0.60, 0.64]


# ---------------------------------------------------------------- the real audit path

def graph_for(
    estimate: Estimate,
    *,
    sufficiency: Estimate | None = None,
    state_vector: dict[str, Estimate] | None = None,
    verification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A minimal well-formed graph carrying the judgment at a chosen site."""
    dump = estimate.model_dump(mode="json", exclude_none=True)
    question = fx_node("question", "Should I take the offer?")
    if state_vector is not None:
        question["state_vector"] = {
            k: v.model_dump(mode="json", exclude_none=True) for k, v in state_vector.items()
        }
    factor = fx_node(
        "factor",
        "salary covers expenses",
        estimate=dump,
        resolution={
            "criteria": "the offer letter states a base salary >= 1.2x monthly expenses",
            "source": "offer letter",
        },
    )
    if sufficiency is not None:
        factor["sufficiency"] = sufficiency.model_dump(mode="json", exclude_none=True)
    residual = fx_node("factor", "other components not listed", residual=True)
    kid = fx_edge("decomposes_into", question, factor)
    if verification is not None:
        kid["verification"] = verification
    return base_graph(
        [question, factor, residual],
        [kid, fx_edge("decomposes_into", question, residual)],
    )


def test_real_client_saved_ledger_and_loaded_graph_agree(tmp_path: Path) -> None:
    """This whole path used to return `card/question mismatch`."""
    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    data = graph_for(result.estimates["relevance"].estimate)

    saved = save_graph(DecisionGraph.model_validate(data), tmp_path / "decision")
    reloaded = load_graph(saved)
    outcome = check_all(reloaded, ledger.call_index())
    assert outcome.ok, [f"{f.check}/{f.subject}: {f.message}" for f in outcome.findings]


def audit(data: dict[str, Any], ledger: Ledger) -> list[str]:
    """FAIL findings only: the note that card binding was not checked is informational here."""
    graph = DecisionGraph.model_validate(data)
    return [f.message for f in check_all(graph, ledger.call_index()).findings if f.severity == "FAIL"]


def test_a_value_that_contradicts_the_ledger_fails(tmp_path: Path) -> None:
    """An estimate of 0.99 over a ledger answer of 0.01 used to pass."""
    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    tampered = result.estimates["relevance"].estimate.model_copy(update={"p": 0.99})
    messages = audit(graph_for(tampered), ledger)
    assert any("replay" in m for m in messages), messages


def test_a_tampered_input_hash_fails(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    estimate = result.estimates["relevance"].estimate
    messages = audit(graph_for(estimate.model_copy(update={"state_sha": "0" * 16})), ledger)
    assert any("state" in m for m in messages), messages


def test_a_judgment_claiming_another_questions_key_fails(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    estimate = result.estimates["relevance"].estimate
    borrowed = estimate.model_copy(update={"question_key": "not_asked"})
    messages = audit(graph_for(borrowed), ledger)
    assert any("not_asked" in m for m in messages), messages


def test_a_judgment_cannot_claim_a_payload_its_own_call_did_not_send(tmp_path: Path) -> None:
    """The schema refuses this before any audit: the primary call defines the payload."""
    result, _ = asked(ledger=Ledger(tmp_path))
    estimate = result.estimates["relevance"].estimate
    data = graph_for(estimate.model_copy(update={"rendered_hash": "rq_0000"}))
    with pytest.raises(ValidationError, match="rendered question"):
        DecisionGraph.model_validate(data)


def test_a_rendered_question_that_was_never_sent_fails(tmp_path: Path) -> None:
    """An internally consistent forgery still has to match what the ledger recorded."""
    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    estimate = result.estimates["relevance"].estimate
    forged = estimate.model_copy(
        update={
            "rendered_hash": "rq_forged0000",
            "contributions": [
                c.model_copy(update={"rendered_hash": "rq_forged0000"}) if c.role == "primary" else c
                for c in estimate.contributions
            ],
        }
    )
    messages = audit(graph_for(forged), ledger)
    assert any("rendered question" in m for m in messages), messages


def ghost_of(estimate: Estimate) -> Estimate:
    """The same judgment, re-pointed at a call that never happened."""
    return estimate.model_copy(
        update={
            "contributions": [
                c.model_copy(update={"call_id": "jc_ghost"}) for c in estimate.contributions
            ]
        }
    )


def test_sufficiency_judgments_are_audited(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    estimate = result.estimates["relevance"].estimate
    other = result.estimates["more_concrete"].estimate
    data = graph_for(estimate, sufficiency=ghost_of(other))
    assert any("jc_ghost" in m for m in audit(data, ledger))
    assert audit(graph_for(estimate, sufficiency=other), ledger) == []


def test_state_vector_judgments_are_audited(tmp_path: Path) -> None:
    """`check_against_ledger` skipped node.state_vector entirely."""
    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    estimate = result.estimates["relevance"].estimate
    ghost = ghost_of(estimate)
    messages = audit(graph_for(estimate, state_vector={"verifiability": ghost}), ledger)
    assert any("jc_ghost" in m for m in messages), messages


def test_edge_verification_judgments_are_audited(tmp_path: Path) -> None:
    """`check_against_ledger` skipped edge.verification entirely."""
    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    estimate = result.estimates["relevance"].estimate
    ghost = ghost_of(estimate)
    data = graph_for(
        estimate, verification={"relevance": ghost.model_dump(mode="json", exclude_none=True)}
    )
    messages = audit(data, ledger)
    assert any("jc_ghost" in m for m in messages), messages


def test_sufficiency_carries_the_question_it_came_from(tmp_path: Path) -> None:
    card = sealed(
        {
            "id": "R7",
            "purpose": "leaf judgment",
            "primitive": "noul",
            "state_pack": ["question"],
            "instructions": f"Answer from the state alone. {SENTINEL}",
            "questions": {
                "leaf": {"primitive": "noul", "text": "Is {question} true?"},
                "enough": {
                    "primitive": "noul",
                    "text": "Is there enough information to answer {question}?",
                    "role": "sufficiency",
                },
            },
        }
    )
    recorder = Recorder({"leaf": [0.9] * 4, "enough": [0.8] * 4})
    ledger = Ledger(tmp_path)
    result = client_for(card, recorder, ledger).ask("R7", {"question": "Is the salary enough?"})
    leaf = result.estimates["leaf"].estimate
    assert leaf.sufficiency_p == pytest.approx(0.8)
    assert leaf.sufficiency_question_key == "enough"
    assert audit(graph_for(leaf), ledger) == []


# ------------------------------------------- stale evidence must not be accepted quietly


def test_a_legacy_ledger_row_without_protocol_identity_fails(tmp_path: Path) -> None:
    """Historical jev-calls rows predate the wire protocol; they cannot certify a judgment."""
    ledger = Ledger(tmp_path)
    result, _ = asked(ledger=ledger)
    estimate = result.estimates["more_concrete"].estimate
    index = ledger.call_index()
    legacy = {k: {i: j for i, j in v.items() if i != "protocol_version"} for k, v in index.items()}
    graph = DecisionGraph.model_validate(graph_for(estimate))
    messages = [f.message for f in check_all(graph, legacy).findings]
    assert any("protocol" in m for m in messages), messages


def test_a_legacy_estimate_without_provenance_is_refused() -> None:
    """An old Estimate payload must fail validation rather than load as unverifiable evidence."""
    legacy = {
        "primitive": "noul",
        "p": 0.8,
        "card_id": "R7",
        "question_hash": "qh_legacy00000001",
        "call_ids": ["jc_legacy0001"],
        "model_id": "jev-1.13.0",
        "transport": "typesafe",
        "source_tier": "jev-judged",
        "calibration_status": {"kind": "outcome", "state": "pending", "n": 0},
        "at": "2026-09-18T00:00:00Z",
    }
    with pytest.raises(ValidationError):
        Estimate.model_validate(legacy)


# ---------------------------------------------------------------- request accounting


def test_the_whole_request_is_held_to_the_request_token_budget() -> None:
    recorder = Recorder(SERIES)
    client = client_for(edge_card(), recorder, max_request_tokens=60)
    with pytest.raises(JevClientError, match="request budget"):
        client.ask("R3", PACK)
    assert recorder.calls == [], "the over-budget request was still sent"


def test_paraphrase_calls_are_counted_in_usage(tmp_path: Path) -> None:
    """Paraphrase runs spent input tokens that never reached usage_input_tokens."""
    card = edge_card()
    plain = Recorder(SERIES)
    client_for(card, plain).ask(card.id, PACK)

    para = Recorder({"relevance": [0.52, 0.56, 0.60, 0.64, 0.5, 0.5], "more_concrete": [0.95]})
    with_para = client_for(card, para).ask(
        card.id, PACK, paraphrases={"relevance": ["Is it related?", "Does it matter?"]}
    )
    assert with_para.requests == 4 + 2
    baseline = client_for(card, Recorder(SERIES)).ask(card.id, PACK).usage_input_tokens
    assert with_para.usage_input_tokens > baseline


def test_paraphrase_runs_still_send_the_shared_instructions() -> None:
    card = edge_card()
    recorder = Recorder({"relevance": [0.52, 0.56, 0.60, 0.64, 0.5], "more_concrete": [0.95]})
    client_for(card, recorder).ask(card.id, PACK, paraphrases={"relevance": ["Is it related?"]})
    assert all(SENTINEL in text for text in recorder.sent_instructions)


# ---------------------------------------------------------------- measurement cache


def test_measurement_cache_is_keyed_by_protocol_identity(tmp_path: Path) -> None:
    """`(item_id, key)` alone let a re-word reuse answers gathered for the old wording."""
    from sextant.calibrate import GoldenItem, measure, measurement_ledger

    card = edge_card()
    items = [
        GoldenItem(id="gi_1", card_id=card.id, key="relevance", pack=dict(PACK), split="fit")
    ]
    root = tmp_path / "calibration-v2"
    ledger = measurement_ledger(card.id, root)

    first = measure(card, client_for(card, Recorder(SERIES), ledger), items, root=root)
    assert first == 1
    again = measure(card, client_for(card, Recorder(SERIES), ledger), items, root=root)
    assert again == 0, "an unchanged protocol should not be re-measured"

    rows = [
        json.loads(line)
        for line in (root / card.id / "responses.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert all(r["protocol_fingerprint"] for r in rows)

    reworded = edge_card(instructions=f"Answer literally. {SENTINEL} Extra clause.")
    after = measure(reworded, client_for(reworded, Recorder(SERIES), ledger), items, root=root)
    assert after == 1, "a changed wire protocol must invalidate the cached measurement"


def test_fitting_refuses_measurements_from_a_superseded_protocol(tmp_path: Path) -> None:
    """A threshold is what lets the runtime call a question calibrated; it needs live evidence."""
    from sextant.calibrate import CalibrationProtocolError, card_dir, fit_card

    card = edge_card()
    directory = card_dir(card.id, tmp_path / "calibration")
    rows = [
        {"item_id": f"gi_{i}", "key": "relevance", "split": "fit", "p": 0.9 if i % 2 == 0 else 0.1}
        for i in range(20)
    ]
    gold = [{"item_id": r["item_id"], "key": "relevance", "gold": r["p"] > 0.5} for r in rows]
    for name, payload in (("responses.jsonl", rows), ("gold.jsonl", gold)):
        (directory / name).write_text("".join(json.dumps(r) + "\n" for r in payload))

    with pytest.raises(CalibrationProtocolError, match="re-measure"):
        fit_card(card, target=0.9, root=tmp_path / "calibration")
