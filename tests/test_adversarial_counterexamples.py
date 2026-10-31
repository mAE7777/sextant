"""Counterexamples from adversarial testing of the decision graph, ledger, and calibration code.

Each test encodes one attack that once succeeded. They were written before the fixes, against
the behaviour the fixes must deliver. Imports of names that did not exist before a fix stay
inside the tests, so one missing name fails one test rather than the whole module.
"""

from __future__ import annotations

import copy
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import pytest

from sextant.config import load_config
from sextant.graph import check_all, diff_graphs, save_graph
from sextant.jev_client import JevClient, JevClientError
from sextant.jev_transport import JevResult
from sextant.ledger import Ledger
from sextant.rolecards import RoleCard, compute_hash
from sextant.schema import DecisionGraph
from sextant.selftest import base_graph
from sextant.selftest import edge as fx_edge
from sextant.selftest import node as fx_node

pytestmark = pytest.mark.unit

PARENT = "Should I take the offer?"
CHILD = "Will the base salary cover my monthly expenses?"
OTHER_CHILD = "Is the commute under forty-five minutes?"


# ---------------------------------------------------------------- builders


def sealed(data: dict[str, Any]) -> RoleCard:
    draft = RoleCard.model_validate(data)
    data = copy.deepcopy(data)
    data["question_hashes"] = {k: compute_hash(draft, k) for k in draft.questions}
    return RoleCard.model_validate(data)


def r3_card(**over: Any) -> RoleCard:
    data: dict[str, Any] = {
        "id": "R3",
        "purpose": "edge verification",
        "primitive": "noul",
        "state_pack": ["parent_question", "child_question", "parent_kind", "child_kind"],
        "instructions": "Judge only what the state says.",
        "questions": {
            "relevance": {
                "primitive": "noul",
                "text": "Does the answer to {child_question} contribute directly to {parent_question}?",
                "criteria_true": "It contributes directly.",
                "criteria_false": "It does not contribute directly.",
            },
            "relevance_neg": {
                "primitive": "noul",
                "text": "Is {child_question} unrelated to {parent_question}?",
                "role": "negation_twin",
            },
        },
    }
    data.update(over)
    return sealed(data)


def choice_card(**over: Any) -> RoleCard:
    data: dict[str, Any] = {
        "id": "RC",
        "purpose": "route a ticket",
        "primitive": "choice",
        "state_pack": ["ticket"],
        "instructions": "Pick the department that owns the ticket.",
        "questions": {
            "department": {
                "primitive": "choice",
                "text": "Which department owns {ticket}?",
                "options": {"billing": "money", "technical": "software", "none_fits": "none of these"},
                "escape": "none_fits",
            }
        },
    }
    data.update(over)
    return sealed(data)


def score_card() -> RoleCard:
    return sealed({
        "id": "RS",
        "purpose": "rate frustration",
        "primitive": "score",
        "state_pack": ["ticket"],
        "instructions": "Rate only what the ticket says.",
        "questions": {
            "frustration": {
                "primitive": "score",
                "text": "How frustrated is the writer of {ticket}?",
                "levels": ["calm", "mild", "clear", "strong"],
            }
        },
    })


def leaf_card() -> RoleCard:
    return sealed({
        "id": "R7",
        "purpose": "leaf judgment",
        "primitive": "noul",
        "state_pack": ["question"],
        "instructions": "Answer from the state alone.",
        "questions": {
            "leaf": {"primitive": "noul", "text": "Is {question} true?"},
            "enough": {"primitive": "noul", "text": "Is there enough to answer {question}?", "role": "sufficiency"},
        },
    })


class Scripted:
    """A transport stand-in. `answer(key, question, call_index)` returns the raw answer dict."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def __call__(self, state: Any, questions: dict[str, dict[str, Any]], model: str) -> JevResult:
        index = len(self.calls)
        self.calls.append({"state": copy.deepcopy(state), "questions": copy.deepcopy(questions)})
        answers = {key: self.answer(key, q, index) for key, q in questions.items()}
        return JevResult(
            model_id=model, answers=answers, usage={"input_tokens": 40, "output_tokens": 4},
            latency_ms=1.0, request_id="scripted",
        )


def noul_series(series: dict[str, list[float]], default: float = 0.9) -> Scripted:
    pending = {k: list(v) for k, v in series.items()}

    def answer(key: str, q: dict[str, Any], index: int) -> dict[str, Any]:
        values = pending.get(key)
        return {"type": "noul", "noul": values.pop(0) if values else default}

    return Scripted(answer)


def client(
    card: RoleCard, caller: Any, ledger: Ledger | None = None, calibration_root: Path | None = None, **cards: RoleCard
) -> JevClient:
    config = copy.deepcopy(load_config())
    config["jev"]["requests_per_minute"] = 1_000_000
    return JevClient(
        config, {card.id: card, **cards}, ledger=ledger, caller=caller, seed=5,
        transport="controlled:scripted", calibration_root=calibration_root,
    )


def r3_pack(parent: str = PARENT, child: str = CHILD) -> dict[str, Any]:
    return {"parent_question": parent, "child_question": child, "parent_kind": "question", "child_kind": "factor"}


def decomposition_graph(verification: dict[str, Any] | None, *, child: str = CHILD) -> dict[str, Any]:
    question = fx_node("question", PARENT)
    factor = fx_node("factor", child)
    residual = fx_node("factor", "other components not listed", residual=True)
    kid = fx_edge("decomposes_into", question, factor)
    if verification is not None:
        kid["verification"] = verification
    return base_graph([question, factor, residual], [kid, fx_edge("decomposes_into", question, residual)])


def dump(estimate: Any) -> dict[str, Any]:
    return estimate.model_dump(mode="json", exclude_none=True)


def fails(findings: list[Any]) -> list[str]:
    return [f"{f.check}: {f.message}" for f in findings if getattr(f, "severity", "FAIL") == "FAIL"]


# ---------------------------------------------------------------- gates


def test_forged_receipts_cannot_assert_a_gate_after_verify(tmp_path: Path) -> None:
    """A hand-built receipt chain passed journal and absorb on a draft with no evidence."""
    from sextant import state as st
    from sextant.schema import Gate, GateAuthority, GateReceipt, GateStatus, receipt_token

    graph = DecisionGraph.model_validate(decomposition_graph(None))
    state = st.init_state(tmp_path, "forge", created="2026-09-22T00:00:00Z")
    tokens: list[str] = []
    for name in st.GATES_IN_ORDER:
        checker = "sextant.graph.release" if name in {"verify", "render", "consistency", "verifier"} else "sextant.graph.check_all"
        try:
            revisions = st.input_revisions(state, graph)
            artifact = st.bound_artifact(graph, {})
        except TypeError:
            revisions, artifact = {"forged": "1"}, "forged"
        receipt = GateReceipt(
            protocol_version=st.WIRE_PROTOCOL_VERSION, checker=checker, input_revisions=revisions,
            artifact_hash=artifact, prerequisites=list(tokens), result="PASS",
        )
        gate = Gate(authority=GateAuthority.checker, status=GateStatus.PASS, at="2026-09-22T00:00:00Z",
                    token=receipt_token(receipt), receipt=receipt)
        state.gates[name] = gate
        tokens.append(gate.token)
    for name in ("journal", "absorb", "values", "aggregate"):
        with pytest.raises(st.GateError):
            st.assert_gate(state, name, graph=graph, call_index={})


def test_only_gates_with_a_real_check_can_be_certified(tmp_path: Path) -> None:
    from sextant import state as st

    graph = DecisionGraph.model_validate(decomposition_graph(None))
    state = st.init_state(tmp_path, "gates", created="2026-09-22T00:00:00Z")
    for name in ("intake", "values", "journal", "absorb"):
        with pytest.raises(st.GateError, match="no stage check"):
            st.evaluate_gate(state, name, graph, {})


# ---------------------------------------------------------------- edge evidence


def _r3_judgment(tmp_path: Path, p: float, child: str = CHILD) -> tuple[Any, Ledger, RoleCard]:
    card = r3_card()
    ledger = Ledger(tmp_path / "decision")
    result = client(card, noul_series({"relevance": [p] * 4, "relevance_neg": [1 - p] * 4}), ledger).ask(
        "R3", r3_pack(child=child)
    )
    return result.estimates["relevance"].estimate, ledger, card


def test_a_no_answer_does_not_support_an_edge(tmp_path: Path) -> None:
    from sextant.graph import check_release, derive_edge_states

    estimate, ledger, card = _r3_judgment(tmp_path, 0.01)
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    states = derive_edge_states(graph, ledger.call_index(), cards={"R3": card}, calibration_root=tmp_path / "cal")
    assert "externally_tested" not in states.values()
    assert check_release(graph, ledger.call_index(), cards={"R3": card}, calibration_root=tmp_path / "cal")


def test_one_judgment_cannot_support_an_edge_it_was_not_about(tmp_path: Path) -> None:
    from sextant.graph import derive_edge_states

    estimate, ledger, card = _r3_judgment(tmp_path, 0.95, child=CHILD)
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}, child=OTHER_CHILD))
    index = ledger.call_index()
    findings = fails(check_all(graph, index, cards={"R3": card}).findings)
    states = derive_edge_states(graph, index, cards={"R3": card}, calibration_root=tmp_path / "cal")
    real = next(e for e in graph.edges if e.verification is not None)
    assert findings or states[real.id] in {"proposed", "structurally_valid", "reviewed"}
    assert states[real.id] not in {"relevance_supported", "externally_tested"}


def test_an_uncalibrated_judgment_never_supports_release(tmp_path: Path) -> None:
    from sextant.graph import check_release

    estimate, ledger, card = _r3_judgment(tmp_path, 0.97)
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    release = check_release(graph, ledger.call_index(), cards={"R3": card}, calibration_root=tmp_path / "cal")
    assert any("calibrat" in f.message for f in release), [f.message for f in release]


def test_release_without_cards_or_calibration_is_refused(tmp_path: Path) -> None:
    from sextant.graph import check_release

    estimate, ledger, _ = _r3_judgment(tmp_path, 0.97)
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    assert check_release(graph, ledger.call_index())


def test_edge_types_without_an_evidence_protocol_do_not_release(tmp_path: Path) -> None:
    from sextant.graph import check_release

    option = fx_node("option", "take the offer")
    factor = fx_node("factor", CHILD)
    question = fx_node("question", PARENT)
    residual = fx_node("factor", "other components not listed", residual=True)
    data = base_graph(
        [question, factor, residual, option],
        [fx_edge("decomposes_into", question, factor), fx_edge("decomposes_into", question, residual),
         fx_edge("decomposes_into", question, option), fx_edge("enables", option, factor)],
    )
    graph = DecisionGraph.model_validate(data)
    messages = [f.message for f in check_release(graph, {}, cards={}, calibration_root=tmp_path)]
    assert any("no evidence protocol" in m for m in messages), messages


# ---------------------------------------------------------------- residual exemption


def test_a_sibling_set_has_exactly_one_residual() -> None:
    question = fx_node("question", PARENT)
    kids = [fx_node("factor", f"component {i}", residual=True) for i in range(3)]
    real = fx_node("factor", CHILD)
    data = base_graph([question, real, *kids], [fx_edge("decomposes_into", question, k) for k in [real, *kids]])
    graph = DecisionGraph.model_validate(data)
    assert any("residual" in m for m in fails(check_all(graph).findings))


def test_the_diff_sees_a_residual_flip() -> None:
    base = decomposition_graph(None)
    a = DecisionGraph.model_validate(base)
    flipped = copy.deepcopy(base)
    flipped["nodes"][1]["residual"] = True
    flipped["version"], flipped["parent_version"] = 2, 1
    b = DecisionGraph.model_validate(flipped)
    changed = json.dumps(diff_graphs(a, b)["changed"])
    assert "residual" in changed


# ---------------------------------------------------------------- traversal


def _chain(n: int, edge_type: str) -> DecisionGraph:
    kind = "question" if edge_type == "decomposes_into" else "factor"
    nodes = [fx_node(kind, f"step {i}") for i in range(n)]
    edges = [fx_edge(edge_type, nodes[i], nodes[i + 1]) for i in range(n - 1)]
    if edge_type != "decomposes_into":
        nodes.insert(0, fx_node("question", PARENT))
    return DecisionGraph.model_validate(base_graph(nodes, edges))


@pytest.mark.parametrize("edge_type", ["decomposes_into", "depends_on"])
def test_a_deep_chain_returns_findings_instead_of_raising(edge_type: str) -> None:
    result = check_all(_chain(1200, edge_type))
    assert isinstance(result.findings, list)


def test_shared_children_do_not_explode() -> None:
    layers, width = 26, 2
    nodes = [fx_node("question", PARENT)]
    previous = [nodes[0]]
    edges = []
    for layer in range(layers):
        current = [fx_node("question", f"layer {layer} node {j}") for j in range(width)]
        nodes.extend(current)
        for parent in previous:
            for kid in current:
                edges.append(fx_edge("decomposes_into", parent, kid))
        previous = current
    graph = DecisionGraph.model_validate(base_graph(nodes, edges))
    started = time.perf_counter()
    check_all(graph)
    assert time.perf_counter() - started < 3.0


# ---------------------------------------------------------------- ledger audit completeness


def _audited(tmp_path: Path) -> tuple[Any, Ledger, RoleCard]:
    card = r3_card()
    ledger = Ledger(tmp_path / "decision")
    result = client(card, noul_series({"relevance": [0.52, 0.56, 0.60, 0.64]}), ledger).ask("R3", r3_pack())
    return result.estimates["relevance"].estimate, ledger, card


def _audit(estimate: Any, ledger: Ledger, card: RoleCard, index: dict[str, Any] | None = None) -> list[str]:
    data = decomposition_graph(None)
    data["nodes"][1]["estimate"] = dump(estimate)
    data["nodes"][1]["resolution"] = {"criteria": "offer letter shows salary >= expenses", "source": "offer letter"}
    graph = DecisionGraph.model_validate(data)
    return fails(check_all(graph, index if index is not None else ledger.call_index(), cards={card.id: card}).findings)


def test_the_untampered_judgment_audits_clean(tmp_path: Path) -> None:
    estimate, ledger, card = _audited(tmp_path)
    assert _audit(estimate, ledger, card) == []


def test_cherry_picking_one_rerun_fails(tmp_path: Path) -> None:
    estimate, ledger, card = _audited(tmp_path)
    best = max(estimate.contributions, key=lambda c: c.value or 0)
    forged = estimate.model_copy(update={
        "p": best.value, "aggregation": "single",
        "contributions": [best.model_copy(update={"role": "primary", "rendered_hash": estimate.rendered_hash})],
    })
    assert _audit(forged, ledger, card)


def test_dropping_a_rerun_fails(tmp_path: Path) -> None:
    estimate, ledger, card = _audited(tmp_path)
    kept = estimate.contributions[:-1]
    forged = estimate.model_copy(update={"contributions": kept, "p": sum(c.value for c in kept) / len(kept)})
    assert _audit(forged, ledger, card)


def test_duplicating_a_call_fails(tmp_path: Path) -> None:
    estimate, ledger, card = _audited(tmp_path)
    doubled = [*estimate.contributions, estimate.contributions[-1], estimate.contributions[-1]]
    forged = estimate.model_copy(update={"contributions": doubled, "p": sum(c.value for c in doubled) / len(doubled)})
    assert _audit(forged, ledger, card)


def test_renaming_the_aggregation_fails(tmp_path: Path) -> None:
    estimate, ledger, card = _audited(tmp_path)
    assert _audit(estimate.model_copy(update={"aggregation": "permutation_mean"}), ledger, card)


def test_a_reused_call_id_is_refused_by_the_ledger(tmp_path: Path) -> None:
    _, ledger, _ = _audited(tmp_path)
    row = ledger.read("jev-calls")[0]
    with pytest.raises(ValueError):
        ledger.record_jev_call(
            card_id=row["card_id"], protocol_version=row["protocol_version"], question_keys=row["question_keys"],
            question_bindings=row["question_bindings"], questions_digest=row["questions_digest"],
            model_id=row["model_id"], transport=row["transport"], state_sha=row["state_sha"],
            request=row["request"], response=row["response"], latency_ms=1.0, usage={},
            run_id=row["run_id"], call_role=row["call_role"], run_protocol=row["run_protocol"], call_id=row["id"],
        )


def test_retracting_an_unfavourable_rerun_fails(tmp_path: Path) -> None:
    estimate, ledger, card = _audited(tmp_path)
    rows = ledger.read("jev-calls")
    worst = min((r for r in rows if r["id"] != estimate.contributions[0].call_id),
                key=lambda r: r["response"]["answers"]["relevance"]["noul"])
    ledger.retract("jev-calls", worst["seq"], "unfavourable")
    kept = [c for c in estimate.contributions if c.call_id != worst["id"]]
    forged = estimate.model_copy(update={"contributions": kept, "p": sum(c.value for c in kept) / len(kept)})
    assert _audit(forged, ledger, card)


def test_a_stripped_recorded_payload_fails(tmp_path: Path) -> None:
    estimate, ledger, card = _audited(tmp_path)
    index = copy.deepcopy(ledger.call_index())
    for row in index.values():
        for question in row["request"]["questions"].values():
            question["instructions"] = "stripped"
    assert _audit(estimate, ledger, card, index)


def test_a_second_favourable_run_cannot_be_chosen_for_release(tmp_path: Path) -> None:
    from sextant.graph import check_release

    card = r3_card()
    ledger = Ledger(tmp_path / "decision")
    client(card, noul_series({"relevance": [0.05] * 4}), ledger).ask("R3", r3_pack())
    second = client(card, noul_series({"relevance": [0.97] * 4}), ledger).ask("R3", r3_pack())
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(second.estimates["relevance"].estimate)}))
    messages = [f.message for f in check_release(graph, ledger.call_index(), cards={"R3": card}, calibration_root=tmp_path)]
    assert any("run" in m for m in messages), messages


# ---------------------------------------------------------------- answers at ingest


def _choice_answer(probabilities: dict[str, float], choice: str | None = None) -> Any:
    def answer(key: str, q: dict[str, Any], index: int) -> dict[str, Any]:
        top = choice or max(probabilities, key=lambda o: probabilities[o])
        return {"type": "choice", "choice": top, "probabilities": dict(probabilities), "confidence": 0.7}

    return answer


def test_two_decimal_rounding_is_accepted_and_audits_clean(tmp_path: Path) -> None:
    card = choice_card(permutations=1)
    ledger = Ledger(tmp_path / "d")
    rounded = {"billing": 0.33, "technical": 0.33, "none_fits": 0.33}
    result = client(card, Scripted(_choice_answer(rounded, "billing")), ledger).ask("RC", {"ticket": "refund please"})
    estimate = result.estimates["department"].estimate
    assert math.isclose(sum(estimate.probabilities.values()), 1.0, abs_tol=1e-9)
    data = base_graph([fx_node("question", PARENT, state_vector={"department": dump(estimate)})], [])
    assert fails(check_all(DecisionGraph.model_validate(data), ledger.call_index(), cards={"RC": card}).findings) == []


def test_a_rounded_score_is_accepted(tmp_path: Path) -> None:
    card = score_card()

    def answer(key: str, q: dict[str, Any], index: int) -> dict[str, Any]:
        return {"type": "score", "score": 2.0, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.49, "3": 0.2}, "confidence": 0.6}

    result = client(card, Scripted(answer), Ledger(tmp_path / "d")).ask("RS", {"ticket": "this is the third time"})
    assert result.estimates["frustration"].estimate.score == 2.0


@pytest.mark.parametrize(
    "probabilities",
    [
        {"billing": 0.1, "zzz": 0.8, "none_fits": 0.1},
        {"billing": 0.7, "none_fits": 0.3},
        {"billing": 1.5, "technical": -0.5, "none_fits": 0.0},
        {"billing": 0.9, "technical": 0.9, "none_fits": 0.9},
    ],
)
def test_an_out_of_domain_choice_answer_is_refused(tmp_path: Path, probabilities: dict[str, float]) -> None:
    card = choice_card(permutations=1)
    with pytest.raises(JevClientError):
        client(card, Scripted(_choice_answer(probabilities)), Ledger(tmp_path / "d")).ask("RC", {"ticket": "x"})


@pytest.mark.parametrize("value", [99.0, float("inf"), -1.0])
def test_an_out_of_range_score_is_refused(tmp_path: Path, value: float) -> None:
    def answer(key: str, q: dict[str, Any], index: int) -> dict[str, Any]:
        return {"type": "score", "score": value, "probabilities": {"0": 0.25, "1": 0.25, "2": 0.25, "3": 0.25}}

    with pytest.raises(JevClientError):
        client(score_card(), Scripted(answer), Ledger(tmp_path / "d")).ask("RS", {"ticket": "x"})


@pytest.mark.parametrize("value", [1.6, -0.4, float("nan")])
def test_an_out_of_range_noul_is_refused(tmp_path: Path, value: float) -> None:
    with pytest.raises(JevClientError):
        client(r3_card(), noul_series({"relevance": [value]}), Ledger(tmp_path / "d")).ask("R3", r3_pack())


def test_choice_derivation_matches_an_independent_oracle() -> None:
    from sextant.wire import replay_choice

    options = ["billing", "technical", "none_fits"]
    rng = random.Random(7)
    for _ in range(2000):
        runs = []
        for _ in range(rng.randint(1, 4)):
            raw = [rng.choice([0.0, 0.25, 0.5, 0.25, rng.random()]) for _ in options]
            if sum(raw) == 0:
                raw[rng.randrange(len(raw))] = 1.0  # ingest refuses a vector with no mass
            total = sum(raw)
            runs.append({o: v / total for o, v in zip(options, raw, strict=True)})
        oracle = {o: sum(r[o] for r in runs) / len(runs) for o in options}
        total = sum(oracle.values())
        oracle = {o: v / total for o, v in oracle.items()}
        best = max(oracle.values())
        winner = next(o for o in options if math.isclose(oracle[o], best, abs_tol=1e-12))
        shuffled = [dict(sorted(r.items(), key=lambda kv: rng.random())) for r in runs]
        got = replay_choice(shuffled, options=options)
        assert got is not None
        probabilities, choice = got[0], got[1]
        assert set(probabilities) == set(options)
        assert all(math.isclose(probabilities[o], oracle[o], abs_tol=1e-9) for o in options)
        assert choice == winner


def test_a_sufficiency_value_is_rederived_not_copied_from_the_primary(tmp_path: Path) -> None:
    card = leaf_card()
    ledger = Ledger(tmp_path / "d")
    result = client(card, noul_series({"leaf": [0.9] * 4, "enough": [0.5, 0.52, 0.54, 0.56]}), ledger).ask(
        "R7", {"question": "Is the salary enough?"}
    )
    leaf = result.estimates["leaf"].estimate
    data = decomposition_graph(None)
    data["nodes"][1]["estimate"] = dump(leaf)
    data["nodes"][1]["resolution"] = {"criteria": "offer letter", "source": "letter"}
    graph = DecisionGraph.model_validate(data)
    assert fails(check_all(graph, ledger.call_index(), cards={"R7": card}).findings) == []


def test_a_choice_that_is_not_the_argmax_still_audits_clean(tmp_path: Path) -> None:
    card = choice_card(permutations=1)
    ledger = Ledger(tmp_path / "d")
    tie = {"billing": 0.5, "technical": 0.5, "none_fits": 0.0}
    result = client(card, Scripted(_choice_answer(tie, "technical")), ledger).ask("RC", {"ticket": "x"})
    estimate = result.estimates["department"].estimate
    assert estimate.contested
    data = base_graph([fx_node("question", PARENT, state_vector={"department": dump(estimate)})], [])
    assert fails(check_all(DecisionGraph.model_validate(data), ledger.call_index(), cards={"RC": card}).findings) == []


def test_every_yes_no_question_is_rerun_by_protocol(tmp_path: Path) -> None:
    card = r3_card()
    scripted = noul_series({"relevance": [0.97] * 4})
    result = client(card, scripted, Ledger(tmp_path / "d")).ask("R3", r3_pack())
    estimate = result.estimates["relevance"].estimate
    assert estimate.stability.n_reruns == card.reruns
    assert len(scripted.calls) == 1 + card.reruns


# ---------------------------------------------------------------- identity and protocol


def test_purpose_and_language_are_part_of_the_measurement_identity() -> None:
    from sextant.wire import measurement_conditions

    card = r3_card()
    other_purpose = r3_card(purpose="something else entirely")
    base = measurement_conditions(card, keys=["relevance"], model_id="m", transport="t", language="en")
    assert base != measurement_conditions(other_purpose, keys=["relevance"], model_id="m", transport="t", language="en")
    assert base != measurement_conditions(card, keys=["relevance"], model_id="m", transport="t", language="zh")


def test_a_judgment_from_a_superseded_protocol_is_not_current_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from sextant import graph as graph_module
    from sextant import wire

    estimate, ledger, card = _audited(tmp_path)
    monkeypatch.setattr(wire, "WIRE_PROTOCOL_VERSION", "sextant-jev-wire/99")
    monkeypatch.setattr(graph_module.wire, "WIRE_PROTOCOL_VERSION", "sextant-jev-wire/99")
    assert any("protocol" in m for m in _audit(estimate, ledger, card))


# ---------------------------------------------------------------- calibration


def _cal_items(n: int, *, lang: str = "en") -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        related = i % 2 == 0
        child = f"{CHILD} (case {i})" if related else f"What is the grain size of basalt sample {i}?"
        out.append({"key": "relevance", "pack": r3_pack(child=child), "expected": related, "lang": lang,
                    "lineage": f"synthetic-case-{i}"})
    return out


def _cal_caller(error_every: int = 0) -> Scripted:
    def answer(key: str, q: dict[str, Any], index: int) -> dict[str, Any]:
        text = q["instructions"]
        related = "salary" in text
        if error_every and "case " in text:
            case = int(text.rsplit("case ", 1)[1].split(")")[0])
            if case % error_every == 0:
                related = not related
        p = 0.95 if related else 0.05
        if key.endswith("_neg"):
            p = 1 - p
        return {"type": "noul", "noul": p}

    return Scripted(answer)


def _calibrated(tmp_path: Path, n: int, error_every: int = 0, card: RoleCard | None = None) -> tuple[RoleCard, Path]:
    from sextant import calibrate as cal

    card = card or r3_card()
    root = tmp_path / "calibration-v2"
    cal.build(card.id, _cal_items(n), root=root)
    cal.record_truth(
        card.id,
        [{"item_id": i.id, "key": i.key, "label": "salary" in i.pack["child_question"]} for i in cal.load_items(card.id, root)],
        card=card, source="generator", method="synthetic ground truth", root=root,
    )
    cal.resolve_gold(card.id, root)
    ledger = cal.measurement_ledger(card.id, root)
    cal.measure(card, client(card, _cal_caller(error_every), ledger, root), cal.load_items(card.id, root), root=root)
    return card, root


def test_a_holdout_that_was_scored_cannot_confirm_again(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    card, root = _calibrated(tmp_path, 60)
    first = {r.key: r for r in cal.fit_card(card, target=0.9, root=root)}["relevance"]
    assert first.deployable is False
    cal.build(card.id, _cal_items(200)[60:], root=root)
    cal.record_truth(
        card.id,
        [{"item_id": i.id, "key": i.key, "label": "salary" in i.pack["child_question"]} for i in cal.load_items(card.id, root)],
        card=card, source="generator", method="synthetic ground truth", root=root,
    )
    cal.resolve_gold(card.id, root)
    cal.measure(card, client(card, _cal_caller(), cal.measurement_ledger(card.id, root), root), cal.load_items(card.id, root), root=root)
    second = {r.key: r for r in cal.fit_card(card, target=0.9, root=root)}["relevance"]
    seen = set(first.confirming_groups)
    assert seen and not (seen & set(second.confirming_groups))


def test_metrics_do_not_read_the_holdout_by_default(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    card, root = _calibrated(tmp_path, 80)
    metrics = cal.compute_metrics(card, root)
    assert set(metrics["split_counts"]) <= {"development", "fit"}
    assert cal.holdout_evaluations(card.id, root) == []


def test_zero_automatic_releases_cannot_deploy(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    rows = [{"item_id": f"i{k}", "key": "relevance", "split": "holdout", "p": 0.02, "gold": False,
             "resolved_by": "deterministic_truth", "group": f"g{k}"} for k in range(400)]
    fit = [{"item_id": f"f{k}", "key": "relevance", "split": "fit", "p": 0.97 if k % 2 else 0.02,
            "gold": bool(k % 2), "resolved_by": "deterministic_truth", "group": f"h{k}"} for k in range(200)]
    result = cal.fit_key(fit + rows, target=0.9, family_size=3)
    assert result.deployable is False


def test_one_case_counts_once_however_many_decisions_it_holds(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    fit = [{"item_id": f"f{k}", "key": "relevance", "split": "fit", "p": 0.97 if k % 2 else 0.02,
            "gold": bool(k % 2), "resolved_by": "deterministic_truth", "group": f"f{k}"} for k in range(200)]
    one_case = [{"item_id": f"h{k}", "key": "relevance", "split": "holdout", "p": 0.97, "gold": True,
                 "resolved_by": "deterministic_truth", "group": "the-only-case"} for k in range(400)]
    result = cal.fit_key(fit + one_case, target=0.9, family_size=3)
    assert result.deployable is False
    assert result.n_eligible == 1


def test_a_low_coverage_holdout_cannot_deploy(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    fit = [{"item_id": f"f{k}", "key": "relevance", "split": "fit", "p": 0.97 if k % 2 else 0.02,
            "gold": bool(k % 2), "resolved_by": "deterministic_truth", "group": f"f{k}"} for k in range(200)]
    hold = [{"item_id": f"h{k}", "key": "relevance", "split": "holdout", "p": 0.97 if k < 200 else 0.5,
             "gold": True, "resolved_by": "deterministic_truth", "group": f"h{k}"} for k in range(1000)]
    assert cal.fit_key(fit + hold, target=0.9, family_size=3).deployable is False


def test_model_majority_gold_cannot_confirm(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    fit = [{"item_id": f"f{k}", "key": "relevance", "split": "fit", "p": 0.97 if k % 2 else 0.02,
            "gold": bool(k % 2), "resolved_by": "deterministic_truth", "group": f"f{k}"} for k in range(200)]
    hold = [{"item_id": f"h{k}", "key": "relevance", "split": "holdout", "p": 0.97 if k % 2 else 0.02,
             "gold": bool(k % 2), "resolved_by": "independent_model_majority", "group": f"h{k}"} for k in range(600)]
    assert cal.fit_key(fit + hold, target=0.9, family_size=3).deployable is False


@pytest.mark.parametrize("label", ["false", "no", "0", "ambiguous", 0, 1])
def test_labels_are_typed_by_the_question_primitive(tmp_path: Path, label: Any) -> None:
    from sextant import calibrate as cal

    card = r3_card()
    root = tmp_path / "calibration-v2"
    cal.build(card.id, _cal_items(4), root=root)
    item = cal.load_items(card.id, root)[0]
    if label == "ambiguous":
        cal.record_truth(card.id, [{"item_id": item.id, "key": item.key, "label": label}], source="s", method="m", root=root, card=card)
        cal.resolve_gold(card.id, root)
        rows = [json.loads(line) for line in (root / card.id / "gold.jsonl").read_text().splitlines() if line.strip()]
        gold = [g for g in rows if g["item_id"] == item.id]
        assert gold and gold[0]["gold"] is None
        return
    with pytest.raises(cal.CalibrationEvidenceError):
        cal.record_truth(card.id, [{"item_id": item.id, "key": item.key, "label": label}], source="s", method="m", root=root, card=card)


def test_withdrawn_gold_revokes_a_certificate(tmp_path: Path) -> None:
    from sextant import calibrate as cal
    from sextant.certificates import Certificate, append_certificate, effective_certificate

    card, root = _calibrated(tmp_path, 40)
    append_certificate(root, card.id, Certificate(key="relevance", status="validated", value=0.8))
    (root / card.id / "gold.jsonl").write_text("")
    try:
        cal.fit_card(card, target=0.9, root=root)
    except cal.CalibrationProtocolError:
        pass
    assert effective_certificate(root, card.id, "relevance").status == "revoked"


def test_a_refit_that_raises_revokes_the_certificate(tmp_path: Path) -> None:
    from sextant import calibrate as cal
    from sextant.certificates import Certificate, append_certificate, effective_certificate

    card, root = _calibrated(tmp_path, 40)
    append_certificate(root, card.id, Certificate(key="relevance", status="validated", value=0.8))
    reworded = r3_card(instructions="Judge only what the state says, literally.")
    with pytest.raises(cal.CalibrationProtocolError):
        cal.fit_card(reworded, target=0.9, root=root)
    assert effective_certificate(root, card.id, "relevance").status == "revoked"


def test_a_hand_written_certificate_is_not_calibrated(tmp_path: Path) -> None:
    from sextant.certificates import Certificate, append_certificate, certificate_state

    card = r3_card()
    root = tmp_path / "calibration-v2"
    append_certificate(root, card.id, Certificate(key="relevance", status="validated", value=0.5))
    verdict = certificate_state(card, "relevance", model_id="jev-1.13.0", transport="typesafe", language="en", root=root,
                                asked_keys=["relevance", "relevance_neg"], input_language="en")
    assert verdict.state != "calibrated"


def test_card_yaml_thresholds_grant_nothing(tmp_path: Path) -> None:
    card = r3_card(thresholds={"relevance": {"value": 0.5, "status": "validated", "n": 999}})
    result = client(card, noul_series({"relevance": [0.97] * 4}), Ledger(tmp_path / "d")).ask("R3", r3_pack())
    assert result.estimates["relevance"].estimate.calibration_status.state != "calibrated"
    assert result.passes("relevance") is None


def test_one_pack_cannot_be_split_through_different_group_keys(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    root = tmp_path / "calibration-v2"
    pack = r3_pack()
    rows = [
        {"key": "relevance", "pack": pack, "pair_id": "pair-1", "split": "fit", "expected": True},
        {"key": "relevance_neg", "pack": pack, "split": "holdout", "expected": False},
    ]
    with pytest.raises(cal.CalibrationSplitError):
        cal.build("R3", rows, root=root)


# ---------------------------------------------------------------- concurrent writers


def _save_graph_race(directory: str, text: str) -> str:
    from sextant.graph import save_graph as save

    data = decomposition_graph(None, child=text)
    try:
        save(DecisionGraph.model_validate(data), Path(directory))
        return "saved"
    except FileExistsError:
        return "refused"


def test_two_graph_writers_cannot_both_save_one_version(tmp_path: Path) -> None:
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    for trial in range(6):
        target = tmp_path / f"t{trial}"
        with context.Pool(2) as pool:
            outcomes = pool.starmap(_save_graph_race, [(str(target), CHILD), (str(target), OTHER_CHILD)])
        assert sorted(outcomes) == ["refused", "saved"], outcomes


def test_a_retraction_cannot_be_recorded_twice(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    row = ledger.event("x")
    ledger.retract("events", row["seq"], "first")
    with pytest.raises(ValueError):
        ledger.retract("events", row["seq"], "second")


def test_non_blocking_lock_reports_another_process(tmp_path: Path) -> None:
    import multiprocessing

    from sextant.ledger import WriterBusy, exclusive

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(target=_hold_lock, args=(str(tmp_path), ready, release))
    holder.start()
    try:
        assert ready.wait(10)
        with pytest.raises(WriterBusy), exclusive(tmp_path, block=False):
            pass
    finally:
        release.set()
        holder.join(10)


def _hold_lock(directory: str, ready: Any, release: Any) -> None:
    from sextant.ledger import exclusive

    with exclusive(Path(directory)):
        ready.set()
        release.wait(10)


# ---------------------------------------------------------------- packaging, config, doctor


def test_the_shipped_config_names_no_machine_path() -> None:
    from sextant.config import PROJECT_ROOT

    text = (PROJECT_ROOT / "config.toml").read_text()
    assert "/Users/" not in text and "/home/" not in text


def test_writing_config_keeps_portable_values(tmp_path: Path) -> None:
    from sextant.config import load_raw_config, write_config

    raw = load_raw_config()
    out = tmp_path / "config.toml"
    write_config(raw, out)
    text = out.read_text()
    assert 'rolecards_dir = "package"' in text and "/Users/" not in text


def test_the_wheel_configuration_does_not_include_resources_twice() -> None:
    import tomllib

    from sextant.config import PROJECT_ROOT

    data = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    wheel = data["tool"]["hatch"]["build"]["targets"]["wheel"]
    packages = wheel.get("packages", [])
    forced = wheel.get("force-include", {})
    assert not any(any(src.startswith(p) for p in packages) for src in forced), forced


def test_a_canary_that_compared_nothing_is_not_a_pass() -> None:
    from sextant.doctor import canary_rung

    assert canary_rung({"n": 3, "compared": 0, "drifted": [], "model_moved": [], "protocol_moved": [], "missing_cards": 3}).status != "PASS"
    assert canary_rung({"n": 0, "compared": 0, "drifted": [], "model_moved": [], "protocol_moved": [], "missing_cards": 0}).status != "PASS"


def test_live_probes_are_not_run_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from sextant import doctor, jev_client, jev_transport

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("doctor attempted a live Jev call without --live")

    monkeypatch.setattr(jev_transport, "call_jev", refuse)
    monkeypatch.setattr(jev_client, "call_jev", refuse)
    monkeypatch.setenv("TYPESAFE_API_KEY", "not-a-real-key")
    rungs = doctor.run_doctor(skip_slow=True, checkout_checks=False)
    live = [r for r in rungs if r.lane == doctor.LIVE]
    assert live and all(r.status in {"NOT_RUN", "NOT-APPLICABLE"} for r in live)


def test_the_release_checklist_covers_hashes_and_linter_and_refuses_duplicates() -> None:
    from sextant.doctor import OFFLINE, RELEASE_CHECKLIST, Rung, release_eligibility

    names = {name for _, name in RELEASE_CHECKLIST}
    assert {"rolecard-hashes", "linter"} <= names
    rungs = [Rung(name, "PASS", "", lane) for lane, name in RELEASE_CHECKLIST]
    rungs.append(Rung("selftest", "FAIL", "", OFFLINE))
    assert release_eligibility(rungs)["eligible"] is False


# ---------------------------------------------------------------- lower-severity items


def test_isolated_nodes_are_reported() -> None:
    question = fx_node("question", PARENT)
    loose = fx_node("factor", "a loose fact")
    graph = DecisionGraph.model_validate(base_graph([question, loose], []))
    assert any("connected" in m or "isolated" in m for m in fails(check_all(graph).findings))


def test_a_null_ledger_answer_is_a_finding_not_a_crash(tmp_path: Path) -> None:
    estimate, ledger, card = _audited(tmp_path)
    index = copy.deepcopy(ledger.call_index())
    for row in index.values():
        row["response"]["answers"]["relevance"] = None
    assert _audit(estimate, ledger, card, index)


def test_a_skip_stamp_does_not_advance_the_stage(tmp_path: Path) -> None:
    from sextant.state import init_state, stamp

    state = init_state(tmp_path, "s", created="2026-09-22T00:00:00Z")
    stamp(state, "absorb", "SKIP", "operator")
    assert state.stage == "init"


def test_paraphrases_are_filled_with_the_item_values(tmp_path: Path) -> None:
    card = sealed({
        "id": "R4",
        "purpose": "coverage",
        "primitive": "noul",
        "state_pack": ["question", "children"],
        "instructions": "Judge the listed children.",
        "questions": {"major": {"primitive": "noul", "text": "Is child {n} major?", "role": "item"}},
    })
    scripted = noul_series({})
    client(card, scripted).ask("R4", {"question": PARENT, "children": ["a", "b"]}, paraphrases={"major": ["Does item {n} matter?"]})
    sent = [q["instructions"] for call in scripted.calls for q in call["questions"].values()]
    assert not any("{n}" in text for text in sent)


def test_the_token_estimate_does_not_undercount_chinese() -> None:
    from sextant.wire import estimate_tokens

    text = "这是一个关于工作选择的问题" * 50
    assert estimate_tokens({"s": text}) >= len(text)


def test_compute_ledger_rows_can_be_retracted(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    row = ledger.record_cost(provider="typesafe", model="jev", request_id="r1", input_tokens=1, output_tokens=1, purpose="t")
    ledger.retract("compute-ledger", row["seq"], "duplicate")


def test_a_torn_last_line_is_reported_before_appending(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path)
    ledger.event("first")
    with ledger.path("events").open("a") as f:
        f.write('{"seq": 2, "kind": "torn')
    with pytest.raises(ValueError, match="incomplete"):
        ledger.event("second")


def test_the_diff_sees_constraint_predicates_and_frames() -> None:
    constraint = fx_node("constraint", "budget", predicate="cost <= 100")
    question = fx_node("question", PARENT)
    a = DecisionGraph.model_validate(base_graph([question, constraint], [fx_edge("constrains", constraint, question)]))
    changed = base_graph([question, dict(constraint, predicate="cost <= 50")], [fx_edge("constrains", constraint, question)])
    changed["frame"]["stakes"] = "high"
    changed["version"], changed["parent_version"] = 2, 1
    b = DecisionGraph.model_validate(changed)
    text = json.dumps(diff_graphs(a, b))
    assert "predicate" in text and "stakes" in text


def test_save_graph_is_exclusive_even_when_checked_first(tmp_path: Path) -> None:
    graph = DecisionGraph.model_validate(decomposition_graph(None))
    save_graph(graph, tmp_path)
    with pytest.raises(FileExistsError):
        save_graph(graph, tmp_path)


# ---------------------------------------------------------------- statistical oracles


def _binomial_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _upper_by_bisection(k: int, n: int, alpha: float) -> float:
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _binomial_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def test_clopper_pearson_matches_a_direct_binomial_inversion() -> None:
    from sextant.certificates import clopper_pearson_lower, clopper_pearson_upper

    for n in (1, 5, 30, 59, 80, 120, 300):
        for k in sorted({0, 1, 2, n // 10, n // 2, n - 1, n}):
            if k > n:
                continue
            for alpha in (0.05, 0.0167, 0.0042):
                expected = _upper_by_bisection(k, n, alpha)
                assert math.isclose(clopper_pearson_upper(k, n, alpha), expected, abs_tol=1e-7), (k, n, alpha)
                mirrored = 1 - _upper_by_bisection(n - k, n, alpha)
                assert math.isclose(clopper_pearson_lower(k, n, alpha), mirrored, abs_tol=1e-7), (k, n, alpha)
    assert math.isclose(clopper_pearson_upper(0, 59, 0.05), 0.0495, abs_tol=5e-4)


def test_false_certification_stays_under_alpha_with_clustered_unequal_cases() -> None:
    """A judge whose per-case error is above the bar must almost never certify.

    Cases hold between 1 and 30 decisions whose correctness is shared inside a case (the
    observed dependence), and the representative decision of each case is what the
    standard counts. Over many simulated holdouts the share that certifies must stay at or below
    alpha; the bound is exact for one decision per independent case.
    """
    from sextant import calibrate as cal

    rng = random.Random(20260922)
    fit = [{"item_id": f"f{k}", "key": "relevance", "split": "fit", "p": 0.97 if k % 2 else 0.02,
            "gold": bool(k % 2), "resolved_by": "deterministic_truth", "group": f"f{k}"} for k in range(200)]
    certified = 0
    simulations = 300
    for sim in range(simulations):
        holdout = []
        for case in range(240):
            size = rng.randint(1, 30)
            truth = rng.random() < 0.5
            wrong = rng.random() < 0.07  # per-case error above the 0.05 bar
            said_yes = truth != wrong
            for item in range(size):
                holdout.append({
                    "item_id": f"s{sim}c{case}i{item}", "key": "relevance", "split": "holdout",
                    "p": 0.97 if said_yes else 0.02, "gold": truth,
                    "resolved_by": "deterministic_truth", "group": f"s{sim}c{case}", "lang": "en",
                })
        if cal.fit_key(fit + holdout, target=0.9, family_size=3).deployable:
            certified += 1
    # The bound is the alpha actually spent on the error bound (0.05 / 2 / (3 x 3) per simulation,
    # 0.83 expected in 300), plus Poisson slack: more than 4 would happen under 0.2% by chance. A
    # mutant that counts every item instead of every case certifies about 10 of 300 here.
    from sextant.certificates import alpha_per_bound

    assert alpha_per_bound(3, 1) * simulations < 1.0
    assert certified <= 4, certified


# ---------------------------------------------------------------- positive control and tamper suite
#
# Everything above shows that bad evidence fails. That alone would also be satisfied by a checker
# that fails everything, so this section first builds one complete synthetic run that really is
# certified in the controlled domain, and then applies each tamper to that passing baseline.


@pytest.fixture(scope="module")
def certified(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    from sextant import calibrate as cal
    from sextant.certificates import effective_certificate

    tmp = tmp_path_factory.mktemp("certified")
    card, root = _calibrated(tmp, 1000)
    results = {r.key: r for r in cal.fit_card(card, target=0.9, root=root)}
    cert = effective_certificate(root, card.id, "relevance")
    return {"card": card, "root": root, "results": results, "cert": cert, "tmp": tmp}


def _ask_certified(env: dict[str, Any], decision: Path, child: str = CHILD, caller: Any = None) -> Any:
    from sextant.wire import identity_keys

    card = env["card"]
    ledger = Ledger(decision)
    c = client(card, caller or noul_series({"relevance": [0.97] * 4, "relevance_neg": [0.03] * 4}), ledger, env["root"])
    return c.ask("R3", r3_pack(child=child), keys=identity_keys(card, "relevance"), input_language="en"), ledger


def test_positive_control_a_complete_run_certifies_and_releases(certified: dict[str, Any], tmp_path: Path) -> None:
    from sextant.graph import check_release, derive_edge_states
    from sextant.state import assert_gate, evaluate_gate, init_state

    fit = certified["results"]["relevance"]
    assert fit.deployable, fit.reason
    assert certified["cert"] is not None and certified["cert"].status == "validated"
    result, ledger = _ask_certified(certified, tmp_path / "decision")
    estimate = result.estimates["relevance"].estimate
    assert estimate.calibration_status.state == "calibrated", estimate.calibration_status.reason
    assert result.passes("relevance") is True
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    cards, root, index = {"R3": certified["card"]}, certified["root"], ledger.call_index()
    real = next(e for e in graph.edges if e.verification is not None)
    assert derive_edge_states(graph, index, cards, root)[real.id] == "direct_contribution_supported"
    assert check_release(graph, index, cards, root) == []
    state = init_state(tmp_path / "decision", "control", created="2026-09-22T00:00:00Z")
    assert evaluate_gate(state, "generate", graph, index, cards=cards, calibration_root=root).status.value == "PASS"
    assert evaluate_gate(state, "verify", graph, index, cards=cards, calibration_root=root).status.value == "PASS"
    assert_gate(state, "verify", graph=graph, call_index=index, cards=cards, calibration_root=root)


def _release_of(certified: dict[str, Any], estimate: Any, index: dict[str, Any]) -> list[str]:
    from sextant.graph import check_release

    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    return [f.message for f in check_release(graph, index, {"R3": certified["card"]}, certified["root"])]


def test_tamper_an_edited_ledger_answer_blocks_release(certified: dict[str, Any], tmp_path: Path) -> None:
    result, ledger = _ask_certified(certified, tmp_path / "d")
    index = copy.deepcopy(ledger.call_index())
    for row in index.values():
        if row["call_role"] == "rerun":
            row["response"]["answers"]["relevance"]["noul"] = 0.2
            break
    assert _release_of(certified, result.estimates["relevance"].estimate, index)


def test_tamper_a_retracted_rerun_blocks_release(certified: dict[str, Any], tmp_path: Path) -> None:
    result, ledger = _ask_certified(certified, tmp_path / "d")
    rerun = next(r for r in ledger.read("jev-calls") if r["call_role"] == "rerun")
    ledger.retract("jev-calls", rerun["seq"], "unfavourable")
    assert _release_of(certified, result.estimates["relevance"].estimate, ledger.call_index())


def test_tamper_a_disagreeing_second_run_blocks_release(certified: dict[str, Any], tmp_path: Path) -> None:
    decision = tmp_path / "d"
    _ask_certified(certified, decision, caller=noul_series({"relevance": [0.02] * 4, "relevance_neg": [0.98] * 4}))
    result, ledger = _ask_certified(certified, decision)
    messages = _release_of(certified, result.estimates["relevance"].estimate, ledger.call_index())
    assert any("runs" in m for m in messages), messages


def test_tamper_an_edited_certificate_invalidates(certified: dict[str, Any], tmp_path: Path) -> None:
    from sextant.certificates import append_certificate, certificate_state

    root = tmp_path / "root"
    import shutil

    shutil.copytree(certified["root"], root)
    genuine = certified["cert"].value
    forged = certified["cert"].model_copy(update={"value": round(genuine + 0.11, 2), "at": None})
    append_certificate(root, "R3", forged)
    verdict = certificate_state(certified["card"], "relevance", model_id="jev-1.13.0", transport="controlled:scripted",
                                language="en", root=root, asked_keys=["relevance", "relevance_neg"], input_language="en")
    assert verdict.state == "invalidated", verdict.reason


def test_tamper_edited_responses_invalidate(certified: dict[str, Any], tmp_path: Path) -> None:
    from sextant.certificates import certificate_state

    root = tmp_path / "root"
    import shutil

    shutil.copytree(certified["root"], root)
    path = root / "R3" / "responses.jsonl"
    lines = path.read_text().splitlines()
    row = json.loads(lines[-1])
    row["p"] = 0.5 if row.get("p") != 0.5 else 0.6
    lines[-1] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n")
    verdict = certificate_state(certified["card"], "relevance", model_id="jev-1.13.0", transport="controlled:scripted",
                                language="en", root=root, asked_keys=["relevance", "relevance_neg"], input_language="en")
    assert verdict.state == "invalidated", verdict.reason


@pytest.mark.parametrize(
    "change",
    [
        {"model_id": "jev-9.9.9"},
        {"language": "zh"},
        {"transport": "typesafe"},
        {"asked_keys": ["relevance"]},
        {"input_language": "zh"},
        {"input_language": None},
    ],
)
def test_tamper_a_different_domain_is_out_of_domain(certified: dict[str, Any], change: dict[str, Any]) -> None:
    from sextant.certificates import certificate_state

    params = {"model_id": "jev-1.13.0", "transport": "controlled:scripted", "language": "en",
              "asked_keys": ["relevance", "relevance_neg"], "input_language": "en", **change}
    verdict = certificate_state(certified["card"], "relevance", root=certified["root"], **params)
    assert verdict.state == "out_of_domain", verdict.reason


def test_tamper_a_scripted_caller_cannot_claim_the_real_transport() -> None:
    config = copy.deepcopy(load_config())
    with pytest.raises(JevClientError):
        JevClient(config, {"R3": r3_card()}, caller=noul_series({}), transport="typesafe")


def test_the_exported_schema_matches_the_model() -> None:
    from sextant.config import PROJECT_ROOT
    from sextant.schema import export_json_schema

    exported = json.loads((PROJECT_ROOT / "schema" / "decision-graph.schema.json").read_text())
    assert exported == export_json_schema()


def test_the_packaged_selftest_fixtures_match_the_generator(tmp_path: Path) -> None:
    """The fixtures shipped in the package are what `selftest --write-fixtures` produces now."""
    from sextant.selftest import write_fixtures

    # Wire-v2 historical fixtures are preserved separately, never rewritten as
    # newly verified evidence when the current wire changes.
    packaged = Path(__file__).resolve().parents[1] / "src" / "sextant" / "fixtures-v3"
    fresh = write_fixtures(tmp_path)
    assert sorted(p.name for p in fresh) == sorted(p.name for p in packaged.iterdir() if p.is_file())
    stale = [p.name for p in fresh if (packaged / p.name).read_bytes() != p.read_bytes()]
    assert stale == [], f"regenerate current fixtures-v3, preserving historical fixtures: {stale}"


def test_lower_a_missing_provider_request_id_is_recorded_as_none() -> None:
    """The SDK raises its own error, not AttributeError, when the request-id header is absent."""
    from typesafe_sdk import TypeSafeError

    from sextant.jev_transport import _request_id

    class NoHeader:
        @property
        def request_id(self) -> str:
            raise TypeSafeError("The response did not include a request ID.")

    class WithHeader:
        request_id = "req_123"

    assert _request_id(NoHeader()) is None
    assert _request_id(object()) is None
    assert _request_id(WithHeader()) == "req_123"


def test_lower_a_card_written_for_the_first_wire_protocol_names_its_migration(tmp_path: Path) -> None:
    from sextant.rolecards import RoleCardError, load_card

    packaged = Path(__file__).resolve().parents[1] / "src" / "sextant" / "resources" / "role-cards" / "R3-edge.yaml"
    legacy = tmp_path / "R3-edge.yaml"
    legacy.write_text(packaged.read_text() + "boundary_band: 0.1\nboundary_reruns: 3\n")
    with pytest.raises(RoleCardError, match="first wire protocol") as info:
        load_card(legacy)
    assert "boundary_band" in str(info.value)
    assert "rename it to `reruns`" in str(info.value)
    assert load_card(packaged).id == "R3"


# ---------------------------------------------------------------- gaps found by re-running the attacks


def test_rerun_a_duplicated_unfavourable_run_cannot_hide_from_release(certified: dict[str, Any], tmp_path: Path) -> None:
    """Appending byte-identical copies of a disagreeing run's rows made them ambiguous, and the run vanished."""
    decision = tmp_path / "d"
    _ask_certified(certified, decision, caller=noul_series({"relevance": [0.02] * 4, "relevance_neg": [0.98] * 4}))
    ledger_file = decision / "jev-calls.jsonl"
    unfavourable = ledger_file.read_text()
    result, ledger = _ask_certified(certified, decision)
    with ledger_file.open("a") as f:
        f.write(unfavourable)  # a raw append; no existing byte changes
    index = ledger.call_index()
    estimate = result.estimates["relevance"].estimate
    messages = _release_of(certified, estimate, index)
    assert any("more than once" in m for m in messages), messages
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    assert any("more than once" in m for m in fails(check_all(graph, index, {"R3": certified["card"]}).findings))


def test_rerun_a_malformed_second_run_is_a_release_finding_not_a_crash(certified: dict[str, Any], tmp_path: Path) -> None:
    from sextant.graph import check_release, derive_edge_states

    decision = tmp_path / "d"
    _ask_certified(certified, decision, caller=noul_series({"relevance": [0.97] * 4, "relevance_neg": [0.03] * 4}))
    result, ledger = _ask_certified(certified, decision)
    estimate = result.estimates["relevance"].estimate
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    assert check_release(graph, ledger.call_index(), {"R3": certified["card"]}, certified["root"]) == []
    for field_name, bad in (("run_protocol", "reruns=3"), ("question_keys", 7), ("response", ["x"])):
        index = copy.deepcopy(ledger.call_index())
        other = next(r for r in index.values() if r["run_id"] != estimate.run_id)
        other[field_name] = bad
        release = [f.message for f in check_release(graph, index, {"R3": certified["card"]}, certified["root"])]
        assert release, field_name
        states = derive_edge_states(graph, index, {"R3": certified["card"]}, certified["root"])
        assert "direct_contribution_supported" not in states.values(), field_name


def test_rerun_generate_without_a_ledger_audits_against_an_empty_one(tmp_path: Path) -> None:
    from sextant.state import GateError, assert_gate, evaluate_gate, init_state

    estimate, _, _ = _r3_judgment(tmp_path, 0.97)
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    state = init_state(tmp_path / "s", "noledger", created="2026-09-22T00:00:00Z")
    issued = evaluate_gate(state, "generate", graph, None)
    assert issued.status.value == "FAIL" and "not in jev-calls ledger" in (issued.note or "")
    with pytest.raises(GateError):
        assert_gate(state, "generate", graph=graph, call_index=None)


def test_rerun_duplicate_ids_built_in_memory_fail_and_are_never_saved(tmp_path: Path) -> None:
    graph = DecisionGraph.model_validate(decomposition_graph(None))
    doubled = graph.model_copy(deep=True)
    doubled.nodes.append(doubled.nodes[0].model_copy(deep=True))
    assert any(f.check == "unique" for f in check_all(doubled).findings)
    with pytest.raises(ValueError, match="would not load back"):
        save_graph(doubled, tmp_path / "d")
    assert not (tmp_path / "d" / "graph.v1.json").exists()


def test_rerun_a_node_judgment_must_be_asked_over_the_node_answer_space(tmp_path: Path) -> None:
    card = choice_card()
    ledger = Ledger(tmp_path / "d")
    answer = Scripted(_choice_answer({"billing": 0.8, "technical": 0.1, "none_fits": 0.1}, "billing"))
    result = client(card, answer, ledger).ask("RC", {"ticket": "I was charged twice."})
    estimate = dump(result.estimates["department"].estimate)
    factor = fx_node("factor", "Which team owns the ticket?",
                     resolution={"criteria": "the owning team confirms", "source": "ticket system"},
                     answer_space={"primitive": "choice", "options": {"billing": None, "legal": None, "sales": None, "none": None},
                                   "escape_option": "none"})
    factor["estimate"] = estimate
    question = fx_node("question", "Who should handle this ticket?")
    residual = fx_node("factor", "other components not listed", residual=True)
    data = base_graph([question, factor, residual],
                      [fx_edge("decomposes_into", question, factor), fx_edge("decomposes_into", question, residual)])
    graph = DecisionGraph.model_validate(data)
    messages = fails(check_all(graph, ledger.call_index(), {"RC": card}).findings)
    assert any("was asked over options" in m for m in messages), messages


def test_rerun_an_edited_threshold_on_a_judgment_is_a_finding(certified: dict[str, Any], tmp_path: Path) -> None:
    result, ledger = _ask_certified(certified, tmp_path / "d")
    estimate = result.estimates["relevance"].estimate
    assert estimate.threshold_ref is not None
    cards, root, index = {"R3": certified["card"]}, certified["root"], ledger.call_index()
    genuine = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    assert fails(check_all(genuine, index, cards, root).findings) == []
    edited = dump(estimate)
    edited["threshold_ref"]["value"] = 0.01
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": edited}))
    messages = fails(check_all(graph, index, cards, root).findings)
    assert any("no issued certificate" in m for m in messages), messages


def test_rerun_the_build_command_takes_the_split_fractions(tmp_path: Path) -> None:
    from sextant import __main__ as cli
    from sextant import calibrate as cal

    # Exercise the retained historical split parser in isolated offline data,
    # not a current V1 authorization or public write command.

    items = tmp_path / "items.json"
    items.write_text(json.dumps(_cal_items(40)))
    root = tmp_path / "calibration-v2"
    assert cli.legacy_main(["calibrate", "build", "R3", "--items", str(items), "--root", str(root),
                     "--holdout-frac", "0.5", "--dev-frac", "0.1", "--release-frac", "0.1"]) == 0
    splits = {it.split for it in cal.load_items("R3", root)}
    assert {"holdout", "development", "release_holdout"} <= splits, splits


@pytest.mark.parametrize("link", ["group_id", "lineage"])
def test_rerun_a_case_link_declared_after_certification_invalidates(certified: dict[str, Any], tmp_path: Path, link: str) -> None:
    """A confirming case later joined to a fit case stayed calibrated until someone refitted."""
    import shutil

    from sextant.certificates import certificate_state

    root = tmp_path / "root"
    shutil.copytree(certified["root"], root)
    cert = certified["cert"]
    confirming, fitted = cert.confirming[0]["item_id"], cert.fit_items[0]
    golden = root / "R3" / "golden.jsonl"
    rows = [json.loads(line) for line in golden.read_text().splitlines() if line.strip()]
    for row in rows:
        if row["id"] in (confirming, fitted):
            row[link] = "joined-after-certification"
    golden.write_text("".join(json.dumps(r) + "\n" for r in rows))
    verdict = certificate_state(certified["card"], "relevance", model_id="jev-1.13.0", transport="controlled:scripted",
                                language="en", root=root, asked_keys=["relevance", "relevance_neg"], input_language="en")
    assert verdict.state == "invalidated" and "spans" in verdict.reason, verdict.reason


def test_rerun_an_unfit_card_never_writes_the_packaged_role_card() -> None:
    from sextant import calibrate as cal
    from sextant.config import packaged_rolecards_dir
    from sextant.rolecards import load_card

    path = packaged_rolecards_dir() / "R3-edge.yaml"
    before = path.read_bytes()
    card = load_card(path)
    unfit = [cal.FitResult("relevance", None, "unfit", reason="nothing reached the target")]
    reason = cal.degrade_if_needed(card, path, unfit, load_config())
    assert reason is not None and "read-only" in reason
    assert path.read_bytes() == before


def test_rerun_a_malformed_row_in_the_cited_run_is_named_not_raised(tmp_path: Path) -> None:
    estimate, ledger, card = _r3_judgment(tmp_path, 0.97)
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(estimate)}))
    index = copy.deepcopy(ledger.call_index())
    next(iter(index.values()))["run_protocol"] = "reruns=3"
    messages = fails(check_all(graph, index, {"R3": card}).findings)
    assert any("malformed call row" in m for m in messages), messages
    assert not any("check raised" in m for m in messages), messages


# ---------------------------------------------------------------- certificates and release under attack
#
# Each test is one executed counterexample, rebuilt from the script that found it.


def _copy_root(certified: dict[str, Any], tmp_path: Path) -> Path:
    import shutil

    root = tmp_path / "calibration-v2"
    shutil.copytree(certified["root"], root)
    return root


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _state(card: RoleCard, root: Path, *, transport: str = "controlled:scripted", input_language: str | None = "en",
           asked: list[str] | None = None) -> Any:
    from sextant.certificates import certificate_state

    return certificate_state(card, "relevance", model_id="jev-1.13.0", transport=transport, language="en",
                             input_language=input_language, root=root, asked_keys=asked or ["relevance", "relevance_neg"])


def test_relabelling_the_transport_in_responses_alone_does_not_certify(tmp_path: Path) -> None:
    from sextant import calibrate as cal
    from sextant.wire import identity_keys, measurement_fingerprint

    card, root = _calibrated(tmp_path, 1000)
    items = {i.id: i for i in cal.load_items(card.id, root)}
    path = root / "R3" / "responses.jsonl"
    rows = _rows(path)
    for r in rows:
        r["transport"] = "typesafe"
        r["measurement_conditions"] = cal.conditions_for(card, r["key"], model_id=r["model_id"], transport="typesafe", language=r["language"])
        r["protocol_fingerprint"] = measurement_fingerprint(card, keys=identity_keys(card, r["key"]), model_id=r["model_id"],
                                                            transport="typesafe", language=r["language"], pack=items[r["item_id"]].pack)
    _write_rows(path, rows)
    with pytest.raises((cal.CalibrationProtocolError, cal.CalibrationEvidenceError)):
        cal.fit_card(card, target=0.9, root=root)
    assert _state(card, root, transport="typesafe").state != "calibrated"


def test_a_response_pointed_at_another_items_run_does_not_certify(tmp_path: Path) -> None:
    import shutil

    from sextant import calibrate as cal

    card, root = _calibrated(tmp_path, 1000, error_every=10)
    honest_root = tmp_path / "honest"
    shutil.copytree(root, honest_root)  # the honest fit spends its own copy's holdout, not this one
    honest = {r.key: r for r in cal.fit_card(card, target=0.85, root=honest_root, write=False)}["relevance"]
    assert honest.deployable is False
    items = {i.id: i for i in cal.load_items(card.id, root)}
    truth = {i: "salary" in it.pack["child_question"] for i, it in items.items()}
    path = root / "R3" / "responses.jsonl"
    rows = _rows(path)
    donor = {}
    for r in rows:
        if items[r["item_id"]].split == "fit" and (r["p"] >= 0.5) == truth[r["item_id"]]:
            donor.setdefault(truth[r["item_id"]], r)
    for r in rows:
        if items[r["item_id"]].split == "holdout" and (r["p"] >= 0.5) != truth[r["item_id"]]:
            for f in ("run_id", "call_ids", "p", "stability", "contested", "confidence"):
                r[f] = donor[truth[r["item_id"]]][f]
    _write_rows(path, rows)
    with pytest.raises(cal.CalibrationEvidenceError, match="pack|run"):
        cal.fit_card(card, target=0.85, root=root)
    assert _state(card, root).state != "calibrated"


def test_a_certificate_naming_only_the_right_cases_is_invalidated(tmp_path: Path) -> None:
    from sextant import calibrate as cal
    from sextant.certificates import (
        CALIBRATION_PROTOCOL_VERSION,
        STANDARD_ID,
        Certificate,
        append_certificate,
        confirm,
        family_of,
    )
    from sextant.wire import WIRE_PROTOCOL_VERSION

    card, root = _calibrated(tmp_path, 1600, error_every=10)  # enough right cases for the forged set to meet the standard
    honest = {r.key: r for r in cal.fit_card(card, target=0.85, root=root)}["relevance"]
    assert honest.deployable is False
    d = root / "R3"
    fit_doc = json.loads((d / "fit.json").read_text())
    rows = cal._joined_rows(card, root, cal.SPLITS)
    by_id = {r["item_id"]: r for r in rows}
    keep = [c for c in honest.confirming if (by_id[c["item_id"]]["p"] >= 0.5) == by_id[c["item_id"]]["gold"]]
    conf = confirm([{**by_id[c["item_id"]], "group": c["group"]} for c in keep], honest.threshold,
                   permits_on=True, family_size=1, attempt=honest.attempt)
    assert conf.deployable
    fit_rows = [r for r in rows if r["split"] == "fit" and not r.get("invalid")]
    used = [r for r in rows if r["item_id"] in {c["item_id"] for c in keep} or (r["split"] == "fit" and not r.get("invalid"))]
    first = rows[0]
    append_certificate(root, "R3", Certificate(
        key="relevance", status="validated", value=honest.threshold, reason=conf.reason,
        experiment_id=fit_doc["experiment_id"], dataset_version=fit_doc["dataset_version"],
        responses_version=fit_doc["responses_version"], calibration_protocol_version=CALIBRATION_PROTOCOL_VERSION,
        wire_protocol_version=WIRE_PROTOCOL_VERSION, standard=STANDARD_ID,
        measurement_conditions=fit_doc["measurement_conditions"]["relevance"], model_id=first["model_id"],
        transport=first["transport"], language=first["language"], input_language="en", permits_on=True,
        asked_keys=first["asked_keys"], family=family_of(card), attempt=honest.attempt,
        alpha_per_bound=conf.alpha_per_bound, target=0.85, fit_items=sorted(r["item_id"] for r in fit_rows),
        confirming=conf.confirming, counts={**conf.counts(), "fit": len(fit_rows)}, bounds=conf.bounds(),
        golden_rows=len(cal.load_items(card.id, root)),
        digests=cal._digests(card, root, "relevance", used, len(cal.load_items(card.id, root))),
    ))
    for r in fit_doc["results"]:
        if r["key"] == "relevance":
            r["deployable"] = True
    (d / "fit.json").write_text(json.dumps(fit_doc, indent=2))
    verdict = _state(card, root)
    assert verdict.state == "invalidated", verdict.reason
    from sextant.calibrate import verify_certificate
    from sextant.certificates import effective_certificate

    problems = verify_certificate(card, effective_certificate(root, card.id, "relevance"), root)
    assert any("representatives of every eligible case" in p for p in problems), problems
    assert any("fit record disagrees" in p for p in problems), problems


def test_confirmation_abstains_where_the_runtime_abstains() -> None:
    from sextant.certificates import confirm
    from sextant.wire import decision

    assert decision(0.8275, 0.5, contested=True) is None
    rows = [{"item_id": f"h{i}", "key": "relevance", "split": "holdout", "group": f"h{i}", "lang": "en", "p": 0.8275,
             "contested": True, "gold": True, "resolved_by": "deterministic_truth"} for i in range(50)]
    rows += [{"item_id": f"k{i}", "key": "relevance", "split": "holdout", "group": f"k{i}", "lang": "en", "p": 0.9,
              "contested": False, "gold": False, "resolved_by": "deterministic_truth"} for i in range(5)]
    conf = confirm(rows, 0.5, permits_on=True, family_size=1)
    assert conf.automatic == 5 and conf.abstentions == 50
    assert conf.releases == 5 and conf.bad_releases == 5 and not conf.deployable


def test_contested_runs_are_abstentions_in_the_certificate(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    def answer(key: str, q: dict[str, Any], index: int) -> dict[str, Any]:
        text = q["instructions"]
        related = "salary" in text
        case = int(text.rsplit("case ", 1)[1].split(")")[0]) if related else 1
        # every fourth related case straddles 0.5 across its re-runs
        p = (0.40 if index % 4 == 3 else 0.97) if related and case % 8 == 0 else (0.97 if related else 0.03)
        return {"type": "noul", "noul": 1 - p if key.endswith("_neg") else p}

    card = r3_card()
    root = tmp_path / "calibration-v2"
    cal.build(card.id, _cal_items(1000), root=root)
    cal.record_truth(card.id, [{"item_id": i.id, "key": i.key, "label": "salary" in i.pack["child_question"]}
                               for i in cal.load_items(card.id, root)],
                     card=card, source="generator", method="synthetic", root=root)
    cal.resolve_gold(card.id, root)
    cal.measure(card, client(card, Scripted(answer), cal.measurement_ledger(card.id, root), root), cal.load_items(card.id, root), root=root)
    rows = {r["item_id"]: r for r in _rows(root / "R3" / "responses.jsonl")}
    fit = {r.key: r for r in cal.fit_card(card, target=0.9, root=root)}["relevance"]
    contested = sum(1 for c in fit.confirming if rows[c["item_id"]].get("contested"))
    assert contested > 0
    assert fit.n_abstentions >= contested


def _pair_items() -> list[dict[str, Any]]:
    out = []
    for i in range(1000):
        related = i % 2 == 0
        en = f"{CHILD} (case {i})" if related else f"What is the grain size of basalt? (case {i})"
        out.append({"key": "relevance", "pack": r3_pack(child=en), "expected": related, "lang": "en",
                    "lineage": f"case-{i}", "pair_id": f"pair-{i}"})
        zh = f"基本工资能覆盖我的每月开支吗？(case {i})" if related else f"玄武岩样本的粒度是多少？(case {i})"
        out.append({"key": "relevance", "pack": r3_pack(parent="我应该接受这个工作吗？", child=zh), "expected": related,
                    "lang": "zh", "lineage": f"case-{i}", "pair_id": f"pair-{i}"})
    return out


def test_a_certificate_covers_only_the_input_language_it_was_confirmed_on(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    def answer(key: str, q: dict[str, Any], index: int) -> dict[str, Any]:
        text = q["instructions"]
        related = "salary" in text or "工资" in text
        if any("一" <= ch <= "鿿" for ch in text):
            related = not related  # wrong on every Chinese input
        p = 0.95 if related else 0.05
        return {"type": "noul", "noul": 1 - p if key.endswith("_neg") else p}

    card = r3_card()
    root = tmp_path / "calibration-v2"
    cal.build(card.id, _pair_items(), root=root)
    items = cal.load_items(card.id, root)
    truth = {i.id: ("salary" in i.pack["child_question"] or "工资" in i.pack["child_question"]) for i in items}
    cal.record_truth(card.id, [{"item_id": i.id, "key": i.key, "label": truth[i.id]} for i in items],
                     card=card, source="generator", method="synthetic", root=root)
    cal.resolve_gold(card.id, root)
    cal.measure(card, client(card, Scripted(answer), cal.measurement_ledger(card.id, root), root), items, root=root)
    cal.fit_card(card, target=0.9, root=root)
    from sextant.certificates import effective_certificate

    cert = effective_certificate(root, card.id, "relevance")
    assert cert is not None and cert.input_language == "en"
    assert _state(card, root).state == "calibrated", _state(card, root).reason
    assert _state(card, root, input_language="zh").state == "out_of_domain"
    assert _state(card, root, input_language=None).state == "out_of_domain"
    zh = next(i for i in items if i.lang == "zh")
    declared = client(card, Scripted(answer), Ledger(tmp_path / "d0"), root).ask("R3", zh.pack, keys=["relevance", "relevance_neg"],
                                                                                 input_language="zh")
    assert declared.estimates["relevance"].estimate.calibration_status.state == "out_of_domain"
    res = client(card, Scripted(answer), Ledger(tmp_path / "d1"), root).ask("R3", zh.pack, keys=["relevance", "relevance_neg"],
                                                                           input_language="en")
    assert res.estimates["relevance"].estimate.calibration_status.state != "calibrated", "Chinese text declared as English"
    assert res.passes("relevance") is None


def test_doctor_rm1_needs_the_composition_it_runs_and_a_decided_rule() -> None:
    from sextant.doctor import judge_rm1
    from sextant.meta import RM1Row

    t = 0.8
    thresholds = {"scoping_defined": t, "asks_computation": t, "answerable_from_fields": t, "criteria_consistent": t}
    row = RM1Row(card_id="R3", key="relevance", scoping_defined=0.95, asks_computation=0.6,
                 answerable_from_fields=0.95, criteria_consistent=0.95, has_scoping_word=True, contested=False)
    assert row.undecided(thresholds) == ["asks_computation"]
    contested = RM1Row(card_id="R3", key="relevance", scoping_defined=0.95, asks_computation=0.05,
                       answerable_from_fields=0.95, criteria_consistent=0.95, has_scoping_word=True, contested=True)
    assert contested.undecided(thresholds)
    assert judge_rm1(executed=True, fitted=True, flagged=False, rows=1, undecided=True) != "PASS"


def test_doctor_counts_a_threshold_only_for_the_rm1_composition(certified: dict[str, Any]) -> None:
    from sextant.doctor import fitted_flag_thresholds

    c = client(certified["card"], noul_series({}), None, certified["root"])
    assert fitted_flag_thresholds(certified["card"], c) == {}


def test_an_unfavourable_run_under_another_label_still_blocks_release(certified: dict[str, Any], tmp_path: Path) -> None:
    from sextant.graph import check_release

    card = certified["card"]
    ledger = Ledger(tmp_path / "d")
    config = copy.deepcopy(load_config())
    config["jev"]["requests_per_minute"] = 1_000_000
    probe = JevClient(config, {"R3": card}, ledger=ledger, caller=noul_series({"relevance": [0.02] * 4, "relevance_neg": [0.98] * 4}),
                      seed=5, transport="controlled:probe", calibration_root=certified["root"])
    probe.ask("R3", r3_pack(), keys=["relevance", "relevance_neg"], input_language="en")
    good, _ = _ask_certified(certified, tmp_path / "d")
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(good.estimates["relevance"].estimate)}))
    messages = [f.message for f in check_release(graph, ledger.call_index(), {"R3": card}, certified["root"])]
    assert any("disagree" in m or "runs on this input" in m for m in messages), messages


def test_an_agreeing_run_of_another_composition_does_not_block_release(certified: dict[str, Any], tmp_path: Path) -> None:
    from sextant.graph import check_release

    decision = tmp_path / "d"
    c = client(certified["card"], noul_series({"relevance": [0.97] * 4}), Ledger(decision), certified["root"])
    first = c.ask("R3", r3_pack(), keys=["relevance"], input_language="en")
    assert first.estimates["relevance"].estimate.calibration_status.state == "out_of_domain"
    good, ledger = _ask_certified(certified, decision)
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(good.estimates["relevance"].estimate)}))
    assert check_release(graph, ledger.call_index(), {"R3": certified["card"]}, certified["root"]) == []


def test_a_torn_measurement_ledger_invalidates_instead_of_raising(certified: dict[str, Any], tmp_path: Path) -> None:
    root = _copy_root(certified, tmp_path)
    with (root / "R3" / "ledger" / "jev-calls.jsonl").open("a") as f:
        f.write('{"seq": 99999, "id": "jc_torn"')
    verdict = _state(certified["card"], root)
    assert verdict.state == "invalidated" and "cannot be read" in verdict.reason, verdict.reason
    res = client(certified["card"], noul_series({"relevance": [0.97] * 4, "relevance_neg": [0.03] * 4}),
                 Ledger(tmp_path / "d"), root).ask("R3", r3_pack(), keys=["relevance", "relevance_neg"], input_language="en")
    assert res.estimates["relevance"].estimate.calibration_status.state == "invalidated"


def test_a_failed_prerequisite_supersedes_the_later_receipt(certified: dict[str, Any], tmp_path: Path) -> None:
    from sextant.certificates import Certificate, append_certificate
    from sextant.state import evaluate_gate, init_state, status_text

    root = _copy_root(certified, tmp_path)
    good, ledger = _ask_certified({**certified, "root": root}, tmp_path / "d")
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(good.estimates["relevance"].estimate)}))
    cards = {"R3": certified["card"]}
    state = init_state(tmp_path / "d", "stage", created="2026-09-22T00:00:00Z")
    assert evaluate_gate(state, "generate", graph, ledger.call_index(), cards=cards, calibration_root=root).status.value == "PASS"
    assert evaluate_gate(state, "verify", graph, ledger.call_index(), cards=cards, calibration_root=root).status.value == "PASS"
    assert state.stage == "verify"
    append_certificate(root, "R3", Certificate(key="relevance", status="revoked", reason="test"))
    assert evaluate_gate(state, "generate", graph, ledger.call_index(), cards=cards, calibration_root=root).status.value == "FAIL"
    assert state.stage == "init"
    assert "verify" not in state.gates
    assert state.notes["superseded_gates"][-1]["gate"] == "verify"
    assert not any(line.split()[:1] == ["verify"] and "PASS" in line for line in status_text(state).splitlines())


def test_fit_and_verification_resolve_truth_the_same_way(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    card = r3_card()
    root = tmp_path / "calibration-v2"
    cal.build(card.id, _cal_items(1000), root=root)
    items = cal.load_items(card.id, root)
    cal.record_truth(card.id, [{"item_id": i.id, "key": i.key, "label": "salary" in i.pack["child_question"]} for i in items],
                     card=card, source="generator", method="synthetic", root=root)
    # an unrelated holdout case: its automatic answer is "no", so the untruthed case costs R, not R+
    h = next(i for i in items if i.split == "holdout" and "salary" not in i.pack["child_question"])
    cal.record_truth(card.id, [{"item_id": h.id, "key": h.key, "label": "salary" not in h.pack["child_question"]}],
                     card=card, kind="human_adjudication", source="reviewer", method="read the pair", root=root)
    cal.resolve_gold(card.id, root)
    gold = {(g["item_id"], g["key"]): g for g in _rows(root / "R3" / "gold.jsonl")}
    assert gold[(h.id, h.key)]["gold"] is None and gold[(h.id, h.key)]["status"] == "unresolved-truth-conflict"
    cal.measure(card, client(card, _cal_caller(), cal.measurement_ledger(card.id, root), root), items, root=root)
    fit = {r.key: r for r in cal.fit_card(card, target=0.9, root=root)}["relevance"]
    assert fit.deployable, fit.reason
    assert _state(card, root).state == "calibrated", _state(card, root).reason


def test_a_verify_receipt_does_not_survive_a_graph_change(certified: dict[str, Any], tmp_path: Path) -> None:
    from sextant.state import GateError, assert_gate, evaluate_gate, init_state

    good, ledger = _ask_certified(certified, tmp_path / "d")
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(good.estimates["relevance"].estimate)}))
    cards, root, index = {"R3": certified["card"]}, certified["root"], ledger.call_index()
    state = init_state(tmp_path / "d", "receipt", created="2026-09-22T00:00:00Z")
    evaluate_gate(state, "generate", graph, index, cards=cards, calibration_root=root)
    evaluate_gate(state, "verify", graph, index, cards=cards, calibration_root=root)
    assert_gate(state, "verify", graph=graph, call_index=index, cards=cards, calibration_root=root)
    changed = graph.model_copy(deep=True)
    changed.metadata["prediction"] = {"salary": "holds"}
    with pytest.raises(GateError, match="does not match"):
        assert_gate(state, "verify", graph=changed, call_index=index, cards=cards, calibration_root=root)


def test_observation_receipts_name_the_transports_behind_them(certified: dict[str, Any], tmp_path: Path) -> None:
    from sextant.state import evaluate_gate, init_state, status_text

    good, ledger = _ask_certified(certified, tmp_path / "d")
    graph = DecisionGraph.model_validate(decomposition_graph({"relevance": dump(good.estimates["relevance"].estimate)}))
    state = init_state(tmp_path / "d", "transports", created="2026-09-22T00:00:00Z")
    g = evaluate_gate(state, "generate", graph, ledger.call_index(), cards={"R3": certified["card"]}, calibration_root=certified["root"])
    assert g.receipt is not None and g.receipt.input_revisions["evidence_transports"] == "controlled:scripted"
    assert "controlled:scripted" in status_text(state)


def test_flipping_invalid_in_responses_does_not_hide_wrong_answers(tmp_path: Path) -> None:
    """A stored `invalid` flag turned wrong holdout answers into abstentions."""
    from sextant import calibrate as cal

    card, root = _calibrated(tmp_path, 1000, error_every=10)
    items = {i.id: i for i in cal.load_items(card.id, root)}
    path = root / "R3" / "responses.jsonl"
    rows = _rows(path)
    for r in rows:
        if items[r["item_id"]].split == "holdout" and (r["p"] >= 0.5) != ("salary" in items[r["item_id"]].pack["child_question"]):
            r["invalid"], r["p"] = True, None
    _write_rows(path, rows)
    with pytest.raises(cal.CalibrationEvidenceError, match="invalid"):
        cal.fit_card(card, target=0.85, root=root)
    assert _state(card, root).state != "calibrated"


def test_a_favourable_extra_run_cannot_replace_the_measured_one(tmp_path: Path) -> None:
    """An extra honest run on a holdout pack, cited by an appended response."""
    from sextant import calibrate as cal
    from sextant.wire import identity_keys

    card, root = _calibrated(tmp_path, 1000, error_every=10)
    items = {i.id: i for i in cal.load_items(card.id, root)}
    path = root / "R3" / "responses.jsonl"
    rows = _rows(path)
    extra = client(card, _cal_caller(), cal.measurement_ledger(card.id, root), root)
    appended = []
    for r in rows:
        it = items[r["item_id"]]
        if it.split == "holdout" and (r["p"] >= 0.5) != ("salary" in it.pack["child_question"]):
            res = extra.ask(card.id, it.pack, keys=identity_keys(card, it.key), variant=f"cal:{it.id}", input_language="en")
            e = res.estimates["relevance"].estimate
            appended.append({**r, "p": e.p, "run_id": e.run_id, "call_ids": e.call_ids, "contested": e.contested})
    with path.open("a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in appended))
    with pytest.raises(cal.CalibrationEvidenceError, match="run"):
        cal.fit_card(card, target=0.85, root=root)
    assert _state(card, root).state != "calibrated"


def test_the_input_language_check_is_symmetric() -> None:
    from sextant.wire import input_language_of

    english = {"child_question": "Will the base salary cover my monthly expenses?"}
    chinese = {"child_question": "基本工资能覆盖我的每月开支吗？"}
    russian = {"child_question": "Покроет ли зарплата мои ежемесячные расходы?"}
    assert input_language_of(english, "en") == "en"
    assert input_language_of(chinese, "en") is None
    assert input_language_of(russian, "en") is None
    assert input_language_of(english, "zh") is None
    assert input_language_of(chinese, "zh") == "zh"
    assert input_language_of(english, None) is None
    assert input_language_of({"n": 3}, "en") == "en"


def test_observation_a_requested_holdout_split_is_refused(tmp_path: Path) -> None:
    from sextant import calibrate as cal

    items = _cal_items(10)
    items[0]["split"] = "holdout"
    with pytest.raises(cal.CalibrationSplitError, match="random"):
        cal.build("R3", items, root=tmp_path / "calibration-v2")
    items[0]["split"] = "fit"
    assert cal.build("R3", items, root=tmp_path / "calibration-v2")["added"] == 10
