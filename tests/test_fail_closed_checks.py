"""Counterexamples: missing evidence must not read as a pass.

These tests name the current entry points. On the pre-fix tree they fail or
raise; after the fix they pass because the same calls fail closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from sextant.graph import (
    check_all,
    check_reified,
    check_release,
    derive_edge_states,
    diff_graphs,
    save_graph,
    world_view,
)
from sextant.schema import (
    DecisionGraph,
    Estimate,
    Gate,
    GateAuthority,
    GateReceipt,
    GateStatus,
    receipt_token,
)
from sextant.selftest import base_graph, edge, fx_ledger_edge_verification, node, sample_estimate
from sextant.state import (
    GATES_IN_ORDER,
    RELEASE_GATES,
    STRUCTURE_CHECKER,
    GateError,
    assert_gate,
    evaluate_gate,
    init_state,
    stamp,
)

pytestmark = pytest.mark.unit


def test_empty_gate_token_cannot_pass(tmp_path: Path) -> None:
    """verify=PASS with token='' is asserted successfully."""
    state = init_state(tmp_path, "demo", created="2026-09-18T00:00:00Z")
    with pytest.raises(ValidationError):
        state.gates["verify"] = Gate.model_validate(
            {"status": "PASS", "at": "2026-09-18T00:00:00Z", "token": ""}
        )
        assert_gate(state, "verify")
        raise AssertionError("empty token was treated as a pass")
    receipt = GateReceipt(
        protocol_version="sextant-jev-wire/1",
        checker="sextant.graph.check_all",
        input_revisions={"graph_version": "1"},
        artifact_hash="artifact-fixture",
        result="PASS",
    )
    with pytest.raises(ValidationError):
        Gate(
            authority=GateAuthority.checker,
            status=GateStatus.PASS,
            at="2026-09-18T00:00:00Z",
            token="",
            receipt=receipt,
        )


def test_invented_token_cannot_pass_render(tmp_path: Path) -> None:
    """A fresh state stamps render with token='made-up' and assert succeeds."""
    state = init_state(tmp_path, "demo", created="2026-09-18T00:00:00Z")
    try:
        stamp(state, "render", "PASS", "made-up")
    except GateError:
        return
    with pytest.raises(GateError):
        assert_gate(state, "render")


def test_verified_edge_without_evidence_does_not_pass() -> None:
    """A real edge set to verified, with a residual sibling and no measurement, passes check_all."""
    question = node("question", "Should I take the offer?")
    real = node("factor", "salary covers expenses")
    residual = node("factor", "other components not listed", residual=True)
    claimed = edge("decomposes_into", question, real)
    claimed["status"] = "verified"
    data = base_graph(
        [question, real, residual],
        [claimed, edge("decomposes_into", question, residual)],
    )
    try:
        graph = DecisionGraph.model_validate(data)
    except ValidationError:
        return
    assert not check_all(graph, {}).ok


def test_cyclic_decomposition_returns_findings() -> None:
    """root→b→c→b on decomposes_into, each sibling set residual, makes check_all raise RecursionError."""
    root = node("question", "Should I take the offer?")
    left = node("question", "Is the salary enough?")
    right = node("question", "Is the commute acceptable?")
    root_residual = node("question", "other parts of the offer question", residual=True)
    left_residual = node("question", "other parts of the salary question", residual=True)
    right_residual = node("question", "other parts of the commute question", residual=True)
    graph = DecisionGraph.model_validate(
        base_graph(
            [root, left, right, root_residual, left_residual, right_residual],
            [
                edge("decomposes_into", root, left),
                edge("decomposes_into", root, root_residual),
                edge("decomposes_into", left, right),
                edge("decomposes_into", left, left_residual),
                edge("decomposes_into", right, left),
                edge("decomposes_into", right, right_residual),
            ],
        )
    )
    result = check_all(graph, {})
    assert any(f.check == "acyclic" and f.severity == "FAIL" for f in result.findings)


def test_world_view_has_no_dangling_edge() -> None:
    """A legal option→factor influences edge is emitted while the option is not in the view."""
    option = node("option", "take the offer")
    factor = node("factor", "salary covers expenses")
    graph = DecisionGraph.model_validate(base_graph([option, factor], [edge("influences", option, factor)]))
    view = world_view(graph)
    present = {item["id"] for item in view["nodes"]}
    for item in view["edges"]:
        assert item["from_id"] in present and item["to_id"] in present


def _changed(diff: dict, field: str) -> bool:
    return any(field in item for item in diff["changed"])


def test_diff_sees_edge_verification_strength_and_method() -> None:
    """The same edge id with different verification, strength, or method is invisible to diff_graphs."""
    question = node("question", "Should I take the offer?")
    factor = node("factor", "salary covers expenses")
    residual = node("factor", "other components not listed", residual=True)
    before_edge = edge("decomposes_into", question, factor)
    after_edge = edge("decomposes_into", question, factor)
    after_edge["verification"] = {"relevance": sample_estimate()}
    after_edge["strength"] = {"direction": "+", "prior_source": "user"}
    after_edge["method"] = {"ordered": True, "subtasks": [factor["id"]], "applicability": "when the letter is in hand"}
    nodes = [question, factor, residual]
    sibling = edge("decomposes_into", question, residual)
    before = DecisionGraph.model_validate(base_graph(nodes, [before_edge, sibling]))
    after = DecisionGraph.model_validate(base_graph(nodes, [after_edge, sibling]))
    diff = diff_graphs(before, after)
    assert _changed(diff, "verification")
    assert _changed(diff, "strength")
    assert _changed(diff, "method")


def test_diff_sees_weights_members_answer_space_and_policy() -> None:
    """Weights, evidence membership, answer space, and policy metadata do not show up in a diff."""
    factor = node("factor", "salary covers expenses")
    evidence = node("evidence", "offer letter, 2026-09-10")
    link = node("evidence_link", "letter supports salary", members=[evidence["id"], factor["id"]], direction="+")
    before = DecisionGraph.model_validate(
        base_graph(
            [factor, evidence, link],
            [edge("supports", link, factor), edge("cites", link, evidence)],
        )
    )
    after = before.model_copy(deep=True)
    after.metadata["weights"] = {"salary": 0.5}
    after.metadata["ranking"] = ["take the offer"]
    after.metadata["tripwires"] = ["walk away if salary drops"]
    after.metadata["prediction"] = {"salary": "holds"}
    for item in after.nodes:
        if item.kind.value == "evidence_link":
            item.members = [factor["id"]]
        if item.kind.value == "factor":
            item.answer_space = {"primitive": "noul"}  # type: ignore[assignment]
    diff = diff_graphs(before, after)
    assert _changed(diff, "weights")
    assert _changed(diff, "members")
    assert _changed(diff, "answer_space")
    assert _changed(diff, "ranking")
    assert _changed(diff, "tripwires")
    assert _changed(diff, "prediction")


def test_choice_probability_vector_must_be_a_distribution() -> None:
    """A choice estimate accepts a negative, non-normalized probability vector."""
    payload = sample_estimate()
    payload["primitive"] = "choice"
    payload["p"] = None
    payload["probabilities"] = {"a": -0.2, "b": 0.2}
    payload["choice"] = "a"
    with pytest.raises(ValidationError):
        Estimate.model_validate(payload)


def test_question_hash_must_match_its_text() -> None:
    """question.hash is not checked against the question text."""
    factor = node("factor", "salary covers expenses")
    data = base_graph([factor], [])
    data["question"]["hash"] = "q_not_the_text"
    with pytest.raises(ValidationError):
        DecisionGraph.model_validate(data)


def test_evidence_link_members_must_match_its_edges() -> None:
    """An evidence link's members and its cites/supports targets are not required to agree."""
    factor = node("factor", "salary covers expenses")
    evidence = node("evidence", "offer letter, 2026-09-10")
    other = node("evidence", "a different document")
    link = node(
        "evidence_link",
        "letter supports salary",
        members=[other["id"], factor["id"]],
        direction="+",
    )
    graph = DecisionGraph.model_validate(
        base_graph(
            [factor, evidence, other, link],
            [edge("supports", link, factor), edge("cites", link, evidence)],
        )
    )
    assert check_reified(graph)


def _draft() -> DecisionGraph:
    question = node("question", "Should I take the offer?")
    real = node("factor", "salary covers expenses")
    residual = node("factor", "other components not listed", residual=True)
    return DecisionGraph.model_validate(
        base_graph(
            [question, real, residual],
            [edge("decomposes_into", question, real), edge("decomposes_into", question, residual)],
        )
    )


def test_choice_must_belong_to_the_answer_space() -> None:
    """A choice judgment is not checked against the node's options."""
    payload = sample_estimate()
    payload["primitive"] = "choice"
    payload["p"] = None
    payload["probabilities"] = {"a": 0.5, "b": 0.5}
    payload["choice"] = "a"
    factor = node(
        "factor",
        "which department",
        estimate=payload,
        resolution={"criteria": "the offer names one department", "source": "offer letter"},
        answer_space={
            "primitive": "choice",
            "options": {"billing": None, "none": "none fits"},
            "escape_option": "none",
        },
    )
    with pytest.raises(ValidationError, match="answer space"):
        DecisionGraph.model_validate(base_graph([factor], []))


def test_user_judgment_is_not_a_checker_pass(tmp_path: Path) -> None:
    state = init_state(tmp_path, "demo", created="2026-09-18T00:00:00Z")
    recorded = stamp(state, "generate", "SKIP", "operator-note", note="approved without evidence")
    assert recorded.authority == GateAuthority.user_judgment
    assert recorded.status != GateStatus.PASS
    with pytest.raises(GateError, match="user judgment"):
        assert_gate(state, "generate")


def test_draft_saves_and_does_not_release(tmp_path: Path) -> None:
    graph = _draft()
    save_graph(graph, tmp_path)
    assert (tmp_path / "graph.v1.json").exists()
    assert check_release(graph, {})
    real = next(edge for edge in graph.edges if not graph.node(edge.to_id).residual)
    assert derive_edge_states(graph, {})[real.id] == "structurally_valid"
    state = init_state(tmp_path, "demo", created="2026-09-18T00:00:00Z")
    assert evaluate_gate(state, "generate", graph, {}).status == GateStatus.PASS
    issued = evaluate_gate(state, "verify", graph, {})
    assert "verify" in RELEASE_GATES and issued.status == GateStatus.FAIL
    with pytest.raises(GateError):
        assert_gate(state, "verify", graph=graph, call_index={})
    # every other gate belongs to a stage with no check yet and cannot be certified at all
    for name in GATES_IN_ORDER:
        if name not in ("generate", "verify"):
            with pytest.raises(GateError, match="no stage check"):
                evaluate_gate(state, name, graph, {})


def test_receipt_invalidates_when_the_graph_changes(tmp_path: Path) -> None:
    graph = _draft()
    state = init_state(tmp_path, "demo", created="2026-09-18T00:00:00Z")
    issued = evaluate_gate(state, "generate", graph, {})
    assert issued.status == GateStatus.PASS
    assert_gate(state, "generate", graph=graph, call_index={})
    changed = graph.model_copy(deep=True)
    changed.metadata["weights"] = {"salary": 1.0}
    with pytest.raises(GateError, match="does not match"):
        assert_gate(state, "generate", graph=changed, call_index={})


def test_stale_protocol_receipt_does_not_pass(tmp_path: Path) -> None:
    state = init_state(tmp_path, "demo", created="2026-09-18T00:00:00Z")
    receipt = GateReceipt(
        protocol_version="sextant-jev-wire/0",
        checker=STRUCTURE_CHECKER,
        input_revisions={"protocol_version": "sextant-jev-wire/0"},
        artifact_hash="artifact-fixture",
        result="PASS",
    )
    state.gates["generate"] = Gate(
        authority=GateAuthority.checker,
        status=GateStatus.PASS,
        at="2026-09-18T00:00:00Z",
        token=receipt_token(receipt),
        receipt=receipt,
    )
    with pytest.raises(GateError, match="protocol"):
        assert_gate(state, "generate")


def test_cycle_evaluation_records_a_failure(tmp_path: Path) -> None:
    root = node("question", "Should I take the offer?")
    left = node("question", "Is the salary enough?")
    right = node("question", "Is the commute acceptable?")
    root_residual = node("question", "other parts of the offer question", residual=True)
    left_residual = node("question", "other parts of the salary question", residual=True)
    right_residual = node("question", "other parts of the commute question", residual=True)
    graph = DecisionGraph.model_validate(
        base_graph(
            [root, left, right, root_residual, left_residual, right_residual],
            [
                edge("decomposes_into", root, left),
                edge("decomposes_into", root, root_residual),
                edge("decomposes_into", left, right),
                edge("decomposes_into", left, left_residual),
                edge("decomposes_into", right, left),
                edge("decomposes_into", right, right_residual),
            ],
        )
    )
    state = init_state(tmp_path, "demo", created="2026-09-18T00:00:00Z")
    issued = evaluate_gate(state, "generate", graph, {})
    assert issued.status == GateStatus.FAIL
    assert issued.authority == GateAuthority.checker


def test_a_replayed_judgment_alone_does_not_pass_a_release_gate(tmp_path: Path) -> None:
    """A model judgment that replays against its ledger is not external evidence.

    This test used to assert that such a judgment made the edge `externally_tested` and passed
    every gate. The judgment here is not even an R3 relevance decision about this edge; it
    replays, and still supports nothing. The calibrated positive path is in test_adversarial_counterexamples.
    """
    data = fx_ledger_edge_verification(False)
    graph = DecisionGraph.model_validate({key: value for key, value in data.items() if not key.startswith("_")})
    index = data["_ledger_index"]
    assert check_all(graph, index).ok, "the judgment itself replays against the ledger"
    real = next(item for item in graph.edges if item.verification is not None)
    assert derive_edge_states(graph, index)[real.id] not in ("externally_tested", "direct_contribution_supported")
    state = init_state(tmp_path, "demo", created="2026-09-18T00:00:00Z")
    assert evaluate_gate(state, "generate", graph, index).status == GateStatus.PASS
    assert evaluate_gate(state, "verify", graph, index).status == GateStatus.FAIL
    with pytest.raises(GateError):
        assert_gate(state, "verify", graph=graph, call_index=index)
