from __future__ import annotations

import pytest
from pydantic import ValidationError

from sextant.schema import (
    LEGAL_ENDPOINTS,
    AnswerSpace,
    DecisionGraph,
    Edge,
    EdgeType,
    Node,
    NodeKind,
    Primitive,
    edge_id,
    export_json_schema,
    node_id,
)
from sextant.selftest import (
    base_graph,
    edge,
    fx_schema_endpoint,
    fx_schema_resolution,
    node,
    sample_estimate,
)

pytestmark = pytest.mark.unit


def test_node_id_is_content_addressed() -> None:
    a = node_id(NodeKind.factor, "salary covers expenses")
    b = node_id("factor", " salary covers expenses ")
    assert a == b
    assert a != node_id(NodeKind.factor, "salary covers expenses", "with a definition")
    assert edge_id(EdgeType.influences, "n_a", "n_b") != edge_id(EdgeType.influences, "n_b", "n_a")


def test_every_edge_type_has_a_legal_matrix() -> None:
    assert set(LEGAL_ENDPOINTS) == set(EdgeType)
    for pairs in LEGAL_ENDPOINTS.values():
        assert pairs


def test_illegal_endpoint_rejected_and_legal_accepted() -> None:
    with pytest.raises(ValidationError, match="not legal"):
        DecisionGraph.model_validate(fx_schema_endpoint(True))
    DecisionGraph.model_validate(fx_schema_endpoint(False))


def test_estimate_without_resolution_rejected() -> None:
    with pytest.raises(ValidationError, match="resolution criteria"):
        DecisionGraph.model_validate(fx_schema_resolution(True))
    DecisionGraph.model_validate(fx_schema_resolution(False))


def test_value_node_never_carries_estimate() -> None:
    with pytest.raises(ValidationError, match="value node never"):
        Node.model_validate(node("value", "autonomy matters", estimate=sample_estimate()))


def test_choice_space_needs_escape_and_caps_options() -> None:
    with pytest.raises(ValidationError, match="escape"):
        AnswerSpace(primitive=Primitive.choice, options={"a": None, "b": None})
    with pytest.raises(ValidationError, match="32"):
        AnswerSpace(primitive=Primitive.choice, options={f"o{i}": None for i in range(33)} | {"none": None}, escape_option="none")
    AnswerSpace(primitive=Primitive.choice, options={"a": None, "none": None}, escape_option="none")


def test_score_space_levels() -> None:
    with pytest.raises(ValidationError):
        AnswerSpace(primitive=Primitive.score, levels=["only one"])
    AnswerSpace(primitive=Primitive.score, levels=["low", "high"])


def test_decomposition_edge_needs_method() -> None:
    q = node("question", "Should I take the offer?")
    f = node("factor", "salary covers expenses")
    raw = edge("decomposes_into", q, f)
    raw.pop("method")
    with pytest.raises(ValidationError, match="method"):
        Edge.model_validate(raw)


def test_conflict_needs_two_members() -> None:
    with pytest.raises(ValidationError, match="exactly two"):
        Node.model_validate(node("conflict", "rent vs savings", members=["n_1"]))


def test_duplicate_ids_rejected() -> None:
    f = node("factor", "salary covers expenses")
    with pytest.raises(ValidationError, match="duplicate node ids"):
        DecisionGraph.model_validate(base_graph([f, dict(f)], []))


def test_exported_schema_carries_anchors() -> None:
    s = export_json_schema()
    assert s["$schema"].endswith("2020-12/schema")
    assert "x-legal-endpoints" in s and "is_value_for" in s["x-legal-endpoints"]
    assert "factor" in s["x-resolvable-kinds"]
