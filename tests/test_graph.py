from __future__ import annotations

from pathlib import Path

import pytest

from sextant.graph import (
    check_acyclic,
    check_all,
    check_budgets,
    check_reified,
    check_residuals,
    diff_graphs,
    latest_version,
    load_graph,
    render_diff,
    save_graph,
    world_view,
)
from sextant.schema import DecisionGraph
from sextant.selftest import fx_acyclic, fx_budget, fx_ledger, fx_reified, fx_residual

pytestmark = pytest.mark.unit


def _g(data: dict) -> DecisionGraph:  # type: ignore[type-arg]
    return DecisionGraph.model_validate({k: v for k, v in data.items() if not k.startswith("_")})


def test_acyclic_check() -> None:
    assert check_acyclic(_g(fx_acyclic(True)))
    assert not check_acyclic(_g(fx_acyclic(False)))


def test_residual_check() -> None:
    assert check_residuals(_g(fx_residual(True)))
    assert not check_residuals(_g(fx_residual(False)))


def test_budget_check_uses_tier() -> None:
    assert check_budgets(_g(fx_budget(True)))
    assert not check_budgets(_g(fx_budget(False)))


def test_reified_check() -> None:
    assert check_reified(_g(fx_reified(True)))
    assert not check_reified(_g(fx_reified(False)))


def test_check_all_with_ledger_index() -> None:
    data = fx_ledger(False)
    g = _g(data)
    assert check_all(g, data["_ledger_index"]).ok
    bad = fx_ledger(True)
    assert not check_all(_g(bad), bad["_ledger_index"]).ok


def test_save_never_overwrites_and_writes_views(tmp_path: Path) -> None:
    g = _g(fx_residual(False))
    p = save_graph(g, tmp_path)
    assert p.name == "graph.v1.json"
    assert (tmp_path / "world.v1.json").exists()
    assert (tmp_path / "policy.v1.json").exists()
    assert (tmp_path / "objective.json").exists()
    with pytest.raises(FileExistsError):
        save_graph(g, tmp_path)
    assert latest_version(tmp_path) == 1
    again = load_graph(p)
    assert again.slug == g.slug and len(again.nodes) == len(g.nodes)
    assert "nodes" in world_view(again)


def test_diff_reports_added_and_removed() -> None:
    a = _g(fx_residual(True))
    b = _g(fx_residual(False))
    b = b.model_copy(update={"version": 2, "parent_version": 1})
    d = diff_graphs(a, b)
    assert d["added_nodes"] and not d["removed_nodes"]
    assert "Diff v1 -> v2" in render_diff(d)
