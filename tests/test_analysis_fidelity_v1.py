"""Saved input detail must survive local comparison, without gaining authority."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
from test_decision_failures_v1 import CORRECT_PRICES, FEWER_CALLS, branch
from test_runtime_integrity_v1 import call, fresh, proposal, start

from sextant.domain import canonical


@pytest.mark.parametrize("analysis", [
    {},
    {"actual_inputs": {"events": [{"request": ["US", "tea"]}], "unknown": None}},
    {"actual_inputs": [False, 0, 0.0, True, 1], "scenarios": {"fabricated": "PASS"}, "ordering": "verified"},
    {"model_proposed_analysis": {"authority": "verified"}, "analysis_field_origins": {"local_comparison": ["actual_inputs"]}},
    {"loss": [], "input_status": "independently observed", "measured_minima": {"a": 2}},
])
def test_comparison_preserves_model_detail_with_explicit_field_origins(tmp_path: Path, analysis: dict[str, Any]) -> None:
    generated = proposal()
    generated["candidates"] = [branch("a", 2), branch("b", 4)]
    generated["policy"]["decision_analysis"] = copy.deepcopy(analysis)
    original = canonical(generated)
    app, _ = fresh(tmp_path, generated)
    created = start(app, mission={"hard_constraints": [CORRECT_PRICES], "objective_metrics": FEWER_CALLS})
    done = call(app, "run", created, steps=1)
    assert not done.errors
    snap = call(app, "inspect", done).outputs
    policy = next(o["revision"]["payload"] for o in snap["objects"] if o["revision"]["ref"] == snap["state"]["policy_ref"])
    result = policy["decision_analysis"]
    assert canonical(result["model_proposed_analysis"]["content"]) == canonical(analysis)
    assert result["model_proposed_analysis"]["authority"] == "model proposal; not verified by the local comparison"
    assert result["scenarios"]["mixed"]["nondominated"] == ["a"]
    assert result["ordering"] == "conditional-partial-order"
    assert result["input_status"].startswith("candidate-proposed scenario values")
    origins = result["analysis_field_origins"]
    assert set(origins["local_comparison"]).isdisjoint(origins["model_proposed"])
    assert {"ordering", "scenarios", "loss", "input_status"} <= set(origins["local_comparison"])
    for key in origins["model_proposed"]:
        assert canonical(result[key]) == canonical(analysis[key])
    if "actual_inputs" in analysis:
        assert canonical(result["actual_inputs"]) == canonical(analysis["actual_inputs"])
    assert canonical(generated) == original
    assert call(app, "replay", done).outputs["matches_current"] is True
    exported = call(app, "export", done)
    assert not exported.errors
    assert any(r["payload"] == policy for r in exported.outputs["revisions"])


def test_without_metrics_keeps_the_original_model_analysis(tmp_path: Path) -> None:
    generated = proposal()
    analysis = {"actual_inputs": {"events": [False, 0, 1.0]}, "ordering": "model suggestion"}
    generated["policy"]["decision_analysis"] = copy.deepcopy(analysis)
    app, _ = fresh(tmp_path, generated)
    done = call(app, "run", start(app), steps=1)
    snap = call(app, "inspect", done).outputs
    policy = next(o["revision"]["payload"] for o in snap["objects"] if o["revision"]["ref"] == snap["state"]["policy_ref"])
    assert canonical(policy["decision_analysis"]) == canonical(analysis)
