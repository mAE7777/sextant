"""Unknown mandatory conditions cannot confer ranking or utility eligibility."""

from __future__ import annotations

import copy
from pathlib import Path

from test_decision_failures_v1 import FEWER_CALLS, branch
from test_runtime_integrity_v1 import call, fresh, proposal, start

from sextant.decision import compare
from sextant.domain import canonical

NATURAL_CORRECTNESS = {
    "id": "correct-prices",
    "requirement": "Every tested response must equal the current origin price for that request's region and SKU; unexecuted repairs remain unverified.",
}


def test_saved_v7_costs_do_not_rank_when_mandatory_correctness_is_unverified() -> None:
    # VA-F01: these are the three actually saved metric projections. They are
    # proposals, not a new mapping from natural language into a predicate.
    metrics = {
        "payload": {"v7_all": {"correct-prices": False, "histories": 32,
                               "price_correct_histories": 16, "origin_calls": {"value": 64, "unit": "calls"}}},
        "watermark": {"v7_all": {"correct-prices": False, "histories": 32,
                                 "price_correct_histories": 24, "origin_calls": {"value": 64, "unit": "calls"}}},
        "validated_fill": {"v7_all": {"correct-prices": True, "histories": 32,
                                      "price_correct_histories": 32, "origin_calls": {"value": 80, "unit": "calls"}}},
    }
    original = canonical(metrics)
    result = compare(metrics, FEWER_CALLS, hard_constraints=[NATURAL_CORRECTNESS])
    scenario = result["scenarios"]["v7_all"]
    assert scenario["nondominated"] == [] and scenario["dominated_by"] == {}
    assert set(scenario["unknown"]) == set(metrics)
    assert scenario["infeasible"] == {}  # Unknown does not invent a failed predicate.
    assert scenario["constraints_unverified"] == {name: ["correct-prices"] for name in metrics}
    assert canonical(metrics) == original


def test_unverified_candidate_cannot_dominate_verified_one_or_gain_cardinal_score() -> None:
    condition = {"id": "correctness", "predicate": {"op": "eq", "args": [{"var": "errors"}, 0]}}
    options = {
        "cheap_unknown": {"s": {"origin_calls": {"value": 1, "unit": "calls"}}},
        "verified": {"s": {"origin_calls": {"value": 4, "unit": "calls"}, "errors": {"value": 0, "unit": "count"}}},
        "known_bad": {"s": {"origin_calls": {"value": 0, "unit": "calls"}, "errors": {"value": 1, "unit": "count"}}},
    }
    mapping = {"kind": "affine-cardinal/1", "metrics": {"origin_calls": {"unit": "calls", "weight": -1, "scale": 1, "offset": 0}}}
    result = compare(options, FEWER_CALLS, hard_constraints=[condition], utility_mapping=mapping)
    scenario = result["scenarios"]["s"]
    assert scenario["nondominated"] == ["verified"]
    assert scenario["dominated_by"] == {"verified": []}
    assert scenario["unknown"] == ["cheap_unknown"]
    assert "known_bad" in scenario["infeasible"]
    assert result["scores_by_scenario"] == {"cheap_unknown": {"s": None}, "verified": {"s": -4}, "known_bad": {"s": None}}


def test_saved_application_analysis_preserves_detail_but_withholds_unknown_ranking(tmp_path: Path) -> None:
    generated = proposal()
    generated["candidates"] = [branch("a", 2), branch("b", 4)]
    raw_analysis = {"actual_inputs": {"missing_measurement": True}, "nondominated": ["a"]}
    generated["policy"]["decision_analysis"] = copy.deepcopy(raw_analysis)
    app, _ = fresh(tmp_path, generated)
    created = start(app, mission={"hard_constraints": [NATURAL_CORRECTNESS], "objective_metrics": FEWER_CALLS})
    done = call(app, "run", created, steps=1)
    assert not done.errors
    snapshot = call(app, "inspect", done).outputs
    policy = next(row["revision"]["payload"] for row in snapshot["objects"]
                  if row["revision"]["ref"] == snapshot["state"]["policy_ref"])
    analysis = policy["decision_analysis"]
    assert analysis["scenarios"]["mixed"]["nondominated"] == []
    assert set(analysis["scenarios"]["mixed"]["unknown"]) == {"a", "b"}
    assert canonical(analysis["model_proposed_analysis"]["content"]) == canonical(raw_analysis)
    assert call(app, "replay", done).outputs["matches_current"] is True
    exported = call(app, "export", done)
    assert not exported.errors and any(row["payload"] == policy for row in exported.outputs["revisions"])
