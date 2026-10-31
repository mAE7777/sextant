from __future__ import annotations

from typing import Any

import pytest

from sextant.decision import observed_scenario_gate


def gate(observed: Any, expected: dict[str, Any]) -> dict[str, Any]:
    return observed_scenario_gate(
        {"measurement_protocol": "scope-control/1", "required_scenarios": ["complete"],
         "scope_bindings": {"phase": "phase"}, "scenario_path": ["scenario"],
         "measurements_path": ["rows"], "input_bindings": [],
         "constraints": [{"predicate": {"op": "eq", "args": [{"var": "valid"}, True]}}]},
        {}, [{"measurement_protocol": "scope-control/1", "ref": "receipt-bound-control",
              "environment_scope": observed,
              "content": {"scenario": "complete", "rows": [{"valid": True}]}}],
        environment=expected,
    )


@pytest.mark.parametrize(("observed", "expected"), [
    ({"phase": False}, {"phase": 0}),
    ({"phase": True}, {"phase": 1}),
    ({"phase": 0.0}, {"phase": 0}),
    ({"phase": 1.0}, {"phase": 1}),
    ({"phase": [False]}, {"phase": [0]}),
    ({"phase": {"value": True}}, {"phase": {"value": 1}}),
    ({}, {}),
    ({}, {"phase": None}),
    ({"phase": None}, {}),
    (None, {}),
    ([], {"phase": 0}),
])
def test_missing_or_different_json_scope_cannot_authorize_effect(observed: Any, expected: dict[str, Any]) -> None:
    result = gate(observed, expected)
    assert result["status"] == "unknown"
    assert result["evidence_refs"] == []
    assert result["missing_scenarios"] == ["complete"]


@pytest.mark.parametrize("phase", [0, 1, "phase-0", None, {"value": 0}, [True, 0]])
def test_explicit_identical_json_scope_remains_supported(phase: Any) -> None:
    result = gate({"phase": phase}, {"phase": phase})
    assert result["status"] == "passed"
    assert result["evidence_refs"] == ["receipt-bound-control"]
