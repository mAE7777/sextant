from __future__ import annotations

from typing import Any

import pytest

from sextant.decision import compare, constraints, observed_scenario_gate
from sextant.domain import DomainError


def test_joint_scenarios_reverse_choice_without_fabricated_probabilities() -> None:
    options = {"a": {"correlated": {"loss": {"value": 9, "unit": "errors"}}, "anticorrelated": {"loss": {"value": 1, "unit": "errors"}}},
               "b": {"correlated": {"loss": {"value": 2, "unit": "errors"}}, "anticorrelated": {"loss": {"value": 8, "unit": "errors"}}}}
    output = compare(options, [{"metric": "loss", "direction": "min", "unit": "errors"}])
    assert output["scenarios"]["correlated"]["nondominated"] == ["b"]
    assert output["scenarios"]["anticorrelated"]["nondominated"] == ["a"]
    assert output["reversal_triggers"] and "no independence" in output["scenario_probabilities"]


def test_ordinal_units_and_hard_constraints_cannot_be_softened() -> None:
    assert constraints([{"requirement": "be correct"}], {})[0]["status"] == "unknown"
    constraint = {"predicate": {"op": "le", "args": [{"var": "errors"}, 0]}}
    assert constraints([constraint], {"errors": 1})[0]["status"] == "violated"
    options = {"a": {"s": {"delay": {"value": 1, "unit": "s"}}}, "b": {"s": {"delay": {"value": 5, "unit": "ms"}}}}
    objective = [{"metric": "delay", "direction": "min", "unit": "s"}]
    assert compare(options, objective)["scenarios"]["s"]["unknown"] == ["b"]
    with pytest.raises(DomainError, match="affine"):
        compare(options, objective, utility_mapping={"rank": {"a": 1, "b": 2}})
    mapped = compare(options, objective, utility_mapping={"kind": "affine-cardinal/1", "metrics": {"delay": {"unit": "s", "weight": -1, "scale": 1, "offset": 0}}})
    assert mapped["scores_by_scenario"]["a"]["s"] == -1


def test_observed_gate_requires_current_scope_full_config_all_replicates() -> None:
    spec = {"measurement_protocol": "executed-service/1", "scenario_path": ["workload"], "measurements_path": ["measurements"],
            "required_scenarios": ["mixed", "stress"], "scope_bindings": {"episode": "episode_id", "phase": "phase"},
            "input_bindings": [{"measurement_path": ["config"], "input_path": ["config_patch"]}],
            "constraints": [{"predicate": {"op": "eq", "args": [{"var": "errors"}, 0]}}]}
    environment = {"episode_id": "ep", "phase": 1}
    def measurement(name: str, errors: int = 0, phase: int = 1, config: Any = None) -> dict[str, Any]:
        return {"ref": name, "environment_scope": {"episode": "ep", "phase": phase}, "measurement_protocol": "executed-service/1",
                "content": {"workload": name, "measurements": [{"config": config or {"workers": 2}, "errors": errors}]}}
    inputs = {"config_patch": {"workers": 2}}
    passing = [measurement("mixed"), measurement("stress")]
    assert observed_scenario_gate(spec, inputs, passing, environment=environment)["status"] == "passed"
    assert observed_scenario_gate(spec, inputs, [measurement("mixed", phase=0), measurement("stress")], environment=environment)["status"] == "unknown"
    assert observed_scenario_gate(spec, inputs, [*passing, measurement("mixed", errors=1)], environment=environment)["status"] == "violated"
    assert observed_scenario_gate(spec, {"config_patch": {}}, passing, environment=environment)["status"] == "unknown"
    assert observed_scenario_gate(spec, inputs, [*passing, measurement("mixed", errors=1, config={"workers": 3})], environment=environment)["status"] == "passed"
