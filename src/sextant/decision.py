"""Finite constraint, partial-order and scenario interfaces; no invented weights."""

from __future__ import annotations

from typing import Any

from sextant.domain import DomainError, canonical, finite_number
from sextant.representations import Unknown, expression


def constraints(requirements: list[dict[str, Any]], subject: dict[str, Any]) -> list[dict[str, Any]]:
    results = []
    for i, requirement in enumerate(requirements):
        record: dict[str, Any] = {"constraint": requirement.get("id", str(i)), "requirement": requirement, "status": "unknown"}
        if "predicate" in requirement:
            try:
                value = expression(requirement["predicate"], subject)
                if not isinstance(value, bool):
                    raise DomainError("constraint_type", "hard constraint must return a boolean")
                record["status"] = "passed" if value else "violated"
            except (Unknown, DomainError) as exc:
                record["reason"] = str(exc)
        else:
            record["reason"] = "natural-language requirement needs a matching measurement/resolver; no implicit PASS"
        results.append(record)
    return results


def measured_violations(requirements: list[dict[str, Any]], records: list[dict[str, Any]],
                        observations: dict[tuple[str, str], dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Hard constraints violated in receipt-backed results of a candidate's own experiments.

    ``records`` are completed-experiment records naming the branch that proposed the
    experiment; ``observations`` maps current (object_id, revision_id) to payload. Only
    active constructed/environment measurements count; declared text never does.
    """
    found: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        branch = record.get("branch")
        if not branch or branch == "policy":
            continue
        for raw in record.get("observations", []):
            payload = observations.get((raw.get("object_id"), raw.get("revision_id")))
            if (not payload or payload.get("status") != "active"
                    or payload.get("provenance") not in ("constructed", "environment")
                    or not isinstance(payload.get("content"), dict)):
                continue
            for check in constraints(requirements, payload["content"]):
                if check["status"] == "violated":
                    found.setdefault(branch, []).append({"constraint": check["constraint"], "observation": raw["object_id"],
                                                         "experiment": record.get("experiment", {}).get("key", "unknown")})
    return found


def compare(options: dict[str, dict[str, dict[str, Any]]], objectives: list[dict[str, Any]], *,
            utility_mapping: dict[str, Any] | None = None, hard_constraints: list[dict[str, Any]] | None = None,
            measured_violations: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, Any]:
    """Compare option->scenario->metric without assuming scenario probabilities.

    A metric is {value,unit}; objectives name metric/direction/unit. Cardinal
    mapping, if explicitly supplied by the owner, requires weight/scale/offset
    for every objective and never consumes ordinal ranks as weights.

    Hard constraints come first: an option violating one in a scenario (by its
    stated metric values) or in a receipt-backed result of its own executed
    experiment is infeasible there and never enters dominance. A constraint that
    cannot be evaluated is reported unverified; that option cannot enter
    dominance or receive an authorized cardinal score until it is established.
    """
    if not objectives or any(o.get("direction") not in ("min", "max") or not o.get("unit") for o in objectives):
        raise DomainError("objective_contract", "objectives require metric, min/max direction and explicit unit")
    scenarios = sorted({s for values in options.values() for s in values})
    measured_violations = measured_violations or {}
    infeasible: dict[str, dict[str, list[str]]] = {s: {} for s in scenarios}
    unverified: dict[str, dict[str, list[str]]] = {s: {} for s in scenarios}
    for name, rows in options.items():
        for scenario in scenarios:
            subject = {metric: m["value"] for metric, m in rows.get(scenario, {}).items() if isinstance(m, dict) and "value" in m}
            for check in constraints(hard_constraints or [], subject):
                if check["status"] == "violated":
                    infeasible[scenario].setdefault(name, []).append(f"{check['constraint']} (stated values)")
                elif check["status"] == "unknown":
                    unverified[scenario].setdefault(name, []).append(check["constraint"])
            for violation in measured_violations.get(name, []):
                infeasible[scenario].setdefault(name, []).append(
                    f"{violation['constraint']} (measured by experiment {violation['experiment']}, observation {violation['observation']})")
    values: dict[str, dict[str, list[float] | None]] = {}
    losses = []
    for name, rows in options.items():
        values[name] = {}
        for scenario in scenarios:
            vector = []
            for objective in objectives:
                measurement = rows.get(scenario, {}).get(objective["metric"])
                if not isinstance(measurement, dict) or measurement.get("unit") != objective["unit"] or not finite_number(measurement.get("value")):
                    losses.append(f"{name}/{scenario}/{objective['metric']}: missing, nonfinite or mismatched unit")
                    break
                vector.append(float(measurement["value"]) * (1 if objective["direction"] == "max" else -1))
            values[name][scenario] = vector if len(vector) == len(objectives) else None
    per_scenario: dict[str, Any] = {}
    for scenario in scenarios:
        feasible = [n for n in options if n not in infeasible[scenario] and n not in unverified[scenario]]
        dominated: dict[str, list[str]] = {name: [] for name in feasible}
        for left in feasible:
            for right in feasible:
                a, b = values[left][scenario], values[right][scenario]
                if left != right and a is not None and b is not None and all(x >= y for x, y in zip(a, b, strict=True)) and any(x > y for x, y in zip(a, b, strict=True)):
                    dominated[right].append(left)
        per_scenario[scenario] = {"nondominated": [n for n in feasible if not dominated[n] and values[n][scenario] is not None],
                                  "dominated_by": dominated,
                                  "unknown": [n for n in options if n not in infeasible[scenario]
                                              and (n in unverified[scenario] or values[n][scenario] is None)],
                                  "infeasible": infeasible[scenario],
                                  "constraints_unverified": unverified[scenario]}
    if any(unverified[s] for s in scenarios):
        losses.append("some hard constraints could not be evaluated against the stated values; affected options remain unverified and are excluded from rankings and cardinal scores")
    result: dict[str, Any] = {"ordering": "conditional-partial-order", "scenarios": per_scenario,
                              "scenario_probabilities": "not supplied; no independence/correlation assumption",
                              "reversal_triggers": [], "loss": losses}
    if len({canonical(r["nondominated"]) for r in per_scenario.values()}) > 1:
        result["reversal_triggers"].append("preferred set changes with scenario; resolve which scenario applies")
    if utility_mapping is not None:
        if utility_mapping.get("kind") != "affine-cardinal/1":
            raise DomainError("utility_mapping", "supported authorized mapping is affine-cardinal/1")
        mappings = utility_mapping.get("metrics", {})
        for objective in objectives:
            mapping = mappings.get(objective["metric"], {})
            if mapping.get("unit") != objective["unit"] or not all(finite_number(mapping.get(k)) for k in ("weight", "scale", "offset")) or mapping["scale"] == 0:
                raise DomainError("utility_mapping", "each metric needs matching units and finite weight/offset/nonzero scale")
        scores: dict[str, dict[str, float | None]] = {}
        for name, rows in options.items():
            scores[name] = {}
            for scenario in scenarios:
                if values[name][scenario] is None or name in infeasible[scenario] or name in unverified[scenario]:
                    scores[name][scenario] = None
                    continue
                scores[name][scenario] = sum(mappings[o["metric"]]["weight"] *
                                               (rows[scenario][o["metric"]]["value"] - mappings[o["metric"]]["offset"]) /
                                               mappings[o["metric"]]["scale"] for o in objectives)
        result.update(ordering="authorized-cardinal", authorized_mapping=utility_mapping, scores_by_scenario=scores)
        canonical(result)
    return result


def observed_scenario_gate(spec: dict[str, Any], proposed_inputs: dict[str, Any],
                           measurements: list[dict[str, Any]], *, environment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generic receipt-backed scenario check, supplied by the task owner.

    The application selects only current environment/constructed observations.
    Spec names paths, required scenario labels, equal-input bindings and predicates;
    it contains no privileged true answer. No observed scenario implies all-world truth.
    """
    from sextant.grounding import field
    required = spec.get("required_scenarios", [])
    covered: set[str] = set()
    failures: list[str] = []
    supporting: list[str] = []
    for observation in measurements:
        if observation["measurement_protocol"] != spec["measurement_protocol"]:
            continue
        try:
            observed_scope = observation.get("environment_scope")
            expected_scope = environment or {}
            bindings = spec.get("scope_bindings", {})
            if bindings and (not isinstance(observed_scope, dict) or any(
                    scope_key not in observed_scope or env_key not in expected_scope
                    or canonical(observed_scope[scope_key]) != canonical(expected_scope[env_key])
                    for scope_key, env_key in bindings.items())):
                continue
            content = observation["content"]
            if any(c["status"] != "passed" for c in constraints(spec.get("observation_constraints", []), content)):
                continue
            scenario = str(field(content, spec["scenario_path"]))
            if scenario not in required:
                continue
            rows = field(content, spec.get("measurements_path", []))
            if not isinstance(rows, list) or not rows:
                continue
            matched = []
            for row in rows:
                bindings = spec.get("input_bindings", [])
                if not all(canonical(field(row, b["measurement_path"])) == canonical(field(proposed_inputs, b["input_path"])) for b in bindings):
                    continue
                checks = constraints(spec["constraints"], row)
                if any(c["status"] != "passed" for c in checks):
                    failures.append(f"{scenario}: matching measurement violates or cannot establish a hard constraint")
                else:
                    matched.append(row)
            if matched:
                covered.add(scenario)
                supporting.append(observation["ref"])
        except (KeyError, TypeError, DomainError):
            continue
    missing = sorted(set(required) - covered)
    return {"status": "violated" if failures else "unknown" if missing or not required else "passed",
            "missing_scenarios": missing, "failures": failures, "evidence_refs": supporting,
            "scope": "only the measured scenarios and exact input bindings; not a universal guarantee"}
