"""Prediction settlement and constraint-preserving policy normalization."""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from sextant.domain import (
    DomainError,
    Mission,
    Observation,
    Policy,
    Prediction,
    PredictionSpec,
    canonical,
    finite_number,
)


def normalize_policy(policy: Policy, mission: Mission) -> Policy:
    result = policy.model_copy(deep=True)
    result.hard_constraints = mission.hard_constraints
    if result.ordering == "authorized-cardinal" and mission.utility_mapping is None:
        result.ordering = "conditional-partial-order"
        result.unresolved.append("No user-authorized cardinal utility mapping; scalar rankings have no decision authority.")
    if not result.reversal_triggers:
        result.reversal_triggers = ["new contrary observation", "changed goal, constraint or authorization"]
    return result


def field(content: Any, path: list[str | int]) -> Any:
    validate_path(path)
    for part in path:
        if not ((type(content) is dict and type(part) is str)
                or (type(content) is list and type(part) is int)):
            raise DomainError("resolver_field", "resolver path does not match the JSON container type")
        try:
            content = cast(Any, content)[part]
        except (KeyError, IndexError, TypeError) as exc:
            raise DomainError("resolver_field", "observation does not contain the frozen resolver field") from exc
    return content


def validate_path(path: Any) -> None:
    if type(path) is not list or any(
        type(part) not in (str, int) or (type(part) is int and part < 0) for part in path
    ):
        raise DomainError("resolver_field", "resolver field must be an explicit list of JSON keys/nonnegative integer indices")


def resolver_field(spec: dict[str, Any]) -> list[str | int]:
    """Explicit selectors only; retain the field_path spelling in stored history."""
    names = [name for name in ("field", "field_path") if name in spec]
    if not names:
        raise DomainError("resolver_field", "resolver needs an explicit field selector; [] explicitly selects the whole value")
    for name in names:
        validate_path(spec[name])
    if len(names) == 2 and canonical(spec["field"]) != canonical(spec["field_path"]):
        raise DomainError("resolver_field", "field and field_path specify conflicting selectors")
    return cast(list[str | int], spec[names[0]])


def validate_resolver_spec(resolver: str, spec: dict[str, Any]) -> None:
    if resolver not in ("field-equals/1", "numeric-threshold/1"):
        return  # Non-empirical questions retain no automatic outcome authority.
    resolver_field(spec)
    for name in ("measurement_protocol", "source", "supports_scope"):
        if type(spec.get(name)) is not str or not spec[name]:
            raise DomainError("resolver_contract", "deterministic resolver requires explicit protocol/source/scope strings")
    if resolver == "field-equals/1":
        if "equals" not in spec:
            raise DomainError("resolver_contract", "field-equals requires an explicit equals value, including null")
        canonical(spec["equals"])
    else:
        if not finite_number(spec.get("threshold")) or spec.get("comparison") not in ("le", "ge"):
            raise DomainError("resolver_number", "numeric resolver needs a finite threshold and le/ge comparison")
        if type(spec.get("unit")) is not str or not spec["unit"] or "unit_field" not in spec:
            raise DomainError("resolver_unit", "numeric resolver requires an explicit unit string and unit_field selector")
        validate_path(spec["unit_field"])


def validate_prediction_contract(prediction: PredictionSpec, *, new_admission: bool = False) -> None:
    """Reject impossible frozen domains; require exactly both Booleans for new questions."""
    validate_resolver_spec(prediction.resolver, prediction.resolver_spec)
    validate_prediction_domain(prediction, new_admission=new_admission)


def validate_prediction_domain(prediction: PredictionSpec, *, new_admission: bool = False) -> None:
    """Check outcome types without moving legacy field grammar ahead of eligibility."""
    if prediction.resolver in ("field-equals/1", "numeric-threshold/1"):
        domain = prediction.outcome_domain
        booleans = {canonical(value) for value in domain if type(value) is bool}
        if not booleans:
            raise DomainError("resolver_outcome_domain",
                              "deterministic resolver cannot produce any member of the frozen outcome domain; "
                              "its result is Boolean")
        if new_admission and (len(domain) != 2 or any(type(value) is not bool for value in domain)
                              or booleans != {"false", "true"}):
            raise DomainError("resolver_outcome_domain",
                              "new deterministic predictions require exactly false and true, each once, as Boolean outcomes")


def resolution_context(prediction: Prediction, observation: Observation, *, now: str) -> dict[str, Any] | None:
    """Check measurement eligibility independently of the frozen field grammar."""
    if prediction.status != "pending":
        raise DomainError("prediction_state", "only a pending prediction can be settled")
    if observation.status != "active":
        raise DomainError("withdrawn", "withdrawn observations cannot settle predictions")
    frozen = datetime.fromisoformat(prediction.frozen_at)
    measured = datetime.fromisoformat(observation.occurred_at or observation.observed_at)
    if frozen.tzinfo is None or measured.tzinfo is None:
        raise DomainError("resolver_time", "prediction and measurement timestamps must have timezones")
    if measured < frozen:
        return {"status": "pending", "reason": "measurement predates the frozen prediction; cannot be its confirming outcome"}
    # A sentence supplied by a person or found in a file proves only that sentence.
    if observation.provenance not in ("constructed", "environment"):
        return {"status": "pending", "reason": "declared/source evidence does not establish the outcome",
                "observation_claim": observation.content}
    spec = prediction.resolver_spec
    if not spec.get("measurement_protocol") or spec["measurement_protocol"] != observation.measurement_protocol:
        raise DomainError("resolver_protocol", "measurement does not match the frozen resolver protocol")
    if spec.get("source") != observation.source:
        raise DomainError("resolver_source", "measurement source does not match the frozen resolver")
    if spec.get("supports_scope") != observation.supports_scope:
        raise DomainError("resolver_scope", "measurement scope does not match the frozen resolver")
    if spec.get("not_before"):
        event_time = observation.occurred_at or observation.observed_at
        try:
            actual = datetime.fromisoformat(event_time)
            earliest = datetime.fromisoformat(spec["not_before"])
            if actual.tzinfo is None or earliest.tzinfo is None or actual < earliest:
                return {"status": "pending", "reason": "measurement precedes the frozen outcome window"}
        except ValueError as exc:
            raise DomainError("resolver_time", "resolver needs timezone-aware ISO timestamps") from exc
    if spec.get("deadline"):
        deadline = datetime.fromisoformat(spec["deadline"])
        if deadline.tzinfo is None:
            raise DomainError("resolver_time", "deadline has no timezone")
        if datetime.fromisoformat(now) > deadline and prediction.censoring_rule == "censor-after-deadline":
            return {"status": "censored", "reason": "frozen deadline passed before resolution"}
    return None


def settle(prediction: Prediction, observation: Observation, *, now: str) -> dict[str, Any]:
    if prediction.status != "pending":
        raise DomainError("prediction_state", "only a pending prediction can be settled")
    validate_prediction_domain(prediction)
    context = resolution_context(prediction, observation, now=now)
    if context is not None:
        return context
    spec = prediction.resolver_spec
    if prediction.resolver not in ("field-equals/1", "numeric-threshold/1"):
        return {"status": "pending", "reason": "declared resolver has no automatic empirical authority"}
    validate_resolver_spec(prediction.resolver, spec)
    value = field(observation.content, resolver_field(spec))
    if prediction.resolver == "field-equals/1":
        outcome: Any = canonical(value) == canonical(spec["equals"])
    elif prediction.resolver == "numeric-threshold/1":
        if not finite_number(value) or not finite_number(spec.get("threshold")):
            raise DomainError("resolver_number", "numeric resolver requires finite numbers")
        if canonical(field(observation.content, spec["unit_field"])) != canonical(spec["unit"]):
            raise DomainError("resolver_unit", "resolver must bind equal, explicit units")
        op = spec.get("comparison")
        if op not in ("le", "ge"):
            raise DomainError("resolver_comparison", "supported comparisons are le/ge")
        outcome = value <= spec["threshold"] if op == "le" else value >= spec["threshold"]
    if canonical(outcome) not in {canonical(v) for v in prediction.outcome_domain}:
        raise DomainError("resolver_domain", "resolved value lies outside the frozen outcome domain")
    return {"status": "resolved", "outcome": outcome, "scope": observation.supports_scope,
            "protocol": prediction.resolver, "resolver_semantics": "explicit-json-field/2",
            "field_selector": resolver_field(spec),
            "resolution_basis": "deterministic within this measurement contract"}
