"""Host-derived structural controls for declared representation experiments.

This measures an intervention, not its usefulness. Parameter updates remain
possible within a fixed vocabulary and executable structure. No model-provided
signature, candidate rename, or prose field can establish the intervention.
"""

from __future__ import annotations

import copy
from contextvars import ContextVar
from typing import Any

from jsonschema import Draft202012Validator, validators
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as SchemaValidationError
from pydantic import Field, model_validator

from sextant.domain import Contract, DomainError, Proposal, Representation, canonical, digest
from sextant.representations import evaluate

# Feasibility bounds for this declared adapter, not an assertion of arbitrary
# schema support. Fail closed rather than letting recursive alternatives consume
# unbounded work. The counter spans all descendants, including local references.
_schema_work: ContextVar[list[int] | None] = ContextVar("submission_schema_work", default=None)
MAX_SCHEMA_STEPS = 20_000


def _schema_step() -> None:
    work = _schema_work.get()
    if work is not None:
        work[0] += 1
        if work[0] > MAX_SCHEMA_STEPS:
            raise DomainError("representation_shape", f"submission schema exceeds {MAX_SCHEMA_STEPS} validation steps")


def _one_of(validator: Any, alternatives: Any, instance: Any, schema: Any) -> Any:
    matches = 0
    for alternative in alternatives:
        _schema_step()
        if next(validator.descend(instance, alternative), None) is None:
            matches += 1
            if matches > 1:
                break
    if matches != 1:
        yield SchemaValidationError("exactly one schema alternative must match")


def _properties(validator: Any, properties: Any, instance: Any, schema: Any) -> Any:
    if not validator.is_type(instance, "object"):
        return
    # Check scalar discriminators before recursive arguments regardless of JSON
    # key order. First-error validation preserves conjunction semantics without
    # expanding every failure tree of an already rejected alternative.
    for key in sorted(properties, key=lambda k: (isinstance(instance.get(k), (dict, list)), k)):
        _schema_step()
        if key in instance:
            error = next(validator.descend(instance[key], properties[key], path=key, schema_path=key), None)
            if error is not None:
                yield error
                return


def _bounded(check: Any) -> Any:
    def validate(validator: Any, value: Any, instance: Any, schema: Any) -> Any:
        _schema_step()
        yield from check(validator, value, instance, schema)
    return validate


_keywords = {**Draft202012Validator.VALIDATORS, "oneOf": _one_of, "properties": _properties}
SubmissionValidator = validators.extend(  # type: ignore[no-untyped-call]  # jsonschema's validator factory lacks typed stubs.
    Draft202012Validator, {key: _bounded(check) for key, check in _keywords.items()},
    # This host contract intentionally tightens JSON Schema's mathematical
    # integer convention: 1.0 and true are not integer-typed JSON input here.
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine("integer", lambda _, value: type(value) is int),
)


def _validate_schema(value: Any, schema: dict[str, Any]) -> None:
    work = _schema_work.set([0])
    try:
        if next(SubmissionValidator(schema).iter_errors(value), None) is not None:
            raise DomainError("representation_shape", "submitted representation does not match the host schema")
    except RecursionError as exc:
        raise DomainError("representation_shape", "submission schema exceeds supported recursion") from exc
    finally:
        _schema_work.reset(work)


class SubmissionProjection(Contract):
    path: list[str] = Field(min_length=1)
    model_schema: dict[str, Any]
    kind_field: str = "kind"
    payload_field: str = "payload"
    table_kind: str = "table"
    expression_kind: str = "expression"
    coordinates_field: str = "variables"
    rows_field: str = "rows"
    row_input_field: str = "inputs"

    @model_validator(mode="after")
    def local_schema(self) -> SubmissionProjection:
        pending: list[Any] = [self.model_schema]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                    if key in item and (not isinstance(item[key], str) or not item[key].startswith("#")):
                        raise ValueError("submission schema may reference only its own document")
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
        try:
            Draft202012Validator.check_schema(self.model_schema)
        except SchemaError as exc:
            raise ValueError("invalid submission model schema") from exc
        return self


def expression_shape(node: Any, depth: int = 0) -> Any:
    if depth > 32:
        raise DomainError("representation_shape", "expression exceeds depth 32")
    if not isinstance(node, dict):
        scalar_types = {type(None): "null", bool: "boolean", int: "integer", float: "number", str: "string"}
        if type(node) not in scalar_types:
            raise DomainError("representation_shape", "expression leaf must be a JSON scalar")
        return {"scalar_type": scalar_types[type(node)]}
    if set(node) == {"var"} and isinstance(node["var"], str):
        return {"var": node["var"]}
    if set(node) == {"literal"}:
        value = node["literal"]
        return {"literal": expression_shape(value, depth + 1) if not isinstance(value, (dict, list)) else {"fixed_data": canonical(value)}}
    if set(node) == {"op", "args"} and isinstance(node["op"], str) and isinstance(node["args"], list):
        return {"op": node["op"], "args": [expression_shape(child, depth + 1) for child in node["args"]]}
    raise DomainError("representation_shape", "invalid expression structure")


def internal_shape(representation: Representation) -> dict[str, Any] | None:
    checked = evaluate(representation, {})
    if checked["status"] == "unsupported":
        return None
    body = representation.body
    if representation.kind == "table":
        structure: Any = {"inputs": body["inputs"]}
    elif representation.kind == "argument_graph":
        structure = {"nodes": [{"id": n["id"], "expression": expression_shape(n["expression"])} for n in body["nodes"]],
                     "outputs": {key: expression_shape(value) for key, value in body["outputs"].items()}}
    elif representation.kind == "conditional_policy":
        structure = {"conditions": [expression_shape(b["when"]) for b in body["branches"]]}
    elif representation.kind == "program":
        # Arbitrary Python does not have a declared parameter/structure split.
        # Its source is fixed; report this stricter treatment explicitly.
        structure = {"fixed_source": body["program"], "treatment": "whole program; parameter changes not separated"}
    else:
        return None
    return {"kind": representation.kind, "version": representation.version, "body": structure,
            "input_interface": representation.input_interface, "output_interface": representation.output_interface}


def submitted_shape(result: Any, projection: SubmissionProjection) -> dict[str, Any] | None:
    value = result
    for key in projection.path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if value is None:
        return None
    if not isinstance(value, dict):
        raise DomainError("representation_shape", "submitted representation must be an object")
    kind, payload = value.get(projection.kind_field), value.get(projection.payload_field)
    if kind == projection.expression_kind:
        shape = expression_shape(payload)
    elif kind == projection.table_kind:
        if not isinstance(payload, dict):
            raise DomainError("representation_shape", "table payload must be an object")
        coordinates, rows = payload.get(projection.coordinates_field), payload.get(projection.rows_field)
        if (not isinstance(coordinates, list) or any(not isinstance(k, str) for k in coordinates)
                or len(set(coordinates)) != len(coordinates) or not isinstance(rows, list)):
            raise DomainError("representation_shape", "table coordinates/rows are malformed")
        assignments = []
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get(projection.row_input_field), dict)
                    or set(row[projection.row_input_field]) != set(coordinates)):
                raise DomainError("representation_shape", "table row differs from its coordinate vocabulary")
            assignments.append(canonical(row[projection.row_input_field]))
        if len(set(assignments)) != len(assignments):
            raise DomainError("representation_shape", "table contains duplicate input assignments")
        shape = {"coordinates": coordinates}
    else:
        raise DomainError("representation_shape", "host projection has no adapter for the submitted kind")
    # Structural/depth checks precede recursive schema evaluation. The validator
    # also bounds work for schemas whose recursion is unrelated to the AST.
    _validate_schema(value, projection.model_schema)
    return {"kind": kind, "structure": shape}


def control(proposal: Proposal, *, enabled: bool, projection: SubmissionProjection | None,
            frozen: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    declaration = digest({"rerepresentation": enabled, "submission_projection": projection})
    prior = frozen or {}
    if prior and prior.get("declaration") != declaration:
        raise DomainError("representation_treatment_changed", "explicitly revise the representation treatment before changing its declaration")
    state = copy.deepcopy(prior) if prior else {"declaration": declaration, "internal": {}, "candidate_keys": [], "external": None}
    actual = {c.key: internal_shape(c.representation) for c in proposal.candidates}
    audit: dict[str, Any] = {"protocol": "structural-representation-control/1", "rerepresentation_enabled": enabled,
                             "declaration": declaration, "internal": {key: digest(shape) if shape is not None else None for key, shape in actual.items()},
                             "external": "not_configured", "authority": "host structural comparison; no quality claim"}
    if not enabled:
        if state["candidate_keys"] and set(actual) != set(state["candidate_keys"]):
            raise DomainError("representation_change", "candidate identity set changed under the fixed representation treatment")
        for key, shape in actual.items():
            if key in state["internal"] and shape != state["internal"][key]:
                raise DomainError("representation_change", f"candidate {key} changed or removed its frozen structure")
            if shape is not None:
                state["internal"][key] = shape
        if state["internal"]:
            state["candidate_keys"] = list(actual)
    if projection is not None:
        try:
            external = submitted_shape(proposal.direct_result, projection)
        except DomainError as exc:
            if not enabled and state["external"] is not None:
                raise DomainError("representation_change", "frozen submitted representation became invalid") from exc
            external = None
            audit["external"] = {"status": "invalid_unfrozen", "reason": str(exc)}
        else:
            audit["external"] = {"status": "observed" if external is not None else "absent", "shape_hash": digest(external) if external is not None else None}
        if not enabled:
            if state["external"] is not None and state["external"] != external:
                raise DomainError("representation_change", "submitted representation changed, moved or removed its frozen structure")
            if external is not None:
                state["external"] = external
    audit["frozen_state_hash"] = digest(state)
    return state, audit
