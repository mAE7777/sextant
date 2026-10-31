"""Lossless JSON transport for native strict Structured Outputs.

Sextant's logical contracts may contain open dictionaries, arbitrary JSON and
optional fields. Keep that expressiveness and validate it locally; the service
only constrains a closed envelope containing the serialized logical result.
"""

from __future__ import annotations

import json
from typing import Any

from jsonschema import Draft202012Validator

from sextant.domain import canonical

TRANSPORT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"json_payload": {"type": "string"}},
    "required": ["json_payload"],
    "additionalProperties": False,
}


def encode_request(request: dict[str, Any]) -> dict[str, Any]:
    """The task data stays unchanged inside an explicit transport packet."""
    canonical(request).encode("utf-8")
    instruction = (
        "Transport contract: return only an object with the required string field json_payload. "
        "That string must contain one JSON serialization of the complete logical result, "
        "conforming to logical_output_schema. Preserve all logical fields and types, "
        "including nested objects, arbitrary JSON values and legitimately omitted optional fields. "
        "The transport envelope itself is not the logical answer. Use no tools."
    )
    return {
        **request,
        "system": request["system"] + "\n" + instruction,
        "prompt": json.dumps(
            {"task_packet": request["prompt"], "logical_output_schema": request["schema"]},
            ensure_ascii=False, allow_nan=False, separators=(",", ":"),
        ),
        "schema": TRANSPORT_SCHEMA,
    }


def _strict_json(text: str) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError("non-JSON numeric constant: " + value)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate logical JSON key: " + key)
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)


def decode_output(envelope: Any, schema: dict[str, Any]) -> Any:
    Draft202012Validator(TRANSPORT_SCHEMA).validate(envelope)
    result = _strict_json(envelope["json_payload"])
    Draft202012Validator(schema).validate(result)
    # JSON's numeric syntax may overflow Python floats even without Infinity
    # literals; escaped lone surrogates also parse but cannot enter the ledger.
    # Reject both before returning an output that receipt hashing cannot record.
    canonical(result).encode("utf-8")
    return result


def decode_output_text(text: str, schema: dict[str, Any]) -> Any:
    return decode_output(_strict_json(text), schema)
