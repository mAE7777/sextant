"""Logical proposal expressiveness survives the native strict-schema subset."""

import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

from sextant.compiler import Proposal
from sextant.providers.codex_transport import (
    TRANSPORT_SCHEMA,
    decode_output,
    decode_output_text,
    encode_request,
)


def test_real_proposal_contract_is_not_replaced_by_closed_service_fields():
    original = {"system": "Bounded task", "prompt": "public quoted task", "schema": Proposal.model_json_schema()}
    encoded = encode_request(original)
    packet = json.loads(encoded["prompt"])
    assert packet == {"task_packet": original["prompt"], "logical_output_schema": original["schema"]}
    assert encoded["schema"] == TRANSPORT_SCHEMA
    assert set(TRANSPORT_SCHEMA["required"]) == set(TRANSPORT_SCHEMA["properties"])
    assert TRANSPORT_SCHEMA["additionalProperties"] is False
    Draft202012Validator.check_schema(encoded["schema"])
    assert original["system"] == "Bounded task"


def test_open_json_nested_values_and_optional_absence_survive_round_trip():
    schema = {"type": "object", "properties": {"required": {}, "optional": {"type": "string"}},
              "required": ["required"], "additionalProperties": True}
    logical = {"required": [None, True, 4, "中文\\\"", {"unknown_future_field": [1.25]}],
               "new_representation": {"operator": {"arbitrary": "program"}}}
    assert decode_output({"json_payload": json.dumps(logical, ensure_ascii=False)}, schema) == logical
    assert "optional" not in logical


@pytest.mark.parametrize("envelope", [
    {"json_payload": '{"ready":false}'},
    {"json_payload": '{"ready":true,"ready":false}'},
    {"json_payload": '{"ready":NaN}'},
    {"json_payload": '{"ready":true} trailing'},
    {"json_payload": {"ready": True}},
    {"json_payload": '{"ready":true}', "unqualified_extra": 1},
])
def test_transport_or_logical_violation_is_withheld(envelope):
    schema = {"type": "object", "required": ["ready"], "properties": {"ready": {"const": True}},
              "additionalProperties": False}
    with pytest.raises((ValueError, ValidationError)):
        decode_output(envelope, schema)


def test_duplicate_outer_envelope_fields_cannot_be_hidden_by_json_parser():
    with pytest.raises(ValueError, match="duplicate"):
        decode_output_text('{"json_payload":"true","json_payload":"false"}', {"type": "boolean"})


@pytest.mark.parametrize("logical", ['{"free":1e999}', r'{"free":"\ud800"}'])
def test_open_json_still_must_be_finite_canonical_utf8(logical):
    with pytest.raises(ValueError):
        decode_output({"json_payload": logical}, {"type": "object"})
