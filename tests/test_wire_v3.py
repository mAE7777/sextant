"""The wire v2 option-order loss, kept as a permanent counterexample."""
from __future__ import annotations

import json

import pytest

from sextant import wire
from sextant.domain import canonical
from sextant.jev_transport import _to_sdk_questions


def test_reversed_options_have_different_render_and_request_hashes() -> None:
    a = {"type": "choice", "instructions": "choose", "criteria": {"a": "first", "b": "second"}}
    b = {"type": "choice", "instructions": "choose", "criteria": {"b": "second", "a": "first"}}
    assert wire.rendered_hash("test", "q", "choice", a) != wire.rendered_hash("test", "q", "choice", b)
    assert wire.request_digest(model_id="test", transport="typesafe", state={}, questions={"q": a}) != wire.request_digest(model_id="test", transport="typesafe", state={}, questions={"q": b})


def test_canonical_json_and_sdk_preserve_actual_treatment_order() -> None:
    original = wire.ordered_payload({"type": "choice", "instructions": "choose", "criteria": {"z": "last label first", "a": "first label last"}})
    restored = json.loads(canonical(original))
    assert list(restored["criteria"]) == ["a", "z"]  # the v2 counterexample
    assert list(wire.ordered_criteria(restored)) == ["z", "a"]
    assert wire.rendered_hash("test", "q", "choice", restored) == wire.rendered_hash("test", "q", "choice", original)
    sdk = _to_sdk_questions({"q": restored})["q"]
    assert list(sdk.model_dump()["criteria"]) == ["z", "a"]
    assert wire.declared_options(restored) == ["z", "a"]


@pytest.mark.parametrize("order", [["a", "a"], ["a"], ["a", "b", "c"], "a,b", [1, "b"]])
def test_incomplete_or_forged_order_is_rejected(order: object) -> None:
    with pytest.raises(wire.AnswerError, match="exactly once"):
        _to_sdk_questions({"q": {"type": "choice", "instructions": "choose", "criteria": {"a": "A", "b": "B"}, "criteria_order": order}})
