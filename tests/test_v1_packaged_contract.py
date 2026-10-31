"""Delivered contract resources must describe the current callable interface."""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

from jsonschema import Draft202012Validator

from sextant.cli import schemas


def test_packaged_protocol_matches_current_interface() -> None:
    packaged = json.loads(resources.files("sextant.resources").joinpath("v1-protocol.schema.json").read_text())
    assert packaged == schemas()


def test_packaged_options_accept_current_codex_examples() -> None:
    packaged = json.loads(resources.files("sextant.resources").joinpath("v1-protocol.schema.json").read_text())
    validator = Draft202012Validator(packaged["schemas"]["CompileOptions"])
    examples = Path(__file__).resolve().parents[1] / "examples" / "v1"
    for name in ("finite-table-payload.json", "finite-table-start.json", "service-cache-payload.json"):
        payload = json.loads((examples / name).read_text())
        options = payload.get("options", payload.get("payload", {}).get("options"))
        assert options is not None
        validator.validate(options)
