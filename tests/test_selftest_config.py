from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from sextant.config import DEFAULTS, dump_toml, load_config, write_config
from sextant.selftest import CHECKS, run_selftest, write_fixtures

pytestmark = pytest.mark.unit


def test_every_check_proves_on_its_plant() -> None:
    outcomes = run_selftest()
    assert len(outcomes) == len(CHECKS)
    assert all(o.verdict == "PASS" for o in outcomes), [o for o in outcomes if o.verdict != "PASS"]


def test_disabled_check_is_reported_broken() -> None:
    outcomes = run_selftest({"residual"})
    by_name = {o.name: o for o in outcomes}
    assert by_name["residual"].verdict == "BROKEN"
    assert by_name["acyclic"].verdict == "PASS"


def test_fixtures_export(tmp_path: Path) -> None:
    paths = write_fixtures(tmp_path)
    assert len(paths) == 2 * len(CHECKS)
    assert any(p.name == "rolecard-hash.plant.yaml" for p in paths)


def test_toml_round_trip(tmp_path: Path) -> None:
    text = dump_toml(DEFAULTS)
    back = tomllib.loads(text)
    assert back["jev"]["model_id"] == DEFAULTS["jev"]["model_id"]
    assert back["tiers"]["T2"]["depth_max"] == 4
    assert back["providers"]["auto_fulfil"] == DEFAULTS["providers"]["auto_fulfil"]
    p = write_config(DEFAULTS, tmp_path / "config.toml")
    loaded = load_config(p)
    assert loaded["calibration"]["outcome_bar"] == 30
