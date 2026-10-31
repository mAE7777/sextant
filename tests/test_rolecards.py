from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sextant.rolecards import (
    RoleCardError,
    Threshold,
    cards_hash,
    compute_hash,
    drifted,
    load_all,
    load_card,
    require_sealed,
    seal,
    set_rung,
    write_threshold,
)
from sextant.schema import Primitive, Rung
from sextant.selftest import fx_rolecard

pytestmark = pytest.mark.unit


def _write(tmp_path: Path, data: dict, name: str = "R2-sufficiency.yaml") -> Path:  # type: ignore[type-arg]
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def test_unhashed_then_seal_then_drift(tmp_path: Path) -> None:
    data = fx_rolecard(False)
    data["question_hashes"] = {}
    p = _write(tmp_path, data)
    card = load_card(p)
    assert drifted(card) == {"sufficient": "unhashed"}
    with pytest.raises(RoleCardError, match="no recorded hash"):
        require_sealed(card, "sufficient")
    changes = seal(p)
    assert "sufficient" in changes
    card = load_card(p)
    assert drifted(card) == {}
    assert require_sealed(card, "sufficient") == compute_hash(card, "sufficient")

    # change the wording: the hash drifts and the threshold is voided on the next seal
    data2 = yaml.safe_load(p.read_text())
    data2["questions"]["sufficient"]["text"] += " Really?"
    _write(tmp_path, data2)
    card = load_card(p)
    assert drifted(card) == {"sufficient": "drifted"}
    with pytest.raises(RoleCardError, match="changed"):
        require_sealed(card, "sufficient")
    seal(p)
    card = load_card(p)
    assert card.thresholds["sufficient"].value is None


def test_write_threshold_requires_sealed(tmp_path: Path) -> None:
    data = fx_rolecard(False)
    data["question_hashes"] = {}
    p = _write(tmp_path, data)
    with pytest.raises(RoleCardError):
        write_threshold(p, "sufficient", Threshold(value=0.72, n=150, date="2026-09-18", primitive=Primitive.noul))
    seal(p)
    write_threshold(p, "sufficient", Threshold(value=0.72, n=150, date="2026-09-18", primitive=Primitive.noul))
    assert load_card(p).thresholds["sufficient"].value == 0.72


def test_set_rung_and_cards_hash(tmp_path: Path) -> None:
    p = _write(tmp_path, fx_rolecard(False))
    set_rung(p, Rung.llm_isolated, "held-out agreement below target", "2026-09-18")
    card = load_card(p)
    assert card.rung == Rung.llm_isolated and card.degraded_on == "2026-09-18"
    cards = load_all(tmp_path)
    assert set(cards) == {"R2"}
    assert cards_hash(cards).startswith("rc_")
