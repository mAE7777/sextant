"""Role cards: the single place a Jev question lives, hash-locked to its fitted threshold.

A card is a YAML file. The runtime refuses to run a question whose current text hashes
differently from the hash recorded next to its threshold, because a threshold fitted on
one wording means nothing for another.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sextant.schema import Primitive, Rung


class QuestionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primitive: Primitive
    text: str
    options: dict[str, str | None] | None = None
    levels: list[str] | None = None
    escape: str | None = None
    role: Literal["main", "negation_twin", "item", "pairwise", "sufficiency"] = "main"
    criteria_true: str | None = None
    criteria_false: str | None = None
    # Which yes/no answer permits the action this question guards (true for "is it relevant",
    # false for "was a level skipped" and for negation twins). Not sent on the wire; it fixes
    # what counts as a release when the question is calibrated, and certificates bind it.
    permits_on: bool = True


ThresholdStatus = Literal["validated", "provisional", "unfit", "unverified"]


class Threshold(BaseModel):
    """A legacy threshold kept on a card. It grants nothing.

    Certificates now live beside their evidence in the calibration root (`certificates.py`)
    and are re-verified against it. A threshold on the card, whatever its status says, was
    written before that and is shown only as history; the runtime never reads it as authority.
    """

    model_config = ConfigDict(extra="forbid")
    value: float | None = None
    fitted_on: str | None = None
    n: int = 0
    date: str | None = None
    target_agreement: float | None = None
    coverage: float | None = None
    primitive: Primitive | None = None
    status: ThresholdStatus = "unverified"
    dataset_version: str | None = None
    responses_version: str | None = None
    experiment_id: str | None = None
    protocol_version: str | None = None
    n_holdout_effective: int = 0
    holdout_agreement: float | None = None
    risk_lower: float | None = None
    risk_upper: float | None = None
    domain: str | None = None
    cleared_reason: str | None = None


class RoleCard(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    purpose: str
    primitive: Primitive
    state_pack: list[str] = Field(default_factory=list)
    instructions: str
    questions: dict[str, QuestionSpec]
    thresholds: dict[str, Threshold] = Field(default_factory=dict)
    rung: Rung = Rung.jev
    degraded_on: str | None = None
    degrade_reason: str | None = None
    question_hashes: dict[str, str] = Field(default_factory=dict)
    permutations: int = 4
    # Every yes/no question is asked 1 + reruns times on every ask. Re-running only near a
    # fitted threshold made the measurement depend on the fit it fed.
    reruns: int = 3
    contested_margin: float = 0.10
    notes_literalness: str = ""


class RoleCardError(RuntimeError):
    pass


def compute_hash(card: RoleCard, key: str) -> str:
    """The seal covers every card-level byte that goes on the wire for this question.

    That includes the purpose line, which the wire prompt sends with every question.
    """
    q = card.questions[key]
    h = hashlib.sha256()
    parts: list[str] = [card.id, key, q.primitive.value, q.text.strip(), card.instructions.strip(), card.purpose.strip()]
    if q.options:
        for k, v in q.options.items():
            parts.append(f"{k}={v or ''}")
    if q.levels:
        parts.extend(q.levels)
    parts.append(q.escape or "")
    parts.append(q.criteria_true or "")
    parts.append(q.criteria_false or "")
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return "qh_" + h.hexdigest()[:16]


# Fields of the first wire protocol that changed meaning; a card carrying them was written for it.
LEGACY_FIELDS = {
    "boundary_band": "remove it; every yes/no question is now re-run on every ask",
    "boundary_reruns": "rename it to `reruns`",
}


def load_card(path: Path) -> RoleCard:
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise RoleCardError(f"{path}: not a mapping")
    legacy = [f"`{name}` ({fix})" for name, fix in LEGACY_FIELDS.items() if name in data]
    if legacy:
        raise RoleCardError(
            f"{path}: written for the first wire protocol; migrate "
            + "; ".join(legacy)
            + ", then reseal with `python -m sextant rolecards seal`"
        )
    return RoleCard.model_validate(data)


def load_all(cards_dir: Path) -> dict[str, RoleCard]:
    cards: dict[str, RoleCard] = {}
    for p in sorted(cards_dir.glob("*.yaml")):
        if p.name.startswith("_"):
            continue
        card = load_card(p)
        if card.id in cards:
            raise RoleCardError(f"duplicate card id {card.id} in {p}")
        cards[card.id] = card
    return cards


def drifted(card: RoleCard) -> dict[str, str]:
    """Questions whose stored hash is missing or differs from the current text."""
    out: dict[str, str] = {}
    for key in card.questions:
        current = compute_hash(card, key)
        stored = card.question_hashes.get(key)
        if stored is None:
            out[key] = "unhashed"
        elif stored != current:
            out[key] = "drifted"
    return out


def require_sealed(card: RoleCard, key: str) -> str:
    """Return the question hash if the question is sealed; refuse otherwise."""
    state = drifted(card).get(key)
    if state == "unhashed":
        raise RoleCardError(f"{card.id}.{key} has no recorded hash; run `rolecards seal` before using it")
    if state == "drifted":
        raise RoleCardError(f"{card.id}.{key} text changed since its hash was recorded; refit before using it")
    return compute_hash(card, key)


def _dump(card: RoleCard, path: Path) -> None:
    data: dict[str, Any] = card.model_dump(mode="json", exclude_none=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=100))


def seal(path: Path) -> dict[str, str]:
    """Record current hashes for every question; thresholds of drifted questions are voided."""
    card = load_card(path)
    changes: dict[str, str] = {}
    for key in card.questions:
        new = compute_hash(card, key)
        old = card.question_hashes.get(key)
        if old != new:
            changes[key] = f"{old or 'none'} -> {new}"
            card.question_hashes[key] = new
            if key in card.thresholds and old is not None:
                card.thresholds[key] = Threshold(
                    primitive=card.questions[key].primitive,
                    status="unfit",
                    cleared_reason=f"question text changed ({old} -> {new}); refit before using it",
                )
    _dump(card, path)
    return changes


def write_threshold(path: Path, key: str, threshold: Threshold) -> None:
    card = load_card(path)
    if key not in card.questions:
        raise RoleCardError(f"{card.id}: no question {key}")
    require_sealed(card, key)
    card.thresholds[key] = threshold
    _dump(card, path)


def clear_threshold(
    path: Path,
    key: str,
    reason: str,
    date: str | None = None,
    status: ThresholdStatus = "unfit",
) -> None:
    """Revoke one key's threshold and say why.

    A refit that fails must not leave the previous bar in place for the runtime to keep using,
    and the failure of one key must not be masked by another key that still fits. No
    seal is required: clearing is always safe, and a drifted question is exactly the case where
    the old number is least trustworthy. `status` records whether the key is unfit or merely
    still provisional; neither carries a value, so neither can be acted on.
    """
    card = load_card(path)
    if key not in card.questions:
        raise RoleCardError(f"{card.id}: no question {key}")
    card.thresholds[key] = Threshold(
        primitive=card.questions[key].primitive,
        status=status,
        cleared_reason=reason,
        date=date,
    )
    _dump(card, path)


def set_rung(path: Path, rung: Rung, reason: str, date: str) -> None:
    card = load_card(path)
    card.rung = rung
    card.degraded_on = date if rung != Rung.jev else None
    card.degrade_reason = reason if rung != Rung.jev else None
    _dump(card, path)


def cards_hash(cards: dict[str, RoleCard]) -> str:
    h = hashlib.sha256()
    for cid in sorted(cards):
        card = cards[cid]
        for key in sorted(card.questions):
            h.update(compute_hash(card, key).encode())
    return "rc_" + h.hexdigest()[:12]
