"""Meta cards applied to the machine itself: RM1 over every role-card question, RM3 over a slice's
acceptance evidence. Jev as a sensor on the machine, beside the deterministic linter, never alone.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from sextant import wire
from sextant.jev_client import JevClient
from sextant.rolecards import RoleCard
from sextant.schema import Primitive

RM1_KEYS = ("scoping_defined", "asks_computation", "answerable_from_fields", "criteria_consistent")
# RM1 reads "false" for scoping_defined when a question has no scoping word at all (the card's own note);
# the deterministic linter decides that case, so only questions with a scoping word are asked it.
SCOPING_HINTS = ("directly", "major", "enough", "concrete", "same thing", "sufficient", "relevant", "obvious", "reliable", "material", "significant", "bounded", "intermediate", "exercised", "real run", "supported", "absent", "contradict", "scoping", "defined", "arithmetic", "change the choice", "prefers", "observable", "particular person")


def rm1_pack(card: RoleCard, key: str) -> dict[str, Any]:
    q = card.questions[key]
    if q.primitive == Primitive.noul:
        criteria = f"true: {q.criteria_true} | false: {q.criteria_false}"
    elif q.primitive == Primitive.choice:
        criteria = "options: " + json.dumps(q.options, ensure_ascii=False)
    else:
        criteria = "levels: " + json.dumps(q.levels, ensure_ascii=False)
    return {
        "card_instructions": card.instructions.strip(),
        "question_text": q.text,
        "state_fields": ", ".join(card.state_pack),
        "criteria": criteria,
    }


# which answer of each RM1 question marks the wording as a problem
RM1_BAD_ANSWER = {"scoping_defined": False, "asks_computation": True, "answerable_from_fields": False, "criteria_consistent": False}


@dataclass
class RM1Row:
    card_id: str
    key: str
    scoping_defined: float | None
    asks_computation: float | None
    answerable_from_fields: float | None
    criteria_consistent: float | None
    has_scoping_word: bool
    contested: bool
    # per RM1 key: whether that key's run straddled 0.5; empty means `contested` applies to all
    contested_keys: tuple[str, ...] = ()
    # per RM1 key: the calibration state of that judgment in the run that produced this row
    states: dict[str, str] = field(default_factory=dict)

    def _considered(self, thresholds: Mapping[str, float]) -> list[str]:
        return [k for k in RM1_KEYS if k in thresholds and (k != "scoping_defined" or self.has_scoping_word)]

    def _decision(self, key: str, threshold: float) -> bool | None:
        value = getattr(self, key)
        if value is None:
            return None
        contested = key in self.contested_keys if self.contested_keys else self.contested
        return wire.decision(float(value), float(threshold), contested=contested)

    def flags(self, thresholds: Mapping[str, float] | None = None) -> list[str]:
        """Flags under the deployed decision rule, only for questions with a certified threshold.

        An empty map is not a pass and not a fixed 0.5 band. A value the rule abstains on (inside
        the band around the threshold, or from a contested run) is not a flag; it is undecided.
        """
        fitted = dict(thresholds or {})
        return [
            f"{k} {getattr(self, k):.2f}" for k in self._considered(fitted)
            if self._decision(k, fitted[k]) is RM1_BAD_ANSWER[k]
        ]

    def undecided(self, thresholds: Mapping[str, float] | None = None) -> list[str]:
        """Certified questions whose answer here the deployed rule abstains on; never a pass."""
        fitted = dict(thresholds or {})
        return [k for k in self._considered(fitted) if self._decision(k, fitted[k]) is None]


def run_rm1(client: JevClient, cards: dict[str, RoleCard], only: list[str] | None = None) -> list[RM1Row]:
    """Ask RM1 about every other card's questions; the card text is English, declared as such."""
    rows: list[RM1Row] = []
    for cid, card in cards.items():
        if only and cid not in only:
            continue
        if cid == "RM1":
            continue  # a card does not grade its own wording
        for key in card.questions:
            res = client.ask("RM1", rm1_pack(card, key), keys=list(RM1_KEYS), input_language="en")
            vals = {k: res.value(k) for k in RM1_KEYS}
            contested_keys = tuple(k for k in RM1_KEYS if res.estimates[k].contested)
            rows.append(RM1Row(
                card_id=cid, key=key,
                scoping_defined=vals["scoping_defined"], asks_computation=vals["asks_computation"],
                answerable_from_fields=vals["answerable_from_fields"], criteria_consistent=vals["criteria_consistent"],
                has_scoping_word=any(h in card.questions[key].text.lower() for h in SCOPING_HINTS),
                contested=bool(contested_keys), contested_keys=contested_keys,
                states={k: res.estimates[k].estimate.calibration_status.state for k in RM1_KEYS},
            ))
    return rows


def _cell(value: float | None) -> str:
    return "  -  " if value is None else f"{value:5.2f}"


def render_rm1(rows: list[RM1Row], thresholds: Mapping[str, float] | None = None) -> str:
    fitted = dict(thresholds or {})
    lines = ["card  question                 scoping  compute  fields  consistent  flags"]
    for r in rows:
        flags = ", ".join(r.flags(fitted)) or "-"
        lines.append(
            f"{r.card_id:<5} {r.key:<24} {_cell(r.scoping_defined)}    {_cell(r.asks_computation)}   "
            f"{_cell(r.answerable_from_fields)}   {_cell(r.criteria_consistent)}      {flags}"
        )
    flagged = [r for r in rows if r.flags(fitted)]
    if fitted:
        note = f"{len(flagged)} flagged against the fitted thresholds"
    else:
        note = f"{len(flagged)} flagged; no validated threshold, so this is not a pass"
    lines.append("")
    lines.append(f"RM1 over {len(rows)} questions: {note}")
    return "\n".join(lines)


def rm3_pack(criterion: str, evidence_log: str, max_chars: int = 6000) -> dict[str, Any]:
    log = evidence_log if len(evidence_log) <= max_chars else evidence_log[: max_chars // 2] + "\n…\n" + evidence_log[-max_chars // 2 :]
    return {"criterion": criterion, "evidence_log": log}


def run_rm3(client: JevClient, criterion: str, evidence_log: str) -> dict[str, float | None]:
    res = client.ask("RM3", rm3_pack(criterion, evidence_log))
    return {"exercised": res.value("exercised"), "real_run": res.value("real_run")}
