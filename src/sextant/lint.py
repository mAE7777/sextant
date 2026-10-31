"""Deterministic linter for role-card questions (the mechanical half of the question style rules).

RM1, the Jev meta card, judges what this cannot (is a scoping word really defined, is the
question single-hop); this catches what code can: question marks, double negation, forbidden
computations, missing escape options, missing legends, undefined scoping words by string match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sextant.rolecards import QuestionSpec, RoleCard, load_all
from sextant.schema import Primitive

SCOPING_WORDS = (
    "directly", "major", "enough", "concrete", "same thing", "sufficient", "relevant", "obvious",
    "reliable", "material", "significant", "bounded", "intermediate", "exercised", "real run",
    "supported", "absent from", "contradict", "scoping word", "defined", "arithmetic",
    "change the choice", "prefers", "observable", "particular person",
)

FORBIDDEN_PHRASES = (
    "how many", "count the", "number of", "before or after", "how long", "sum of", "average of",
    "percent", "earlier than", "later than", "older than", "newer than", "how much later",
    "add up", "multiply", "divide", "subtract",
)

DOUBLE_NEGATION = (
    re.compile(r"\bnot\b[^?.;]*\b(?:un\w+|not|never|no)\b", re.IGNORECASE),
    re.compile(r"\bn't\b[^?.;]*\bnot\b", re.IGNORECASE),
    re.compile(r"\bnever\b[^?.;]*\bnot\b", re.IGNORECASE),
)

MAX_QUESTION_CHARS = 240


@dataclass
class LintFinding:
    card_id: str
    key: str
    code: str
    message: str


def _scoping_words_in(text: str) -> list[str]:
    low = text.lower()
    return [w for w in SCOPING_WORDS if w in low]


def lint_question(card: RoleCard, key: str, q: QuestionSpec) -> list[LintFinding]:
    out: list[LintFinding] = []
    text = q.text.strip()

    def add(code: str, msg: str) -> None:
        out.append(LintFinding(card.id, key, code, msg))

    is_placeholder = re.fullmatch(r"\{[a-z_]+\}", text) is not None  # rendered from the node at call time
    if not text.endswith("?") and not is_placeholder:
        add("Q001", "question text must end with a question mark")
    if len(text) > MAX_QUESTION_CHARS:
        add("Q009", f"question text is {len(text)} chars; keep it under {MAX_QUESTION_CHARS}")
    for pat in DOUBLE_NEGATION:
        if pat.search(text):
            add("Q002", "double negation in the question; ask it the other way round in plain words")
            break
    low = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        if phrase in low:
            add("Q004", f"asks for a computation Jev cannot do ({phrase!r}); do it in code")
    instr = card.instructions.lower()
    for w in _scoping_words_in(text):
        if w not in instr:
            add("Q003", f"scoping word {w!r} is used but not defined in the instructions")
    if q.primitive == Primitive.choice:
        opts = q.options or {}
        if not (2 <= len(opts) <= 32):
            add("Q005", f"choice needs 2 to 32 options, has {len(opts)}")
        if not q.escape or q.escape not in opts:
            add("Q005", "choice needs an escape option that is one of its options")
        for name, desc in opts.items():
            if not desc or len(desc.split()) < 2:
                add("Q005", f"option {name!r} needs a plain-sentence description")
    if q.primitive == Primitive.score:
        levels = q.levels or []
        if not (2 <= len(levels) <= 10):
            add("Q006", f"score needs 2 to 10 levels, has {len(levels)}")
        for i, lv in enumerate(levels):
            if len(lv.split()) < 3:
                add("Q006", f"level {i} legend is too short to stand on its own: {lv!r}")
    if q.primitive == Primitive.noul:
        if not q.criteria_true or not q.criteria_false:
            add("Q007", "noul needs criteria_true and criteria_false")
        elif q.criteria_true.strip().lower() == q.criteria_false.strip().lower():
            add("Q007", "criteria_true and criteria_false are identical")
    if q.role == "item" and "{n}" not in text:
        add("Q008", "an item question must reference the item as {n}")
    if q.role == "pairwise" and not ("{i}" in text and "{j}" in text):
        add("Q008", "a pairwise question must reference both items as {i} and {j}")
    if q.role == "negation_twin":
        base = key[:-4] if key.endswith("_neg") else None
        if base is None or base not in card.questions:
            add("Q010", "a negation twin's key must be <base>_neg with <base> present")
    return out


def lint_card(card: RoleCard) -> list[LintFinding]:
    out: list[LintFinding] = []
    if not card.instructions.strip():
        out.append(LintFinding(card.id, "-", "C001", "instructions are empty"))
    if not card.state_pack:
        out.append(LintFinding(card.id, "-", "C002", "state_pack is empty"))
    if not card.questions:
        out.append(LintFinding(card.id, "-", "C003", "no questions"))
    for key, q in card.questions.items():
        out.extend(lint_question(card, key, q))
    return out


def lint_dir(cards_dir: Path) -> dict[str, list[LintFinding]]:
    cards = load_all(cards_dir)
    return {cid: lint_card(card) for cid, card in cards.items()}


def render(results: dict[str, list[LintFinding]]) -> str:
    lines: list[str] = []
    total = 0
    for cid in sorted(results):
        fs = results[cid]
        total += len(fs)
        lines.append(f"{'PASS' if not fs else 'FAIL':<5} {cid:<5} {len(fs)} finding(s)")
        for f in fs:
            lines.append(f"      {f.code} {f.key}: {f.message}")
    lines.append("")
    lines.append(f"lint: {len(results)} cards, {total} findings")
    return "\n".join(lines)
