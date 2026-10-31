"""The one versioned Jev wire protocol: how a role card becomes a request, how each raw answer
is checked, and how the answers of one run collapse back into one judgment.

Both halves carry the same version because they are one contract. A judgment recorded under
one version is not evidence under another: the model saw different bytes, or its answers were
combined by different rules, so a threshold, a metric, or a cached measurement taken under the
old version says nothing about the new one.

Three hashes are deliberately distinct:
  template hash  - the sealed role-card question (`rolecards.compute_hash`); survives rendering
  rendered hash  - the exact per-question payload handed to the transport, protocol included
  request digest - the whole request: protocol, model, transport, state and every question

Version 2 changed the run: every yes/no question is asked `1 + reruns` times on every ask
(re-running only near a fitted threshold made the measurement depend on the fit it fed), each
answer is validated against its declared answer space before anything is derived from it,
choice vectors are normalised over the declared options with ties broken by declared order, and
a judgment is re-derived from every call of its run rather than from the calls it names.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Literal

from sextant.ledger import sha_of
from sextant.rolecards import RoleCard, compute_hash

WIRE_PROTOCOL_VERSION = "sextant-jev-wire/3"

# How the contributing answers of one question become the reported value.
Aggregation = Literal["single", "mean_of_runs", "permutation_mean"]
CallRole = Literal["primary", "permutation", "rerun", "paraphrase"]

# Jev reports probabilities rounded to two decimals, so a vector can miss 1 by up to half a
# hundredth per option; one of 90 historical answers summed to 0.99.
ROUNDING_PER_OPTION = 0.005


class AnswerError(ValueError):
    """A raw answer that does not fit the answer space its question declared."""


def compose_instructions(
    card: RoleCard, question_key: str, question_text: str, *, language: str = "en"
) -> str:
    """The instruction string actually sent for one question.

    The card's shared instructions are repeated per question because `instructions` is the
    only channel the transport reads per question; sending them once beside the state would
    not put them in front of the question that depends on them.
    """
    return "\n".join(
        [
            f"[{WIRE_PROTOCOL_VERSION}] role card {card.id} — {card.purpose.strip()}",
            f"Answer language: {language}",
            "Shared card instructions (they apply to every question in this request):",
            card.instructions.strip(),
            f"Question ({card.id}.{question_key}):",
            question_text.strip(),
        ]
    )


def rendered_hash(card_id: str, question_key: str, primitive: str, sdk: dict[str, Any]) -> str:
    """Bind the exact question payload, so a re-worded card cannot reuse an old judgment."""
    return "rq_" + sha_of([WIRE_PROTOCOL_VERSION, card_id, question_key, primitive, ordered_payload(sdk)])


def ordered_payload(sdk: dict[str, Any]) -> dict[str, Any]:
    """Bind order explicitly so sorted JSON artifacts/brokers cannot erase it."""
    result = dict(sdk)
    if isinstance(result.get("criteria"), dict):
        result.setdefault("criteria_order", list(result["criteria"]))
    return result


def ordered_criteria(sdk: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact SDK insertion order after any canonical transport."""
    criteria = sdk.get("criteria") or {}
    if not isinstance(criteria, dict):
        raise AnswerError("ordered criteria requires a mapping")
    order = sdk.get("criteria_order", list(criteria))
    if not isinstance(order, list) or not all(isinstance(k, str) for k in order) or len(order) != len(set(order)) or set(order) != set(criteria):
        raise AnswerError("criteria_order must list every criterion exactly once")
    return {k: criteria[k] for k in order}


def questions_digest(bindings: dict[str, dict[str, str]]) -> str:
    """One digest over every question in a single request."""
    return "qs_" + sha_of([WIRE_PROTOCOL_VERSION, bindings])


def request_digest(
    *,
    model_id: str,
    transport: str,
    state: Any,
    questions: dict[str, dict[str, Any]],
    input_language: str | None = None,
) -> str:
    """The whole request. A declared input language is part of it; an undeclared one adds nothing,
    so rows recorded before the declaration existed keep their digest."""
    parts: list[Any] = [WIRE_PROTOCOL_VERSION, model_id, transport, state,
                        {key: ordered_payload(question) for key, question in questions.items()}]
    if input_language is not None:
        parts.append({"input_language": input_language})
    return "rqst_" + sha_of(parts)


def _obj(value: Any) -> dict[str, Any]:
    """`value` if it is a mapping, else an empty one: recorded rows are data and may be malformed."""
    return value if isinstance(value, dict) else {}


def payload_problems(row: dict[str, Any]) -> list[str]:
    """A recorded call must hash to what it says was sent: every rendered question, the question
    bindings, the whole request and the state."""
    problems: list[str] = []
    request = _obj(row.get("request"))
    questions = _obj(request.get("questions"))
    bindings = _obj(row.get("question_bindings"))
    name = f"call {row.get('id')}"
    keys = row.get("question_keys")
    for key in keys if isinstance(keys, list) else []:
        sdk = _obj(questions.get(key))
        binding = _obj(bindings.get(key))
        if isinstance(sdk.get("criteria"), dict):
            if "criteria_order" not in sdk:
                problems.append(f"{name}: {key} lacks the explicit ordered-wire criteria declaration")
            try:
                ordered_criteria(sdk)
            except AnswerError as exc:
                problems.append(f"{name}: {key} {exc}")
        if rendered_hash(str(row.get("card_id")), key, str(sdk.get("type")), sdk) != binding.get("rendered_hash"):
            problems.append(f"{name}: the recorded payload for {key} does not hash to the rendered question it claims")
    digest = request_digest(
        model_id=str(row.get("model_id")), transport=str(row.get("transport")),
        state=request.get("state"), questions=questions, input_language=row.get("input_language"),
    )
    if digest != request.get("request_digest"):
        problems.append(f"{name}: the recorded request does not hash to its request digest")
    if questions_digest(bindings) != row.get("questions_digest"):
        problems.append(f"{name}: the question bindings do not hash to the recorded digest")
    if sha_of(request.get("state")) != row.get("state_sha"):
        problems.append(f"{name}: the recorded state does not hash to its state hash")
    return problems


# The script each declared input language is written in. The check is at script level: within
# one script (English and Spanish, say) the language is only a declaration.
_SCRIPT_OF_LANGUAGE = {
    **{lang: "latin" for lang in ("en", "es", "fr", "de", "it", "pt", "nl", "sv", "da", "no", "fi", "pl", "cs", "tr", "id", "vi")},
    **{lang: "cjk" for lang in ("zh", "ja", "ko")},
    **{lang: "cyrillic" for lang in ("ru", "uk", "bg", "sr")},
    "ar": "arabic", "he": "hebrew", "el": "greek", "hi": "devanagari", "th": "thai",
}
SCRIPT_SHARE = 0.9


def _script_of(ch: str) -> str | None:
    o = ord(ch)
    if ("a" <= ch <= "z") or ("A" <= ch <= "Z") or 0x00C0 <= o <= 0x024F:
        return "latin" if ch.isalpha() else None
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7AF:
        return "cjk"
    if 0x0400 <= o <= 0x04FF:
        return "cyrillic"
    if 0x0600 <= o <= 0x06FF:
        return "arabic"
    if 0x0590 <= o <= 0x05FF:
        return "hebrew"
    if 0x0370 <= o <= 0x03FF:
        return "greek"
    if 0x0900 <= o <= 0x097F:
        return "devanagari"
    if 0x0E00 <= o <= 0x0E7F:
        return "thai"
    return None


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def input_language_of(state: Any, declared: str | None) -> str | None:
    """The input language a run may claim: the declaration, if the text is written in its script.

    The same predicate decides which calibration items fall in a certified stratum and whether a
    runtime input falls in it. At least 90% of the letters must be in the declared language's
    script; a pack without letters keeps its declaration. An undeclared language, an unknown one,
    or text in another script gives None, which no certificate covers.
    """
    if declared is None:
        return None
    expected = _SCRIPT_OF_LANGUAGE.get(declared)
    if expected is None:
        return None
    counts: dict[str, int] = {}
    for text in _strings(state):
        for ch in text:
            script = _script_of(ch)
            if script is not None:
                counts[script] = counts.get(script, 0) + 1
    letters = sum(counts.values())
    if letters == 0:
        return declared
    return declared if counts.get(expected, 0) >= SCRIPT_SHARE * letters else None


def estimate_tokens(obj: Any) -> int:
    """A deliberately high token estimate for a JSON-serialisable payload.

    `chars / 4` under-counted all 58 historical spike calls, Chinese by about 2.5x. This
    counts a third of a token per ASCII character, one per other character, plus a fixed
    overhead for the request frame. It is a feasibility limit, not a price limit.
    """
    text = json.dumps(obj, ensure_ascii=False)
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return math.ceil(ascii_chars / 3) + (len(text) - ascii_chars) + 400


def request_tokens(state: Any, questions: dict[str, dict[str, Any]]) -> int:
    """Estimated size of the whole request, not just the state pack."""
    return estimate_tokens({"state": state, "questions": questions})


def identity_keys(card: RoleCard, item_key: str) -> list[str]:
    """The questions whose wording fixes a measurement of `item_key`.

    R7 is always asked with its sufficiency twin; any other question with its negation twin.
    """
    if card.id == "R7":
        return sorted(card.questions)
    base_key = item_key.split("[")[0]
    keys = [base_key]
    if f"{base_key}_neg" in card.questions:
        keys.append(f"{base_key}_neg")
    return keys


def run_protocol(card: RoleCard) -> dict[str, Any]:
    """The re-test settings a run is held to; recorded on every call of the run."""
    return {
        "permutations": card.permutations,
        "reruns": card.reruns,
        "contested_margin": card.contested_margin,
    }


def _condition_parts(
    card: RoleCard, keys: list[str], model_id: str, transport: str, language: str
) -> list[Any]:
    return [
        WIRE_PROTOCOL_VERSION,
        card.id,
        {k: compute_hash(card, k) for k in sorted(keys)},
        card.rung.value,
        run_protocol(card),
        model_id,
        transport,
        language,
    ]


def measurement_conditions(
    card: RoleCard, *, keys: list[str], model_id: str, transport: str, language: str
) -> str:
    """Everything that fixes a measurement except which input pack was shown.

    Two measurements of different items belong to one experiment only when this matches. A
    re-worded question or purpose, a re-sealed card, a changed rung or re-test setting, a
    different model, transport or answer language all produce a different value.
    """
    return "mc_" + sha_of(_condition_parts(card, keys, model_id, transport, language))


def measurement_fingerprint(
    card: RoleCard,
    *,
    keys: list[str],
    model_id: str,
    transport: str,
    language: str,
    pack: dict[str, Any],
    leaf_space: dict[str, Any] | None = None,
) -> str:
    """Identity of one measurement: the conditions plus the input it was shown."""
    return "mf_" + sha_of(
        [*_condition_parts(card, keys, model_id, transport, language), pack, leaf_space]
    )


# ---------------------------------------------------------------- answer validation


def declared_options(sdk: dict[str, Any]) -> list[str]:
    """The answer space a question declared on the wire, in declared order."""
    kind = sdk.get("type")
    if kind == "choice":
        try:
            return list(ordered_criteria(sdk))
        except AnswerError:
            return []
    if kind == "score":
        return [str(i) for i in range(len(sdk.get("criteria") or []))]
    return []


def _finite_unit(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0.0 <= float(value) <= 1.0
    )


def validate_answer(sdk: dict[str, Any], answer: Any) -> list[str]:
    """Every way a raw answer can fail to fit the answer space its question declared."""
    if not isinstance(answer, dict):
        return ["the answer is not an object"]
    kind = sdk.get("type")
    if answer.get("type") != kind:
        return [f"answer type {answer.get('type')!r} for a {kind} question"]
    if kind == "noul":
        return [] if _finite_unit(answer.get("noul")) else [f"noul {answer.get('noul')!r} is not in [0, 1]"]
    problems: list[str] = []
    options = declared_options(sdk)
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict):
        return ["the answer carries no probability vector"]
    keys = [str(k) for k in probabilities]
    if sorted(keys) != sorted(options):
        problems.append(f"probability keys {sorted(keys)} differ from the declared {sorted(options)}")
    if not all(_finite_unit(v) for v in probabilities.values()):
        problems.append("a probability is not a finite number in [0, 1]")
    elif abs(sum(float(v) for v in probabilities.values()) - 1.0) > ROUNDING_PER_OPTION * len(options) + 1e-9:
        problems.append(f"probabilities sum to {sum(float(v) for v in probabilities.values()):.4f}")
    if kind == "choice":
        if answer.get("choice") is not None and str(answer.get("choice")) not in options:
            problems.append(f"choice {answer.get('choice')!r} is not a declared option")
    elif kind == "score":
        score = answer.get("score")
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= len(options) - 1
        ):
            problems.append(f"score {score!r} is outside the level range 0..{len(options) - 1}")
    else:
        problems.append(f"unknown question type {kind!r}")
    return problems


def normalised(probabilities: dict[str, Any], options: list[str]) -> dict[str, float]:
    """A validated vector rescaled to sum to one over the declared options, in declared order."""
    total = sum(float(probabilities.get(o, 0.0)) for o in options)
    if total <= 0:
        raise AnswerError("the probability vector has no mass")
    return {o: float(probabilities.get(o, 0.0)) / total for o in options}


# ---------------------------------------------------------------- aggregation


def primary_value(answer: dict[str, Any]) -> float | None:
    """The one number a single raw answer contributes, whatever its primitive."""
    if not isinstance(answer, dict):
        return None
    kind = answer.get("type")
    try:
        if kind == "noul":
            return float(answer["noul"])
        if kind == "score":
            return float(answer["score"])
        if kind == "choice":
            probabilities = answer.get("probabilities") or {}
            choice = answer.get("choice")
            return float(probabilities.get(choice, 0.0)) if choice is not None else None
    except (KeyError, TypeError, ValueError):
        return None
    return None


def replay_scalar(values: list[float], aggregation: Aggregation) -> float | None:
    """Recompute a noul or score from its contributing answers."""
    if not values:
        return None
    if aggregation == "single":
        return values[0] if len(values) == 1 else None
    return statistics.fmean(values)


def replay_choice(
    runs: list[dict[str, float]], options: list[str] | None = None
) -> tuple[dict[str, float], str, bool] | None:
    """Average the normalised runs over the declared options; the winner and whether it tied.

    Options come from the declaration, not from whichever run came first, so an option one run
    omitted is not dropped. A tie is broken by declared order and reported, never hidden.
    """
    if not runs:
        return None
    order = list(options) if options is not None else list(runs[0].keys())
    if not order:
        return None
    averaged = {o: statistics.fmean(float(run.get(o, 0.0)) for run in runs) for o in order}
    total = sum(averaged.values())
    if total <= 0:
        return None
    averaged = {o: v / total for o, v in averaged.items()}
    best = max(averaged.values())
    leaders = [o for o in order if math.isclose(averaged[o], best, abs_tol=1e-12)]
    return averaged, leaders[0], len(leaders) > 1


# ---------------------------------------------------------------- run derivation


@dataclass
class Contribution:
    call_id: str
    variant: str
    role: str
    rendered_hash: str
    value: float | None
    contributes_to_value: bool


@dataclass
class Derived:
    """One question's judgment, re-derived from every call of its run."""

    key: str
    primitive: str
    rendered_hash: str
    template_hash: str
    p: float | None = None
    probabilities: dict[str, float] | None = None
    choice: str | None = None
    score: float | None = None
    contested: bool = False
    aggregation: Aggregation = "single"
    contributions: list[Contribution] = field(default_factory=list)
    rerun_sd: float | None = None
    permutation_sd: float | None = None
    paraphrase_sd: float | None = None
    n_reruns: int = 0
    n_permutations: int = 0
    n_paraphrases: int = 0
    problems: list[str] = field(default_factory=list)


def _answer(row: dict[str, Any], key: str) -> Any:
    return ((row.get("response") or {}).get("answers") or {}).get(key)


def _sdk(row: dict[str, Any], key: str) -> dict[str, Any]:
    found = ((row.get("request") or {}).get("questions") or {}).get(key)
    return found if isinstance(found, dict) else {}


def row_shape_problems(row: Any) -> list[str]:
    """Structural problems of one recorded call row, so a malformed row is reported, not raised on."""
    if not isinstance(row, dict):
        return ["a call row is not an object"]
    name = f"call {row.get('id')}"
    problems = [
        f"{name}: {field_name} is not {kind}"
        for field_name, kind, expected in (
            ("question_keys", "a list", list), ("question_bindings", "an object", dict),
            ("run_protocol", "an object", dict), ("request", "an object", dict), ("response", "an object", dict),
        )
        if not isinstance(row.get(field_name) if row.get(field_name) is not None else expected(), expected)
    ]
    if not problems:
        for part, inner in (("request", "questions"), ("response", "answers")):
            value = row[part].get(inner)
            if value is not None and not isinstance(value, dict):
                problems.append(f"{name}: {part}.{inner} is not an object")
        margin = row["run_protocol"].get("contested_margin", 0.1) if isinstance(row.get("run_protocol"), dict) else 0.1
        if not isinstance(margin, (int, float)) or isinstance(margin, bool):
            problems.append(f"{name}: run_protocol.contested_margin is not a number")
    return problems


def derive_question(rows: list[dict[str, Any]], key: str) -> Derived:
    """Collapse every call of one run that asked `key` into one judgment.

    `rows` are that run's calls in the order they were made. Problems are collected, not
    raised, so a checker can report them; a client treats any problem as a hard error.
    """
    malformed = [p for r in rows for p in row_shape_problems(r)]
    if malformed:
        return Derived(key, "", "", "", problems=malformed)
    asked = [r for r in rows if key in (r.get("question_keys") or [])]
    primaries = [r for r in asked if r.get("call_role") == "primary"]
    if len(primaries) != 1:
        return Derived(key, "", "", "", problems=[f"{len(primaries)} primary calls asked {key}"])
    primary = primaries[0]
    sdk = _sdk(primary, key)
    binding = (primary.get("question_bindings") or {}).get(key) or {}
    out = Derived(
        key=key,
        primitive=str(sdk.get("type")),
        rendered_hash=str(binding.get("rendered_hash", "")),
        template_hash=str(binding.get("template_hash", "")),
    )
    options = declared_options(_sdk(primary, key))
    margin = float((primary.get("run_protocol") or {}).get("contested_margin", 0.1))
    values: list[float] = []
    vectors: list[dict[str, float]] = []
    rerun_vectors: list[dict[str, float]] = []
    paraphrase_values: list[float] = []
    for row in asked:
        role = str(row.get("call_role"))
        answer = _answer(row, key)
        row_sdk = _sdk(row, key)
        problems = validate_answer(row_sdk, answer)
        row_binding = (row.get("question_bindings") or {}).get(key) or {}
        if problems:
            out.problems.extend(f"call {row.get('id')}: {p}" for p in problems)
            continue
        if out.primitive == "choice":
            vector = normalised(answer["probabilities"], declared_options(row_sdk))
            feeds = role in ("primary", "permutation")
            if feeds:
                vectors.append(vector)
            elif role == "rerun":
                rerun_vectors.append(vector)
            out.contributions.append(Contribution(str(row.get("id")), str(row.get("variant")), role,
                                                  str(row_binding.get("rendered_hash", "")), None, feeds))
        elif out.primitive == "noul":
            value = float(answer["noul"])
            feeds = role in ("primary", "rerun")
            if feeds:
                values.append(value)
            elif role == "paraphrase":
                paraphrase_values.append(value)
            out.contributions.append(Contribution(str(row.get("id")), str(row.get("variant")), role,
                                                  str(row_binding.get("rendered_hash", "")), value, feeds))
        elif out.primitive == "score":
            value = float(answer["score"])
            feeds = role == "primary"
            out.contributions.append(Contribution(str(row.get("id")), str(row.get("variant")), role,
                                                  str(row_binding.get("rendered_hash", "")), value, feeds))
            if feeds:
                out.score = value
                out.probabilities = normalised(answer["probabilities"], declared_options(row_sdk))
        else:
            out.problems.append(f"unknown primitive {out.primitive!r}")
    out.n_reruns = sum(1 for c in out.contributions if c.role == "rerun")
    out.n_permutations = sum(1 for c in out.contributions if c.role == "permutation")
    out.n_paraphrases = sum(1 for c in out.contributions if c.role == "paraphrase")
    if out.problems:
        return out
    if out.primitive == "noul":
        out.p = statistics.fmean(values)
        out.aggregation = "mean_of_runs" if len(values) > 1 else "single"
        out.rerun_sd = statistics.pstdev(values) if len(values) > 1 else None
        out.contested = any(v >= 0.5 for v in values) and any(v < 0.5 for v in values)
        if paraphrase_values:
            out.paraphrase_sd = statistics.pstdev([out.p, *paraphrase_values])
    elif out.primitive == "choice":
        replayed = replay_choice(vectors, options)
        if replayed is None:
            out.problems.append("the choice runs have no probability mass")
            return out
        out.probabilities, out.choice, tied = replayed
        out.aggregation = "permutation_mean" if len(vectors) > 1 else "single"
        support = [v[out.choice] for v in vectors]
        out.permutation_sd = statistics.pstdev(support) if len(support) > 1 else None
        feeding = [c for c in out.contributions if c.contributes_to_value]
        for c, vector in zip(feeding, vectors, strict=True):
            c.value = vector[out.choice]
        stability_runs = [c for c in out.contributions if c.role == "rerun"]
        for c, vector in zip(stability_runs, rerun_vectors, strict=True):
            c.value = vector[out.choice]
        out.contested = choice_contested(out.probabilities, vectors, out.choice, tied, margin)
        rerun_support = [support[0], *(v[out.choice] for v in rerun_vectors)]
        out.rerun_sd = statistics.pstdev(rerun_support) if len(rerun_support) > 1 else None
    return out


def choice_contested(
    probabilities: dict[str, float], vectors: list[dict[str, float]], choice: str, tied: bool, margin: float
) -> bool:
    """A choice is contested when it tied, won by less than the margin, or flipped with order.

    Decided from the primary and permutation runs only, so whether stability re-runs were
    owed is fixed before they happen and an audit can check they were made.
    """
    top2 = sorted(probabilities.values(), reverse=True)[:2]
    thin = len(top2) == 2 and top2[0] - top2[1] < margin
    if len(vectors) < 2:
        return tied or thin
    winners = [max(v, key=lambda o: v[o]) for v in vectors]
    unstable = sum(1 for w in winners if w != choice) >= max(1, len(vectors) // 2)
    support = [v[choice] for v in vectors]
    unsteady = statistics.pstdev(support) > margin
    return tied or thin or unstable or unsteady


def answer_language(row: dict[str, Any], key: str) -> str:
    """The answer language one recorded call asked for, read from the instructions it sent."""
    text = str(_sdk(row, key).get("instructions") or "")
    for line in text.splitlines():
        if line.startswith("Answer language: "):
            return line.removeprefix("Answer language: ").strip()
    return ""


def asked_base_keys(rows: list[dict[str, Any]]) -> list[str]:
    """The questions the primary call of a run asked, by card key (items and pairs folded)."""
    primary = next((r for r in rows if r.get("call_role") == "primary"), None)
    keys = (primary or {}).get("question_keys") or []
    return sorted({str(k).split("[")[0] for k in keys})


def decision(p: float, threshold: float, *, contested: bool) -> bool | None:
    """The automatic yes/no answer at a threshold, or None to abstain.

    This is the one decision rule: certification, threshold selection, the runtime and release
    all use it, so what a certificate bounds is what gets deployed. A run whose answers straddle
    0.5 (`contested`) abstains. At t = 0.5 a value of exactly 0.5 would be both yes and no; it
    abstains too.
    """
    if contested:
        return None
    yes = p >= threshold and p > 1.0 - threshold
    no = p <= 1.0 - threshold and p < threshold
    if yes:
        return True
    if no:
        return False
    return None


def negation_gap(rows: list[dict[str, Any]], key: str) -> float | None:
    """|p(key) + p(key_neg) - 1| over the run's derived values, when the twin was asked."""
    twin = f"{key}_neg"
    if not any(twin in (r.get("question_keys") or []) for r in rows):
        return None
    a, b = derive_question(rows, key), derive_question(rows, twin)
    if a.p is None or b.p is None or a.problems or b.problems:
        return None
    return abs(a.p + b.p - 1.0)
