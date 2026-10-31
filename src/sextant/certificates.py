"""Calibration certificates: what gives a Jev threshold authority, and how that is checked.

A threshold is a claim that a question's automatic answers are right often enough to act on.
The claim lives beside the evidence that supports it, in `<calibration root>/<card>/
certificates.jsonl`, never in the role card: the card is the sealed wording that goes on the
wire, and package resources are not a place to store evidence-derived authority.

`certificate_state` is the only way the runtime learns whether a question is calibrated. It does
not trust a stored status. A `validated` entry grants authority only while the data it names
still exist on disk and reproduce it: the fit record, the confirming cases, the responses, the
gold, the measurement ledger rows and the holdout log (the re-verification itself lives in
`calibrate.verify_certificate`). Signed, runtime-only issuance is not built yet. Until it is, this
re-verification is the guard that keeps a hand-written certificate from granting anything.

The certification standard is the provisional first-version standard, applied with the case as
the unit. Each confirming case contributes exactly one decision, its
representative item, so the three quantities are exact binomial proportions over independent
cases:

- eligible: every unseen holdout case with an item for the key in the certified input-language
  stratum; a representative that was never measured, or whose answer was invalid, abstains
  rather than leaving the denominator
- automatic (A): eligible cases decided without abstaining (`wire.decision`: p >= t or p <= 1 - t,
  a tie at t = 0.5 abstains, and a contested run, whose answers straddle 0.5, abstains exactly as
  it does at runtime)
- releases: automatic decisions whose answer is the one that permits the action (`permits_on`)
- R = wrong automatic decisions / A; an automatic decision without deterministic or human truth
  counts as wrong
- R+ = releases whose truth says not to release / releases; zero releases leaves R+ undefined
- C = A / eligible

Deployable only if the one-sided Clopper-Pearson upper bounds of R and R+ are at most 0.05 and
the lower bound of C is at least 0.50. Alpha is spent across attempts (attempt j of a key under
one set of measurement conditions confirms at 0.05 / 2^j, so repeated tries on fresh cases cannot
add up past 0.05) and split by Bonferroni over the three bounds and the card's fixed family of
yes/no `main` questions. The intersection of the three bounds would not need the split. It is
kept so the standard reads literally.

The stratum is the canonical input language: only holdout items declared in it, with text in its
script (`wire.input_language_of`), are eligible, and at runtime only inputs that pass the same
predicate are covered. Within one script the language is a declaration, not a measurement.

The claim is about the per-case decision error under the assumption that cases are independent:
items that share a group, a translation pair, a pack or a generation lineage are one case, and a
case contributes one decision. It says nothing about a decision-weighted error when several
decisions share a case. Isolation of the evidence is "none, self-logged": the runtime wrote it
and nothing outside the process attests it.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from sextant.ledger import exclusive, now_iso
from sextant.rolecards import RoleCard
from sextant.schema import Primitive, Rung

CALIBRATION_PROTOCOL_VERSION = "sextant-calibration/2"
# v2: a contested run abstains in certification exactly as at runtime, and a certificate covers
# one input-language stratum (the canonical language) checked at script level.
STANDARD_ID = "validation-4-provisional/per-case-v2"
R_MAX = 0.05
R_PLUS_MAX = 0.05
C_MIN = 0.50
ALPHA = 0.05
BOUNDS_PER_KEY = 3
TRUTH_KINDS = frozenset({"deterministic_truth", "human_adjudication"})

CertificateStatus = Literal["validated", "provisional", "unfit", "revoked"]
CalibrationState = Literal[
    "pending", "provisional", "calibrated", "degraded", "invalidated", "out_of_domain"
]


class Certificate(BaseModel):
    """One entry in a card's certificate log. The last entry for a key is in effect."""

    model_config = ConfigDict(extra="forbid")
    key: str
    status: CertificateStatus
    value: float | None = None
    reason: str = ""
    at: str | None = None
    experiment_id: str | None = None
    dataset_version: str | None = None
    responses_version: str | None = None
    calibration_protocol_version: str | None = None
    wire_protocol_version: str | None = None
    standard: str | None = None
    measurement_conditions: str | None = None
    model_id: str | None = None
    transport: str | None = None
    language: str | None = None
    # the certified input-language stratum; inputs outside it are out of domain
    input_language: str | None = None
    permits_on: bool | None = None
    asked_keys: list[str] = Field(default_factory=list)
    family: list[str] = Field(default_factory=list)
    attempt: int = 1
    alpha_per_bound: float | None = None
    languages: dict[str, int] = Field(default_factory=dict)
    # golden rows present when the fit ran; the eligible cases are re-derived from this prefix
    golden_rows: int = 0
    isolation: str = "none, self-logged"
    target: float | None = None
    fit_items: list[str] = Field(default_factory=list)
    confirming: list[dict[str, str]] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)
    bounds: dict[str, float | None] = Field(default_factory=dict)
    digests: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------- store


def certificates_path(root: Path, card_id: str) -> Path:
    return root / card_id / "certificates.jsonl"


def read_certificates(root: Path, card_id: str) -> list[Certificate]:
    path = certificates_path(root, card_id)
    if not path.exists():
        return []
    return [
        Certificate.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def append_certificate(root: Path, card_id: str, certificate: Certificate) -> Certificate:
    """Append-only: a revocation or refit is a new entry, the old one stays readable."""
    path = certificates_path(root, card_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = certificate if certificate.at else certificate.model_copy(update={"at": now_iso()})
    with exclusive(path.parent), path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
    _CACHE.clear()
    return entry


def effective_certificate(root: Path, card_id: str, key: str) -> Certificate | None:
    found = None
    for entry in read_certificates(root, card_id):
        if entry.key == key:
            found = entry
    return found


def revoke_validated(root: Path, card_id: str, reason: str, keys: list[str] | None = None) -> list[str]:
    """Withdraw authority from every validated key of a card (or the named keys)."""
    revoked: list[str] = []
    latest: dict[str, Certificate] = {}
    for entry in read_certificates(root, card_id):
        latest[entry.key] = entry
    for key, entry in sorted(latest.items()):
        if entry.status != "validated" or (keys is not None and key not in keys):
            continue
        append_certificate(root, card_id, Certificate(key=key, status="revoked", reason=reason))
        revoked.append(key)
    return revoked


# ---------------------------------------------------------------- exact binomial bounds


def _log_pmf(i: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if i == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if i == n else -math.inf
    return (
        math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        + i * math.log(p) + (n - i) * math.log1p(-p)
    )


def _cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space for stability."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    terms = [_log_pmf(i, n, p) for i in range(k + 1)]
    top = max(terms)
    if top == -math.inf:
        return 0.0
    return min(1.0, math.exp(top) * sum(math.exp(t - top) for t in terms))


def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """One-sided exact upper bound: the largest p with P(X <= k; n, p) >= alpha."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided exact lower bound: the smallest p with P(X >= k; n, p) >= alpha."""
    if n <= 0 or k <= 0:
        return 0.0
    return 1.0 - clopper_pearson_upper(n - k, n, alpha)


# ---------------------------------------------------------------- the standard


@dataclass
class Confirmation:
    """What one holdout says about one threshold, under the per-case standard."""

    eligible: int = 0
    automatic: int = 0
    wrong: int = 0
    releases: int = 0
    bad_releases: int = 0
    abstentions: int = 0
    should_not_release: int = 0
    released_when_should_not: int = 0
    should_release: int = 0
    rejected_when_should: int = 0
    untruthed_automatic: int = 0
    alpha_per_bound: float = 0.0
    r_upper: float | None = None
    r_plus_upper: float | None = None
    c_lower: float | None = None
    deployable: bool = False
    reason: str = ""
    confirming: list[dict[str, str]] = field(default_factory=list)

    @property
    def fpr(self) -> float | None:
        return self.released_when_should_not / self.should_not_release if self.should_not_release else None

    @property
    def false_rejection_rate(self) -> float | None:
        return self.rejected_when_should / self.should_release if self.should_release else None

    def counts(self) -> dict[str, int]:
        return {
            "eligible": self.eligible, "automatic": self.automatic, "wrong": self.wrong,
            "releases": self.releases, "bad_releases": self.bad_releases,
            "abstentions": self.abstentions, "should_not_release": self.should_not_release,
            "released_when_should_not": self.released_when_should_not,
            "should_release": self.should_release, "rejected_when_should": self.rejected_when_should,
            "untruthed_automatic": self.untruthed_automatic,
        }

    def bounds(self) -> dict[str, float | None]:
        return {"r_upper": self.r_upper, "r_plus_upper": self.r_plus_upper, "c_lower": self.c_lower}


def alpha_per_bound(family_size: int, attempt: int = 1) -> float:
    """Alpha for one bound of one key on its `attempt`-th confirmation under fixed conditions."""
    return ALPHA / float(1 << max(1, attempt)) / (BOUNDS_PER_KEY * max(1, family_size))


def family_of(card: RoleCard) -> list[str]:
    """The card's fixed family: every yes/no `main` question. Never chosen by the caller."""
    return sorted(k for k, q in card.questions.items() if q.primitive == Primitive.noul and q.role == "main")


def representatives(rows: list[dict[str, Any]], canonical_language: str = "en") -> list[dict[str, Any]]:
    """One row per case: the canonical-language item first, then the lowest item id."""
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(str(r["group"]), []).append(r)
    chosen: list[dict[str, Any]] = []
    for group in sorted(by_group):
        members = sorted(
            by_group[group],
            key=lambda r: (r.get("lang", canonical_language) != canonical_language, str(r["item_id"])),
        )
        chosen.append(members[0])
    return chosen


def resolve_truth(rows: list[dict[str, Any]]) -> tuple[Any, str | None, str]:
    """One truth resolution for gold and for certificate verification.

    `rows` are deterministic-truth and human-adjudication labels (`kind`, `label`). Every one
    must agree for the item to resolve; a marker label makes it ambiguous; any disagreement is a
    conflict. A conflict never falls back to a model majority. Returns (label, kind, outcome).
    """
    truth = [r for r in rows if r.get("kind") in TRUTH_KINDS]
    if not truth:
        return None, None, "none"
    labels = [r.get("label") for r in truth]
    if any(isinstance(label, str) and label in ("ambiguous", "insufficient") for label in labels):
        return None, None, "ambiguous"
    if len({json.dumps(label, sort_keys=True) for label in labels}) != 1:
        return None, None, "conflict"
    kind = "deterministic_truth" if any(r.get("kind") == "deterministic_truth" for r in truth) else "human_adjudication"
    return labels[0], kind, "resolved"


def truth_of(row: dict[str, Any]) -> bool | None:
    """Deterministic or human truth as a strict boolean; anything else is not truth."""
    gold = row.get("gold")
    if row.get("resolved_by") in TRUTH_KINDS and type(gold) is bool:
        return gold
    return None


def confirm(
    rows: list[dict[str, Any]],
    threshold: float,
    *,
    permits_on: bool,
    family_size: int,
    attempt: int = 1,
    canonical_language: str = "en",
) -> Confirmation:
    """Apply the per-case standard to unseen holdout rows (the caller filters out seen cases).

    Every case counts as eligible. A representative with no usable measurement (never
    measured, or its answer was invalid) abstains, which lowers coverage; it cannot vanish
    from the denominator.
    """
    from sextant.wire import decision

    out = Confirmation(alpha_per_bound=alpha_per_bound(family_size, attempt))
    for row in representatives(rows, canonical_language):
        out.confirming.append({"group": str(row["group"]), "item_id": str(row["item_id"]), "lang": str(row.get("lang", canonical_language))})
        out.eligible += 1
        truth = truth_of(row)
        if truth is not None:
            if truth == permits_on:
                out.should_release += 1
            else:
                out.should_not_release += 1
        p = row.get("p")
        answer = (
            decision(float(p), threshold, contested=bool(row.get("contested")))
            if p is not None and math.isfinite(float(p)) else None
        )
        if answer is None:
            out.abstentions += 1
            continue
        said_yes = answer
        out.automatic += 1
        release = said_yes == permits_on
        if truth is None:
            out.untruthed_automatic += 1
            out.wrong += 1
            if release:
                out.releases += 1
                out.bad_releases += 1
            continue
        if said_yes != truth:
            out.wrong += 1
        if release:
            out.releases += 1
            if truth != permits_on:
                out.bad_releases += 1
                out.released_when_should_not += 1
        elif truth == permits_on:
            out.rejected_when_should += 1
    a = out.alpha_per_bound
    if out.automatic:
        out.r_upper = clopper_pearson_upper(out.wrong, out.automatic, a)
    if out.releases:
        out.r_plus_upper = clopper_pearson_upper(out.bad_releases, out.releases, a)
    if out.eligible:
        out.c_lower = clopper_pearson_lower(out.automatic, out.eligible, a)
    problems: list[str] = []
    if not out.eligible:
        problems.append("no eligible unseen holdout case")
    if not out.releases:
        problems.append("no automatic release, so the release risk is undefined")
    if out.r_upper is None or out.r_upper > R_MAX:
        problems.append(f"error upper bound {out.r_upper} exceeds {R_MAX}")
    if out.r_plus_upper is not None and out.r_plus_upper > R_PLUS_MAX:
        problems.append(f"release-risk upper bound {out.r_plus_upper} exceeds {R_PLUS_MAX}")
    if out.c_lower is None or out.c_lower < C_MIN:
        problems.append(f"coverage lower bound {out.c_lower} is under {C_MIN}")
    out.deployable = not problems
    out.reason = (
        f"{out.automatic} of {out.eligible} cases decided automatically, {out.wrong} wrong, "
        f"{out.releases} releases, {out.bad_releases} bad; alpha per bound {a:.5f}"
        + ("" if out.deployable else "; " + "; ".join(problems))
    )
    return out


# ---------------------------------------------------------------- runtime state


@dataclass
class Verdict:
    state: CalibrationState
    reason: str
    certificate: Certificate | None = None


_CACHE: dict[tuple[Any, ...], Verdict] = {}


EVIDENCE_FILES = (
    "certificates.jsonl", "responses.jsonl", "truth.jsonl", "labels.jsonl", "gold.jsonl",
    "golden.jsonl", "holdout_evaluations.jsonl", "fit.json", "ledger/jev-calls.jsonl",
)


def _evidence_stamp(root: Path, card_id: str) -> tuple[tuple[str, str], ...]:
    """Content hashes of every evidence file; a same-size rewrite cannot reuse a verdict."""
    stamp: list[tuple[str, str]] = []
    for name in EVIDENCE_FILES:
        path = root / card_id / name
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except FileNotFoundError:
            digest = "missing"
        stamp.append((name, digest))
    superseded = root / card_id / "superseded"
    if superseded.exists():
        for path in sorted(superseded.glob("fit-*.json")):
            stamp.append((path.name, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(stamp)


def certificate_state(
    card: RoleCard,
    key: str,
    *,
    model_id: str,
    transport: str,
    language: str,
    root: Path | None,
    asked_keys: list[str],
    input_language: str | None,
) -> Verdict:
    """How much authority this question's threshold has right now, re-derived from evidence.

    `asked_keys` is the composition of the run being judged (the questions asked together);
    a certificate covers only the composition it was measured under. `input_language` is the
    run's effective input language (`wire.input_language_of`); None, or a language other than
    the certified stratum, is out of domain. Evidence that cannot even be read is invalidated,
    never an exception.
    """
    try:
        return _certificate_state(card, key, model_id=model_id, transport=transport, language=language,
                                  root=root, asked_keys=asked_keys, input_language=input_language)
    except Exception as exc:  # noqa: BLE001 - unreadable evidence grants nothing and must not crash an ask
        return Verdict("invalidated", f"evidence cannot be read: {type(exc).__name__}: {str(exc)[:200]}")


def _certificate_state(
    card: RoleCard,
    key: str,
    *,
    model_id: str,
    transport: str,
    language: str,
    root: Path | None,
    asked_keys: list[str],
    input_language: str | None,
) -> Verdict:
    if card.rung != Rung.jev:
        return Verdict("degraded", f"card runs at rung {card.rung.value}")
    if root is None:
        return Verdict("pending", "no calibration root is configured")
    question = card.questions.get(key)
    if question is None:
        return Verdict("invalidated", f"{card.id} has no question {key}")
    entry = effective_certificate(root, card.id, key)
    if entry is None:
        return Verdict("pending", "no certificate has been issued for this question")
    if entry.status == "provisional":
        return Verdict("provisional", entry.reason or "the holdout has not shown the target", entry)
    if entry.status in ("unfit", "revoked"):
        return Verdict("invalidated", f"{entry.status}: {entry.reason}", entry)
    from sextant import wire  # local: wire is the protocol owner and imports role cards

    if entry.calibration_protocol_version != CALIBRATION_PROTOCOL_VERSION or entry.wire_protocol_version != wire.WIRE_PROTOCOL_VERSION:
        return Verdict("invalidated", "certificate was issued under a superseded protocol", entry)
    if entry.standard != STANDARD_ID:
        return Verdict("invalidated", "certificate was issued under a different standard", entry)
    if question.primitive != Primitive.noul or entry.permits_on != question.permits_on:
        return Verdict("invalidated", "the card's release polarity differs from the certificate's", entry)
    now = wire.measurement_conditions(
        card, keys=wire.identity_keys(card, key), model_id=model_id, transport=transport, language=language
    )
    if entry.measurement_conditions != now:
        return Verdict(
            "out_of_domain",
            "certificate conditions differ from the current card, model, transport or language",
            entry,
        )
    if sorted(asked_keys) != sorted(entry.asked_keys):
        return Verdict(
            "out_of_domain",
            f"this run asked {sorted(asked_keys)} together; the certificate covers {sorted(entry.asked_keys)}",
            entry,
        )
    if input_language != entry.input_language:
        shown = "undeclared, or not written in its declared script" if input_language is None else input_language
        return Verdict(
            "out_of_domain",
            f"the input language is {shown}; the certificate covers {entry.input_language} inputs",
            entry,
        )
    cache_key = (str(root), card.id, key, entry.model_dump_json(), _evidence_stamp(root, card.id))
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    from sextant import calibrate  # local: calibration imports the client, which imports this

    problems = calibrate.verify_certificate(card, entry, root)
    verdict = (
        Verdict("invalidated", "certificate does not reproduce from its evidence: " + "; ".join(problems[:3]), entry)
        if problems
        else Verdict("calibrated", "certificate reproduces from its evidence", entry)
    )
    _CACHE[cache_key] = verdict
    return verdict
