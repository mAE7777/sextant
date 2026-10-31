"""Calibration harness: golden sets, typed labels, Jev measurements recorded in a replayable
ledger, proper-scoring metrics, threshold fitting, and certificates that re-verify.

Layout: <calibration root>/<card>/
  golden.jsonl        items (pack + question key + author's expected label + split + lang + group + lineage)
  labels.jsonl        grader labels (provider, family, isolation, provenance recorded per label)
  truth.jsonl         deterministic truth and human adjudications, kept apart from model votes
  gold.jsonl          resolved labels, with what resolved them and how many independent sources
  disagreements.md    the items a person spot-checks
  ledger/             the measurement ledger: every Jev call of every measurement, replayable
  requests/           the grader handshake for this card
  responses.jsonl     Jev's derived answer per item, with its run id and measurement identity
  holdout_evaluations.jsonl  every scoring of locked items, by item, so a case confirms once
  certificates.jsonl  threshold certificates, append-only (see certificates.py)
  metrics.json        Brier, log loss, binned Murphy decomposition, AUROC, coverage curve
  fit.json            the latest fit, with what it rested on
  superseded/         previous metrics.json and fit.json, kept rather than overwritten
  report.md           the human-readable report (internal; never published)
<calibration root>/canary.jsonl   frozen items with stored answers and their measurement identity
Numbers here never leave the calibration folder (TypeSafe's customer agreement).

Five things are deliberately different fields and never collapse into each other: what the
author expected, what a model voted, what a human adjudicated, what a deterministic procedure
established, and what was finally resolved. A split is taken over real cases, never over rows:
items sharing a group, a translation pair, a pack or a generation lineage are one case. A
threshold is a certificate re-derived from its evidence, not a float on a card. The calibration
root comes from config or the caller; the historical `calibration/` directory is never a default,
and writers refuse a card directory that holds rows from an earlier protocol.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from sextant import wire
from sextant.certificates import (
    CALIBRATION_PROTOCOL_VERSION,
    STANDARD_ID,
    TRUTH_KINDS,
    Certificate,
    append_certificate,
    confirm,
    effective_certificate,
    family_of,
    representatives,
    resolve_truth,
    revoke_validated,
)
from sextant.config import packaged_rolecards_dir
from sextant.jev_client import JevAnswerError, JevClient
from sextant.ledger import Ledger, exclusive, now_iso, sha_of
from sextant.rolecards import RoleCard, compute_hash, set_rung
from sextant.schema import Primitive, Rung
from sextant.wire import (
    WIRE_PROTOCOL_VERSION,
    identity_keys,
    measurement_conditions,
    measurement_fingerprint,
)

# How a metric is computed and reported. Bumping this supersedes previous metric claims.
METRIC_PROTOCOL_VERSION = "sextant-calibration-metric/2"

#: Development is where choices are made, `fit` is what a threshold is fitted on, `holdout`
#: confirms it once per case, and `release_holdout` stays locked. Keeping them apart is the only
#: reason a held-out number means anything.
SPLITS: tuple[str, ...] = ("development", "fit", "holdout", "release_holdout")
LOCKED_SPLITS: tuple[str, ...] = ("holdout", "release_holdout")
#: Explicit non-answers a labeller may give; they never resolve an item.
MARKERS: tuple[str, ...] = ("ambiguous", "insufficient")

TruthKind = Literal["deterministic_truth", "human_adjudication"]


class CalibrationProtocolError(RuntimeError):
    """Raised rather than fitting a threshold onto measurements of unknown provenance."""


class CalibrationIdentityError(CalibrationProtocolError):
    """The stored measurements were taken under conditions that no longer exist."""


class CalibrationSplitError(RuntimeError):
    """A case would be, or already is, cut between two splits."""


class CalibrationEvidenceError(RuntimeError):
    """A label, measurement or certificate no longer matches the evidence it rests on."""


class GoldenItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    card_id: str
    key: str
    pack: dict[str, Any]
    leaf_space: dict[str, Any] | None = None
    expected: bool | str | int | None = None
    lang: str = "en"
    split: str = "fit"
    source: str = "synthetic"
    pair_id: str | None = None
    group_id: str | None = None
    # The generation template or source family. Items from one lineage are one case; an item
    # with no declared lineage can be fitted on but cannot help confirm a certificate.
    lineage: str | None = None
    notes: str = ""

    @field_validator("split")
    @classmethod
    def _known_split(cls, v: str) -> str:
        if v not in SPLITS:
            raise ValueError(f"split {v!r} is not one of {', '.join(SPLITS)}")
        return v


# ---------------------------------------------------------------- roots and files


def default_root() -> Path:
    """The configured calibration root. A missing config file is refused, not guessed."""
    from sextant.config import ConfigError, load_config, require_config_file

    try:
        require_config_file()
    except ConfigError as exc:
        raise CalibrationProtocolError(f"no calibration root: {exc}") from exc
    return Path(load_config()["paths"]["calibration_dir"])


def _root(root: Path | None) -> Path:
    return root if root is not None else default_root()


def card_dir(card_id: str, root: Path | None = None) -> Path:
    d = _root(root) / card_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def measurement_ledger(card_id: str, root: Path | None = None) -> Ledger:
    """The per-card ledger every calibration measurement must be recorded in."""
    return Ledger(card_dir(card_id, root) / "ledger")


def _read_jsonl(p: Path) -> list[dict[str, Any]]:
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _append_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    with exclusive(p.parent), p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_jsonl(p: Path, rows: list[dict[str, Any]]) -> None:
    with exclusive(p.parent):
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def _supersede(p: Path) -> dict[str, Any] | None:
    """Move the current result aside before writing a new one.

    A recomputed metric or a refitted threshold is a new claim, not a correction of the old
    record. The old file keeps its own path and hash so a report that cited it can still be
    checked, and the new file names what it replaced.
    """
    if not p.exists():
        return None
    text = p.read_text()
    archive = p.parent / "superseded"
    archive.mkdir(parents=True, exist_ok=True)
    dest = archive / f"{p.stem}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}Z{p.suffix}"
    dest.write_text(text)
    return {"path": str(dest), "sha": sha_of(text), "superseded_at": now_iso()}


def _supersede_if_changed(p: Path, new_text: str) -> dict[str, Any] | None:
    if p.exists() and p.read_text() == new_text:
        return None
    return _supersede(p)


def _refuse_legacy(d: Path) -> None:
    """New-protocol writers never add to a directory that holds earlier-protocol evidence."""
    stale_responses = [r for r in _read_jsonl(d / "responses.jsonl") if r.get("protocol_version") != WIRE_PROTOCOL_VERSION]
    stale_gold = [g for g in _read_jsonl(d / "gold.jsonl") if g.get("protocol_version") != CALIBRATION_PROTOCOL_VERSION]
    if stale_responses or stale_gold:
        raise CalibrationProtocolError(
            f"{d} holds evidence from an earlier protocol ({len(stale_responses)} responses, "
            f"{len(stale_gold)} gold rows); it is historical, so use a new calibration root"
        )


# ---------------------------------------------------------------- cases


def _link_keys(item: GoldenItem) -> list[str]:
    keys = ["pack:" + sha_of([item.card_id, item.pack, item.leaf_space])]
    if item.group_id:
        keys.append(f"group:{item.group_id}")
    if item.pair_id:
        keys.append(f"pair:{item.pair_id}")
    if item.lineage:
        keys.append(f"lineage:{item.lineage}")
    return keys


def group_map(items: list[GoldenItem]) -> dict[str, str]:
    """Item id -> case label: connected components over group, pair, pack and lineage.

    Taking the first matching key instead of joining them let one pack be cut across splits
    when one of its items had a pair id and another did not. A case is everything
    reachable through any shared key.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for it in items:
        keys = _link_keys(it)
        for k in keys:
            parent.setdefault(k, k)
        for k in keys[1:]:
            a, b = find(keys[0]), find(k)
            if a != b:
                parent[max(a, b)] = min(a, b)
    return {it.id: "case_" + sha_of(find(_link_keys(it)[0])) for it in items}


def group_of(item: GoldenItem) -> str:
    """The case label of one item on its own (use `group_map` for a whole set)."""
    return group_map([item])[item.id]


def _draw_split(seed: int, group: str, dev_frac: float, release_frac: float, holdout_frac: float) -> str:
    """One draw per case, from the case's own label, so item order cannot change the split."""
    u = random.Random(sha_of([seed, group])).random()
    if u < dev_frac:
        return "development"
    if u < dev_frac + release_frac:
        return "release_holdout"
    if u < dev_frac + release_frac + holdout_frac:
        return "holdout"
    return "fit"


def split_leakage(items: list[GoldenItem]) -> list[dict[str, Any]]:
    """Cases that appear in more than one split: the holdout is not out of sample for these."""
    groups = group_map(items)
    by_group: dict[str, dict[str, list[str]]] = {}
    for it in items:
        by_group.setdefault(groups[it.id], {}).setdefault(it.split, []).append(it.id)
    return [
        {
            "group": group,
            "splits": sorted(splits),
            "item_ids": sorted(i for ids in splits.values() for i in ids),
        }
        for group, splits in sorted(by_group.items())
        if len(splits) > 1
    ]


def build(
    card_id: str,
    items: list[dict[str, Any]],
    *,
    seed: int = 20260918,
    holdout_frac: float = 0.25,
    dev_frac: float = 0.0,
    release_frac: float = 0.0,
    root: Path | None = None,
) -> dict[str, int]:
    """Add items to the card's golden set: content-addressed ids, one split per case.

    New items are joined with the existing set before anything is assigned, so an item that
    links two cases, or two new items of one case asking for different splits, is refused
    rather than cut.
    """
    if dev_frac + release_frac + holdout_frac > 1.0:
        raise CalibrationSplitError(
            f"development {dev_frac} + release {release_frac} + holdout {holdout_frac} exceeds the whole set"
        )
    d = card_dir(card_id, root)
    with exclusive(d):
        _refuse_legacy(d)
        existing = [GoldenItem.model_validate(r) for r in _read_jsonl(d / "golden.jsonl")]
        known = {it.id for it in existing}
        fresh: list[GoldenItem] = []
        wanted: dict[str, str | None] = {}
        for raw in items:
            item = dict(raw)
            item.setdefault("card_id", card_id)
            item["id"] = "gi_" + sha_of([card_id, item["key"], item["pack"], item.get("lang", "en")])
            if item["id"] in known or any(f.id == item["id"] for f in fresh):
                continue
            if raw.get("split") in LOCKED_SPLITS:
                raise CalibrationSplitError(
                    f"{item['id']} asks to be put in {raw.get('split')!r}; a holdout is drawn at random so no one "
                    "picks the cases a certificate is confirmed on. Ask for fit or development, or leave the split out"
                )
            wanted[item["id"]] = raw.get("split")
            fresh.append(GoldenItem.model_validate({**item, "split": raw.get("split") or "fit"}))
        groups = group_map(existing + fresh)
        assigned: dict[str, str] = {}
        for it in existing:
            prior = assigned.setdefault(groups[it.id], it.split)
            if prior != it.split:
                raise CalibrationSplitError(
                    f"case {groups[it.id]} is already cut across {prior} and {it.split}; rebuild the set"
                )
        for it in fresh:
            asked = wanted[it.id]
            group = groups[it.id]
            if asked is not None:
                earlier = assigned.get(group)
                if earlier is not None and earlier != asked:
                    raise CalibrationSplitError(
                        f"{it.id} asks for split {asked!r} but its case {group} is already {earlier!r}"
                    )
                assigned[group] = asked
        added: list[dict[str, Any]] = []
        for it in fresh:
            group = groups[it.id]
            split = assigned.setdefault(group, _draw_split(seed, group, dev_frac, release_frac, holdout_frac))
            final = it.model_copy(update={"split": split})
            added.append(final.model_dump(mode="json", exclude_none=True))
        _append_jsonl(d / "golden.jsonl", added)
        return {"added": len(added), "total": len(existing) + len(added), "groups": len(set(groups.values()))}


def load_items(card_id: str, root: Path | None = None, split: str | None = None) -> list[GoldenItem]:
    rows = _read_jsonl(card_dir(card_id, root) / "golden.jsonl")
    items = [GoldenItem.model_validate(r) for r in rows]
    return [i for i in items if split is None or i.split == split]


# ---------------------------------------------------------------- labels


def _question_for(card: RoleCard, key: str) -> tuple[Primitive, list[str]]:
    base = key.split("[")[0]
    q = card.questions.get(base)
    if q is None:
        raise CalibrationEvidenceError(f"{card.id} has no question {base}")
    if q.primitive == Primitive.choice:
        return q.primitive, list(q.options or {})
    if q.primitive == Primitive.score:
        return q.primitive, [str(i) for i in range(len(q.levels or []))]
    return q.primitive, []


def typed_label(card: RoleCard, key: str, label: Any, leaf_space: dict[str, Any] | None = None) -> Any:
    """A label of the type its question asks for, or an explicit marker; nothing is coerced.

    `bool("false")` is True, which is how a string label once became a positive.
    """
    if isinstance(label, str) and label in MARKERS:
        return label
    primitive, options = _question_for(card, key)
    if leaf_space and card.questions[key.split("[")[0]].text.strip() == "{question}":
        primitive = Primitive(leaf_space["primitive"])
        options = list(leaf_space.get("options") or {}) if primitive == Primitive.choice else [
            str(i) for i in range(len(leaf_space.get("levels") or []))
        ]
    if primitive == Primitive.noul and type(label) is bool:
        return label
    if primitive == Primitive.choice and isinstance(label, str) and label in options:
        return label
    if primitive == Primitive.score and type(label) is int and 0 <= label < len(options):
        return label
    raise CalibrationEvidenceError(
        f"{card.id}.{key}: label {label!r} is not a {primitive.value} answer"
        + (f" (one of {options})" if options else " (a JSON boolean)")
        + f" or one of the markers {list(MARKERS)}"
    )


def label_schema(card: RoleCard) -> dict[str, Any]:
    """The grader answer schema for one card, with each question's label typed.

    The handshake rejects a mistyped label at acceptance, before import ever reads it.
    """
    rules: list[dict[str, Any]] = []
    for key, q in card.questions.items():
        pattern = f"^{key}(\\\\[.*\\\\])?$".replace("\\\\\\\\", "\\\\")
        if q.primitive == Primitive.noul:
            label: dict[str, Any] = {"anyOf": [{"type": "boolean"}, {"enum": list(MARKERS)}]}
        elif q.primitive == Primitive.choice:
            label = {"enum": [*list(q.options or {}), *MARKERS]}
        else:
            label = {"anyOf": [{"type": "integer", "minimum": 0, "maximum": len(q.levels or []) - 1}, {"enum": list(MARKERS)}]}
        rules.append({"if": {"properties": {"key": {"pattern": pattern}}}, "then": {"properties": {"label": label}}})
    return {
        "type": "object",
        "required": ["labels"],
        "additionalProperties": False,
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "key", "label", "confidence", "rationale"],
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "key": {"type": "string"},
                        "label": {"type": ["boolean", "string", "integer"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "rationale": {"type": "string", "maxLength": 300},
                    },
                    "allOf": rules,
                },
            }
        },
    }


def grader_prompt(card: RoleCard, items: list[GoldenItem]) -> str:
    lines = [
        "You are labeling items for a calibration set. Judge each item LITERALLY by the card's instructions and the question's criteria, exactly as a careful reader who has only the state shown. Do not use outside knowledge about the topic. Output one label per item and key.",
        f"If the state does not settle an item, answer one of {list(MARKERS)} instead of guessing.",
        "",
        f"CARD {card.id}: {card.purpose}",
        "INSTRUCTIONS:",
        card.instructions.strip(),
        "",
        "QUESTIONS (key: text | criteria):",
    ]
    for key, q in card.questions.items():
        if q.primitive == Primitive.noul:
            lines.append(f"- {key}: {q.text} | true: {q.criteria_true} | false: {q.criteria_false}  (label: the JSON boolean true or false, not a string)")
        elif q.primitive == Primitive.choice:
            lines.append(f"- {key}: {q.text} | options: {json.dumps(q.options, ensure_ascii=False)}  (label: one option name)")
        else:
            lines.append(f"- {key}: {q.text} | levels: {json.dumps(q.levels, ensure_ascii=False)}  (label: the level index as an integer, 0-based)")
    lines.append("")
    lines.append("ITEMS:")
    for it in items:
        lines.append(json.dumps({"id": it.id, "key": it.key, "state": it.pack}, ensure_ascii=False))
    lines.append("")
    lines.append("Return {labels: [{id, key, label, confidence, rationale}]} with one entry per item, in the same order. Keys with an index such as major[2] refer to the numbered item of the list field.")
    return "\n".join(lines)


def label_requests(
    card: RoleCard,
    items: list[GoldenItem],
    decision_dir: Path,
    graders: list[str],
    *,
    batch: int = 20,
) -> list[str]:
    """Create one handshake request per batch per grader; labels arrive through `import_labels`."""
    from sextant.handshake import create_request

    ids: list[str] = []
    for g in graders:
        for start in range(0, len(items), batch):
            chunk = items[start : start + batch]
            req = create_request(
                decision_dir,
                role="grader",
                purpose=f"calibration labels {card.id} items {start + 1}-{start + len(chunk)}",
                prompt=grader_prompt(card, chunk),
                answer_schema=label_schema(card),
                provider=g,
                node_ids=[it.id for it in chunk],
                constraints={"card_id": card.id, "grader": g},
            )
            ids.append(req.id)
    return ids


def _item_leaf_space(card_id: str, item_id: str, root: Path | None) -> dict[str, Any] | None:
    for it in load_items(card_id, root):
        if it.id == item_id:
            return it.leaf_space
    return None


def record_labels(card_id: str, rows: list[dict[str, Any]], *, card: RoleCard, root: Path | None = None) -> int:
    """Append model labels whose identity is declared by the caller rather than derived.

    `import_labels` can show that a label came from a specific accepted answer; a label written
    here cannot. Both carry `provenance`, and only handshake labels count toward independent
    sources. Labels are typed by their question.
    """
    d = card_dir(card_id, root)
    with exclusive(d):
        _refuse_legacy(d)
        have = {
            (r["item_id"], r["key"], r.get("provider"), json.dumps(r["label"], sort_keys=True))
            for r in _read_jsonl(d / "labels.jsonl")
        }
        out: list[dict[str, Any]] = []
        for raw in rows:
            missing = [f for f in ("item_id", "key", "label", "provider", "family") if raw.get(f) is None]
            if missing:
                raise CalibrationEvidenceError(f"a label needs {', '.join(missing)}: {raw}")
            label = typed_label(card, raw["key"], raw["label"], _item_leaf_space(card_id, raw["item_id"], root))
            row = {
                "item_id": raw["item_id"], "key": raw["key"], "label": label,
                "confidence": raw.get("confidence"), "rationale": raw.get("rationale", ""),
                "request_id": raw.get("request_id"), "provider": raw["provider"], "family": raw["family"],
                "isolation": raw.get("isolation", "n/a"), "model": raw.get("model"),
                "provenance": "declared", "at": now_iso(),
            }
            marker = (row["item_id"], row["key"], row["provider"], json.dumps(row["label"], sort_keys=True))
            if marker in have:
                continue
            have.add(marker)
            out.append(row)
        _append_jsonl(d / "labels.jsonl", out)
        return len(out)


def record_truth(
    card_id: str,
    rows: list[dict[str, Any]],
    *,
    card: RoleCard,
    kind: TruthKind = "deterministic_truth",
    source: str,
    method: str,
    root: Path | None = None,
) -> int:
    """Record a resolving judgment: a deterministic procedure, or a human adjudication.

    These are the only two kinds of vote that settle an item on their own, and they are stored
    apart from model labels so nobody can later count them as more agreement. `method`
    has to say how the label was obtained; "the author thinks so" is not a method, it stays in
    the golden item's `expected`. Labels are typed by their question.
    """
    d = card_dir(card_id, root)
    with exclusive(d):
        _refuse_legacy(d)
        have = {
            (r["item_id"], r["key"], r["kind"], r["source"], json.dumps(r["label"], sort_keys=True))
            for r in _read_jsonl(d / "truth.jsonl")
        }
        out: list[dict[str, Any]] = []
        for raw in rows:
            if raw.get("item_id") is None or raw.get("key") is None or "label" not in raw:
                raise CalibrationEvidenceError(f"a truth row needs item_id, key and label: {raw}")
            label = typed_label(card, raw["key"], raw["label"], _item_leaf_space(card_id, raw["item_id"], root))
            row = {
                "item_id": raw["item_id"], "key": raw["key"], "label": label,
                "kind": raw.get("kind", kind), "source": raw.get("source", source),
                "method": raw.get("method", method), "note": raw.get("note", ""), "at": now_iso(),
            }
            if row["kind"] not in TRUTH_KINDS:
                raise CalibrationEvidenceError(f"truth kind {row['kind']!r} is not one of {sorted(TRUTH_KINDS)}")
            marker = (row["item_id"], row["key"], row["kind"], row["source"], json.dumps(row["label"], sort_keys=True))
            if marker in have:
                continue
            have.add(marker)
            out.append(row)
        _append_jsonl(d / "truth.jsonl", out)
        return len(out)


def import_labels(card_id: str, decision_dir: Path, root: Path | None = None, *, card: RoleCard) -> int:
    """Read every accepted grader answer for this card and append its labels (idempotent).

    The answer file is re-read long after it was accepted, and it is an ordinary writable file.
    Its payload hash must equal the `answer_sha` the ledger recorded at acceptance, and the
    grader identity (provider, family, isolation) is taken from that ledger row, never from the
    file: a file edited afterwards stops the import loudly.
    """
    from sextant.providers.base import read_answer, read_request

    d = card_dir(card_id, root)
    with exclusive(d):
        _refuse_legacy(d)
        have = {(r["item_id"], r["key"], r.get("request_id")) for r in _read_jsonl(d / "labels.jsonl")}
        ledger = Ledger(decision_dir)
        accepted = [r for r in ledger.active("llm-calls") if r.get("status") == "accepted" and r.get("role") == "grader"]
        rows: list[dict[str, Any]] = []
        for r in accepted:
            try:
                req = read_request(decision_dir, r["request_id"])
            except FileNotFoundError:
                continue
            if req.constraints.get("card_id") != card_id:
                continue
            ans = read_answer(decision_dir, r["request_id"])
            current = sha_of(ans.answer)
            if current != r.get("answer_sha"):
                raise CalibrationEvidenceError(
                    f"{r['request_id']}: the answer file hashes to {current} but was accepted as "
                    f"{r.get('answer_sha')}; it changed after acceptance and cannot be imported"
                )
            declared = (ans.provider, ans.family, ans.isolation)
            recorded = (r.get("provider"), r.get("family"), r.get("isolation"))
            if declared != recorded:
                raise CalibrationEvidenceError(
                    f"{r['request_id']}: the answer file names grader {declared} but was accepted from "
                    f"{recorded}; the identity changed after acceptance"
                )
            for lab in ans.answer.get("labels", []):
                key3 = (lab["id"], lab["key"], r["request_id"])
                if key3 in have:
                    continue
                label = typed_label(card, lab["key"], lab["label"], _item_leaf_space(card_id, lab["id"], root))
                rows.append({
                    "item_id": lab["id"], "key": lab["key"], "label": label, "confidence": lab.get("confidence"),
                    "rationale": lab.get("rationale", ""), "request_id": r["request_id"],
                    "provider": r.get("provider"), "family": r.get("family"), "isolation": r.get("isolation"),
                    "model": ans.model, "provenance": "handshake", "answer_sha": current, "at": now_iso(),
                })
                have.add(key3)
        _append_jsonl(d / "labels.jsonl", rows)
        return len(rows)


def _independence_group(label_row: dict[str, Any]) -> str:
    """Two graders of one model family share their failure modes; that is one sample, not two.

    `isolation` is a declared field about process, not a measurement of error independence,
    so it never promotes a second same-family vote into a second source.
    """
    family = label_row.get("family") or label_row.get("provider") or "unknown"
    return f"family:{family}"


def _vote_trail(
    expected: Any,
    votes: list[dict[str, Any]],
    det: list[dict[str, Any]],
    adj: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Every vote in one list, each still carrying the kind it is, for the human read."""
    trail: list[dict[str, Any]] = []
    if expected is not None:
        trail.append({"who": "author", "kind": "author_expectation", "label": expected})
    trail += [{"who": v["who"], "kind": "model_label", "label": v["label"]} for v in votes]
    trail += [{"who": t["source"], "kind": "deterministic_truth", "label": t["label"]} for t in det]
    trail += [{"who": t["source"], "kind": "human_adjudication", "label": t["label"]} for t in adj]
    return trail


def resolve_gold(card_id: str, root: Path | None = None) -> dict[str, int]:
    """Resolve each item from votes that can actually settle it, keeping every kind apart.

    A deterministic procedure or a human adjudicator settles an item alone. Model labels settle
    one only when at least two independent families, each with a handshake-provenance label,
    agree; declared labels are shown but do not count. A marker (`ambiguous`, `insufficient`)
    never resolves anything. The author's expectation never settles anything: it is what the set
    was built to test.
    """
    d = card_dir(card_id, root)
    with exclusive(d):
        items = {i.id: i for i in load_items(card_id, root)}
        labels = _read_jsonl(d / "labels.jsonl")
        truths = _read_jsonl(d / "truth.jsonl")

        author: dict[tuple[str, str], Any] = {}
        model_votes: dict[tuple[str, str], list[dict[str, Any]]] = {}
        deterministic: dict[tuple[str, str], list[dict[str, Any]]] = {}
        adjudications: dict[tuple[str, str], list[dict[str, Any]]] = {}
        subjects: set[tuple[str, str]] = set()

        for it in items.values():
            if it.expected is not None:
                author[(it.id, it.key)] = it.expected
                subjects.add((it.id, it.key))
        for lab in labels:
            marker = (lab["item_id"], lab["key"])
            subjects.add(marker)
            model_votes.setdefault(marker, []).append({
                "who": f"{lab.get('provider')}:{lab.get('isolation')}",
                "provider": lab.get("provider"), "family": lab.get("family"),
                "isolation": lab.get("isolation"), "independence_group": _independence_group(lab),
                "provenance": lab.get("provenance", "declared"), "request_id": lab.get("request_id"),
                "label": lab["label"],
            })
        for t in truths:
            marker = (t["item_id"], t["key"])
            subjects.add(marker)
            kind = t.get("kind", "deterministic_truth")
            bucket = deterministic if kind == "deterministic_truth" else adjudications
            bucket.setdefault(marker, []).append(
                {"source": t.get("source"), "method": t.get("method"), "label": t["label"], "kind": kind}
            )

        gold_rows: list[dict[str, Any]] = []
        dis_lines = [
            "# Disagreements to spot-check", "",
            (
                f"card {card_id}; one line per item; gold is only set by deterministic truth, human "
                "adjudication, or agreeing independent model families with handshake provenance; "
                "'unresolved' means none of those held."
            ),
            "",
        ]
        n_dis = 0
        for iid, key in sorted(subjects):
            marker = (iid, key)
            det = deterministic.get(marker, [])
            adj = adjudications.get(marker, [])
            votes = model_votes.get(marker, [])
            expected = author.get(marker)

            counted = [
                v for v in votes
                if v["provenance"] == "handshake" and not (isinstance(v["label"], str) and v["label"] in MARKERS)
            ]
            by_group: dict[str, list[Any]] = {}
            for v in counted:
                by_group.setdefault(v["independence_group"], []).append(v["label"])
            source_labels: dict[str, Any] = {}
            split_groups: list[str] = []
            for group, group_labels in by_group.items():
                distinct = {json.dumps(x, sort_keys=True) for x in group_labels}
                if len(distinct) == 1:
                    source_labels[group] = group_labels[0]
                else:
                    split_groups.append(group)
            independent_sources = len(source_labels)

            gold: Any = None
            resolved_by: str | None = None
            status: str
            if det or adj:
                # one resolution for gold and for certificate verification; a conflict between
                # deterministic truth and human adjudication resolves nothing
                value, kind, outcome = resolve_truth(det + adj)
                if outcome == "resolved":
                    gold, resolved_by, status = value, kind, "resolved"
                else:
                    status = f"unresolved-truth-{outcome}"
            elif source_labels:
                counts: dict[str, int] = {}
                for label in source_labels.values():
                    encoded = json.dumps(label, sort_keys=True)
                    counts[encoded] = counts.get(encoded, 0) + 1
                top, top_n = max(counts.items(), key=lambda kv: kv[1])
                if independent_sources < 2:
                    status = "unresolved-single-source"
                elif top_n > independent_sources / 2:
                    gold, resolved_by, status = json.loads(top), "independent_model_majority", "resolved"
                else:
                    status = "unresolved-disagreement"
            elif split_groups:
                status = "unresolved-disagreement"
            elif votes:
                status = "unresolved-declared-only"
            elif expected is not None:
                status = "unresolved-author-only"
            else:
                status = "unresolved-no-votes"

            every_label = (
                ([expected] if expected is not None else [])
                + [v["label"] for v in votes]
                + [t["label"] for t in det + adj]
            )
            dissent = len({json.dumps(x, sort_keys=True) for x in every_label}) > 1
            gold_rows.append({
                "item_id": iid, "key": key, "gold": gold, "resolved_by": resolved_by, "status": status,
                "author_expectation": expected, "model_votes": votes,
                "human_adjudication": adj, "deterministic_truth": det,
                "independent_sources": independent_sources,
                "internally_split_sources": sorted(split_groups),
                "n_votes": len(every_label), "dissent": dissent,
                "votes": _vote_trail(expected, votes, det, adj),
                "protocol_version": CALIBRATION_PROTOCOL_VERSION,
            })
            if dissent:
                n_dis += 1
                dis_item = items.get(iid)
                shown = [(v["who"], v["label"]) for v in gold_rows[-1]["votes"]]
                dis_lines.append(f"- {iid} {key}: votes {shown} -> gold {gold if gold is not None else 'unresolved'} ({status})")
                if dis_item is not None:
                    dis_lines.append(f"  state: {json.dumps(dis_item.pack, ensure_ascii=False)[:400]}")

        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in gold_rows)
        _supersede_if_changed(d / "gold.jsonl", text)
        _write_jsonl(d / "gold.jsonl", gold_rows)
        (d / "disagreements.md").write_text("\n".join(dis_lines) + "\n")
        by_status: dict[str, int] = {}
        for g in gold_rows:
            by_status[g["status"]] = by_status.get(g["status"], 0) + 1
        return {
            "resolved": sum(1 for g in gold_rows if g["gold"] is not None),
            "unresolved": sum(1 for g in gold_rows if g["gold"] is None),
            "dissent": n_dis,
            "author_only": by_status.get("unresolved-author-only", 0),
            "single_source": by_status.get("unresolved-single-source", 0),
        }


# ---------------------------------------------------------------- measurement


def asked_keys(card: RoleCard, item_key: str) -> list[str]:
    """The questions one item's measurement sends: the key with its twin (R7: the whole card)."""
    return identity_keys(card, item_key)


def conditions_for(card: RoleCard, item_key: str, *, model_id: str, transport: str, language: str) -> str:
    """The measurement conditions one item's question would be measured under right now."""
    return measurement_conditions(
        card, keys=identity_keys(card, item_key), model_id=model_id, transport=transport, language=language
    )


def measure(card: RoleCard, client: JevClient, items: list[GoldenItem], root: Path | None = None, paraphrases: dict[str, list[str]] | None = None) -> int:
    """Ask Jev every item's question (with its twin); store one response row per attempt.

    The client must record into this card's measurement ledger, so every response can later be
    replayed from the calls that produced it. Idempotence is keyed on the measurement
    fingerprint. An answer outside its declared space is recorded as an invalid
    attempt, not skipped, so a case that Jev could not answer still counts in the denominator.
    """
    root = _root(root)
    d = card_dir(card.id, root)
    expected_ledger = (d / "ledger").resolve()
    if client.ledger is None or client.ledger.dir.resolve() != expected_ledger:
        raise CalibrationProtocolError(
            f"calibration measurements of {card.id} must be recorded in {expected_ledger}; "
            "pass a client built with that ledger so each response can be replayed"
        )
    _refuse_legacy(d)
    groups = group_map(load_items(card.id, root))

    def fingerprint_of(it: GoldenItem) -> str:
        return measurement_fingerprint(
            card, keys=identity_keys(card, it.key), model_id=client.model_id, transport=client.transport,
            language=client.language, pack=it.pack, leaf_space=it.leaf_space,
        )

    done = {(r["item_id"], r["key"], r.get("protocol_fingerprint")) for r in _read_jsonl(d / "responses.jsonl")}
    added = 0
    for it in items:
        fingerprint = fingerprint_of(it)
        if (it.id, it.key, fingerprint) in done:
            continue
        base_key = it.key.split("[")[0]
        keys = identity_keys(card, it.key)
        common = {
            "item_id": it.id, "key": it.key, "lang": it.lang, "split": it.split, "pair_id": it.pair_id,
            "group": groups.get(it.id), "model_id": client.model_id, "transport": client.transport,
            "language": client.language, "protocol_version": WIRE_PROTOCOL_VERSION,
            "protocol_fingerprint": fingerprint,
            "measurement_conditions": conditions_for(
                card, it.key, model_id=client.model_id, transport=client.transport, language=client.language
            ),
            "asked_keys": sorted(keys),
        }
        row: dict[str, Any]
        try:
            res = client.ask(card.id, it.pack, keys=keys, leaf_space=it.leaf_space, paraphrases=paraphrases,
                             variant=f"cal:{it.id}", input_language=it.lang)
        except JevAnswerError as exc:
            row = {**common, "invalid": True, "error": str(exc)[:500], "run_id": exc.run_id,
                   "call_ids": list(exc.call_ids), "p": None, "measured_at": now_iso()}
        else:
            qe = res.estimates.get(it.key) or res.estimates.get(base_key)
            if qe is None:
                continue
            e = qe.estimate
            row = {
                **common, "invalid": False,
                "p": e.p, "choice": e.choice, "score": e.score, "probabilities": e.probabilities,
                "confidence": e.jev_confidence, "contested": e.contested,
                "stability": e.stability.model_dump(mode="json"),
                "run_id": e.run_id, "call_ids": e.call_ids, "measured_at": e.at,
                "question_hash": e.question_hash, "rendered_hash": e.rendered_hash, "state_sha": e.state_sha,
            }
        with exclusive(d):
            current = {(r["item_id"], r["key"], r.get("protocol_fingerprint")) for r in _read_jsonl(d / "responses.jsonl")}
            if (it.id, it.key, fingerprint) in current:
                continue
            _append_jsonl(d / "responses.jsonl", [row])
        done.add((it.id, it.key, fingerprint))
        added += 1
    return added


# ---------------------------------------------------------------- metrics


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-4) -> float:
    q = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))


def _average_ranks(v: np.ndarray) -> np.ndarray:
    """Ranks from 1, with tied values sharing the mean of the ranks they span."""
    order = np.argsort(v, kind="mergesort")
    sorted_v = v[order]
    ranks = np.empty(len(v), dtype=float)
    start = 0
    while start < len(sorted_v):
        stop = start
        while stop + 1 < len(sorted_v) and sorted_v[stop + 1] == sorted_v[start]:
            stop += 1
        ranks[order[start : stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return ranks


def auroc(p: np.ndarray, y: np.ndarray) -> float | None:
    """P(positive ranked above negative) + half the tied pairs, by the Mann-Whitney identity.

    Undefined (None) with a single class or with non-finite scores; labels must be 0 or 1.
    """
    if len(p) != len(y) or not np.all(np.isfinite(p)) or not np.all((y == 0) | (y == 1)):
        return None
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = _average_ranks(np.concatenate([pos, neg]))
    r_pos = float(ranks[: len(pos)].sum())
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def murphy(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict[str, Any]:
    """The binned Murphy decomposition, reported as binned, with its approximation residual.

    Reliability minus resolution plus uncertainty equals the Brier score of the *binned*
    forecasts, not of the original ones. An empty input has no decomposition.
    """
    if len(p) == 0:
        return {"binned": True, "bins": bins, "binning": "equal-width", "n": 0, "defined": False, "table": []}
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ybar = float(y.mean())
    rel = res = 0.0
    table: list[dict[str, Any]] = []
    for b in range(bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            table.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": 0})
            continue
        pb, yb = float(p[m].mean()), float(y[m].mean())
        rel += n * (pb - yb) ** 2
        res += n * (yb - ybar) ** 2
        table.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n, "mean_p": round(pb, 4), "hit_rate": round(yb, 4)})
    N = len(p)
    reliability, resolution = rel / N, res / N
    uncertainty = ybar * (1 - ybar)
    combined = reliability - resolution + uncertainty
    raw = brier(p, y)
    return {
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "binned": True,
        "bins": bins,
        "binning": "equal-width",
        "n": N,
        "defined": True,
        "brier": raw,
        "brier_from_bins": combined,
        "approximation_residual": raw - combined,
        "table": table,
    }


def coverage_curve(
    p: np.ndarray, y: np.ndarray, thresholds: list[float] | None = None, contested: list[bool] | None = None
) -> list[dict[str, float]]:
    """Coverage and agreement per threshold under the deployed rule (contested runs abstain)."""
    ts = thresholds or [round(x, 2) for x in np.arange(0.50, 1.0, 0.05)]
    flags = contested if contested is not None else [False] * len(p)
    out: list[dict[str, float]] = []
    for t in ts:
        answers = [wire.decision(float(v), float(t), contested=c) for v, c in zip(p, flags, strict=True)]
        covered = [(a, bool(truth)) for a, truth in zip(answers, y, strict=True) if a is not None]
        if not covered:
            out.append({"t": float(t), "coverage": 0.0, "agreement": float("nan")})
            continue
        agree = sum(1 for a, truth in covered if a == truth) / len(covered)
        out.append({"t": float(t), "coverage": len(covered) / len(p), "agreement": agree})
    return out


def holdout_evaluations(card_id: str, root: Path | None = None) -> list[dict[str, Any]]:
    """Every scoring of locked items, by item id. A case scored once is no longer unseen."""
    return _read_jsonl(card_dir(card_id, root) / "holdout_evaluations.jsonl")


def _seen_items(card_id: str, root: Path | None) -> set[str]:
    return {i for e in holdout_evaluations(card_id, root) for i in e.get("item_ids", [])}


def _log_evaluation(card_id: str, root: Path | None, entry: dict[str, Any]) -> None:
    if entry.get("item_ids"):
        _append_jsonl(card_dir(card_id, root) / "holdout_evaluations.jsonl", [{"at": now_iso(), "card_id": card_id, **entry}])


def _current_responses(d: Path) -> list[dict[str, Any]]:
    """The latest response per (item, key); earlier attempts stay in the file as history."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in _read_jsonl(d / "responses.jsonl"):
        latest[(r["item_id"], r["key"])] = r
    return list(latest.values())


def _gold_index(d: Path) -> dict[tuple[str, str], dict[str, Any]]:
    return {(g["item_id"], g["key"]): g for g in _read_jsonl(d / "gold.jsonl")}


def _refuse_stale(card_id: str, responses: list[dict[str, Any]], gold: dict[tuple[str, str], dict[str, Any]]) -> None:
    stale = [r for r in responses if r.get("protocol_version") != WIRE_PROTOCOL_VERSION]
    if stale:
        versions = sorted({str(r.get("protocol_version")) for r in stale})
        raise CalibrationProtocolError(
            f"{card_id}: {len(stale)} of {len(responses)} measurements were taken under "
            f"{versions} rather than {WIRE_PROTOCOL_VERSION}; re-measure before fitting"
        )
    legacy = [g for g in gold.values() if g.get("protocol_version") != CALIBRATION_PROTOCOL_VERSION]
    if legacy:
        raise CalibrationProtocolError(
            f"{card_id}: {len(legacy)} gold rows were resolved under an earlier calibration protocol; "
            "re-resolve them from typed labels before using them"
        )


def dataset_version(rows: list[dict[str, Any]]) -> str:
    """Identity of the labelled data a claim rests on: item, key, gold, source, split and case."""
    return "ds_" + sha_of(sorted(
        [r["item_id"], r["key"], json.dumps(r.get("gold"), sort_keys=True), str(r.get("resolved_by")),
         str(r.get("split")), str(r.get("group"))]
        for r in rows
    ))


def responses_version(rows: list[dict[str, Any]]) -> str:
    """Identity of the measurements a claim rests on: fingerprints, values, runs and calls."""
    return "rs_" + sha_of(sorted(
        [r["item_id"], r["key"], str(r.get("protocol_fingerprint")), json.dumps(r.get("p")),
         json.dumps(r.get("choice")), json.dumps(r.get("score")), str(r.get("run_id")),
         sorted(r.get("call_ids") or []), bool(r.get("invalid"))]
        for r in rows
    ))


def experiment_id(*, conditions: list[str], dataset: str, responses: str, purpose: str) -> str:
    """One name for "this claim, on this data, from these measurements, under these rules"."""
    return "ex_" + sha_of([
        CALIBRATION_PROTOCOL_VERSION,
        METRIC_PROTOCOL_VERSION,
        WIRE_PROTOCOL_VERSION,
        sorted(set(conditions)),
        dataset,
        responses,
        purpose,
    ])


def _joined_rows(card: RoleCard, root: Path, splits: tuple[str, ...]) -> list[dict[str, Any]]:
    """Responses joined with their golden item and gold, their values re-derived from the ledger.

    A response row is only an index from an item to the run that measured it. The value, choice,
    score, contested and invalid flags and the measurement identity all come from that run
    (`_bind_responses`); a stored copy that disagrees is an evidence error, not a correction.
    """
    d = card_dir(card.id, root)
    items = load_items(card.id, root)
    by_id = {it.id: it for it in items}
    groups = group_map(items)
    gold = _gold_index(d)
    responses = _current_responses(d)
    _refuse_stale(card.id, responses, gold)
    for r in responses:
        it = by_id.get(r["item_id"])
        if it is None:
            raise CalibrationEvidenceError(f"{card.id}: a response names item {r['item_id']}, which the golden set lacks")
        if r.get("split") != it.split:
            raise CalibrationSplitError(
                f"{card.id}: response {r['item_id']} was stored as {r.get('split')!r} but the item is now "
                f"{it.split!r}; the golden set decides the split, so re-measure"
            )
    bound, problems = _bind_responses(card, responses, measurement_ledger(card.id, root), by_id)
    if problems:
        raise CalibrationEvidenceError(
            f"{card.id}: responses do not bind to their measurement-ledger runs: " + "; ".join(problems[:3])
        )
    rows: list[dict[str, Any]] = []
    for r in bound:
        it = by_id[r["item_id"]]
        if it.split not in splits:
            continue
        g = gold.get((r["item_id"], r["key"])) or {}
        rows.append({
            **r, "split": it.split, "group": groups[it.id], "lineage": it.lineage, "lang": it.lang,
            "gold": g.get("gold"), "resolved_by": g.get("resolved_by"),
        })
    return rows


def _present(stability: list[dict[str, Any]], name: str) -> list[float]:
    """The recorded values of one stability measure, skipping runs that did not measure it."""
    return [float(s[name]) for s in stability if s.get(name) is not None]


def compute_metrics(card: RoleCard, root: Path | None = None, *, include_holdout: bool = False) -> dict[str, Any]:
    """Descriptive metrics over development and fit data.

    Locked splits are read only when asked, and reading them is logged: a case whose metrics
    were looked at can no longer confirm a certificate.
    """
    root = _root(root)
    splits: tuple[str, ...] = SPLITS if include_holdout else ("development", "fit")
    items = load_items(card.id, root)
    leakage = split_leakage(items)
    if include_holdout and leakage:
        raise CalibrationSplitError(f"{card.id}: {len(leakage)} case(s) span splits; locked metrics would mislead")
    rows = _joined_rows(card, root, splits)
    locked = [r["item_id"] for r in rows if r["split"] in LOCKED_SPLITS]
    split_counts: dict[str, int] = {}
    for r in rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    dataset = dataset_version(rows)
    responses = responses_version(rows)
    ex_id = experiment_id(
        conditions=[str(r.get("measurement_conditions")) for r in rows],
        dataset=dataset, responses=responses, purpose=f"metrics:{','.join(splits)}",
    )
    _log_evaluation(card.id, root, {"purpose": "metrics", "experiment_id": ex_id, "item_ids": sorted(locked)})
    out: dict[str, Any] = {
        "card_id": card.id,
        "n_joined": len(rows),
        "splits_read": list(splits),
        "metric_protocol_version": METRIC_PROTOCOL_VERSION,
        "calibration_protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "wire_protocol_version": WIRE_PROTOCOL_VERSION,
        "dataset_version": dataset,
        "responses_version": responses,
        "experiment_id": ex_id,
        "computed_at": now_iso(),
        "split_counts": split_counts,
        "split_leakage": leakage,
        "resolved_by": sorted({str(r.get("resolved_by")) for r in rows}),
        "supersedes": None,
        "keys": {},
    }
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault(r["key"].split("[")[0], []).append(r)
    for key, rs in by_key.items():
        q = card.questions.get(key)
        if q is None:
            continue
        valid = [r for r in rs if not r.get("invalid")]
        entry: dict[str, Any] = {"n": len(rs), "n_invalid": len(rs) - len(valid), "primitive": q.primitive.value}
        if q.primitive == Primitive.noul:
            labelled = [r for r in valid if type(r.get("gold")) is bool and r.get("p") is not None]
            p = np.array([float(r["p"]) for r in labelled])
            y = np.array([1.0 if r["gold"] is True else 0.0 for r in labelled])
            flags = [bool(r.get("contested")) for r in labelled]
            entry["n_labelled"] = len(labelled)
            if labelled:
                entry.update({
                    "brier": brier(p, y), "log_loss": log_loss(p, y), "auroc": auroc(p, y), "base_rate": float(y.mean()),
                    "murphy": murphy(p, y), "coverage_curve": coverage_curve(p, y, contested=flags),
                    # descriptive: an abstention (tie or contested run) counts as not agreeing
                    "accuracy_at_half": float(np.mean([
                        wire.decision(float(v), 0.5, contested=c) == bool(t) for v, t, c in zip(p, y, flags, strict=True)
                    ])),
                })
        elif q.primitive == Primitive.choice:
            labelled = [r for r in valid if isinstance(r.get("gold"), str) and r["gold"] not in MARKERS]
            entry["n_labelled"] = len(labelled)
            if labelled:
                entry["accuracy"] = float(np.mean([1.0 if str(r["choice"]) == str(r["gold"]) else 0.0 for r in labelled]))
                entry["contested_rate"] = float(np.mean([1.0 if r.get("contested") else 0.0 for r in labelled]))
        else:
            labelled = [r for r in valid if type(r.get("gold")) is int]
            entry["n_labelled"] = len(labelled)
            if labelled:
                entry["mae_levels"] = float(np.mean([abs(float(r["score"]) - float(r["gold"])) for r in labelled]))
        stab = [r.get("stability") or {} for r in valid]
        sd, ps, ng, pp = (_present(stab, name) for name in ("rerun_sd", "permutation_sd", "negation_gap", "paraphrase_sd"))
        entry["stability"] = {
            "rerun_sd_mean": statistics.mean(sd) if sd else None,
            "permutation_sd_mean": statistics.mean(ps) if ps else None,
            "negation_gap_mean": statistics.mean(ng) if ng else None,
            "paraphrase_sd_mean": statistics.mean(pp) if pp else None,
        }
        pairs: dict[str, dict[str, float]] = {}
        for r in valid:
            if r.get("pair_id") and r.get("p") is not None:
                pairs.setdefault(r["pair_id"], {})[r["lang"]] = float(r["p"])
        deltas = [abs(v["en"] - v["zh"]) for v in pairs.values() if "en" in v and "zh" in v]
        entry["zh_en_delta_mean"] = statistics.mean(deltas) if deltas else None
        entry["zh_en_pairs"] = len(deltas)
        out["keys"][key] = entry
    path = card_dir(card.id, root) / "metrics.json"
    out["supersedes"] = _supersede(path)
    path.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------- fitting


@dataclass
class FitResult:
    """What a fit established. Only `deployable` authorizes anything."""

    key: str
    threshold: float | None
    status: str  # fitted | unfit | no-data
    coverage_fit: float = 0.0
    agreement_fit: float = 0.0
    n_fit: int = 0
    n_holdout: int = 0
    provisional: bool = False
    deployable: bool = False
    attempt: int = 1
    alpha_per_bound: float | None = None
    n_eligible: int = 0
    n_automatic: int = 0
    n_wrong: int = 0
    n_releases: int = 0
    n_bad_releases: int = 0
    n_abstentions: int = 0
    r_upper: float | None = None
    r_plus_upper: float | None = None
    c_lower: float | None = None
    fpr: float | None = None
    false_rejection_rate: float | None = None
    confirming: list[dict[str, str]] = field(default_factory=list)
    reason: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def confirming_groups(self) -> list[str]:
        return [c["group"] for c in self.confirming]

    def as_record(self) -> dict[str, Any]:
        return {k: v for k, v in vars(self).items()}


def select_threshold(fit: list[dict[str, Any]], target: float) -> tuple[float, float, float] | None:
    """The threshold with the most coverage whose agreement on the fit split reaches `target`."""
    y = [r["gold"] is True for r in fit]
    best: tuple[float, float, float] | None = None
    for t in np.arange(0.50, 0.995, 0.01):
        answers = [wire.decision(float(r["p"]), float(t), contested=bool(r.get("contested"))) for r in fit]
        covered = [(a, truth) for a, truth in zip(answers, y, strict=True) if a is not None]
        if not covered:
            continue
        agree = sum(1 for a, truth in covered if a == truth) / len(covered)
        cov = len(covered) / len(fit)
        if agree >= target and (best is None or cov > best[0] or (cov == best[0] and float(t) < best[1])):
            best = (cov, round(float(t), 2), agree)
    if best is None:
        return None
    return best[1], best[0], best[2]


def fit_key(
    rows: list[dict[str, Any]],
    target: float,
    tolerance: float = 0.0,
    *,
    family_size: int,
    permits_on: bool = True,
    attempt: int = 1,
    canonical_language: str = "en",
) -> FitResult:
    """Select a threshold on the fit split; confirm it on unseen holdout cases (one per case).

    `rows` carry `split`, `p`, `gold`, `resolved_by`, `group` and optionally `lang`. Only fit
    and holdout rows are read. `tolerance` is accepted for older callers and ignored: the
    confirmation standard is fixed (certificates.py), not relaxed by the caller.
    """
    del tolerance
    key = rows[0]["key"].split("[")[0] if rows else "?"
    fit = [r for r in rows if r.get("split") == "fit" and r.get("p") is not None and type(r.get("gold")) is bool]
    hold = [r for r in rows if r.get("split") == "holdout"]
    if len(fit) < 10:
        return FitResult(key, None, "no-data", n_fit=len(fit), n_holdout=len(hold), attempt=attempt,
                         reason=f"only {len(fit)} labelled items in the fit split; at least 10 are needed")
    chosen = select_threshold(fit, target)
    if chosen is None:
        return FitResult(key, None, "unfit", n_fit=len(fit), n_holdout=len(hold), attempt=attempt,
                         reason=f"no threshold reached {target:.2f} agreement on the fit split")
    t, coverage, agreement = chosen
    conf = confirm(hold, t, permits_on=permits_on, family_size=family_size, attempt=attempt,
                   canonical_language=canonical_language)
    return FitResult(
        key, t, "fitted", coverage_fit=coverage, agreement_fit=agreement, n_fit=len(fit), n_holdout=len(hold),
        provisional=not conf.deployable, deployable=conf.deployable, attempt=attempt,
        alpha_per_bound=conf.alpha_per_bound, n_eligible=conf.eligible, n_automatic=conf.automatic,
        n_wrong=conf.wrong, n_releases=conf.releases, n_bad_releases=conf.bad_releases,
        n_abstentions=conf.abstentions, r_upper=conf.r_upper, r_plus_upper=conf.r_plus_upper,
        c_lower=conf.c_lower, fpr=conf.fpr, false_rejection_rate=conf.false_rejection_rate,
        confirming=conf.confirming, reason=conf.reason,
    )


def _runs_by_id(index: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Every recorded call grouped by run, retracted ones included, in recorded order."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for row in index.values():
        if not isinstance(row, dict) or row.get("_ambiguous") or "run_id" not in row:
            continue
        runs.setdefault(str(row["run_id"]), []).append(row)
    for rows in runs.values():
        rows.sort(key=_seq_of)
    return runs


def _seq_of(row: dict[str, Any]) -> int:
    seq = row.get("seq")
    return seq if isinstance(seq, int) else 0


def _run_signature(run: list[dict[str, Any]]) -> tuple[Any, ...]:
    """What makes two runs measurements of the same thing: card, pack, model, transport, answer
    language, the questions asked together and the sealed wording they were asked in."""
    head = next((r for r in run if r.get("call_role") == "primary"), run[0])
    raw_keys = head.get("question_keys")
    keys: list[Any] = raw_keys if isinstance(raw_keys, list) else []
    raw_bindings = head.get("question_bindings")
    bindings: dict[str, Any] = raw_bindings if isinstance(raw_bindings, dict) else {}
    seals = sorted(str(b.get("template_hash")) for b in bindings.values() if isinstance(b, dict))
    language = wire.answer_language(head, str(keys[0])) if keys else ""
    return (
        head.get("card_id"), head.get("state_sha"), head.get("model_id"), head.get("transport"), language,
        tuple(sorted({str(k).split("[")[0] for k in keys})), tuple(seals),
    )


def _bind_responses(
    card: RoleCard, responses: list[dict[str, Any]], ledger: Ledger, items: dict[str, GoldenItem]
) -> tuple[list[dict[str, Any]], list[str]]:
    """Bind every response to the one run that measured its item; derive everything from the run.

    Each response names an item and a run. The run must exist whole in the measurement ledger
    (no retracted, ambiguous or malformed call, every payload hashing to what it says was sent),
    must have been asked about this item's pack in this item's declared input language, and must
    be the only run on that pack under the same conditions: an honest `measure` makes exactly
    one, so a second run is either a retry nobody recorded or a favourable run being chosen.
    The response's own copies of the value and the identity must agree with the derivation.
    """
    index = ledger.call_index()
    runs = _runs_by_id(index)
    problems: list[str] = []
    ambiguous = sorted(cid for cid, row in index.items() if row.get("_ambiguous"))
    if ambiguous:
        problems.append(f"the measurement ledger records {len(ambiguous)} call id(s) more than once (first {ambiguous[0]})")
    by_signature: dict[tuple[Any, ...], list[str]] = {}
    for run_id, run in runs.items():
        if any(wire.row_shape_problems(row) for row in run):
            continue
        by_signature.setdefault(_run_signature(run), []).append(run_id)
    cited: dict[str, str] = {}
    bound: list[dict[str, Any]] = []
    for r in responses:
        name = str(r.get("item_id"))
        item = items.get(name)
        run_id = str(r.get("run_id"))
        run = runs.get(run_id, []) if r.get("run_id") else []
        if item is None or not run:
            problems.append(f"{name}: run {r.get('run_id')} is not in the measurement ledger")
            continue
        if run_id in cited:
            problems.append(f"{name}: run {run_id} is also cited by {cited[run_id]}")
            continue
        cited[run_id] = name
        shape = [p for row in run for p in wire.row_shape_problems(row)]
        payload = [p for row in run for p in wire.payload_problems(row)] if not shape else []
        if shape or payload:
            problems.append(f"{name}: {(shape or payload)[0]}")
            continue
        if any(row.get("_retracted") for row in run):
            problems.append(f"{name}: run {run_id} contains a retracted call")
            continue
        primaries = [row for row in run if row.get("call_role") == "primary"]
        if len(primaries) != 1:
            problems.append(f"{name}: run {run_id} has {len(primaries)} primary calls")
            continue
        primary = primaries[0]
        if primary.get("card_id") != card.id or primary.get("state_sha") != sha_of(item.pack):
            problems.append(f"{name}: run {run_id} was asked about another card or pack than this item's")
            continue
        if primary.get("input_language") != item.lang:
            problems.append(f"{name}: run {run_id} declared input language {primary.get('input_language')!r}, the item is {item.lang!r}")
            continue
        if item.leaf_space is not None and item.leaf_space.get("primitive") in ("choice", "score"):
            sdk = ((primary.get("request") or {}).get("questions") or {}).get(r["key"]) or {}
            if item.leaf_space["primitive"] == "choice":
                asked_space, item_space = sorted(wire.declared_options(sdk)), sorted(item.leaf_space.get("options") or {})
            else:
                asked_space, item_space = list(sdk.get("criteria") or []), list(item.leaf_space.get("levels") or [])
            if asked_space != item_space:
                problems.append(f"{name}: run {run_id} was asked over another answer space than the item's")
                continue
        same = by_signature.get(_run_signature(run), [])
        if len(same) > 1:
            problems.append(
                f"{name}: {len(same)} runs were made on this item's pack under the same conditions; "
                "an item is measured once, so the favourable one cannot be chosen"
            )
            continue
        d = wire.derive_question(run, r["key"])
        invalid = bool(d.problems)
        derived: dict[str, Any] = {
            "invalid": invalid,
            "p": None if invalid else d.p,
            "choice": None if invalid else d.choice,
            "score": None if invalid else d.score,
            "contested": False if invalid else d.contested,
            "call_ids": sorted(str(row.get("id")) for row in run) if invalid else sorted(c.call_id for c in d.contributions),
        }
        language = wire.answer_language(primary, r["key"])
        base = r["key"].split("[")[0]
        binding = ((primary.get("question_bindings") or {}).get(r["key"]) or {})
        identity = {
            "transport": primary.get("transport"), "model_id": primary.get("model_id"), "language": language,
            "asked_keys": wire.asked_base_keys(run), "input_language": primary.get("input_language"),
            "conditions": conditions_for(card, r["key"], model_id=str(primary.get("model_id")),
                                         transport=str(primary.get("transport")), language=language),
            "current": binding.get("template_hash") == compute_hash(card, base)
            and primary.get("run_protocol") == wire.run_protocol(card),
        }
        mismatched = [
            field_name for field_name, stored, now in (
                ("invalid", bool(r.get("invalid")), invalid),
                ("transport", r.get("transport"), identity["transport"]),
                ("model", r.get("model_id"), identity["model_id"]),
                ("language", r.get("language"), identity["language"]),
                ("asked keys", sorted(r.get("asked_keys") or []), identity["asked_keys"]),
                # conditions are computed with the current card, so they can only be compared for a
                # run of the current wording; a stale run is refused as stale, not as tampered
                ("measurement conditions", r.get("measurement_conditions"),
                 identity["conditions"] if identity["current"] else r.get("measurement_conditions")),
                ("call ids", sorted(r.get("call_ids") or []), derived["call_ids"]),
            ) if stored != now
        ]
        if not invalid:
            for field_name in ("choice", "contested"):
                if (bool(r.get(field_name)) if field_name == "contested" else r.get(field_name)) != derived[field_name]:
                    mismatched.append(field_name)
            for field_name in ("p", "score"):
                stored_value, value = r.get(field_name), derived[field_name]
                if stored_value is None or value is None:
                    if (stored_value is None) != (value is None):
                        mismatched.append(field_name)
                elif not math.isclose(float(stored_value), float(value), abs_tol=1e-9):
                    mismatched.append(field_name)
        if mismatched:
            problems.append(f"{name}: the response's {', '.join(mismatched)} disagree with its run {run_id}")
            continue
        bound.append({**r, **derived, "_identity": identity})
    return bound, problems


def _is_real_transport(label: Any) -> bool:
    return not str(label or "").startswith("controlled:")


def _check_measurement_identity(card: RoleCard, key: str, rs: list[dict[str, Any]]) -> str:
    """The conditions every response of a key was measured under, read from its ledger run."""
    stale = [r["item_id"] for r in rs if not r["_identity"]["current"]]
    if stale:
        raise CalibrationIdentityError(
            f"{card.id}.{key}: {len(stale)} measurement(s) (first {stale[0]}) were taken under measurement "
            "conditions the card no longer has (its wording or re-test settings changed); re-measure before fitting"
        )
    recorded = sorted({str(r["_identity"]["conditions"]) for r in rs})
    if len(recorded) > 1:
        raise CalibrationIdentityError(
            f"{card.id}.{key}: measurements come from {len(recorded)} different measurement "
            f"conditions ({', '.join(recorded)}); a model, wording, rung, re-test, transport or "
            "language change makes them different experiments, so they cannot be pooled"
        )
    return recorded[0]


def _confirm_attempts(card_id: str, root: Path, key: str, conditions: str) -> int:
    return sum(
        1 for e in holdout_evaluations(card_id, root)
        if e.get("purpose") == "confirm" and e.get("key") == key and e.get("conditions") == conditions
    )


def fit_card(card: RoleCard, *, target: float, root: Path | None = None, write: bool = True) -> list[FitResult]:
    """Fit every yes/no `main` question of the card and issue the certificates the evidence supports.

    Any failure revokes the card's validated certificates before it raises: a refit that cannot
    re-establish the evidence must not leave the previous authority in place.
    """
    root = _root(root)
    d = card_dir(card.id, root)
    with exclusive(d):
        try:
            return _fit_card_locked(card, target=target, root=root, write=write)
        except (CalibrationProtocolError, CalibrationSplitError, CalibrationEvidenceError) as exc:
            revoke_validated(root, card.id, f"refit failed: {str(exc)[:300]}")
            raise


def _stratum(
    items: list[GoldenItem], groups: dict[str, str], key: str, seen_groups: set[str], language: str
) -> tuple[list[GoldenItem], int]:
    """Holdout items of `key` in unseen cases that fall in the certified input-language stratum.

    The stratum predicate is `wire.input_language_of`, the same one the runtime applies to an
    input, so the certified population is the deployed one. Also returns how many unseen
    holdout items fell outside it.
    """
    candidates = [
        it for it in items
        if it.key.split("[")[0] == key and it.split == "holdout" and groups[it.id] not in seen_groups
    ]
    inside = [it for it in candidates if wire.input_language_of(it.pack, it.lang) == language]
    return inside, len(candidates) - len(inside)


def _stratum_fit_rows(
    key_rows: list[dict[str, Any]], items: dict[str, GoldenItem], language: str
) -> list[dict[str, Any]]:
    """Usable fit-split rows in the certified stratum: the threshold is selected where it is claimed."""
    return [
        r for r in key_rows
        if r["split"] == "fit" and not r.get("invalid")
        and wire.input_language_of(items[r["item_id"]].pack, items[r["item_id"]].lang) == language
    ]


def _holdout_rows(
    stratum: list[GoldenItem],
    groups: dict[str, str],
    responded: dict[str, dict[str, Any]],
    truths: dict[tuple[str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Confirming rows: values from the bound runs, truth from truth.jsonl through `resolve_truth`."""
    rows: list[dict[str, Any]] = []
    for it in stratum:
        resp = responded.get(it.id)
        label, kind, _ = resolve_truth(truths.get((it.id, it.key), []))
        rows.append({
            "item_id": it.id, "key": it.key, "split": "holdout", "group": groups[it.id], "lang": it.lang,
            "lineage": it.lineage, "p": None if resp is None or resp.get("invalid") else resp.get("p"),
            "contested": bool(resp is not None and resp.get("contested")),
            "gold": label, "resolved_by": kind, "_measured": resp is not None,
        })
    return rows


def _fit_card_locked(card: RoleCard, *, target: float, root: Path, write: bool) -> list[FitResult]:
    d = card_dir(card.id, root)
    items = load_items(card.id, root)
    leakage = split_leakage(items)
    if leakage:
        raise CalibrationSplitError(
            f"{card.id}: {len(leakage)} case(s) are split across two splits, so the holdout is "
            "not out of sample: "
            + "; ".join(f"{leak['group']} in {'/'.join(leak['splits'])}" for leak in leakage[:5])
        )
    rows = _joined_rows(card, root, SPLITS)  # bound to their ledger runs, or refused
    by_id = {it.id: it for it in items}
    groups = group_map(items)
    family = family_of(card)
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_key.setdefault(r["key"].split("[")[0], []).append(r)
    conditions = {key: _check_measurement_identity(card, key, rs) for key, rs in sorted(by_key.items()) if key in family}
    dataset = dataset_version(rows)
    responses_v = responses_version(rows)
    purpose = f"fit:target={target};protocol={CALIBRATION_PROTOCOL_VERSION};standard={STANDARD_ID}"
    ex_id = experiment_id(conditions=list(conditions.values()), dataset=dataset, responses=responses_v, purpose=purpose)
    seen = _seen_items(card.id, root)
    seen_groups = {groups[i] for i in seen if i in groups}
    truths = _truth_rows(d)
    today = datetime.now(UTC).date().isoformat()
    results: list[FitResult] = []
    for key in family:
        q = card.questions[key]
        key_rows = by_key.get(key, [])
        if not key_rows:
            if write and (cert := effective_certificate(root, card.id, key)) is not None and cert.status == "validated":
                append_certificate(root, card.id, Certificate(key=key, status="revoked", reason="evidence withdrawn: no measured items remain"))
            results.append(FitResult(key, None, "no-data", reason="no measured items for this question"))
            continue
        identity = key_rows[0]["_identity"]
        stratum_language = str(identity["language"])  # the canonical language, part of the conditions
        responded = {r["item_id"]: r for r in key_rows}
        stratum, outside = _stratum(items, groups, key, seen_groups, stratum_language)
        holdout_rows = _holdout_rows(stratum, groups, responded, truths)
        reps = representatives(holdout_rows, stratum_language)
        unmeasured = [rep["item_id"] for rep in reps if not rep["_measured"]]
        if unmeasured:
            raise CalibrationEvidenceError(
                f"{card.id}.{key}: {len(unmeasured)} holdout case representative(s) were never measured "
                f"(first {unmeasured[0]}); measure every case so the easy ones cannot certify alone"
            )
        fit_rows = _stratum_fit_rows(key_rows, by_id, stratum_language)
        attempt = _confirm_attempts(card.id, root, key, conditions[key]) + 1
        fr = fit_key(fit_rows + holdout_rows, target, family_size=len(family), permits_on=q.permits_on,
                     attempt=attempt, canonical_language=stratum_language)
        undeclared = [rep["item_id"] for rep in reps if not rep.get("lineage")]
        if fr.deployable and undeclared:
            fr.deployable = False
            fr.provisional = True
            fr.reason += f"; {len(undeclared)} confirming case(s) declare no lineage, so their independence is not shown"
        if outside:
            fr.reason += f"; {outside} unseen holdout item(s) outside the {stratum_language} input-language stratum were not counted"
        fr.evidence = [
            {"item_id": r["item_id"], "key": r["key"], "split": r["split"], "run_id": r.get("run_id"),
             "call_ids": r.get("call_ids", [])}
            for r in key_rows
        ]
        if stratum:
            _log_evaluation(card.id, root, {
                "purpose": "confirm", "key": key, "conditions": conditions[key], "experiment_id": ex_id,
                "attempt": attempt, "stratum": stratum_language, "golden_rows": len(items),
                "item_ids": sorted(it.id for it in stratum), "groups": sorted({groups[it.id] for it in stratum}),
                "write": write,
            })
        results.append(fr)
        if not write:
            continue
        if fr.deployable and fr.threshold is not None:
            confirming_ids = {c["item_id"] for c in fr.confirming}
            used = [r for r in key_rows if r["item_id"] in confirming_ids] + fit_rows
            languages: dict[str, int] = {}
            for c in fr.confirming:
                languages[c.get("lang", stratum_language)] = languages.get(c.get("lang", stratum_language), 0) + 1
            append_certificate(root, card.id, Certificate(
                key=key, status="validated", value=fr.threshold, reason=fr.reason, at=now_iso(),
                experiment_id=ex_id, dataset_version=dataset, responses_version=responses_v,
                calibration_protocol_version=CALIBRATION_PROTOCOL_VERSION,
                wire_protocol_version=WIRE_PROTOCOL_VERSION, standard=STANDARD_ID,
                measurement_conditions=conditions[key], model_id=str(identity["model_id"]),
                transport=str(identity["transport"]), language=stratum_language,
                input_language=stratum_language, permits_on=q.permits_on,
                asked_keys=list(identity["asked_keys"]), family=family, attempt=fr.attempt,
                alpha_per_bound=fr.alpha_per_bound, languages=languages, golden_rows=len(items), target=target,
                fit_items=sorted(r["item_id"] for r in fit_rows), confirming=fr.confirming,
                counts=_counts(fr), bounds={"r_upper": fr.r_upper, "r_plus_upper": fr.r_plus_upper, "c_lower": fr.c_lower},
                digests=_digests(card, root, key, used, len(items)),
            ))
        else:
            previous = effective_certificate(root, card.id, key)
            if fr.status == "no-data":
                if previous is not None and previous.status == "validated":
                    append_certificate(root, card.id, Certificate(
                        key=key, status="revoked", reason=f"evidence withdrawn: {fr.reason}", at=now_iso(),
                        experiment_id=ex_id,
                    ))
                continue
            status: Literal["provisional", "unfit"] = "provisional" if fr.status == "fitted" else "unfit"
            append_certificate(root, card.id, Certificate(
                key=key, status=status, reason=f"{fr.status}: {fr.reason}", at=now_iso(),
                experiment_id=ex_id, counts=_counts(fr),
                bounds={"r_upper": fr.r_upper, "r_plus_upper": fr.r_plus_upper, "c_lower": fr.c_lower},
            ))
    payload = {
        "card_id": card.id,
        "calibration_protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "wire_protocol_version": WIRE_PROTOCOL_VERSION,
        "standard": STANDARD_ID,
        "experiment_id": ex_id,
        "dataset_version": dataset,
        "responses_version": responses_v,
        "measurement_conditions": conditions,
        "target": target,
        "fitted_at": now_iso(),
        "date": today,
        "family": family,
        "golden_rows": len(items),
        "write": write,
        "results": [fr.as_record() for fr in results],
        "supersedes": None,
    }
    fit_path = d / "fit.json"
    if write:
        payload["supersedes"] = _supersede(fit_path)
        fit_path.write_text(json.dumps(payload, indent=2))
    return results


def _counts(fr: FitResult) -> dict[str, int]:
    return {
        "eligible": fr.n_eligible, "automatic": fr.n_automatic, "wrong": fr.n_wrong,
        "releases": fr.n_releases, "bad_releases": fr.n_bad_releases, "abstentions": fr.n_abstentions,
        "fit": fr.n_fit,
    }


def _truth_rows(d: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in _read_jsonl(d / "truth.jsonl"):
        out.setdefault((t["item_id"], t["key"]), []).append(t)
    return out


def _ledger_rows_for(ledger: Ledger, call_ids: list[str]) -> list[dict[str, Any]]:
    index = ledger.call_index()
    return [index.get(c, {"id": c, "_missing": True}) for c in sorted(set(call_ids))]


def _golden_prefix(d: Path, n: int) -> tuple[list[GoldenItem], str]:
    """The first `n` golden rows (the set as it stood when a fit ran) and their digest.

    The golden set is append-only, so the rows a certificate was fitted on stay a prefix.
    """
    path = d / "golden.jsonl"
    lines = [x for x in path.read_text().splitlines() if x.strip()] if path.exists() else []
    prefix = lines[:n]
    return [GoldenItem.model_validate(json.loads(x)) for x in prefix], sha_of(prefix)


def _digests(card: RoleCard, root: Path, key: str, used: list[dict[str, Any]], golden_rows: int) -> dict[str, str]:
    d = card_dir(card.id, root)
    ids = sorted({r["item_id"] for r in used})
    gold = _gold_index(d)
    truths = _truth_rows(d)
    call_ids = [c for r in used for c in (r.get("call_ids") or [])]
    return {
        "responses": sha_of(sorted((json.dumps(r, sort_keys=True) for r in _current_responses(d) if r["item_id"] in ids), key=str)),
        "gold": sha_of([gold.get((i, key)) for i in ids]),
        "truth": sha_of([truths.get((i, key)) for i in ids]),
        "ledger": sha_of(_ledger_rows_for(measurement_ledger(card.id, root), call_ids)),
        "golden": _golden_prefix(d, golden_rows)[1],
    }


def _fit_record_problems(d: Path, cert: Certificate) -> list[str]:
    """The fit record of the certificate's experiment must say what the certificate says."""
    paths = [d / "fit.json", *sorted((d / "superseded").glob("fit-*.json"))]
    for p in paths:
        if not p.exists():
            continue
        doc = json.loads(p.read_text())
        if doc.get("experiment_id") != cert.experiment_id:
            continue
        found = next((r for r in doc.get("results", []) if r.get("key") == cert.key), None)
        if found is None:
            return [f"the fit record of experiment {cert.experiment_id} has no result for {cert.key}"]
        stated = {
            "deployable": True, "threshold": cert.value, "attempt": cert.attempt, "confirming": cert.confirming,
            "n_eligible": cert.counts.get("eligible"), "n_automatic": cert.counts.get("automatic"),
            "n_wrong": cert.counts.get("wrong"), "n_releases": cert.counts.get("releases"),
            "n_bad_releases": cert.counts.get("bad_releases"), "n_abstentions": cert.counts.get("abstentions"),
        }
        differ = [name for name, value in stated.items() if found.get(name) != value]
        return [f"the fit record disagrees with the certificate on {', '.join(differ)}"] if differ else []
    return [f"no fit record of experiment {cert.experiment_id}"]


def _grouping_problems(cert: Certificate, items: list[GoldenItem]) -> list[str]:
    """The case grouping as the golden set declares it now, checked against the certificate.

    A link added after certification (a shared group, pair or lineage) can join a confirming
    case to a fit case, or two confirming cases into one. Either breaks the one-decision-per-
    independent-case premise, so the certificate no longer reproduces.
    """
    groups = group_map(items)
    used = set(cert.fit_items) | {c["item_id"] for c in cert.confirming}
    problems = [
        f"case {leak['group']} now spans {'/'.join(leak['splits'])} and holds certified items"
        for leak in split_leakage(items)
        if used & set(leak["item_ids"])
    ]
    confirming_cases = [groups.get(c["item_id"]) for c in cert.confirming]
    if len(set(confirming_cases)) != len(confirming_cases):
        problems.append("two confirming items now belong to one case")
    return problems


def verify_certificate(card: RoleCard, cert: Certificate, root: Path) -> list[str]:
    """Re-derive a validated certificate from the evidence it names; list every mismatch.

    Nothing the certificate states is taken on trust, and no single evidence file is either:
    - every used response binds to its one ledger run, and the run (not the response row) gives
      the value, the contested flag and the measurement identity, which must be the certificate's;
    - the fit items re-select the threshold under the one decision rule;
    - the eligible cases are re-derived from the golden rows present at fit time and the holdout
      log (every unseen holdout case of the key in the stratum), the log entry of this
      experiment must list exactly those items, and the certificate's confirming cases must be
      exactly their representatives;
    - the confirming truth comes from truth.jsonl through the same resolution gold uses;
    - counts and bounds re-derive, the fit record agrees, and a real transport left a provider
      request id on every call.
    A consistent rewrite of several evidence files is not detectable here. Signed records are
    not built yet.
    """
    problems: list[str] = []
    d = root / card.id
    key = cert.key
    if not d.exists():
        return [f"no evidence directory {d}"]
    if cert.family != family_of(card):
        problems.append("the card's family of yes/no questions differs from the certificate's")
    problems.extend(_fit_record_problems(d, cert))
    items_all = load_items(card.id, root)
    by_id = {it.id: it for it in items_all}
    prefix, _ = _golden_prefix(d, cert.golden_rows)
    if cert.golden_rows <= 0 or len(prefix) != cert.golden_rows:
        problems.append("the golden set no longer holds the rows the certificate was fitted on")
    problems.extend(_grouping_problems(cert, items_all))
    responses = _current_responses(d)
    bound, bind_problems = _bind_responses(card, responses, measurement_ledger(card.id, root), by_id)
    problems.extend(bind_problems[:3])
    derived = {r["item_id"]: r for r in bound if r["key"].split("[")[0] == key}
    used_ids = [*cert.fit_items, *(c["item_id"] for c in cert.confirming)]
    for item_id in used_ids:
        r = derived.get(item_id)
        if r is None:
            continue
        idn = r["_identity"]
        stated = (cert.transport, cert.model_id, cert.language, cert.measurement_conditions, sorted(cert.asked_keys))
        found = (idn["transport"], idn["model_id"], idn["language"], idn["conditions"], sorted(idn["asked_keys"]))
        if tuple(map(str, found)) != tuple(map(str, stated)):
            problems.append(f"{item_id}: its run was measured under other conditions than the certificate states")
            break
    gold = _gold_index(d)
    prefix_ids = {it.id for it in prefix}
    stratum_fit = _stratum_fit_rows(
        [{**r, "split": by_id[i].split} for i, r in derived.items() if i in by_id and i in prefix_ids],
        by_id, str(cert.input_language),
    )
    if sorted(r["item_id"] for r in stratum_fit) != sorted(cert.fit_items):
        problems.append("the certificate's fit items are not every usable fit item of the stratum")
    fit_rows: list[dict[str, Any]] = []
    for item_id in cert.fit_items:
        it = by_id.get(item_id)
        r = derived.get(item_id)
        if it is None or it.split != "fit":
            problems.append(f"fit item {item_id} is no longer a fit item")
            continue
        if r is None or r.get("invalid") or r.get("p") is None:
            problems.append(f"fit item {item_id} has no usable response")
            continue
        fit_rows.append({**r, "gold": (gold.get((item_id, r["key"])) or {}).get("gold"), "split": "fit"})
    labelled = [r for r in fit_rows if type(r.get("gold")) is bool]
    chosen = select_threshold(labelled, float(cert.target or 0.0)) if labelled else None
    if chosen is None or chosen[0] != cert.value:
        problems.append(f"the fit items re-select threshold {chosen[0] if chosen else None}, not {cert.value}")
    log = holdout_evaluations(card.id, root)
    first = next((n for n, e in enumerate(log) if e.get("experiment_id") == cert.experiment_id), None)
    entry = next(
        (n for n, e in enumerate(log)
         if e.get("experiment_id") == cert.experiment_id and e.get("purpose") == "confirm"
         and e.get("key") == key and e.get("attempt") == cert.attempt),
        None,
    )
    if first is None or entry is None:
        problems.append("the holdout log has no confirmation of this key by this experiment")
    else:
        seen = {i for e in log[:first] for i in e.get("item_ids", [])}
        groups = group_map(prefix)
        seen_groups = {groups[i] for i in seen if i in groups}
        stratum, _ = _stratum(prefix, groups, key, seen_groups, str(cert.input_language))
        if sorted(it.id for it in stratum) != sorted(log[entry].get("item_ids", [])):
            problems.append("the holdout log does not list exactly the eligible cases this experiment scored")
        prior = sum(
            1 for e in log[:entry]
            if e.get("purpose") == "confirm" and e.get("key") == key and e.get("conditions") == cert.measurement_conditions
        )
        if prior != cert.attempt - 1:
            problems.append(f"this was attempt {prior + 1} under these conditions, the certificate says {cert.attempt}")
        hold = _holdout_rows(stratum, groups, derived, _truth_rows(d))
        reps = representatives(hold, str(cert.input_language))
        if [(c["item_id"], c["group"]) for c in cert.confirming] != [(r["item_id"], r["group"]) for r in reps]:
            problems.append("the certificate's confirming cases are not exactly the representatives of every eligible case")
        if any(not r.get("lineage") for r in reps):
            problems.append("a confirming case declares no lineage, so its independence is not shown")
        if cert.value is not None:
            conf = confirm(hold, cert.value, permits_on=bool(cert.permits_on), family_size=len(cert.family),
                           attempt=cert.attempt, canonical_language=str(cert.input_language))
            recomputed = conf.counts()
            for name, count in cert.counts.items():
                if name != "fit" and recomputed.get(name) != count:
                    problems.append(f"count {name} re-derives as {recomputed.get(name)}, the certificate says {count}")
            for name, bound_value in conf.bounds().items():
                stored = cert.bounds.get(name)
                if (bound_value is None) != (stored is None) or (
                    bound_value is not None and stored is not None and not math.isclose(bound_value, stored, abs_tol=1e-12)
                ):
                    problems.append(f"bound {name} re-derives as {bound_value}, the certificate says {stored}")
            if not conf.deployable:
                problems.append("the confirming cases do not meet the standard")
    used = [derived[i] for i in used_ids if i in derived]
    now = _digests(card, root, key, used, cert.golden_rows)
    for name, digest in cert.digests.items():
        if now.get(name) != digest:
            problems.append(f"the {name} evidence changed since the certificate was issued")
    if _is_real_transport(cert.transport):
        index = measurement_ledger(card.id, root).call_index()
        for r in used:
            for call_id in r.get("call_ids") or []:
                row = index.get(call_id) or {}
                if not ((row.get("response") or {}).get("request_id")):
                    problems.append(f"call {call_id} from a real transport carries no provider request id")
                    break
    return problems


def degrade_if_needed(card: RoleCard, card_path: Path, results: list[FitResult], config: dict[str, Any]) -> str | None:
    """A card whose gating questions all failed to fit runs at the next rung; recorded on the card."""
    gating = [r for r in results if r.status != "no-data"]
    if gating and all(r.status == "unfit" for r in gating):
        reason = "no threshold reached the target agreement on the fit split for any gating question"
        if card_path.resolve().is_relative_to(packaged_rolecards_dir().resolve()):
            # Calibration never writes package resources; the degradation is reported only.
            return reason + " (not recorded: the packaged role cards are read-only)"
        set_rung(card_path, Rung.llm_isolated, reason, datetime.now(UTC).date().isoformat())
        return reason
    return None


# ---------------------------------------------------------------- report


def report(card: RoleCard, config: dict[str, Any], root: Path | None = None) -> Path:
    from sextant.certificates import read_certificates

    root = _root(root)
    d = card_dir(card.id, root)
    metrics = json.loads((d / "metrics.json").read_text()) if (d / "metrics.json").exists() else {}
    fit_doc = json.loads((d / "fit.json").read_text()) if (d / "fit.json").exists() else {}
    fits = fit_doc.get("results", []) if isinstance(fit_doc, dict) else []
    items = load_items(card.id, root)
    gold = _read_jsonl(d / "gold.jsonl")
    labels = _read_jsonl(d / "labels.jsonl")
    leakage = split_leakage(items)
    seen = len(_seen_items(card.id, root))
    fams = sorted({f"{lab.get('family')}:{lab.get('isolation')}:{lab.get('provenance')}" for lab in labels})
    resolved_by: dict[str, int] = {}
    for g in gold:
        resolved_by[str(g.get("resolved_by"))] = resolved_by.get(str(g.get("resolved_by")), 0) + 1
    counts = {s: sum(1 for i in items if i.split == s) for s in SPLITS}
    lines = [f"# Calibration report: {card.id}", "", f"purpose: {card.purpose}", "",
             f"items: {len(items)} (" + ", ".join(f"{s} {counts[s]}" for s in SPLITS) + ")"
             + f"; cases: {len(set(group_map(items).values()))}"
             + f"; gold resolved: {sum(1 for g in gold if g.get('gold') is not None)}; dissent: {sum(1 for g in gold if g.get('dissent'))}",
             f"gold resolved by: {', '.join(f'{k} {v}' for k, v in sorted(resolved_by.items())) or 'nothing'}"
             + " (an author's expectation never resolves an item)",
             f"split leakage: {'none' if not leakage else str(len(leakage)) + ' case(s) across splits; the holdout is not out of sample'}",
             f"locked items already scored: {seen}; those cases can no longer confirm a certificate",
             f"label families: {', '.join(fams) if fams else 'none'}",
             f"rung: {card.rung.value}" + (f" (degraded {card.degraded_on}: {card.degrade_reason})" if card.rung != Rung.jev else ""),
             f"constants: target_agreement_T2={config['calibration']['target_agreement_T2']}, reruns={card.reruns}, permutations={card.permutations}, contested_margin={card.contested_margin}, min_items_per_bin={config['calibration']['min_items_per_bin']}",
             "", "## Per question", ""]
    for key, m in metrics.get("keys", {}).items():
        lines.append(f"### {key} ({m['primitive']}, n={m['n']}, labelled {m.get('n_labelled', 0)}, invalid {m.get('n_invalid', 0)})")
        if m["primitive"] == "noul" and "brier" in m:
            lines.append(f"- Brier {m['brier']:.3f}, log loss {m['log_loss']:.3f}, AUROC {m['auroc'] if m['auroc'] is None else round(m['auroc'], 3)}, base rate {m['base_rate']:.2f}, accuracy at 0.5 {m['accuracy_at_half']:.2f}")
            mu = m["murphy"]
            if mu.get("defined"):
                lines.append(f"- Murphy ({mu.get('binning', 'equal-width')} binned, {mu.get('bins', 10)} bins): reliability {mu['reliability']:.4f}, resolution {mu['resolution']:.4f}, uncertainty {mu['uncertainty']:.4f}")
                lines.append(
                    f"- the three terms combine to {mu['brier_from_bins']:.4f} against a raw Brier of "
                    f"{mu['brier']:.4f}; binning residual {mu['approximation_residual']:+.4f}. "
                    "The decomposition holds for the binned forecasts, not the original ones."
                )
            lines.append("- coverage curve (t: coverage / agreement): " + ", ".join(f"{c['t']:.2f}: {c['coverage']:.2f}/{c['agreement']:.2f}" for c in m["coverage_curve"] if not math.isnan(c["agreement"])))
        elif m["primitive"] == "choice" and "accuracy" in m:
            lines.append(f"- accuracy {m['accuracy']:.2f}, contested rate {m['contested_rate']:.2f}")
        elif "mae_levels" in m:
            lines.append(f"- mean absolute error in levels {m['mae_levels']:.2f}")
        st = m["stability"]
        lines.append(f"- stability: rerun sd {st['rerun_sd_mean']}, permutation sd {st['permutation_sd_mean']}, negation gap {st['negation_gap_mean']}, paraphrase sd {st['paraphrase_sd_mean']}; zh-en delta {m['zh_en_delta_mean']} over {m['zh_en_pairs']} pairs")
        lines.append("")
    if fits:
        lines.append("## Latest fit")
        lines.append("")
        lines.append(f"experiment {fit_doc.get('experiment_id')} on data {fit_doc.get('dataset_version')} and measurements {fit_doc.get('responses_version')}; standard {fit_doc.get('standard')}")
        lines.append("")
        lines.append("| key | status | deployable | threshold | attempt | eligible cases | automatic | wrong | releases | bad releases | R upper | R+ upper | C lower | why |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for f in fits:
            def fmt(v: Any) -> str:
                return "-" if v is None else f"{v:.3f}"
            lines.append(
                f"| {f['key']} | {f['status']} | {'yes' if f.get('deployable') else 'no'} | {f.get('threshold')} | {f.get('attempt')} | "
                f"{f.get('n_eligible', 0)} | {f.get('n_automatic', 0)} | {f.get('n_wrong', 0)} | {f.get('n_releases', 0)} | "
                f"{f.get('n_bad_releases', 0)} | {fmt(f.get('r_upper'))} | {fmt(f.get('r_plus_upper'))} | {fmt(f.get('c_lower'))} | {f.get('reason', '')} |"
            )
        lines.append("")
    certs = read_certificates(root, card.id)
    if certs:
        lines.append("## Certificate log")
        lines.append("")
        for c in certs:
            lines.append(f"- {c.at} {c.key}: {c.status}" + (f" at {c.value}" if c.value is not None else "") + (f" ({c.reason[:160]})" if c.reason else ""))
        lines.append("")
        lines.append(
            "Only a validated certificate that re-derives from its evidence authorizes anything; the "
            "runtime checks that every time. A provisional entry means the holdout could not confirm "
            "the target, only fail to rule it out."
        )
    lines.append("")
    lines.append("Internal document: calibration numbers are not published (customer agreement).")
    p = d / "report.md"
    p.write_text("\n".join(lines) + "\n")
    return p


# ---------------------------------------------------------------- canary


def canary_freeze(card_ids: list[str], per_card: int, root: Path | None = None, seed: int = 20260918) -> int:
    """Freeze up to `per_card` measured items per card with their stored answers and identity."""
    root = _root(root)
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for cid in card_ids:
        resp = [r for r in _current_responses(card_dir(cid, root)) if not r.get("invalid")]
        items = {i.id: i for i in load_items(cid, root)}
        candidates = [r for r in resp if r["item_id"] in items and r.get("p") is not None]
        rng.shuffle(candidates)
        for r in candidates[:per_card]:
            it = items[r["item_id"]]
            rows.append({
                "card_id": cid, "item_id": it.id, "key": it.key, "pack": it.pack, "lang": it.lang,
                "leaf_space": it.leaf_space, "p": r["p"], "model_id": r["model_id"],
                "transport": r.get("transport"), "language": r.get("language"),
                "protocol_version": r.get("protocol_version"),
                "protocol_fingerprint": r.get("protocol_fingerprint"),
                "measurement_conditions": r.get("measurement_conditions"),
                "question_hash": r.get("question_hash"),
                "frozen_at": datetime.now(UTC).date().isoformat(),
            })
    p = root / "canary.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(p, rows)
    return len(rows)


def canary_check(client: JevClient, cards: dict[str, RoleCard], band: float, root: Path | None = None) -> dict[str, Any]:
    """Re-run the frozen items and report drift, but only where drift is even defined.

    A different model is reported as moved before anything else, a missing card is counted,
    and changed card, protocol, transport or language conditions are reported as moved and not
    compared. `compared` says how many rows were actually re-run; zero is not a pass.
    """
    root = _root(root)
    p = root / "canary.jsonl"
    rows = _read_jsonl(p)
    drifted: list[dict[str, Any]] = []
    model_moved: list[str] = []
    protocol_moved: list[dict[str, Any]] = []
    missing_cards: list[str] = []
    compared = 0
    for r in rows:
        card = cards.get(r["card_id"])
        if card is None:
            missing_cards.append(f"{r['card_id']}:{r['item_id']}")
            continue
        if r.get("model_id") != client.model_id:
            model_moved.append(f"{r['card_id']}:{r['item_id']}: {r.get('model_id')} -> {client.model_id}")
            continue
        base_key = r["key"].split("[")[0]
        now_conditions = conditions_for(
            card, r["key"], model_id=client.model_id, transport=client.transport, language=client.language
        )
        if r.get("measurement_conditions") != now_conditions:
            protocol_moved.append({
                "card_id": r["card_id"], "item_id": r["item_id"], "key": r["key"],
                "frozen": r.get("measurement_conditions"), "now": now_conditions,
            })
            continue
        res = client.ask(card.id, r["pack"], keys=asked_keys(card, r["key"]), leaf_space=r.get("leaf_space"), variant="canary",
                         allow_degraded=True, input_language=r.get("lang"))
        qe = res.estimates.get(r["key"]) or res.estimates.get(base_key)
        if qe is None or qe.estimate.p is None:
            continue
        compared += 1
        delta = abs(qe.estimate.p - float(r["p"]))
        if delta > band:
            drifted.append({"card_id": r["card_id"], "item_id": r["item_id"], "key": r["key"], "frozen_p": r["p"], "now_p": qe.estimate.p, "delta": round(delta, 3)})
    return {
        "n": len(rows), "compared": compared, "drifted": drifted, "model_moved": model_moved,
        "protocol_moved": protocol_moved, "missing_cards": len(missing_cards), "band": band,
    }
