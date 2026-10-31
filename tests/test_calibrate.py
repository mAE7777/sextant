"""Calibration harness tests on a synthetic golden set with a scripted Jev caller."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sextant import calibrate as cal
from sextant.config import load_config, packaged_rolecards_dir
from sextant.jev_client import JevClient
from sextant.jev_transport import JevResult
from sextant.rolecards import load_all
from sextant.state import init_state

CARDS_DIR = packaged_rolecards_dir()

pytestmark = pytest.mark.unit


def _items(n: int) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        related = i % 2 == 0
        child = f"Will the base salary cover expense type {i}?" if related else f"What is fact number {i} about geology?"
        out.append({
            "key": "relevance",
            "pack": {"parent_question": "Should I accept this job offer?", "child_question": child, "parent_kind": "question", "child_kind": "factor"},
            "expected": related,
            "lang": "en",
        })
    return out


def _client(cards: dict[str, Any], by_child: dict[str, bool], root: Path) -> JevClient:
    config = load_config()
    config["jev"]["requests_per_minute"] = 1_000_000  # the scripted caller is local; do not sleep
    return JevClient(
        config, cards, ledger=cal.measurement_ledger("R3", root), caller=_caller_for(by_child), seed=1,
        transport="controlled:scripted", calibration_root=root,
    )


def _caller_for(items_by_child: dict[str, bool], noise: float = 0.05) -> Any:
    rng = np.random.default_rng(0)

    def caller(state: Any, questions: dict[str, dict[str, Any]], model: str) -> JevResult:
        related = items_by_child.get(state["child_question"], True)
        out: dict[str, dict[str, Any]] = {}
        for k in questions:
            base = 0.88 if related else 0.12
            if k.endswith("_neg"):
                base = 1 - base
            out[k] = {"type": "noul", "noul": float(np.clip(base + rng.normal(0, noise), 0.01, 0.99))}
        return JevResult(model_id=model, answers=out, usage={"input_tokens": 40, "output_tokens": 4}, latency_ms=1.0, request_id="t")

    return caller


def test_metrics_math() -> None:
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1.0, 1.0, 0.0, 0.0])
    assert cal.brier(p, y) == pytest.approx(0.025)
    assert cal.auroc(p, y) == 1.0
    m = cal.murphy(p, y, bins=10)
    assert m["reliability"] == pytest.approx(0.025, abs=1e-6)
    assert m["uncertainty"] == pytest.approx(0.25)
    curve = cal.coverage_curve(p, y, [0.5, 0.85])
    assert curve[0]["coverage"] == 1.0 and curve[0]["agreement"] == 1.0
    assert curve[1]["coverage"] == 0.5


def test_fit_key_finds_threshold_and_respects_holdout() -> None:
    rows = []
    for i in range(80):
        y = i % 2 == 0
        p = 0.9 if y else 0.1
        if i % 10 == 0:  # a few wrong-side items
            p = 1 - p
        rows.append({"key": "relevance", "item_id": f"i{i}", "group": f"i{i}", "p": p, "gold": y,
                     "resolved_by": "deterministic_truth", "split": "fit" if i < 60 else "holdout"})
    fr = cal.fit_key(rows, target=0.85, family_size=1)
    assert fr.status == "fitted" and fr.threshold is not None and fr.coverage_fit == 1.0
    fr2 = cal.fit_key(rows, target=0.99, family_size=1)
    assert fr2.status == "unfit" and fr2.threshold is None


def test_end_to_end_synthetic(tmp_path: Path) -> None:
    """The whole path on the packaged R3 card, with assertions that once encoded defects.

    This test used to assert that the author's expectations resolved gold, that a
    15-item holdout was written onto the card, and later that a refit on a holdout the first fit
    had already scored was validated. The path is the same; it now asserts that gold
    needs a resolving source, that a small holdout certifies nothing, and that a refit confirms
    only on cases it has never scored. Deployability itself is shown by the positive controls in
    test_adversarial_counterexamples.py and test_calibration_lifecycle.py.
    """
    root = tmp_path / "calibration-v2"
    all_items = _items(240)
    items = all_items[:60]
    out = cal.build("R3", items, root=root)
    assert out["added"] == 60 and cal.build("R3", items, root=root)["added"] == 0
    loaded = cal.load_items("R3", root)
    assert {i.split for i in loaded} == {"fit", "holdout"}
    assert cal.split_leakage(loaded) == []
    # the author's expectations alone leave every item unresolved
    g = cal.resolve_gold("R3", root)
    assert g["resolved"] == 0 and g["author_only"] == 60
    cards = load_all(CARDS_DIR)
    cal.record_truth(
        "R3",
        [{"item_id": i.id, "key": i.key, "label": bool(i.expected)} for i in loaded],
        card=cards["R3"], source="synthetic-generator", method="the generator decided relatedness", root=root,
    )
    g = cal.resolve_gold("R3", root)
    assert g["resolved"] == 60 and g["dissent"] == 0

    by_child = {i["pack"]["child_question"]: bool(i["expected"]) for i in all_items}
    client = _client(cards, by_child, root)
    n = cal.measure(cards["R3"], client, loaded, root)
    assert n == 60 and cal.measure(cards["R3"], client, loaded, root) == 0
    m = cal.compute_metrics(cards["R3"], root)
    rel = m["keys"]["relevance"]
    assert rel["n"] == sum(1 for i in loaded if i.split == "fit")
    assert rel["auroc"] == 1.0 and rel["brier"] < 0.05
    assert rel["stability"]["negation_gap_mean"] is not None
    assert m["dataset_version"] and m["experiment_id"] and m["split_leakage"] == []

    results = cal.fit_card(cards["R3"], target=0.9, root=root)
    fitted = {r.key: r for r in results}
    # a small holdout cannot confirm the target, so nothing is deployed
    assert fitted["relevance"].status == "fitted" and fitted["relevance"].provisional is True
    assert fitted["relevance"].deployable is False
    first_cases = set(fitted["relevance"].confirming_groups)
    assert first_cases

    more = all_items[60:]
    cal.build("R3", more, root=root)
    grown = cal.load_items("R3", root)
    cal.record_truth(
        "R3",
        [{"item_id": i.id, "key": i.key, "label": bool(i.expected)} for i in grown],
        card=cards["R3"], source="synthetic-generator", method="the generator decided relatedness", root=root,
    )
    cal.resolve_gold("R3", root)
    cal.measure(cards["R3"], client, grown, root)
    second = {r.key: r for r in cal.fit_card(cards["R3"], target=0.9, root=root)}["relevance"]
    assert second.attempt == 2, "a second confirmation under the same conditions spends alpha again"
    assert not first_cases & set(second.confirming_groups), "a scored case confirmed twice"

    rep = cal.report(cards["R3"], load_config(), root)
    text = rep.read_text()
    assert "Latest fit" in text and "relevance" in text and "Internal document" in text
    assert "binning residual" in text and "locked items already scored" in text
    # canary: freeze and re-check with the same caller -> no drift, and no protocol move
    assert cal.canary_freeze(["R3"], 5, root) == 5
    chk = cal.canary_check(client, cards, 0.05, root)
    assert chk["n"] == 5 and chk["compared"] == 5 and chk["model_moved"] == [] and chk["protocol_moved"] == []
    # the scripted caller adds noise with sd 0.05 against a 0.05 band, so small crossings are expected;
    # a real drift would be large
    assert all(d["delta"] < 0.25 for d in chk["drifted"])
    wide = cal.canary_check(client, cards, 0.3, root)
    assert wide["drifted"] == []


def test_label_requests_and_import(tmp_path: Path) -> None:
    root = tmp_path / "calibration-v2"
    cal.build("R3", _items(6), root=root)
    cards = load_all(CARDS_DIR)
    d = tmp_path / "labels"
    d.mkdir()
    init_state(d, "labels")
    ids = cal.label_requests(cards["R3"], cal.load_items("R3", root), d, ["claude-code-subagent"], batch=4)
    assert len(ids) == 2
    # answer the first request as a grader would
    from sextant.handshake import ingest
    from sextant.providers.base import Answer, read_request, write_answer

    req = read_request(d, ids[0])
    labels = [{"id": nid, "key": "relevance", "label": True, "confidence": 0.9, "rationale": "salary bears on the offer"} for nid in req.node_ids]
    write_answer(d, Answer(request_id=ids[0], answer={"labels": labels}, provider="claude-code-subagent", family="anthropic", isolation="isolated"))
    ingest(d, ids[0])
    n = cal.import_labels("R3", d, root, card=cards["R3"])
    assert n == 4
    g = cal.resolve_gold("R3", root)
    assert g["dissent"] >= 1  # the grader said True for the unrelated child too
    dis = (root / "R3" / "disagreements.md").read_text()
    assert "votes" in dis
    rows = [json.loads(line) for line in (root / "R3" / "labels.jsonl").read_text().splitlines()]
    assert rows[0]["family"] == "anthropic" and rows[0]["isolation"] == "isolated"
    assert rows[0]["provenance"] == "handshake"
