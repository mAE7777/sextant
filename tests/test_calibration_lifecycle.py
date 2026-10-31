"""Contract tests: a measurement, label, metric or threshold must not outrank its evidence.

Each test is a minimal counterexample in one of five areas: group splits, whose vote is truth,
what may be deployed, metric arithmetic, and experiment identity.

Certificates live in the calibration root and re-verify against their evidence, measurements
are recorded in the card's measurement ledger, labels are typed, a case confirms once, and the
per-case validation standard decides deployability. A hand-set `validated` status is not
calibrated, and a holdout that was already scored does not confirm.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from sextant import calibrate as cal
from sextant.certificates import (
    Certificate,
    append_certificate,
    certificate_state,
    effective_certificate,
)
from sextant.config import load_config
from sextant.jev_client import JevClient
from sextant.jev_transport import JevResult
from sextant.ledger import Ledger
from sextant.rolecards import RoleCard, Threshold, compute_hash
from sextant.schema import CalibrationStatus, Estimate, ThresholdRef

pytestmark = pytest.mark.unit

RELATED = "Will the base salary cover my monthly expenses?"
UNRELATED = "What is the mean grain size of basalt?"


# ---------------------------------------------------------------- fixtures


def _card_data(instructions: str = "Judge only what the state says.") -> dict[str, Any]:
    return {
        "id": "R3",
        "purpose": "edge verification",
        "primitive": "noul",
        "state_pack": ["parent_question", "child_question"],
        "instructions": instructions,
        "questions": {
            "relevance": {
                "primitive": "noul",
                "text": "Does answering {child_question} contribute to {parent_question}?",
                "criteria_true": "It contributes directly.",
                "criteria_false": "It does not contribute at all.",
            }
        },
    }


def sealed(data: dict[str, Any]) -> RoleCard:
    draft = RoleCard.model_validate(data)
    data = copy.deepcopy(data)
    data["question_hashes"] = {k: compute_hash(draft, k) for k in draft.questions}
    return RoleCard.model_validate(data)


def edge_card(instructions: str = "Judge only what the state says.") -> RoleCard:
    return sealed(_card_data(instructions))


def scripted(error_period: int = 0) -> Any:
    """Answers high when the child question mentions salary, low otherwise.

    With `error_period` n, every case number divisible by n is answered against its truth.
    """

    def caller(state: Any, questions: dict[str, dict[str, Any]], model: str) -> JevResult:
        child = state["child_question"]
        related = "salary" in child
        case = int(child.rsplit("case ", 1)[1].rstrip(")")) if "case " in child else 1
        if error_period and case % error_period == 0:
            related = not related
        answers = {k: {"type": "noul", "noul": 0.93 if related else 0.07, "confidence": 0.8} for k in questions}
        return JevResult(model_id=model, answers=answers, usage={"input_tokens": 40, "output_tokens": 4}, latency_ms=1.0, request_id="scripted")

    return caller


def client_for(card: RoleCard, caller: Any = None, *, ledger: Ledger | None = None, root: Path | None = None) -> JevClient:
    config = copy.deepcopy(load_config())
    config["jev"]["requests_per_minute"] = 1_000_000  # the scripted caller is local; do not sleep
    return JevClient(
        config, {card.id: card}, ledger=ledger, caller=caller or scripted(), seed=11,
        transport="controlled:scripted", calibration_root=root,
    )


def items(n: int, *, key: str = "relevance", start: int = 0) -> list[dict[str, Any]]:
    out = []
    for i in range(start, start + n):
        related = i % 2 == 0
        child = f"{RELATED} (case {i})" if related else f"{UNRELATED} (case {i})"
        out.append({
            "key": key,
            "pack": {"parent_question": "Should I take the offer?", "child_question": child},
            "expected": related,
            "lang": "en",
            "lineage": f"synthetic-{key}-{i}",
        })
    return out


def truth_rows(card_id: str, root: Path) -> list[dict[str, Any]]:
    """Deterministic truth for the synthetic set: the generator decided it, not an author's guess."""
    return [
        {"item_id": it.id, "key": it.key, "label": "salary" in it.pack["child_question"]}
        for it in cal.load_items(card_id, root)
    ]


def measure_all(card: RoleCard, root: Path, caller: Any = None) -> int:
    ledger = cal.measurement_ledger(card.id, root)
    return cal.measure(card, client_for(card, caller, ledger=ledger, root=root), cal.load_items(card.id, root), root)


def measured_card(tmp_path: Path, n: int = 200, card: RoleCard | None = None, caller: Any = None) -> tuple[RoleCard, Path]:
    """A card with a case-split golden set, resolved gold from deterministic truth, and measurements."""
    card = card or edge_card()
    root = tmp_path / "calibration-v2"
    cal.build(card.id, items(n), root=root)
    cal.record_truth(card.id, truth_rows(card.id, root), card=card, source="generator", method="synthetic", root=root)
    cal.resolve_gold(card.id, root)
    measure_all(card, root, caller)
    return card, root


@pytest.fixture(scope="module")
def certifiable(tmp_path_factory: pytest.TempPathFactory) -> tuple[RoleCard, Path]:
    """Enough error-free cases to meet the per-case standard; copied before each use."""
    return measured_card(tmp_path_factory.mktemp("certifiable"), 1000)


def fresh_copy(source: tuple[RoleCard, Path], tmp_path: Path) -> tuple[RoleCard, Path]:
    card, root = source
    target = tmp_path / "calibration-v2"
    shutil.copytree(root, target)
    return card, target


def gold_row(root: Path, item_id: str) -> dict[str, Any]:
    rows = [json.loads(x) for x in (cal.card_dir("R3", root) / "gold.jsonl").read_text().splitlines() if x.strip()]
    return next(r for r in rows if r["item_id"] == item_id)


def handshake_label(root: Path, item_id: str, key: str, label: Any, provider: str, family: str) -> None:
    """A label as `import_labels` records it after an accepted handshake answer."""
    path = cal.card_dir("R3", root) / "labels.jsonl"
    row = {"item_id": item_id, "key": key, "label": label, "provider": provider, "family": family,
           "isolation": "isolated", "provenance": "handshake", "request_id": f"rq_{provider}"}
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def verdict(card: RoleCard, root: Path, key: str = "relevance") -> Any:
    return certificate_state(card, key, model_id="jev-1.13.0", transport="controlled:scripted", language="en",
                             root=root, asked_keys=[key], input_language="en")


# ================================================================ group splits


def test_translations_of_one_scene_stay_in_one_split(tmp_path: Path) -> None:
    """R3 golden.jsonl:19/:20 are the en fit and zh holdout of `three-ideas-pay`."""
    root = tmp_path / "calibration-v2"
    raw = []
    for i in range(60):
        pack_en = {"parent_question": "Should I take the offer?", "child_question": f"{RELATED} (case {i})"}
        pack_zh = {"parent_question": "我应该接受这个 offer 吗？", "child_question": f"基本工资够覆盖每月开销吗？（{i}）"}
        raw.append({"key": "relevance", "pack": pack_en, "lang": "en", "pair_id": f"p{i:02d}", "expected": True})
        raw.append({"key": "relevance", "pack": pack_zh, "lang": "zh", "pair_id": f"p{i:02d}", "expected": True})
    cal.build("R3", raw, root=root)
    by_pair: dict[str, set[str]] = {}
    for it in cal.load_items("R3", root):
        by_pair.setdefault(it.pair_id or "", set()).add(it.split)
    split_pairs = {k: v for k, v in by_pair.items() if len(v) > 1}
    assert split_pairs == {}, f"a translation pair was cut between splits: {split_pairs}"


def test_several_questions_about_one_scene_stay_in_one_split(tmp_path: Path) -> None:
    """R4 had 20 of 28 unique packs appearing in both splits."""
    root = tmp_path / "calibration-v2"
    raw = []
    for i in range(40):
        pack = {"parent_question": "Should I take the offer?", "child_question": f"{RELATED} (scene {i})"}
        for key in ("relevance", "more_concrete"):
            raw.append({"key": key, "pack": pack, "lang": "en", "expected": True})
    cal.build("R3", raw, root=root)
    by_pack: dict[str, set[str]] = {}
    for it in cal.load_items("R3", root):
        by_pack.setdefault(it.pack["child_question"], set()).add(it.split)
    leaked = {k: v for k, v in by_pack.items() if len(v) > 1}
    assert leaked == {}, f"one scene was cut between splits: {leaked}"


def test_split_leakage_is_detected_and_fitting_refuses_it(tmp_path: Path) -> None:
    """A case that spans train and holdout must fail, not quietly inflate the holdout."""
    card, root = measured_card(tmp_path, 120)
    golden = cal.card_dir(card.id, root) / "golden.jsonl"
    rows = [json.loads(line) for line in golden.read_text().splitlines() if line.strip()]
    assert cal.split_leakage(cal.load_items(card.id, root)) == []
    fit_row = next(r for r in rows if r["split"] == "fit")
    hold_row = next(r for r in rows if r["split"] == "holdout")
    fit_row["group_id"] = hold_row["group_id"] = "leaked-group"
    golden.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    leak = cal.split_leakage(cal.load_items(card.id, root))
    assert len(leak) == 1 and sorted(leak[0]["splits"]) == ["fit", "holdout"]
    assert set(leak[0]["item_ids"]) == {fit_row["id"], hold_row["id"]}
    with pytest.raises(cal.CalibrationSplitError, match="split across"):
        cal.fit_card(card, target=0.9, root=root)


def test_development_calibration_and_release_holdout_stay_distinct(tmp_path: Path) -> None:
    """Development, calibration and a locked release holdout are three different sets."""
    root = tmp_path / "calibration-v2"
    cal.build("R3", items(400), root=root, dev_frac=0.2, release_frac=0.2, holdout_frac=0.2)
    loaded = cal.load_items("R3", root)
    seen = {i.split for i in loaded}
    assert seen == {"development", "fit", "holdout", "release_holdout"}
    assert cal.split_leakage(loaded) == []
    # the release holdout and development are not silently pulled into a fit
    rows = [{"key": i.key, "item_id": i.id, "group": i.id, "split": i.split, "p": 0.9, "gold": True,
             "resolved_by": "deterministic_truth"} for i in loaded]
    fr = cal.fit_key(rows, target=0.8, family_size=1)
    assert fr.n_fit == sum(1 for i in loaded if i.split == "fit")
    assert fr.n_holdout == sum(1 for i in loaded if i.split == "holdout")


def test_a_reused_holdout_is_not_reported_as_unseen(tmp_path: Path) -> None:
    """A holdout that was already looked at is no longer an unseen test. A scored case never confirms again."""
    card, root = measured_card(tmp_path)
    first = {r.key: r for r in cal.fit_card(card, target=0.9, root=root)}["relevance"]
    assert first.n_eligible > 0 and first.attempt == 1
    second = {r.key: r for r in cal.fit_card(card, target=0.9, root=root)}["relevance"]
    assert second.n_eligible == 0 and second.deployable is False
    assert not set(first.confirming_groups) & set(second.confirming_groups)
    log = cal.holdout_evaluations(card.id, root)
    assert log and log[0]["purpose"] == "confirm" and log[0]["item_ids"]


# ================================================================ whose vote is truth


def test_author_expectation_alone_is_not_gold(tmp_path: Path) -> None:
    """R4's 84 and R5's 100 gold rows were all n_votes=1, source author."""
    root = tmp_path / "calibration-v2"
    cal.build("R3", items(20), root=root)
    out = cal.resolve_gold("R3", root)
    assert out["resolved"] == 0
    rows = [json.loads(line) for line in (cal.card_dir("R3", root) / "gold.jsonl").read_text().splitlines() if line.strip()]
    assert rows and all(r["gold"] is None for r in rows)
    assert {r["status"] for r in rows} == {"unresolved-author-only"}
    assert all(r["author_expectation"] is not None for r in rows)
    assert all(r["independent_sources"] == 0 for r in rows)


def test_author_expectation_model_vote_and_truth_are_distinct_fields(tmp_path: Path) -> None:
    """Author expectation, model agreement, human adjudication, and deterministic truth stay separate."""
    card = edge_card()
    root = tmp_path / "calibration-v2"
    cal.build("R3", items(2), root=root)
    it = cal.load_items("R3", root)[0]
    cal.record_labels("R3", [{"item_id": it.id, "key": it.key, "label": True, "provider": "claude-code-subagent", "family": "anthropic", "isolation": "isolated"}], card=card, root=root)
    cal.record_truth("R3", [{"item_id": it.id, "key": it.key, "label": True}], card=card, source="generator", method="synthetic", root=root)
    cal.resolve_gold("R3", root)
    row = gold_row(root, it.id)
    assert row["author_expectation"] is True
    assert [v["family"] for v in row["model_votes"]] == ["anthropic"]
    assert [v["provenance"] for v in row["model_votes"]] == ["declared"]
    assert row["deterministic_truth"] and row["human_adjudication"] == []
    assert row["resolved_by"] == "deterministic_truth"
    assert row["gold"] is True


def test_same_family_relabel_is_not_an_independent_vote(tmp_path: Path) -> None:
    """R3's author and its 280 grader labels all come from one model family."""
    card = edge_card()
    root = tmp_path / "calibration-v2"
    cal.build("R3", items(2), root=root)
    it = cal.load_items("R3", root)[0]
    handshake_label(root, it.id, it.key, True, "claude-code-subagent", "anthropic")
    handshake_label(root, it.id, it.key, True, "claude-code-headless", "anthropic")
    cal.resolve_gold("R3", root)
    row = gold_row(root, it.id)
    assert row["independent_sources"] == 1, "two members of one family are not two samples"
    assert row["gold"] is None and row["status"] == "unresolved-single-source"

    handshake_label(root, it.id, it.key, True, "codex-headless", "openai")
    cal.resolve_gold("R3", root)
    row = gold_row(root, it.id)
    assert row["independent_sources"] == 2
    assert row["gold"] is True and row["resolved_by"] == "independent_model_majority"

    # a declared label names a family nobody checked; it is shown but never counted
    other = cal.load_items("R3", root)[1]
    cal.record_labels("R3", [
        {"item_id": other.id, "key": other.key, "label": True, "provider": "x", "family": "anthropic"},
        {"item_id": other.id, "key": other.key, "label": True, "provider": "y", "family": "openai"},
    ], card=card, root=root)
    cal.resolve_gold("R3", root)
    assert gold_row(root, other.id)["status"] == "unresolved-declared-only"


def test_human_adjudication_resolves_where_models_disagree(tmp_path: Path) -> None:
    card = edge_card()
    root = tmp_path / "calibration-v2"
    cal.build("R3", items(2), root=root)
    it = cal.load_items("R3", root)[0]
    handshake_label(root, it.id, it.key, True, "a", "anthropic")
    handshake_label(root, it.id, it.key, False, "b", "openai")
    cal.resolve_gold("R3", root)
    row = gold_row(root, it.id)
    assert row["gold"] is None and row["status"] == "unresolved-disagreement" and row["dissent"] is True

    cal.record_truth("R3", [{"item_id": it.id, "key": it.key, "label": False}], card=card, kind="human_adjudication", source="eric", method="spot-check", root=root)
    cal.resolve_gold("R3", root)
    row = gold_row(root, it.id)
    assert row["gold"] is False and row["resolved_by"] == "human_adjudication"


def test_import_labels_refuses_an_answer_that_no_longer_matches_its_accepted_sha(tmp_path: Path) -> None:
    """import_labels re-read an overwritable answer file without re-checking answer_sha."""
    from sextant.handshake import ingest
    from sextant.providers.base import Answer, read_request, write_answer
    from sextant.state import init_state

    root = tmp_path / "calibration-v2"
    cal.build("R3", items(4), root=root)
    card = edge_card()
    d = tmp_path / "labels"
    d.mkdir()
    init_state(d, "labels")
    ids = cal.label_requests(card, cal.load_items("R3", root), d, ["grader-a"], batch=8)
    req = read_request(d, ids[0])
    labels = [{"id": nid, "key": "relevance", "label": True, "confidence": 0.9, "rationale": "ok"} for nid in req.node_ids]
    write_answer(d, Answer(request_id=ids[0], answer={"labels": labels}, provider="grader-a", family="anthropic", isolation="isolated"))
    ingest(d, ids[0])
    assert cal.import_labels("R3", d, root, card=card) == 4

    tampered = [dict(x, label=False) for x in labels]
    write_answer(d, Answer(request_id=ids[0], answer={"labels": tampered}, provider="grader-a", family="anthropic", isolation="isolated"))
    with pytest.raises(cal.CalibrationEvidenceError, match=ids[0]):
        cal.import_labels("R3", d, root, card=card)

    # the answer's declared grader identity may not drift from what the ledger accepted either
    write_answer(d, Answer(request_id=ids[0], answer={"labels": labels}, provider="grader-a", family="openai", isolation="isolated"))
    with pytest.raises(cal.CalibrationEvidenceError, match="identity"):
        cal.import_labels("R3", d, root, card=card)


# ================================================================ what may be deployed


def test_a_provisional_fit_is_not_written_onto_the_card(tmp_path: Path) -> None:
    """10 fit items and 0 holdout returned fitted+provisional and the runtime said calibrated."""
    rows = [{"key": "relevance", "item_id": f"f{i}", "group": f"f{i}", "split": "fit", "p": 0.9 if i % 2 == 0 else 0.1,
             "gold": i % 2 == 0, "resolved_by": "deterministic_truth"} for i in range(10)]
    fr = cal.fit_key(rows, target=0.9, family_size=1)
    assert fr.status == "fitted" and fr.provisional is True
    assert fr.deployable is False and fr.n_holdout == 0

    card, root = measured_card(tmp_path, 60)
    small = {r.key: r for r in cal.fit_card(card, target=0.9, root=root)}["relevance"]
    assert 0 < small.n_eligible < 94
    assert small.status == "fitted" and small.provisional is True and small.deployable is False
    entry = effective_certificate(root, card.id, "relevance")
    assert entry is not None and entry.status == "provisional" and entry.value is None
    assert verdict(card, root).state == "provisional"


def test_a_confirmed_fit_is_written_as_a_validated_certificate(certifiable: tuple[RoleCard, Path], tmp_path: Path) -> None:
    card, root = fresh_copy(certifiable, tmp_path)
    fr = {r.key: r for r in cal.fit_card(card, target=0.9, root=root)}["relevance"]
    assert fr.status == "fitted" and fr.provisional is False and fr.deployable is True, fr.reason
    cert = effective_certificate(root, card.id, "relevance")
    assert cert is not None and cert.status == "validated" and cert.value == fr.threshold
    assert cert.counts["automatic"] >= 94 and cert.counts["releases"] >= 94
    assert cert.dataset_version and cert.experiment_id and cert.calibration_protocol_version
    assert cert.bounds["r_upper"] is not None and cert.bounds["r_upper"] <= 0.05
    assert cert.confirming and cert.digests and cert.family == ["relevance"]
    assert verdict(card, root).state == "calibrated"


def test_a_failed_refit_clears_the_previous_threshold(certifiable: tuple[RoleCard, Path], tmp_path: Path) -> None:
    """A fit first gives .5. After the data is replaced the question no longer fits, yet the file kept .5."""
    card, root = fresh_copy(certifiable, tmp_path)
    cal.fit_card(card, target=0.9, root=root)
    assert verdict(card, root).state == "calibrated"

    # the same key, refit against a target no threshold reaches
    results = cal.fit_card(card, target=1.01, root=root)
    assert {r.key: r for r in results}["relevance"].status == "unfit"
    entry = effective_certificate(root, card.id, "relevance")
    assert entry is not None and entry.status == "unfit" and entry.value is None, "a failed refit left the old bar in place"
    assert verdict(card, root).state == "invalidated"


def test_one_key_failing_is_not_masked_by_another_key_succeeding(tmp_path: Path) -> None:
    """One key's result must be recorded on its own, not inherited from a sibling key."""
    data = _card_data()
    data["questions"]["more_concrete"] = {"primitive": "noul", "text": "Is {child_question} more concrete than {parent_question}?"}
    card = sealed(data)
    root = tmp_path / "calibration-v2"
    cal.build(card.id, items(1000) + items(200, key="more_concrete", start=5000), root=root)
    cal.record_truth(card.id, truth_rows(card.id, root), card=card, source="generator", method="synthetic", root=root)
    cal.resolve_gold(card.id, root)

    def mixed(state: Any, questions: dict[str, dict[str, Any]], model: str) -> JevResult:
        related = "salary" in state["child_question"]
        answers = {}
        for k in questions:
            wrong = k == "more_concrete" and int(state["child_question"].rsplit("case ", 1)[1].rstrip(")")) % 5 == 0
            answers[k] = {"type": "noul", "noul": 0.93 if related != wrong else 0.07}
        return JevResult(model_id=model, answers=answers, usage={"input_tokens": 40}, latency_ms=1.0, request_id="s")

    measure_all(card, root, mixed)
    results = {r.key: r for r in cal.fit_card(card, target=0.75, root=root)}
    assert results["relevance"].deployable is True, results["relevance"].reason
    assert results["more_concrete"].deployable is False
    assert effective_certificate(root, card.id, "relevance").status == "validated"
    assert effective_certificate(root, card.id, "more_concrete").status in ("provisional", "unfit")
    assert certificate_state(card, "more_concrete", model_id="jev-1.13.0", transport="controlled:scripted",
                             language="en", root=root, asked_keys=["more_concrete"], input_language="en").state != "calibrated"


def test_a_calibrated_estimate_requires_a_validated_threshold(tmp_path: Path) -> None:
    """JevClient set calibration_status=calibrated for any threshold_ref at all."""
    base = {
        "primitive": "noul",
        "p": 0.9,
        "card_id": "R3",
        "question_key": "relevance",
        "question_hash": "qh_x",
        "rendered_hash": "rq_x",
        "state_sha": "sha_x",
        "protocol_version": "sextant-jev-wire/2",
        "run_id": "run_x",
        "language": "en",
        "aggregation": "single",
        "contested": False,
        "contributions": [{"call_id": "jc_1", "variant": "base", "role": "primary", "rendered_hash": "rq_x", "value": 0.9}],
        "model_id": "jev-1.13.0",
        "transport": "typesafe",
        "source_tier": "jev-judged",
        "at": "2026-09-18T00:00:00Z",
    }
    provisional = ThresholdRef(card_id="R3", key="relevance", value=0.7, status="provisional")
    with pytest.raises(ValidationError):
        Estimate.model_validate({**base, "threshold_ref": provisional.model_dump(), "calibration_status": {"kind": "agreement", "state": "calibrated"}})
    ok = Estimate.model_validate({**base, "threshold_ref": provisional.model_dump(), "calibration_status": {"kind": "agreement", "state": "provisional"}})
    assert ok.calibration_status.state == "provisional"
    with pytest.raises(ValidationError):
        Estimate.model_validate({**base, "calibration_status": CalibrationStatus(kind="agreement", state="calibrated").model_dump()})


def test_the_runtime_reports_provisional_and_refuses_the_threshold_decision(tmp_path: Path) -> None:
    card = edge_card()
    root = tmp_path / "calibration-v2"
    append_certificate(root, card.id, Certificate(key="relevance", status="provisional", reason="holdout too small"))
    res = client_for(card, root=root).ask("R3", {"parent_question": "Should I take the offer?", "child_question": RELATED}, keys=["relevance"])
    est = res.estimates["relevance"].estimate
    assert est.calibration_status.state == "provisional"
    assert res.passes("relevance") is None, "a provisional threshold must not decide"

    # A hand-written `validated` entry used to make the runtime say calibrated.
    append_certificate(root, card.id, Certificate(key="relevance", status="validated", value=0.7))
    res2 = client_for(card, root=root).ask("R3", {"parent_question": "Should I take the offer?", "child_question": RELATED}, keys=["relevance"])
    assert res2.estimates["relevance"].estimate.calibration_status.state == "invalidated"
    assert res2.passes("relevance") is None


def test_a_legacy_threshold_without_a_certificate_is_not_calibrated(tmp_path: Path) -> None:
    """Do not preserve compatibility by silently accepting unverifiable legacy evidence."""
    data = _card_data()
    data["thresholds"] = {"relevance": {"value": 0.5, "n": 8, "date": "2026-09-18", "primitive": "noul", "status": "validated"}}
    card = sealed(data)
    res = client_for(card, root=tmp_path / "calibration-v2").ask("R3", {"parent_question": "Should I take the offer?", "child_question": RELATED})
    assert res.estimates["relevance"].estimate.calibration_status.state == "pending"
    assert res.passes("relevance") is None


def test_the_wilson_bound_is_recorded_as_not_rejected_not_as_shown(tmp_path: Path) -> None:
    """Failing to reject a hypothesis is not confirmed performance. Eight clean cases cannot show the target."""
    rows = [{"key": "relevance", "item_id": f"f{i}", "group": f"f{i}", "split": "fit", "p": 0.9 if i % 2 == 0 else 0.1,
             "gold": i % 2 == 0, "resolved_by": "deterministic_truth"} for i in range(40)]
    rows += [{"key": "relevance", "item_id": f"h{i}", "group": f"h{i}", "split": "holdout", "p": 0.9 if i % 2 == 0 else 0.1,
              "gold": i % 2 == 0, "resolved_by": "deterministic_truth"} for i in range(8)]
    fr = cal.fit_key(rows, target=0.9, family_size=1)
    assert fr.status == "fitted" and fr.n_wrong == 0 and fr.n_automatic == 8
    assert fr.r_upper is not None and fr.r_upper > 0.05
    assert fr.deployable is False
    assert "exceeds" in fr.reason


# ================================================================ metric arithmetic


def _auroc_oracle(p: list[float], y: list[float]) -> float | None:
    """Independent arithmetic definition: P(pos > neg) + 0.5 P(pos == neg), counted pairwise."""
    pos = [a for a, b in zip(p, y, strict=True) if b == 1]
    neg = [a for a, b in zip(p, y, strict=True) if b == 0]
    if not pos or not neg:
        return None
    total = 0.0
    for a in pos:
        for b in neg:
            total += 1.0 if a > b else (0.5 if a == b else 0.0)
    return total / (len(pos) * len(neg))


def test_auroc_handles_a_tie() -> None:
    """p=[.5,.5], y=[1,0] returned 0.0; the tie contributes 0.5."""
    assert cal.auroc(np.array([0.5, 0.5]), np.array([1.0, 0.0])) == pytest.approx(0.5)


def test_auroc_matches_an_independent_arithmetic_oracle() -> None:
    cases: list[tuple[list[float], list[float]]] = [
        ([0.5, 0.5], [1.0, 0.0]),
        ([0.3, 0.3, 0.3, 0.3], [1.0, 1.0, 0.0, 0.0]),  # all ties
        ([0.1, 0.2, 0.8, 0.9], [1.0, 1.0, 0.0, 0.0]),  # reverse ranking
        ([0.9, 0.8, 0.2, 0.1], [1.0, 1.0, 0.0, 0.0]),
        ([0.6, 0.6, 0.6, 0.4, 0.4, 0.9], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
        ([0.0, 1.0, 0.5, 0.5, 0.5], [0.0, 1.0, 1.0, 0.0, 1.0]),
    ]
    for p, y in cases:
        got = cal.auroc(np.array(p), np.array(y))
        want = _auroc_oracle(p, y)
        assert got == pytest.approx(want), f"p={p} y={y}: {got} != {want}"


def test_auroc_of_a_reversed_ranking_is_zero() -> None:
    assert cal.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0.0)


def test_all_ties_score_one_half() -> None:
    assert cal.auroc(np.array([0.3, 0.3, 0.3, 0.3]), np.array([1.0, 1.0, 0.0, 0.0])) == pytest.approx(0.5)


def test_auroc_refuses_non_finite_scores_and_non_binary_labels() -> None:
    assert cal.auroc(np.array([np.nan, 0.2]), np.array([1.0, 0.0])) is None
    assert cal.auroc(np.array([0.8, 0.2]), np.array([1.0, 0.5])) is None
    assert cal.murphy(np.array([]), np.array([]))["defined"] is False


def test_murphy_is_reported_as_binned_with_its_residual() -> None:
    """p=.81/.89, y=0/1 -> raw Brier .3341, the three binned terms combine to .3725."""
    p = np.array([0.81, 0.89])
    y = np.array([0.0, 1.0])
    m = cal.murphy(p, y, bins=10)
    assert m["binned"] is True and m["bins"] == 10
    combined = m["reliability"] - m["resolution"] + m["uncertainty"]
    assert m["brier_from_bins"] == pytest.approx(combined)
    assert m["brier"] == pytest.approx(cal.brier(p, y))
    assert m["brier"] == pytest.approx(0.3341)
    assert m["brier_from_bins"] == pytest.approx(0.3725)
    assert m["approximation_residual"] == pytest.approx(m["brier"] - combined)
    assert m["approximation_residual"] != 0.0


def test_metrics_are_versioned_and_supersede_the_old_claim(tmp_path: Path) -> None:
    """Recomputing a metric keeps the old version and never overwrites a historical claim."""
    card, root = measured_card(tmp_path, 60)
    first = cal.compute_metrics(card, root)
    assert first["metric_protocol_version"] == cal.METRIC_PROTOCOL_VERSION
    assert first["dataset_version"] and first["experiment_id"]
    assert first["supersedes"] is None

    cal.build(card.id, items(80, start=60), root=root)
    cal.record_truth(card.id, truth_rows(card.id, root), card=card, source="generator", method="synthetic", root=root)
    cal.resolve_gold(card.id, root)
    measure_all(card, root)
    second = cal.compute_metrics(card, root)
    assert second["dataset_version"] != first["dataset_version"]
    assert second["supersedes"] is not None
    archived = Path(second["supersedes"]["path"])
    assert archived.exists()
    assert json.loads(archived.read_text())["dataset_version"] == first["dataset_version"]


# ================================================================ experiment identity


def test_a_response_row_carries_its_measurement_conditions(tmp_path: Path) -> None:
    card, root = measured_card(tmp_path, 20)
    rows = [json.loads(x) for x in (cal.card_dir(card.id, root) / "responses.jsonl").read_text().splitlines() if x.strip()]
    assert rows
    assert all(r["measurement_conditions"].startswith("mc_") for r in rows)
    assert all(r["protocol_fingerprint"].startswith("mf_") for r in rows)
    assert all(r["run_id"] and r["call_ids"] for r in rows)
    assert len({r["measurement_conditions"] for r in rows}) == 1, "one card, one model, one transport"


def test_a_reworded_card_invalidates_the_old_responses_loudly(certifiable: tuple[RoleCard, Path], tmp_path: Path) -> None:
    """Seal clears the threshold, but the next fit could restore one from old responses."""
    card, root = fresh_copy(certifiable, tmp_path)
    cal.fit_card(card, target=0.9, root=root)
    assert verdict(card, root).state == "calibrated"

    reworded = sealed(_card_data("Judge only what the state says. An added clause changes the wording."))
    with pytest.raises(cal.CalibrationIdentityError, match="measurement conditions"):
        cal.fit_card(reworded, target=0.9, root=root)
    assert effective_certificate(root, card.id, "relevance").status == "revoked"
    assert verdict(card, root).state == "invalidated"


def test_a_changed_model_invalidates_the_old_responses_loudly(tmp_path: Path) -> None:
    """Half the items re-measured under another pinned model: two experiments, never pooled.

    This used to edit `model_id` in responses.jsonl. The identity now comes from each ledger run,
    so the mix is made with real runs; an edited response alone is refused as tampered evidence.
    """
    card, root = measured_card(tmp_path, 40)
    config = copy.deepcopy(load_config())
    config["jev"]["requests_per_minute"] = 1_000_000
    config["jev"]["model_id"] = "jev-2.0.0"
    other = JevClient(config, {card.id: card}, ledger=cal.measurement_ledger(card.id, root), caller=scripted(),
                      seed=11, transport="controlled:scripted", calibration_root=root)
    assert cal.measure(card, other, cal.load_items(card.id, root)[:20], root) == 20
    with pytest.raises(cal.CalibrationIdentityError, match="mc_"):
        cal.fit_card(card, target=0.9, root=root)


def test_a_response_with_no_recorded_conditions_is_never_fitted(tmp_path: Path) -> None:
    """A response row whose recorded conditions disagree with its run is refused, not fitted."""
    card, root = measured_card(tmp_path, 40)
    d = cal.card_dir(card.id, root)
    rows = [json.loads(x) for x in (d / "responses.jsonl").read_text().splitlines() if x.strip()]
    for r in rows:
        r.pop("measurement_conditions", None)
    (d / "responses.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    with pytest.raises(cal.CalibrationEvidenceError, match="measurement conditions"):
        cal.fit_card(card, target=0.9, root=root)


def test_the_fit_records_a_traceable_chain_back_to_the_calls(tmp_path: Path) -> None:
    """Raw calls, derived responses and fits must form a traceable chain."""
    card, root = measured_card(tmp_path, 200)
    cal.fit_card(card, target=0.9, root=root)
    fit = json.loads((cal.card_dir(card.id, root) / "fit.json").read_text())
    assert fit["calibration_protocol_version"] == cal.CALIBRATION_PROTOCOL_VERSION
    assert fit["wire_protocol_version"]
    assert fit["experiment_id"] and fit["dataset_version"] and fit["responses_version"]
    entry = next(r for r in fit["results"] if r["key"] == "relevance")
    assert entry["evidence"], "the fit must name the responses it used"
    index = cal.measurement_ledger(card.id, root).call_index()
    for e in entry["evidence"]:
        assert e["call_ids"] and all(c in index for c in e["call_ids"])
        assert all(index[c]["run_id"] == e["run_id"] for c in e["call_ids"])


def test_an_old_fit_is_superseded_not_overwritten(tmp_path: Path) -> None:
    card, root = measured_card(tmp_path, 200)
    cal.fit_card(card, target=0.9, root=root)
    first = json.loads((cal.card_dir(card.id, root) / "fit.json").read_text())
    cal.fit_card(card, target=0.95, root=root)
    second = json.loads((cal.card_dir(card.id, root) / "fit.json").read_text())
    assert second["supersedes"] is not None
    archived = Path(second["supersedes"]["path"])
    assert archived.exists()
    assert json.loads(archived.read_text())["target"] == first["target"]


def test_the_canary_binds_the_full_protocol_identity(tmp_path: Path) -> None:
    """A canary used to store only model_id and p, without the full protocol identity."""
    card, root = measured_card(tmp_path, 20)
    assert cal.canary_freeze([card.id], 5, root) == 5
    frozen = [json.loads(x) for x in ((root / "canary.jsonl").read_text().splitlines()) if x.strip()]
    assert all(r["measurement_conditions"] and r["protocol_version"] and r["question_hash"] for r in frozen)

    same = cal.canary_check(client_for(card, root=root), {card.id: card}, 0.05, root)
    assert same["protocol_moved"] == [] and same["model_moved"] == [] and same["drifted"] == []
    assert same["compared"] == same["n"] == 5

    reworded = sealed(_card_data("Judge only what the state says. And one more clause."))
    moved = cal.canary_check(client_for(reworded, root=root), {reworded.id: reworded}, 0.05, root)
    assert moved["protocol_moved"], "a re-worded card must not be compared against the old frozen answer"
    assert moved["compared"] == 0

    other_model = copy.deepcopy(load_config())
    other_model["jev"]["model_id"] = "jev-9.9.9"
    later = JevClient(other_model, {card.id: card}, caller=scripted(), transport="controlled:scripted", calibration_root=root)
    assert len(cal.canary_check(later, {card.id: card}, 0.05, root)["model_moved"]) == 5


def test_unverifiable_legacy_thresholds_and_measurements_are_not_accepted(tmp_path: Path) -> None:
    """A legacy card threshold plus legacy responses must not combine into a deployable claim."""
    thr = Threshold(value=0.5, n=8, date="2026-01-01", primitive="noul")
    assert thr.status == "unverified"
    assert thr.dataset_version is None and thr.experiment_id is None
    ref = ThresholdRef(card_id="R3", key="relevance", value=0.5)
    assert ref.status == "unverified"
    legacy_root = tmp_path / "old"
    d = legacy_root / "R3"
    d.mkdir(parents=True)
    (d / "responses.jsonl").write_text(json.dumps({"item_id": "i", "key": "relevance", "p": 0.9}) + "\n")
    with pytest.raises(cal.CalibrationProtocolError, match="earlier protocol"):
        cal.build("R3", items(2), root=legacy_root)
