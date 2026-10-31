"""`doctor`: separate offline integrity, optional live health, and product acceptance.

Offline rungs can PASS or FAIL. The checkout-only rungs (tests, lint, types) need the source
checkout and its development tools; in an installed copy they are NOT-APPLICABLE, while the
product rungs (selftest, card hashes, linter) always run. Live probes run only with `--live`;
otherwise, or without a key, they are NOT_RUN. A canary that compared nothing is NOT_RUN, not a
pass. An RM1 observation without re-verified certificates for every RM1 question is OBSERVED,
never PASS. Product slices that are not built are NOT-APPLICABLE. Release eligibility is the
checklist in `release_eligibility`; the absence of FAIL is not eligibility.

Doctor writes nothing unless asked (`--record PATH`). A record carries the code identity it was
taken on.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sextant import __version__
from sextant.config import PROJECT_ROOT, load_config
from sextant.jev_transport import key_present
from sextant.rolecards import RoleCard, cards_hash, drifted, load_all
from sextant.selftest import render, run_selftest

OFFLINE = "offline_integrity"
LIVE = "live_health"
PRODUCT = "product_acceptance"

CHECKOUT_RUNGS = ("pytest", "ruff", "mypy")

RELEASE_CHECKLIST = (
    (OFFLINE, "selftest"),
    (OFFLINE, "rolecard-hashes"),
    (OFFLINE, "linter"),
    (OFFLINE, "pytest"),
    (OFFLINE, "ruff"),
    (OFFLINE, "mypy"),
    (LIVE, "canary-drift"),
    (LIVE, "RM1-cards"),
    (PRODUCT, "goldrun-replay"),
    (PRODUCT, "render-checks"),
    (PRODUCT, "verifier-fixture"),
    (PRODUCT, "registry-sweeps"),
)


@dataclass
class Rung:
    name: str
    status: str
    detail: str
    lane: str = OFFLINE


def is_checkout() -> bool:
    """True when running from the source checkout that carries the tests and tool config."""
    return (PROJECT_ROOT / "tests").is_dir() and (PROJECT_ROOT / "pyproject.toml").is_file()


def _run(cmd: list[str], timeout: int = 900) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=str(PROJECT_ROOT), check=False)
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode, out[-1200:]


def fitted_flag_thresholds(card: RoleCard, client: Any = None) -> dict[str, float]:
    """Threshold values of RM1 questions whose certificates re-verify now; nothing else counts.

    A thresholds dict on the card, or a certificate that does not reproduce from its evidence,
    is not a fit. The composition is the one `run_rm1` asks (all four RM1 questions together)
    and the input language is English, the language of the card text it reads; a certificate
    measured under another composition covers none of those runs.
    """
    from sextant.certificates import certificate_state
    from sextant.meta import RM1_KEYS

    if client is None:
        return {}
    fitted: dict[str, float] = {}
    for key in card.questions:
        verdict = certificate_state(
            card, key, model_id=client.model_id, transport=client.transport,
            language=client.language, root=client.calibration_root,
            asked_keys=list(RM1_KEYS), input_language="en",
        )
        if verdict.state == "calibrated" and verdict.certificate is not None and verdict.certificate.value is not None:
            fitted[key] = float(verdict.certificate.value)
    return fitted


def judge_rm1(*, executed: bool, fitted: bool, flagged: bool, rows: int = 1, undecided: bool = False) -> str:
    """RM1 is a PASS only when every question is certified and the rule decided every answer.

    Without re-verified certificates for every question it is an observation; an answer the
    certified rule abstains on (near the threshold, or from a contested run) is not a pass.
    """
    if not executed or rows == 0:
        return "NOT_RUN"
    if not fitted or undecided:
        return "OBSERVED"
    return "FAIL" if flagged else "PASS"


def live_probe_rung(name: str, *, executed: bool, reason: str, ok: bool | None = None) -> Rung:
    if not executed:
        return Rung(name, "NOT_RUN", reason, LIVE)
    return Rung(name, "PASS" if ok else "FAIL", reason, LIVE)


def canary_rung(out: dict[str, Any]) -> Rung:
    """PASS only when every frozen row was re-run and nothing drifted or moved."""
    n, compared = int(out.get("n", 0)), int(out.get("compared", 0))
    moved = len(out.get("model_moved", [])) + len(out.get("protocol_moved", []))
    detail = (
        f"{compared} of {n} items compared; drifted {len(out.get('drifted', []))}; moved {moved}; "
        f"missing cards {out.get('missing_cards', 0)}; band {out.get('band')}"
    )
    if out.get("drifted") or moved:
        return Rung("canary-drift", "FAIL", detail, LIVE)
    if n == 0 or compared < n:
        return Rung("canary-drift", "NOT_RUN", detail + "; not every frozen item was compared", LIVE)
    return Rung("canary-drift", "PASS", detail, LIVE)


def release_eligibility(rungs: list[Rung]) -> dict[str, Any]:
    """Eligible only when every required rung is PASS in its own lane, and named once."""
    names = [r.name for r in rungs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    by_name = {rung.name: rung for rung in rungs}
    checklist: list[dict[str, Any]] = []
    for lane, name in RELEASE_CHECKLIST:
        rung = by_name.get(name)
        status = rung.status if rung is not None else "NOT_RUN"
        met = status == "PASS" and rung is not None and rung.lane == lane and name not in duplicates
        checklist.append({"lane": lane, "name": name, "required": "PASS", "status": status, "met": met})
    return {
        "eligible": not duplicates and all(item["met"] for item in checklist),
        "rule": "checklist",
        "duplicates": duplicates,
        "checklist": checklist,
    }


def _cards(cards_dir: Path) -> dict[str, RoleCard] | None:
    if cards_dir.exists() and any(cards_dir.glob("*.yaml")):
        return load_all(cards_dir)
    return None


def run_doctor(skip_slow: bool = False, *, live: bool = False, checkout_checks: bool | None = None) -> list[Rung]:
    rungs: list[Rung] = []
    checkout = is_checkout() if checkout_checks is None else checkout_checks

    outcomes = run_selftest()
    broken = [o for o in outcomes if o.verdict == "BROKEN"]
    detail = f"{len(outcomes) - len(broken)} checks proved on their plants"
    if broken:
        detail += "; BROKEN: " + ", ".join(o.name for o in broken)
    rungs.append(Rung("selftest", "FAIL" if broken else "PASS", detail, OFFLINE))

    config = load_config()
    cards_dir = Path(config["paths"]["rolecards_dir"])
    cards = _cards(cards_dir)
    if cards is None:
        rungs.append(Rung("rolecard-hashes", "FAIL", f"no role cards under {cards_dir}", OFFLINE))
        rungs.append(Rung("linter", "FAIL", "no role cards to lint", OFFLINE))
    else:
        bad = {cid: drifted(card) for cid, card in cards.items() if drifted(card)}
        hash_detail = f"{len(cards)} cards" + (f"; unsealed or drifted: {bad}" if bad else ", all sealed")
        rungs.append(Rung("rolecard-hashes", "FAIL" if bad else "PASS", hash_detail, OFFLINE))
        from sextant.lint import lint_dir

        results = lint_dir(cards_dir)
        n_find = sum(len(v) for v in results.values())
        rungs.append(Rung("linter", "PASS" if n_find == 0 else "FAIL", f"{len(results)} cards, {n_find} findings", OFFLINE))

    if not checkout:
        for name in CHECKOUT_RUNGS:
            rungs.append(Rung(name, "NOT-APPLICABLE", "not a source checkout; tests and tool config are not installed", OFFLINE))
    else:
        if skip_slow:
            rungs.append(Rung("pytest", "NOT_RUN", "skipped by flag", OFFLINE))
        else:
            code, out = _run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-m", "not live"])
            last = out.splitlines()[-1] if out else ""
            rungs.append(Rung("pytest", "PASS" if code == 0 else "FAIL", last, OFFLINE))
        code, out = _run([sys.executable, "-m", "ruff", "check", "src", "tests", "spike"])
        rungs.append(Rung("ruff", "PASS" if code == 0 else "FAIL", out.splitlines()[-1] if out else "clean", OFFLINE))
        code, out = _run([sys.executable, "-m", "mypy"])
        rungs.append(Rung("mypy", "PASS" if code == 0 else "FAIL", out.splitlines()[-1] if out else "clean", OFFLINE))

    root = Path(config["paths"]["calibration_dir"])
    canary = root / "canary.jsonl"
    if not live:
        rungs.append(live_probe_rung("canary-drift", executed=False, reason="live probes not requested (--live)"))
    elif not canary.exists():
        rungs.append(live_probe_rung("canary-drift", executed=False, reason=f"no canary set frozen at {canary}"))
    elif not key_present():
        rungs.append(live_probe_rung("canary-drift", executed=False, reason="no live Jev key"))
    else:
        from sextant.calibrate import canary_check
        from sextant.jev_client import JevClient

        assert cards is not None
        client = JevClient(config, cards, ledger=None, calibration_root=root)
        rungs.append(canary_rung(canary_check(client, cards, float(config["calibration"]["canary_band"]), root)))

    if cards is None or "RM1" not in cards:
        rungs.append(Rung("RM1-cards", "NOT-APPLICABLE", "no RM1 card is loaded", LIVE))
    elif not live:
        rungs.append(live_probe_rung("RM1-cards", executed=False, reason="live probes not requested (--live)"))
    elif not key_present():
        rungs.append(live_probe_rung("RM1-cards", executed=False, reason="no live Jev key"))
    else:
        from sextant.jev_client import JevClient
        from sextant.meta import RM1_KEYS, run_rm1

        client = JevClient(config, cards, ledger=None, calibration_root=root)
        rows = run_rm1(client, cards)
        fitted = fitted_flag_thresholds(cards["RM1"], client)
        # certified for the composition RM1 is asked in, and calibrated in every run that was made
        complete = all(k in fitted for k in RM1_KEYS) and all(r.states.get(k) == "calibrated" for r in rows for k in RM1_KEYS)
        flagged = [f"{r.card_id}.{r.key}: {', '.join(r.flags(fitted))}" for r in rows if r.flags(fitted)]
        undecided = [f"{r.card_id}.{r.key}: {', '.join(r.undecided(fitted))}" for r in rows if r.undecided(fitted)]
        status = judge_rm1(executed=True, fitted=complete, flagged=bool(flagged), rows=len(rows), undecided=bool(undecided))
        detail = f"{len(rows)} questions, {len(flagged)} flagged, {len(undecided)} undecided"
        if not complete:
            detail += "; not every RM1 question has a re-verified certificate for this composition, so this is not a pass"
        elif undecided:
            detail += "; the certified rule abstains on some answers, so this is not a pass: " + "; ".join(undecided[:4])
        elif flagged:
            detail += ": " + "; ".join(flagged[:4])
        rungs.append(Rung("RM1-cards", status, detail, LIVE))
        if status == "OBSERVED":
            rungs.append(Rung("rm1-note", "INFORMATIONAL", "RM1 scores are measurements, not a release decision", LIVE))

    for name in ("goldrun-replay", "render-checks", "verifier-fixture", "registry-sweeps"):
        rungs.append(Rung(name, "NOT-APPLICABLE", "not built in this slice", PRODUCT))
    return rungs


def exit_code(rungs: list[Rung], *, require_release: bool = False) -> int:
    """Non-zero on any FAIL or any applicable offline rung that did not PASS."""
    if any(r.status == "FAIL" for r in rungs):
        return 1
    if any(r.lane == OFFLINE and r.status not in ("PASS", "NOT-APPLICABLE") for r in rungs):
        return 1
    if require_release and not release_eligibility(rungs)["eligible"]:
        return 1
    return 0


def render_rungs(rungs: list[Rung]) -> str:
    width = max(len(r.name) for r in rungs)
    return "\n".join(f"{r.lane:<20} {r.status:<16} {r.name:<{width}}  {r.detail}" for r in rungs)


def render_release(decision: dict[str, Any]) -> str:
    word = "yes" if decision["eligible"] else "no"
    lines = [f"release eligible: {word} (checklist, not the absence of FAIL)"]
    if decision.get("duplicates"):
        lines.append(f"  duplicate rung names: {', '.join(decision['duplicates'])}")
    for item in decision["checklist"]:
        mark = "met" if item["met"] else "unmet"
        lines.append(f"  {mark:<5} {item['lane']:<20} {item['name']:<18} {item['status']}")
    return "\n".join(lines)


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(PROJECT_ROOT), check=False, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def code_identity() -> dict[str, Any]:
    """What a doctor run was taken on: package, commit when known, cards and protocols."""
    from sextant.certificates import CALIBRATION_PROTOCOL_VERSION
    from sextant.schema import SCHEMA_VERSION
    from sextant.wire import WIRE_PROTOCOL_VERSION

    config = load_config()
    cards = _cards(Path(config["paths"]["rolecards_dir"]))
    return {
        "package_version": __version__,
        "git_commit": _git_commit(),
        "cards_hash": cards_hash(cards) if cards else None,
        "wire_protocol_version": WIRE_PROTOCOL_VERSION,
        "calibration_protocol_version": CALIBRATION_PROTOCOL_VERSION,
        "schema_version": SCHEMA_VERSION,
    }


def record_run(rungs: list[Rung], path: Path) -> Path:
    """Append one JSON line per run: the rungs, the release decision and the code identity."""
    entry = {
        "at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "code": code_identity(),
        "rungs": [vars(r) for r in rungs],
        "release": release_eligibility(rungs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


def selftest_text() -> str:
    return render(run_selftest())


def summary(rungs: list[Rung]) -> dict[str, Any]:
    def count(status: str) -> int:
        return sum(rung.status == status for rung in rungs)

    return {
        "pass": count("PASS"),
        "fail": count("FAIL"),
        "not_run": count("NOT_RUN"),
        "observed": count("OBSERVED"),
        "informational": count("INFORMATIONAL"),
        "not_applicable": count("NOT-APPLICABLE"),
        "skip": count("SKIP"),
    }
