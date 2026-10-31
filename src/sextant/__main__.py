"""Dispatch only: `python -m sextant <command>` routes to a module function. No logic lives here."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from sextant.config import load_config, load_raw_config, write_config
from sextant.schema import export_json_schema

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _decision_dir(config: dict[str, Any], slug: str) -> Path:
    if not SLUG_RE.match(slug):
        raise SystemExit(f"invalid slug {slug!r}: lowercase letters, digits and hyphens only")
    return Path(config["paths"]["decisions_dir"]) / slug


def cmd_setup(args: argparse.Namespace) -> int:
    from sextant.setup_cmd import run_setup

    text, result = run_setup(probe_only=args.probe_only, with_smoke=not args.no_smoke)
    print(text)
    return 0 if result.get("jev", {}).get("available") is not False else 1


def cmd_schema(args: argparse.Namespace) -> int:
    schema = export_json_schema()
    if args.export:
        out = Path(args.export)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(schema, indent=2))
        print(f"schema written: {out}")
    else:
        print(json.dumps(schema, indent=2))
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    from sextant.selftest import CHECKS, render, run_selftest, write_fixtures

    known = {c.name for c in CHECKS}
    unknown = [d for d in (args.disable or []) if d not in known]
    if unknown:
        print(f"unknown check name(s): {', '.join(unknown)}; known: {', '.join(sorted(known))}")
        return 2
    if args.write_fixtures:
        paths = write_fixtures(Path(args.write_fixtures))
        print(f"{len(paths)} fixture files written under {args.write_fixtures}")
    outcomes = run_selftest(set(args.disable or []))
    print(render(outcomes))
    return 0 if all(o.verdict == "PASS" for o in outcomes) else 1


def cmd_selfcheck(args: argparse.Namespace) -> int:
    """Run every structural check against the latest saved graph of a decision, with its ledger."""
    from sextant.graph import check_all, graph_path, latest_version, load_graph
    from sextant.ledger import Ledger

    config = load_config()
    d = _decision_dir(config, args.slug)
    v = args.version or latest_version(d)
    if v == 0:
        print(f"SKIP selfcheck: no graph saved under {d}")
        return 1
    graph = load_graph(graph_path(d, v))
    result = check_all(graph, Ledger(d).call_index(), _cards(config), Path(config["paths"]["calibration_dir"]))
    by_check: dict[str, list[str]] = {}
    for f in result.findings:
        by_check.setdefault(f.check, []).append(f"{f.severity} {f.subject}: {f.message}")
    from sextant.graph import STRUCTURAL_CHECKS

    for name in [*STRUCTURAL_CHECKS.keys(), "ledger"]:
        items = by_check.get(name, [])
        fails = [i for i in items if i.startswith("FAIL")]
        status = "FAIL" if fails else ("WARN" if items else "PASS")
        print(f"{status:<5} {name:<12} " + ("; ".join(items[:3]) if items else "clean"))
    print(f"\nselfcheck graph.v{v}: {'PASS' if result.ok else 'FAIL'}")
    return 0 if result.ok else 1


def cmd_doctor(args: argparse.Namespace) -> int:
    from sextant.doctor import (
        exit_code,
        record_run,
        release_eligibility,
        render_release,
        render_rungs,
        run_doctor,
    )

    rungs = run_doctor(skip_slow=args.skip_slow, live=args.live)
    decision = release_eligibility(rungs)
    print(render_rungs(rungs))
    print()
    print(render_release(decision))
    if args.record:
        written = record_run(rungs, Path(args.record))
        print(f"\nrecorded to {written}")
    print("offline PASS is not live Jev or generative-model verification, and not release eligibility")
    return exit_code(rungs, require_release=args.require_release)


def cmd_config(args: argparse.Namespace) -> int:
    config = load_config()
    if args.write_defaults:
        written = write_config(load_raw_config())
        print(f"config written: {written}")
    else:
        print(json.dumps(config, indent=2))
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    from sextant.config import require_config_file
    from sextant.state import init_state

    require_config_file()
    config = load_config()
    d = _decision_dir(config, args.slug)
    d.mkdir(parents=True, exist_ok=True)
    st = init_state(d, args.slug)
    if args.question:
        (d / "question.md").write_text(args.question.strip() + "\n")
    print(f"initialized {d} (created {st.created})")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from sextant.handshake import pending
    from sextant.state import load_state, status_text

    config = load_config()
    d = _decision_dir(config, args.slug)
    st = load_state(d)
    print(status_text(st))
    pend = pending(d)
    if pend:
        print("requests awaiting answers: " + ", ".join(pend))
    return 0


def cmd_stamp(args: argparse.Namespace) -> int:
    from sextant.ledger import Ledger
    from sextant.state import load_state, save_state, stamp

    config = load_config()
    d = _decision_dir(config, args.slug)
    st = load_state(d)
    g = stamp(st, args.gate, args.status, args.token, args.note)
    save_state(st, d)
    Ledger(d).event("gate", gate=args.gate, status=g.status.value, token=g.token, note=args.note)
    print(f"{args.gate} {g.status.value} token={g.token}")
    return 0


def cmd_assert(args: argparse.Namespace) -> int:
    from sextant.graph import graph_path, latest_version, load_graph
    from sextant.ledger import Ledger
    from sextant.state import GateError, assert_gate, load_state

    config = load_config()
    decision = _decision_dir(config, args.slug)
    st = load_state(decision)
    version = latest_version(decision)
    graph = load_graph(graph_path(decision, version)) if version else None
    try:
        assert_gate(
            st, args.gate, graph=graph, call_index=Ledger(decision).call_index(),
            cards=_cards(config), calibration_root=Path(config["paths"]["calibration_dir"]),
        )
    except GateError as e:
        print(f"ASSERT FAIL: {e}")
        return 1
    print(f"ASSERT PASS: {args.gate}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Run a gate's checker chain on the latest graph and store the receipt it produced."""
    from sextant.config import require_config_file
    from sextant.graph import graph_path, latest_version, load_graph
    from sextant.ledger import Ledger
    from sextant.state import evaluate_gate, load_state, save_state

    require_config_file()
    config = load_config()
    decision = _decision_dir(config, args.slug)
    st = load_state(decision)
    version = latest_version(decision)
    if not version:
        print(f"NOT FOUND: no graph saved under {decision}")
        return 1
    graph = load_graph(graph_path(decision, version))
    issued = evaluate_gate(
        st, args.gate, graph, Ledger(decision).call_index(),
        cards=_cards(config), calibration_root=Path(config["paths"]["calibration_dir"]),
    )
    save_state(st, decision)
    print(f"{args.gate} {issued.status.value} token={issued.token}" + (f"\n{issued.note}" if issued.note else ""))
    return 0 if issued.status.value == "PASS" else 1


def _load_schema_arg(s: str) -> dict[str, Any]:
    if s.startswith("@"):
        return json.loads(Path(s[1:]).read_text())  # type: ignore[no-any-return]
    return json.loads(s)  # type: ignore[no-any-return]


def cmd_request(args: argparse.Namespace) -> int:
    from sextant.handshake import create_request, fulfil

    config = load_config()
    d = _decision_dir(config, args.slug)
    provider = args.provider or config["providers"].get(args.role, "claude-code")
    req = create_request(
        d, role=args.role, purpose=args.purpose, prompt=args.prompt, answer_schema=_load_schema_arg(args.schema), provider=provider,
    )
    print(f"request {req.id} written for provider {provider}")
    if args.fulfil:
        ans = fulfil(d, req.id, config)
        if ans is not None:
            print(f"answer accepted from {ans.provider} ({ans.family}, {ans.isolation})")
    return 0


def cmd_fulfil(args: argparse.Namespace) -> int:
    from sextant.handshake import fulfil

    config = load_config()
    d = _decision_dir(config, args.slug)
    ans = fulfil(d, args.request_id, config, provider_override=args.provider)
    if ans is None:
        return 0
    print(f"answer accepted from {ans.provider} ({ans.family}, {ans.isolation})")
    print(json.dumps(ans.answer, ensure_ascii=False, indent=2)[:2000])
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    from sextant.handshake import HandshakeError, ingest

    config = load_config()
    d = _decision_dir(config, args.slug)
    try:
        ans = ingest(d, args.request_id)
    except HandshakeError as e:
        if type(e) is HandshakeError:  # a plain rejection (re-issued); subclasses go to main()
            print(f"REJECTED: {e}")
            return 1
        raise
    print(f"answer {args.request_id} accepted from {ans.provider} ({ans.family}, {ans.isolation})")
    return 0


def cmd_pending(args: argparse.Namespace) -> int:
    from sextant.handshake import pending

    config = load_config()
    pend = pending(_decision_dir(config, args.slug))
    print("\n".join(pend) if pend else "no pending requests")
    return 0


def cmd_rolecards(args: argparse.Namespace) -> int:
    from sextant.rolecards import drifted, load_all, load_card, seal

    config = load_config()
    target = Path(args.path) if args.path else Path(config["paths"]["rolecards_dir"])
    if not target.exists():
        print(f"no role cards at {target}; nothing to {args.action}")
        return 1
    if target.is_dir() and not any(p for p in target.glob("*.yaml") if not p.name.startswith("_")):
        print(f"no role-card files under {target}; nothing to {args.action}")
        return 1
    if args.action == "seal":
        paths = [target] if target.is_file() else sorted(p for p in target.glob("*.yaml") if not p.name.startswith("_"))
        for p in paths:
            changes = seal(p)
            print(f"{p.name}: " + (", ".join(f"{k}: {v}" for k, v in changes.items()) if changes else "already sealed"))
        return 0
    cards = {load_card(target).id: load_card(target)} if target.is_file() else load_all(target)
    bad = 0
    for cid, card in cards.items():
        d = drifted(card)
        bad += len(d)
        print(f"{cid}: {'sealed' if not d else d}")
    return 0 if bad == 0 else 1


def _cards_dir(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["rolecards_dir"])


def _cards(config: dict[str, Any]) -> dict[str, Any]:
    from sextant.rolecards import load_all

    return load_all(_cards_dir(config))


def _client(config: dict[str, Any], decision_dir: Path | None) -> Any:
    from sextant.jev_client import JevClient
    from sextant.ledger import Ledger
    from sextant.rolecards import load_all

    cards = load_all(_cards_dir(config))
    return JevClient(config, cards, ledger=Ledger(decision_dir) if decision_dir else None, transport=str(config["jev"]["transport"]))


def cmd_lint(args: argparse.Namespace) -> int:
    from sextant.lint import lint_dir, render

    config = load_config()
    results = lint_dir(Path(args.path) if args.path else _cards_dir(config))
    print(render(results))
    return 0 if not any(results.values()) else 1


def cmd_jev_ask(args: argparse.Namespace) -> int:
    """Debug helper: ask one card on a pack from a JSON file; records to the decision's ledger when given."""
    config = load_config()
    d = _decision_dir(config, args.slug) if args.slug else None
    pack = json.loads(Path(args.pack).read_text())
    leaf_space = json.loads(Path(args.leaf_space).read_text()) if args.leaf_space else None
    client = _client(config, d)
    res = client.ask(args.card, pack, keys=args.keys or None, leaf_space=leaf_space, allow_degraded=args.allow_degraded,
                     input_language=args.input_language)
    print(f"{res.card_id}: {res.requests} request(s), {res.usage_input_tokens} input tokens, model {res.model_id}")
    for k in sorted(res.estimates):
        qe = res.estimates[k]
        e = qe.estimate
        val = res.value(k)
        print(f"  {k:<22} {e.primitive.value:<6} {'-' if val is None else f'{val:.3f}':<7} choice={e.choice} conf={e.jev_confidence} contested={qe.contested} stab={e.stability.model_dump(exclude_none=True)}")
    return 0


def cmd_meta(args: argparse.Namespace) -> int:
    """Jev's meta cards on the machine: RM1 over the role cards, RM3 over a slice's acceptance evidence."""
    from sextant.meta import render_rm1, run_rm1, run_rm3
    from sextant.rolecards import load_all

    config = load_config()
    cards = load_all(_cards_dir(config))
    client = _client(config, _decision_dir(config, args.slug) if args.slug else None)
    if args.action == "rm1":
        from sextant.doctor import fitted_flag_thresholds

        rows = run_rm1(client, cards, only=args.cards or None)
        fitted = fitted_flag_thresholds(cards["RM1"], client) if "RM1" in cards else {}
        print(render_rm1(rows, fitted))
        if not fitted:
            print("RM1 has no validated threshold; the lines above are not a pass")
        return 0
    if args.action == "rm3":
        log = Path(args.log).read_text() if args.log else ""
        out = run_rm3(client, args.criterion or "", log)
        print(json.dumps(out))
        return 0
    return 2


def _labels_dir(card_id: str, root: Path | None) -> Path:
    """The grader handshake for a card lives under its calibration directory, not decisions/."""
    from sextant import calibrate as cal
    from sextant.state import init_state

    d = cal.card_dir(card_id, root) / "labels"
    if not (d / "state.json").exists():
        d.mkdir(parents=True, exist_ok=True)
        init_state(d, f"{card_id}-labels")
    return d


def cmd_calibrate(args: argparse.Namespace) -> int:
    from sextant import calibrate as cal
    from sextant.rolecards import load_all, load_card

    config = load_config()
    cards_dir = _cards_dir(config)
    cards = load_all(cards_dir)
    root = Path(args.root) if args.root else None
    if args.action == "build":
        items = json.loads(Path(args.items).read_text())
        if isinstance(items, dict) and "answer" in items:  # an accepted handshake answer file
            items = items["answer"]
        items = items.get("items", items) if isinstance(items, dict) else items
        out = cal.build(
            args.card, items, root=root,
            holdout_frac=args.holdout_frac, dev_frac=args.dev_frac, release_frac=args.release_frac,
        )
        print(f"{args.card}: added {out['added']}, total {out['total']}")
        return 0
    if args.action == "label":
        d = _labels_dir(args.card, root)
        card = cards[args.card]
        items = cal.load_items(args.card, root)
        ids = cal.label_requests(card, items, d, args.graders or [config["providers"]["grader"]], batch=args.batch)
        print(f"{len(ids)} grader request(s) written under {d}: {', '.join(ids)}")
        return 0
    if args.action == "import":
        d = _labels_dir(args.card, root)
        n = cal.import_labels(args.card, d, root, card=cards[args.card])
        out = cal.resolve_gold(args.card, root)
        print(f"{args.card}: imported {n} label rows; gold resolved {out['resolved']}, unresolved {out['unresolved']}, dissent {out['dissent']}")
        return 0
    if args.action == "gold":
        out = cal.resolve_gold(args.card, root)
        print(json.dumps(out))
        return 0
    if args.action == "measure":
        from sextant.jev_client import JevClient

        ledger = cal.measurement_ledger(args.card, root)
        client = JevClient(config, cards, ledger=ledger, calibration_root=root)
        items = cal.load_items(args.card, root, split=args.split)
        n = cal.measure(cards[args.card], client, items, root)
        print(f"{args.card}: measured {n} new item(s)")
        return 0
    if args.action == "metrics":
        m = cal.compute_metrics(cards[args.card], root)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ('murphy', 'coverage_curve')} for k, v in m['keys'].items()}, indent=2))
        return 0
    if args.action == "fit":
        card_path = next(p for p in cards_dir.glob("*.yaml") if load_card(p).id == args.card)
        target = float(args.target or config["calibration"]["target_agreement_T2"])
        results = cal.fit_card(cards[args.card], target=target, root=root, write=not args.dry_run)
        for r in results:
            state = "deployable" if r.deployable else ("provisional" if r.status == "fitted" else r.status)
            print(
                f"{r.key:<24} {state:<11} t={r.threshold} attempt={r.attempt} cases={r.n_eligible} "
                f"automatic={r.n_automatic} wrong={r.n_wrong} releases={r.n_releases} "
                f"R<={r.r_upper} R+<={r.r_plus_upper} C>={r.c_lower}"
            )
        if args.dry_run:
            print("dry run: no certificate written; the holdout cases scored above are now spent")
        if not args.dry_run:
            reason = cal.degrade_if_needed(cards[args.card], card_path, results, config)
            if reason:
                print(f"DEGRADED {args.card}: {reason}")
        return 0
    if args.action == "report":
        p = cal.report(load_card(next(p for p in cards_dir.glob("*.yaml") if load_card(p).id == args.card)), config, root)
        print(f"report written: {p}")
        return 0
    if args.action == "canary-freeze":
        n = cal.canary_freeze(args.cards or list(cards), args.per_card, root)
        print(f"canary frozen: {n} items")
        return 0
    if args.action == "canary-check":
        client = _client(config, None)
        out = cal.canary_check(client, cards, float(config["calibration"]["canary_band"]), root)
        print(json.dumps(out, indent=2))
        clean = not out["drifted"] and not out["model_moved"] and not out["protocol_moved"] and not out["missing_cards"]
        return 0 if clean and out["n"] > 0 and out["compared"] == out["n"] else 1
    return 2


def cmd_ledger(args: argparse.Namespace) -> int:
    from sextant.ledger import Ledger

    config = load_config()
    rows = Ledger(_decision_dir(config, args.slug)).read(args.name)
    for r in rows[-args.n :]:
        print(json.dumps(r, ensure_ascii=False)[:400])
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="sextant", description="decision compiler runtime (records and refuses; you reason)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="probe providers and Jev, write config.toml")
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--no-smoke", action="store_true")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("schema", help="print or export the decision-graph JSON Schema")
    p.add_argument("--export", nargs="?", const="schema/decision-graph.schema.json")
    p.set_defaults(fn=cmd_schema)

    p = sub.add_parser("selftest", help="prove every deterministic check on its planted violation")
    p.add_argument("--disable", action="append", help="turn a check into a no-op to show BROKEN detection")
    p.add_argument("--write-fixtures", metavar="DIR")
    p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("selfcheck", help="run the structural checks on a decision's latest graph")
    p.add_argument("slug")
    p.add_argument("--version", type=int)
    p.set_defaults(fn=cmd_selfcheck)

    p = sub.add_parser("doctor", help="run the health ladder; offline unless --live, writes nothing unless --record")
    p.add_argument("--skip-slow", action="store_true", help="skip the test suite")
    p.add_argument("--live", action="store_true", help="also run the live Jev probes (canary drift, RM1)")
    p.add_argument("--record", help="write this run, with code identity, to the given JSON file")
    p.add_argument("--require-release", action="store_true", help="exit non-zero unless release-eligible")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("config", help="print the effective config, or write defaults")
    p.add_argument("--write-defaults", action="store_true")
    p.set_defaults(fn=cmd_config)

    p = sub.add_parser("init", help="create a decision folder and its state.json")
    p.add_argument("slug")
    p.add_argument("--question")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("status", help="plain state of a decision")
    p.add_argument("slug")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("stamp", help="record a user judgment; a checker PASS is refused")
    p.add_argument("slug")
    p.add_argument("gate")
    p.add_argument("status", choices=["PASS", "FAIL", "SKIP"])
    p.add_argument("--token", required=True)
    p.add_argument("--note")
    p.set_defaults(fn=cmd_stamp)

    p = sub.add_parser("assert", help="refuse unless a gate is a checker PASS for the current graph")
    p.add_argument("slug")
    p.add_argument("gate")
    p.set_defaults(fn=cmd_assert)

    p = sub.add_parser("gate", help="run a certifiable gate's checker chain and store its receipt")
    p.add_argument("slug")
    p.add_argument("gate")
    p.set_defaults(fn=cmd_gate)

    p = sub.add_parser("request", help="write a request file for an LLM step")
    p.add_argument("slug")
    p.add_argument("--role", required=True)
    p.add_argument("--purpose", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--schema", required=True, help="JSON Schema text or @file")
    p.add_argument("--provider")
    p.add_argument("--fulfil", action="store_true", help="also ask the provider to fulfil it now")
    p.set_defaults(fn=cmd_request)

    p = sub.add_parser("fulfil", help="have the configured provider fulfil a request")
    p.add_argument("slug")
    p.add_argument("request_id")
    p.add_argument("--provider")
    p.set_defaults(fn=cmd_fulfil)

    p = sub.add_parser("answer", help="validate and ingest an answer file")
    p.add_argument("slug")
    p.add_argument("request_id")
    p.set_defaults(fn=cmd_answer)

    p = sub.add_parser("pending", help="list request files without answers")
    p.add_argument("slug")
    p.set_defaults(fn=cmd_pending)

    p = sub.add_parser("rolecards", help="seal or check role-card question hashes")
    p.add_argument("action", choices=["seal", "check"])
    p.add_argument("path", nargs="?")
    p.set_defaults(fn=cmd_rolecards)

    p = sub.add_parser("lint", help="run the deterministic question linter over the role cards")
    p.add_argument("path", nargs="?")
    p.set_defaults(fn=cmd_lint)

    p = sub.add_parser("jev", help="debug: ask one card on a pack")
    p.add_argument("action", choices=["ask"])
    p.add_argument("card")
    p.add_argument("--pack", required=True, help="JSON file with the state pack")
    p.add_argument("--leaf-space", help="JSON file with the node's answer space (R7)")
    p.add_argument("--keys", nargs="*")
    p.add_argument("--slug", help="record calls into this decision's ledger")
    p.add_argument("--allow-degraded", action="store_true")
    p.add_argument("--input-language", help="the language the pack is written in (e.g. en); undeclared inputs are never calibrated")
    p.set_defaults(fn=cmd_jev_ask)

    p = sub.add_parser("meta", help="Jev meta cards on the machine itself")
    p.add_argument("action", choices=["rm1", "rm3"])
    p.add_argument("--cards", nargs="*", help="rm1: limit to these card ids")
    p.add_argument("--criterion", help="rm3: the acceptance criterion text")
    p.add_argument("--log", help="rm3: file with the evidence log")
    p.add_argument("--slug", help="record calls into this decision's ledger")
    p.set_defaults(fn=cmd_meta)

    p = sub.add_parser("calibrate", help="golden sets, labels, measurement, fitting, reports, canary")
    p.add_argument("action", choices=["build", "label", "import", "gold", "measure", "metrics", "fit", "report", "canary-freeze", "canary-check"])
    p.add_argument("card", nargs="?")
    p.add_argument("--items", help="JSON file with items (build)")
    p.add_argument("--slug", help="decision folder used for label requests or measurement ledgers")
    p.add_argument("--graders", nargs="*", help="provider names for label requests")
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--split", choices=["fit", "holdout"])
    p.add_argument("--target", type=float)
    p.add_argument("--holdout-frac", type=float, default=0.25, help="build: share of cases held out for confirmation")
    p.add_argument("--dev-frac", type=float, default=0.0, help="build: share of cases for development")
    p.add_argument("--release-frac", type=float, default=0.0, help="build: share of cases for the release holdout")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--cards", nargs="*")
    p.add_argument("--per-card", type=int, default=6)
    p.add_argument("--root", help="alternate calibration root (tests)")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("ledger", help="tail a ledger")
    p.add_argument("slug")
    p.add_argument("name")
    p.add_argument("-n", type=int, default=10)
    p.set_defaults(fn=cmd_ledger)
    return ap


def legacy_main(argv: list[str] | None = None) -> int:
    """Every command exits with a plain line, never a traceback, on the errors the runtime raises on purpose."""
    from sextant.calibrate import (
        CalibrationEvidenceError,
        CalibrationProtocolError,
        CalibrationSplitError,
    )
    from sextant.config import ConfigError
    from sextant.handshake import BlockerError, HandshakeError, ProviderError, SupersededError
    from sextant.jev_client import JevClientError
    from sextant.rolecards import RoleCardError
    from sextant.state import GateError

    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except BlockerError as e:
        print(f"BLOCKER: {e}")
        return 2
    except ProviderError as e:
        print(f"PROVIDER FAIL: {e}")
        return 1
    except SupersededError as e:
        print(f"SUPERSEDED: {e}")
        return 1
    except (
        HandshakeError, GateError, RoleCardError, ConfigError, JevClientError,
        CalibrationProtocolError, CalibrationSplitError, CalibrationEvidenceError,
    ) as e:
        print(f"REFUSED: {e}")
        return 1
    except FileNotFoundError as e:
        print(f"NOT FOUND: {e}")
        return 1
    except FileExistsError as e:
        print(f"EXISTS: {e}")
        return 1
    except ValueError as e:
        print(f"INVALID: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    from sextant.cli import COMMANDS
    from sextant.cli import main as v1_main
    arguments = sys.argv[1:] if argv is None else argv
    if not arguments or arguments[0] in (*COMMANDS, "agent", "protocol-schema", "backup", "examples", "guide", "legacy") or arguments[0].startswith("--"):
        return v1_main(arguments)
    return v1_main(["legacy", *arguments])


if __name__ == "__main__":
    sys.exit(main())
