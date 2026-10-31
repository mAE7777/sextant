"""Self-test: every deterministic check must catch its planted violation and pass its clean twin.

A check that passes both is BROKEN, and BROKEN is reported louder than FAIL, because a
checker that can report green while broken is worse than no checker. There is exactly one
level of this; the limit is stated here rather than hidden.

`selftest --disable <check>` turns one check into a no-op so the BROKEN path can be shown.
`selftest --write-fixtures <dir>` exports every plant and twin as JSON for inspection.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from sextant.graph import (
    Finding,
    check_acyclic,
    check_against_ledger,
    check_budgets,
    check_ids,
    check_reified,
    check_residuals,
    validate_graph_dict,
)
from sextant.ledger import sha_of
from sextant.providers.base import Answer, Request, validate_answer
from sextant.rolecards import RoleCard, compute_hash, drifted
from sextant.schema import (
    SCHEMA_VERSION,
    Gate,
    GateAuthority,
    GateReceipt,
    GateStatus,
    edge_id,
    node_id,
    question_hash,
    receipt_token,
)
from sextant.state import STRUCTURE_CHECKER
from sextant.wire import (
    WIRE_PROTOCOL_VERSION,
    compose_instructions,
    questions_digest,
    rendered_hash,
    request_digest,
    run_protocol,
)
from sextant.wire import estimate_tokens as wire_estimate_tokens

# ---------------------------------------------------------------- fixture builders

CALL_ID = "jc_fixture0001"
RUN_ID = "run_fixture0001"
CARD = "R7"
QKEY = "leaf"
MODEL = "jev-1.13.0"
TRANSPORT = "controlled:fixture"
LANGUAGE = "en"
FIXTURE_STATE = {"question": "Does the salary cover expenses?"}


def fixture_card() -> RoleCard:
    """A one-question card with no re-runs, so a fixture run is a single recorded call."""
    draft = RoleCard.model_validate({
        "id": CARD,
        "purpose": "leaf judgment fixture",
        "primitive": "noul",
        "state_pack": ["question"],
        "instructions": "Answer from the state alone.",
        "questions": {QKEY: {"primitive": "noul", "text": "Is it true that {question}"}},
        "reruns": 0,
        "permutations": 1,
    })
    return draft.model_copy(update={"question_hashes": {QKEY: compute_hash(draft, QKEY)}})


def sample_call(p: float = 0.8, question_keys: list[str] | None = None) -> dict[str, Any]:
    """The jev-calls row that `sample_estimate` must be able to prove itself against.

    Built with the wire protocol's own functions, so its recorded payload re-hashes exactly.
    """
    card = fixture_card()
    keys = question_keys or [QKEY]
    text = card.questions[QKEY].text.format(question=FIXTURE_STATE["question"])
    sdk = {"type": "noul", "instructions": compose_instructions(card, QKEY, text, language=LANGUAGE)}
    questions = {k: sdk for k in keys}
    bindings = {
        k: {"template_hash": compute_hash(card, QKEY), "rendered_hash": rendered_hash(CARD, k, "noul", sdk)}
        for k in keys
    }
    return {
        "seq": 1,
        "at": "2026-09-18T00:00:00Z",
        "id": CALL_ID,
        "card_id": CARD,
        "protocol_version": WIRE_PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "call_role": "primary",
        "run_protocol": run_protocol(card),
        "question_keys": keys,
        "question_bindings": bindings,
        "questions_digest": questions_digest(bindings),
        "model_id": MODEL,
        "transport": TRANSPORT,
        "state_sha": sha_of(FIXTURE_STATE),
        "variant": "base",
        "request": {
            "state": FIXTURE_STATE,
            "questions": questions,
            "request_digest": request_digest(model_id=MODEL, transport=TRANSPORT, state=FIXTURE_STATE, questions=questions),
        },
        "response": {"answers": {k: {"type": "noul", "noul": p} for k in keys}, "usage": {}, "request_id": "fixture"},
    }


def sample_estimate(p: float = 0.8, call_id: str = CALL_ID, rendered_hash_override: str | None = None) -> dict[str, Any]:
    """A judgment derived from `sample_call` exactly as the client would record it."""
    row = sample_call(p)
    binding = row["question_bindings"][QKEY]
    rendered = rendered_hash_override or binding["rendered_hash"]
    return {
        "primitive": "noul",
        "p": p,
        "card_id": CARD,
        "question_key": QKEY,
        "question_hash": binding["template_hash"],
        "rendered_hash": rendered,
        "state_sha": row["state_sha"],
        "protocol_version": WIRE_PROTOCOL_VERSION,
        "run_id": RUN_ID,
        "language": LANGUAGE,
        "aggregation": "single",
        "contested": False,
        "contributions": [
            {
                "call_id": call_id,
                "variant": "base",
                "role": "primary",
                "rendered_hash": rendered,
                "value": p,
                "contributes_to_value": True,
            }
        ],
        "model_id": MODEL,
        "transport": TRANSPORT,
        "source_tier": "jev-judged",
        "calibration_status": {"kind": "outcome", "state": "pending", "n": 0, "bar": 30},
        "at": "2026-09-18T00:00:00Z",
    }


def node(kind: str, text: str, **extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"id": node_id(kind, text), "kind": kind, "text_original": text, "text_en": text}
    d.update(extra)
    return d


def edge(etype: str, a: dict[str, Any], b: dict[str, Any], **extra: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"id": edge_id(etype, a["id"], b["id"]), "type": etype, "from_id": a["id"], "to_id": b["id"]}
    if etype == "decomposes_into":
        d.setdefault("method", {"ordered": False, "subtasks": [b["id"]], "applicability": ""})
    d.update(extra)
    return d


def base_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], tier: str = "T2") -> dict[str, Any]:
    q = "Should I take the offer?"
    return {
        "schema_version": SCHEMA_VERSION,
        "slug": "fixture",
        "version": 1,
        "created": "2026-09-18T00:00:00Z",
        "question": {"text_original": q, "text_en": q, "hash": question_hash(q)},
        "frame": {"x": "now", "y": "goal", "decision_type": "choose", "tier": tier},
        "nodes": nodes,
        "edges": edges,
    }


def fx_schema_endpoint(plant: bool) -> dict[str, Any]:
    v = node("value", "autonomy matters")
    c = node("criterion", "autonomy")
    f = node("factor", "commute time")
    target = f if plant else c
    return base_graph([v, c, f], [edge("is_value_for", v, target)])


def fx_schema_resolution(plant: bool) -> dict[str, Any]:
    f = node("factor", "salary covers expenses", estimate=sample_estimate())
    if not plant:
        f["resolution"] = {"criteria": "offer letter states base salary >= 1.2x monthly expenses", "source": "offer letter", "resolve_by": "2026-10-01"}
    return base_graph([f], [])


def fx_schema_escape(plant: bool) -> dict[str, Any]:
    opts = {"billing": None, "technical": None, "sales": None}
    space: dict[str, Any] = {"primitive": "choice", "options": dict(opts)}
    if not plant:
        space["options"]["none_of_these"] = "none of the departments fits"
        space["escape_option"] = "none_of_these"
    f = node("factor", "which department", answer_space=space)
    return base_graph([f], [])


def fx_schema_value_node(plant: bool) -> dict[str, Any]:
    v = node("value", "family proximity matters")
    if plant:
        v["estimate"] = sample_estimate()
    return base_graph([v], [])


def fx_acyclic(plant: bool) -> dict[str, Any]:
    a = node("factor", "rent level")
    b = node("factor", "savings rate")
    edges = [edge("depends_on", a, b)]
    if plant:
        edges.append(edge("depends_on", b, a))
    return base_graph([a, b], edges)


def fx_residual(plant: bool) -> dict[str, Any]:
    q = node("question", "Should I take the offer?")
    k1 = node("factor", "salary covers expenses")
    k2 = node("factor", "commute under 45 minutes")
    nodes = [q, k1, k2]
    edges = [edge("decomposes_into", q, k1), edge("decomposes_into", q, k2)]
    if not plant:
        r = node("factor", "other components not listed", residual=True)
        nodes.append(r)
        edges.append(edge("decomposes_into", q, r))
    return base_graph(nodes, edges)


def fx_budget(plant: bool) -> dict[str, Any]:
    q = node("question", "Should I take the offer?")
    n = 9 if plant else 4
    kids = [node("factor", f"component {i}") for i in range(n)]
    r = node("factor", "other components not listed", residual=True)
    edges = [edge("decomposes_into", q, k) for k in kids] + [edge("decomposes_into", q, r)]
    return base_graph([q, *kids, r], edges, tier="T2")


def fx_reified(plant: bool) -> dict[str, Any]:
    f = node("factor", "salary covers expenses")
    ev = node("evidence", "offer letter, 2026-09-10: base salary $180k")
    link = node("evidence_link", "offer letter supports salary factor", members=[ev["id"], f["id"]], direction="+")
    edges = [edge("supports", link, f)]
    if not plant:
        edges.append(edge("cites", link, ev))
    return base_graph([f, ev, link], edges)


RESOLUTION = {
    "criteria": "offer letter states base salary >= 1.2x monthly expenses",
    "source": "offer letter",
}


def _judged(estimate: dict[str, Any]) -> dict[str, Any]:
    return node("factor", "salary covers expenses", estimate=estimate, resolution=RESOLUTION)


def _judged_graph(estimate: dict[str, Any]) -> dict[str, Any]:
    """A structurally valid graph (root, judged factor, residual) carrying one node judgment."""
    q = node("question", "Should I take the offer?")
    f = _judged(estimate)
    r = node("factor", "other components not listed", residual=True)
    return base_graph([q, f, r], [edge("decomposes_into", q, f), edge("decomposes_into", q, r)])


def fx_ledger(plant: bool) -> dict[str, Any]:
    g = _judged_graph(sample_estimate(call_id="jc_missing0001" if plant else CALL_ID))
    g["_ledger_index"] = {CALL_ID: sample_call()}
    return g


def fx_ledger_replay(plant: bool) -> dict[str, Any]:
    """A judgment whose number contradicts the answer the ledger recorded for it."""
    g = _judged_graph(sample_estimate(p=0.99 if plant else 0.8))
    g["_ledger_index"] = {CALL_ID: sample_call(p=0.8)}
    return g


def fx_ledger_state_vector(plant: bool) -> dict[str, Any]:
    """A state-vector judgment; auditing used to stop at estimate and sufficiency."""
    q = node("question", "Should I take the offer?")
    q["state_vector"] = {
        "verifiability": sample_estimate(call_id="jc_missing0001" if plant else CALL_ID)
    }
    g = base_graph([q], [])
    g["_ledger_index"] = {CALL_ID: sample_call()}
    return g


def fx_ledger_edge_verification(plant: bool) -> dict[str, Any]:
    """An edge that cites a Jev judgment for its own verification."""
    q = node("question", "Should I take the offer?")
    f = node("factor", "salary covers expenses")
    r = node("factor", "other components not listed", residual=True)
    kid = edge("decomposes_into", q, f)
    kid["verification"] = {
        "relevance": sample_estimate(call_id="jc_missing0001" if plant else CALL_ID)
    }
    g = base_graph([q, f, r], [kid, edge("decomposes_into", q, r)])
    g["_ledger_index"] = {CALL_ID: sample_call()}
    return g


def fx_ids(plant: bool) -> dict[str, Any]:
    f = node("factor", "salary covers expenses")
    if plant:
        f["id"] = "n_000000000000"
    return base_graph([f], [])


def fx_rolecard(plant: bool) -> dict[str, Any]:
    card = {
        "id": "R2",
        "purpose": "sufficiency",
        "primitive": "noul",
        "state_pack": ["question", "answer_space", "context_items"],
        "instructions": "Treat the context items as the only information that exists.",
        "questions": {
            "sufficient": {
                "primitive": "noul",
                "text": "Is there enough information in the context items to answer the question?",
                "criteria_true": "A careful reader could pick one answer using only the context items.",
                "criteria_false": "The context items do not settle the question.",
            }
        },
        "thresholds": {"sufficient": {"value": 0.7, "n": 150, "date": "2026-09-18", "primitive": "noul"}},
        "question_hashes": {},
    }
    rc = RoleCard.model_validate(card)
    real = compute_hash(rc, "sufficient")
    card["question_hashes"] = {"sufficient": "qh_stale000000000" if plant else real}
    return card


def fx_answer(plant: bool) -> dict[str, Any]:
    req = {
        "id": "rq_fixture01",
        "role": "generator",
        "purpose": "propose children",
        "prompt": "Propose three child questions.",
        "answer_schema": {
            "type": "object",
            "required": ["children"],
            "additionalProperties": False,
            "properties": {
                "children": {"type": "array", "minItems": 3, "maxItems": 5, "items": {"type": "object", "required": ["text"], "properties": {"text": {"type": "string", "minLength": 5}}}}
            },
        },
        "provider": "claude-code",
        "created": "2026-09-18T00:00:00Z",
    }
    children = [{"text": "Does the salary cover expenses?"}, {"text": "Is the commute under 45 minutes?"}]
    if not plant:
        children.append({"text": "Is the team one I would learn from?"})
    ans = {"request_id": "rq_fixture01", "answer": {"children": children}, "provider": "claude-code", "family": "anthropic", "isolation": "same-session"}
    return {"request": req, "answer": ans}


def fx_gate_token(plant: bool) -> dict[str, Any]:
    receipt = GateReceipt(
        protocol_version=WIRE_PROTOCOL_VERSION,
        checker=STRUCTURE_CHECKER,
        input_revisions={"graph_version": "1", "protocol_version": WIRE_PROTOCOL_VERSION},
        artifact_hash="artifact-fixture",
        prerequisites=[],
        result="PASS",
    )
    gate = Gate(
        authority=GateAuthority.checker,
        status=GateStatus.PASS,
        at="2026-09-18T00:00:00Z",
        token=receipt_token(receipt),
        receipt=receipt,
    )
    payload = json.loads(gate.model_dump_json())
    if plant:
        payload["token"] = ""
    return {"gates": {"intake": payload}}


def fx_pack_size(plant: bool) -> dict[str, Any]:
    text = "x" * (200_000 if plant else 2_000)
    return {"pack": {"parent": "Should I take the offer?", "child": text}, "max_state_tokens": 30000}


# ---------------------------------------------------------------- checks over fixtures


def _schema_check(data: dict[str, Any]) -> list[Finding]:
    _, findings = validate_graph_dict(data)
    return findings


def _graph_check(fn: Callable[..., list[Finding]]) -> Callable[[dict[str, Any]], list[Finding]]:
    def run(data: dict[str, Any]) -> list[Finding]:
        clean = {k: v for k, v in data.items() if not k.startswith("_")}
        g, findings = validate_graph_dict(clean)
        if g is None:
            return findings
        return fn(g)

    return run


def _ledger_check(data: dict[str, Any]) -> list[Finding]:
    clean = {k: v for k, v in data.items() if not k.startswith("_")}
    g, findings = validate_graph_dict(clean)
    if g is None:
        return findings
    return [f for f in check_against_ledger(g, data.get("_ledger_index", {})) if f.severity == "FAIL"]


def _rolecard_check(data: dict[str, Any]) -> list[Finding]:
    card = RoleCard.model_validate(data)
    return [Finding("rolecard-hash", f"{card.id}.{k}", v) for k, v in drifted(card).items()]


def _answer_check(data: dict[str, Any]) -> list[Finding]:
    req = Request.model_validate(data["request"])
    ans = Answer.model_validate(data["answer"])
    return [Finding("answer-schema", req.id, e) for e in validate_answer(req, ans)]


def _gate_token_check(data: dict[str, Any]) -> list[Finding]:
    """The production gate rule: PASS is a current-protocol checker receipt, not a token string."""
    findings: list[Finding] = []
    for name, raw in data["gates"].items():
        try:
            gate = Gate.model_validate(raw)
        except ValidationError as exc:
            findings.append(Finding("gate-token", str(name), str(exc.errors()[0]["msg"])))
            continue
        if gate.authority != GateAuthority.checker or gate.status != GateStatus.PASS or gate.receipt is None:
            findings.append(Finding("gate-token", str(name), "gate is not a checker PASS"))
        elif gate.receipt.protocol_version != WIRE_PROTOCOL_VERSION:
            findings.append(Finding("gate-token", str(name), "gate receipt protocol is stale"))
    return findings


def estimate_tokens(obj: Any) -> int:
    return wire_estimate_tokens(obj)


def _pack_size_check(data: dict[str, Any]) -> list[Finding]:
    n = estimate_tokens(data["pack"])
    limit = int(data["max_state_tokens"])
    return [Finding("pack-size", "pack", f"about {n} tokens exceeds the state budget {limit}")] if n > limit else []


@dataclass
class Check:
    name: str
    rule: str
    fixture: Callable[[bool], dict[str, Any]]
    run: Callable[[dict[str, Any]], list[Finding]]


CHECKS: list[Check] = [
    Check("ids", "a node id is the hash of (kind, text_en, definition); an edge id of (type, from, to)", fx_ids, _graph_check(check_ids)),
    Check("schema-endpoint", "an edge must join legal node kinds for its type", fx_schema_endpoint, _schema_check),
    Check("schema-resolution", "an estimate on a factor or probe needs resolution criteria", fx_schema_resolution, _schema_check),
    Check("schema-escape", "a Choice answer space carries an escape option", fx_schema_escape, _schema_check),
    Check("schema-value-node", "a value node never carries a Jev estimate", fx_schema_value_node, _schema_check),
    Check("acyclic", "decomposition and dependency edges form no cycle", fx_acyclic, _graph_check(check_acyclic)),
    Check("residual", "every sibling set has a residual node", fx_residual, _graph_check(check_residuals)),
    Check("budget", "children per parent stay within the tier's breadth budget", fx_budget, _graph_check(check_budgets)),
    Check("reified", "an evidence link has both a supports edge and a cites edge", fx_reified, _graph_check(check_reified)),
    Check("ledger", "every contributing call exists in the jev-calls ledger with matching card, protocol, question and model", fx_ledger, _ledger_check),
    Check("ledger-replay", "a reported value must replay from the answers its own calls recorded", fx_ledger_replay, _ledger_check),
    Check("ledger-state-vector", "a state-vector judgment is audited against the ledger too", fx_ledger_state_vector, _ledger_check),
    Check("ledger-edge-verification", "an edge's verification judgment is audited against the ledger too", fx_ledger_edge_verification, _ledger_check),
    Check("rolecard-hash", "a role-card question runs only with its recorded hash", fx_rolecard, _rolecard_check),
    Check("answer-schema", "an answer file must conform to its request's JSON Schema", fx_answer, _answer_check),
    Check(
        "gate-token",
        "a gate PASS is a checker receipt bound to the current wire protocol",
        fx_gate_token,
        _gate_token_check,
    ),
    Check("pack-size", "a state pack stays within the token budget", fx_pack_size, _pack_size_check),
]


@dataclass
class CheckOutcome:
    name: str
    verdict: str  # PASS | BROKEN
    detail: str


def run_selftest(disable: set[str] | None = None) -> list[CheckOutcome]:
    disable = disable or set()
    outcomes: list[CheckOutcome] = []
    for chk in CHECKS:
        run = chk.run if chk.name not in disable else (lambda data: [])
        plant_findings = run(chk.fixture(True))
        clean_findings = run(chk.fixture(False))
        if not plant_findings:
            outcomes.append(CheckOutcome(chk.name, "BROKEN", "planted violation not detected"))
        elif clean_findings:
            outcomes.append(CheckOutcome(chk.name, "BROKEN", "clean twin failed: " + clean_findings[0].message))
        else:
            outcomes.append(CheckOutcome(chk.name, "PASS", f"plant caught: {plant_findings[0].message}"))
    return outcomes


def render(outcomes: list[CheckOutcome]) -> str:
    width = max(len(o.name) for o in outcomes)
    lines = [f"{o.verdict:<6} {o.name:<{width}}  {o.detail}" for o in outcomes]
    broken = sum(1 for o in outcomes if o.verdict == "BROKEN")
    lines.append("")
    lines.append(f"selftest: {len(outcomes) - broken} PASS, {broken} BROKEN" + ("" if broken == 0 else "  <- a broken checker is worse than none"))
    return "\n".join(lines)


def write_fixtures(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for chk in CHECKS:
        for label, flag in (("plant", True), ("clean", False)):
            data = chk.fixture(flag)
            p = out_dir / f"{chk.name}.{label}.{'yaml' if chk.name == 'rolecard-hash' else 'json'}"
            if p.suffix == ".yaml":
                p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
            else:
                p.write_text(json.dumps({"rule": chk.rule, "expect": "FAIL" if flag else "PASS", "data": data}, indent=2, ensure_ascii=False))
            paths.append(p)
    return paths
