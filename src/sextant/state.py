"""The per-decision state spine (state.json): gates, pending requests, versions, harness pins.

A checker gate is a receipt issued by `evaluate_gate` after that checker has run. The
receipt binds the wire protocol, the checker name and version, the input revisions (card seal
and certificate store included), the artifact hash and the prerequisite receipts. `stamp`
records a human approval as a user judgment and cannot mint PASS. `assert_gate` accepts only a
checker PASS that still matches the graph it is shown, and re-runs every checker in the chain.

Only gates whose stage has a real check can be certified. Every other gate name belongs to a
stage that does not exist yet; a PASS there would declare product completion nobody verified,
so those gates refuse to certify until their stage and its check exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from sextant import __version__
from sextant.ledger import exclusive, now_iso, sha_of
from sextant.schema import (
    SCHEMA_VERSION,
    DecisionGraph,
    Gate,
    GateAuthority,
    GateReceipt,
    GateStatus,
    receipt_token,
)
from sextant.wire import WIRE_PROTOCOL_VERSION

GATES_IN_ORDER = (
    "intake",
    "frame",
    "statevec",
    "tier",
    "sufficiency",
    "generate",
    "verify",
    "leaf",
    "stop",
    "ground",
    "values",
    "aggregate",
    "render",
    "consistency",
    "verifier",
    "journal",
    "absorb",
)


class GateError(RuntimeError):
    pass


class StaleState(RuntimeError):
    """A writer saved a snapshot older than the one already on disk."""


class HarnessPins(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime_version: str = __version__
    schema_version: str = SCHEMA_VERSION
    skill_version: str | None = None
    rolecards_hash: str | None = None
    executor_contract_version: str | None = None
    jev_model: str | None = None
    transport: str | None = None


class Versions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    graph: int = 0
    policy: int = 0
    pass_no: int = 0


class State(BaseModel):
    model_config = ConfigDict(extra="forbid")
    slug: str
    created: str
    stage: str = "init"
    gates: dict[str, Gate] = Field(default_factory=dict)
    pending_requests: list[str] = Field(default_factory=list)
    versions: Versions = Field(default_factory=Versions)
    harness: HarnessPins = Field(default_factory=HarnessPins)
    blockers: list[str] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)
    write_generation: int = 0


def state_path(decision_dir: Path) -> Path:
    return decision_dir / "state.json"


def load_state(decision_dir: Path) -> State:
    p = state_path(decision_dir)
    if not p.exists():
        raise FileNotFoundError(f"no state.json under {decision_dir}; run init first")
    return State.model_validate_json(p.read_text())


def save_state(state: State, decision_dir: Path) -> Path:
    """Write state under the single-writer lock, refusing a stale snapshot.

    The lock serializes the write. `write_generation` refuses a second writer
    who loaded an older snapshot and would otherwise drop a pending update.
    Ledger rows and this file are still not one crash-safe transaction.
    """
    decision_dir.mkdir(parents=True, exist_ok=True)
    with exclusive(decision_dir):
        path = state_path(decision_dir)
        if path.exists():
            current = State.model_validate_json(path.read_text())
            if current.write_generation != state.write_generation:
                raise StaleState(
                    f"stale state write_generation {state.write_generation}; "
                    f"current is {current.write_generation}; reload before saving"
                )
        state.write_generation += 1
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(state.model_dump_json(indent=2))
        temporary.replace(path)
        return path


def init_state(decision_dir: Path, slug: str, created: str | None = None) -> State:
    if state_path(decision_dir).exists():
        raise FileExistsError(f"{state_path(decision_dir)} exists")
    st = State(slug=slug, created=created or now_iso())
    save_state(st, decision_dir)
    return st


CallIndex = dict[str, dict[str, Any]]

STRUCTURE_CHECKER = "sextant.graph.structure@2"
EDGE_EVIDENCE_CHECKER = "sextant.graph.edge-evidence@2"

# The gates that have a real check in this version: what the checker certifies, and which
# gates must already hold a PASS on the same inputs.
CERTIFIABLE: dict[str, tuple[str, tuple[str, ...]]] = {
    "generate": (STRUCTURE_CHECKER, ()),
    "verify": (EDGE_EVIDENCE_CHECKER, ("generate",)),
}
RELEASE_GATES = frozenset({"verify"})


def _checker_name(gate: str) -> str:
    entry = CERTIFIABLE.get(gate)
    if entry is None:
        raise GateError(
            f"gate {gate} has no stage check in this version; it cannot be certified until its "
            "stage and its check exist"
        )
    return entry[0]


def _chain(gate: str) -> list[str]:
    """The gate and every prerequisite beneath it, deepest first."""
    order: list[str] = []
    for name in CERTIFIABLE[gate][1]:
        for inner in _chain(name):
            if inner not in order:
                order.append(inner)
    order.append(gate)
    return order


def bound_artifact(graph: DecisionGraph, call_index: CallIndex | None) -> str:
    return sha_of({"graph": graph.model_dump(mode="json"), "call_index": call_index or {}})


def _certificates_digest(cards: dict[str, Any] | None, calibration_root: Path | None) -> str:
    if not cards or calibration_root is None:
        return "none"
    parts = []
    for card_id in sorted(cards):
        path = calibration_root / card_id / "certificates.jsonl"
        parts.append([card_id, path.read_text() if path.exists() else ""])
    return sha_of(parts)


def input_revisions(
    state: State,
    graph: DecisionGraph,
    cards: dict[str, Any] | None = None,
    calibration_root: Path | None = None,
) -> dict[str, str]:
    from sextant.graph import judgment_sites
    from sextant.rolecards import cards_hash

    transports = sorted({est.transport for _, _, est in judgment_sites(graph)})
    return {
        "schema_version": graph.schema_version,
        "graph_version": str(graph.version),
        "graph_slug": graph.slug,
        "policy_version": str(state.versions.policy),
        "pass_no": str(state.versions.pass_no),
        "question_hash": graph.question.hash,
        "protocol_version": WIRE_PROTOCOL_VERSION,
        "cards": cards_hash(cards) if cards else "none",
        "certificates": _certificates_digest(cards, calibration_root),
        # the transports behind every judgment the graph cites; a controlled transport shows here
        "evidence_transports": ",".join(transports) if transports else "none",
    }


def _findings_for(
    gate: str,
    graph: DecisionGraph,
    call_index: CallIndex | None,
    cards: dict[str, Any] | None,
    calibration_root: Path | None,
) -> list[Any]:
    from sextant.graph import Finding, check_all, check_release

    # No ledger is an empty ledger here: skipping the audit let a judgment citing a call that
    # never ran pass `generate` through the API.
    ledger = call_index if call_index is not None else {}
    findings: list[Finding] = list(check_all(graph, ledger, cards, calibration_root).findings)
    if CERTIFIABLE[gate][0] == EDGE_EVIDENCE_CHECKER:
        findings.extend(check_release(graph, ledger, cards, calibration_root))
    return findings


def stamp(state: State, gate: str, status: GateStatus | str, token: str, note: str | None = None) -> Gate:
    """Record a user judgment. This never issues a checker PASS."""
    if gate not in GATES_IN_ORDER:
        raise GateError(f"unknown gate {gate}; known: {', '.join(GATES_IN_ORDER)}")
    parsed = GateStatus(status)
    if parsed == GateStatus.PASS:
        raise GateError(
            "a human approval is a user judgment, never a checker PASS; "
            "a checker receipt is issued only by evaluate_gate"
        )
    if not token.strip():
        raise GateError(f"gate {gate} needs a non-empty token")
    g = Gate(
        authority=GateAuthority.user_judgment,
        status=parsed,
        at=now_iso(),
        token=token.strip(),
        note=note,
    )
    # A user judgment is recorded; it does not move the stage, which only a checker PASS can.
    state.gates[gate] = g
    if note:
        state.notes.setdefault("gate_notes", {})[gate] = note
    return g


def evaluate_gate(
    state: State,
    gate: str,
    graph: DecisionGraph,
    call_index: CallIndex | None = None,
    *,
    cards: dict[str, Any] | None = None,
    calibration_root: Path | None = None,
) -> Gate:
    """Run the checker for `gate` and store the receipt it actually produced."""
    if gate not in GATES_IN_ORDER:
        raise GateError(f"unknown gate {gate}; known: {', '.join(GATES_IN_ORDER)}")
    checker = _checker_name(gate)
    revisions = input_revisions(state, graph, cards, calibration_root)
    artifact = bound_artifact(graph, call_index)
    prerequisites: list[str] = []
    for name in CERTIFIABLE[gate][1]:
        prior = state.gates.get(name)
        _require_prerequisite(gate, name, prior, artifact, revisions)
        assert prior is not None
        prerequisites.append(prior.token)
    findings: list[Any] = []
    for name in _chain(gate):
        findings.extend(_findings_for(name, graph, call_index, cards, calibration_root))
    failures = [item for item in findings if getattr(item, "severity", "FAIL") == "FAIL"]
    passed = not failures
    receipt = GateReceipt(
        protocol_version=WIRE_PROTOCOL_VERSION,
        checker=checker,
        input_revisions=revisions,
        artifact_hash=artifact,
        prerequisites=prerequisites,
        result="PASS" if passed else "FAIL",
    )
    note = "; ".join(f"{item.check}: {item.message}" for item in failures) or None
    g = Gate(
        authority=GateAuthority.checker,
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        at=now_iso(),
        token=receipt_token(receipt),
        receipt=receipt,
        note=note[:500] if note else None,
    )
    previous = state.gates.get(gate)
    state.gates[gate] = g
    if previous is None or previous.token != g.token or not passed:
        # a receipt downstream was issued on the old one; it no longer stands
        for name in _dependents(gate):
            old = state.gates.get(name)
            if old is not None and old.authority == GateAuthority.checker:
                state.notes.setdefault("superseded_gates", []).append({
                    "gate": name, "token": old.token, "at": old.at, "status": old.status.value,
                    "reason": f"{gate} was re-evaluated ({g.status.value}, token {g.token})",
                })
                del state.gates[name]
    state.stage = current_stage(state)
    return g


def _dependents(gate: str) -> list[str]:
    return [name for name in CERTIFIABLE if name != gate and gate in _chain(name)]


def current_stage(state: State) -> str:
    """The deepest certifiable gate whose checker PASS still stands on its prerequisites' receipts."""
    stage = "init"
    for name in GATES_IN_ORDER:
        if name not in CERTIFIABLE:
            continue
        g = state.gates.get(name)
        if g is None or g.authority != GateAuthority.checker or g.status != GateStatus.PASS or g.receipt is None:
            break
        prerequisites = [state.gates.get(p) for p in CERTIFIABLE[name][1]]
        if any(p is None or p.token not in g.receipt.prerequisites for p in prerequisites):
            break
        stage = name
    return stage


def _require_prerequisite(
    gate: str, name: str, prior: Gate | None, artifact: str, revisions: dict[str, str]
) -> None:
    if (
        prior is None
        or prior.authority != GateAuthority.checker
        or prior.status != GateStatus.PASS
        or prior.receipt is None
    ):
        raise GateError(f"gate {gate} is missing prerequisite checker PASS {name}")
    if prior.receipt.checker != CERTIFIABLE[name][0]:
        raise GateError(f"prerequisite {name} was not issued by {CERTIFIABLE[name][0]}")
    if prior.token != receipt_token(prior.receipt):
        raise GateError(f"prerequisite {name} token does not match its receipt")
    if prior.receipt.artifact_hash != artifact or prior.receipt.input_revisions != revisions:
        raise GateError(f"prerequisite {name} was issued for different inputs")
    if prior.receipt.protocol_version != WIRE_PROTOCOL_VERSION:
        raise GateError(f"prerequisite {name} uses a stale wire protocol")


def assert_gate(
    state: State,
    gate: str,
    *,
    graph: DecisionGraph | None = None,
    call_index: CallIndex | None = None,
    cards: dict[str, Any] | None = None,
    calibration_root: Path | None = None,
) -> None:
    """Accept only a checker PASS whose receipt still matches, re-running every checker in its chain.

    Re-running only the gate's own checker let a hand-built chain of receipts pass a later gate
    whose prerequisites would have failed; the whole chain is checked again here.
    """
    if gate not in GATES_IN_ORDER:
        raise GateError(f"unknown gate {gate}; known: {', '.join(GATES_IN_ORDER)}")
    checker = _checker_name(gate)
    g = state.gates.get(gate)
    if g is None:
        raise GateError(f"gate {gate} has no checker receipt; run `python -m sextant gate <slug> {gate}` to issue one")
    if g.authority != GateAuthority.checker or g.receipt is None:
        raise GateError(f"gate {gate} is a user judgment, not a checker PASS")
    if g.status != GateStatus.PASS or g.receipt.result != "PASS":
        raise GateError(f"gate {gate} is {g.status.value}, not PASS")
    if g.receipt.protocol_version != WIRE_PROTOCOL_VERSION:
        raise GateError(
            f"gate {gate} receipt uses protocol {g.receipt.protocol_version}, not {WIRE_PROTOCOL_VERSION}"
        )
    if g.receipt.checker != checker:
        raise GateError(f"gate {gate} was not issued by {checker}")
    if g.token != receipt_token(g.receipt):
        raise GateError(f"gate {gate} token does not match its receipt")
    if graph is None:
        raise GateError(f"gate {gate} cannot be asserted without the graph its receipt binds")
    revisions = input_revisions(state, graph, cards, calibration_root)
    artifact = bound_artifact(graph, call_index)
    if g.receipt.artifact_hash != artifact or g.receipt.input_revisions != revisions:
        raise GateError(f"gate {gate} receipt does not match the current inputs")
    for name in CERTIFIABLE[gate][1]:
        prior = state.gates.get(name)
        _require_prerequisite(gate, name, prior, artifact, revisions)
        assert prior is not None
        if prior.token not in g.receipt.prerequisites:
            raise GateError(f"gate {gate} is missing prerequisite checker PASS {name}")
    for name in _chain(gate):
        failures = [
            item for item in _findings_for(name, graph, call_index, cards, calibration_root)
            if getattr(item, "severity", "FAIL") == "FAIL"
        ]
        if failures:
            raise GateError(f"gate {gate} checker chain no longer passes at {name}: {failures[0].check}: {failures[0].message}")


def add_pending(state: State, request_id: str) -> None:
    if request_id not in state.pending_requests:
        state.pending_requests.append(request_id)


def clear_pending(state: State, request_id: str) -> None:
    state.pending_requests = [r for r in state.pending_requests if r != request_id]


def status_text(state: State) -> str:
    lines = [f"decision: {state.slug}", f"stage: {state.stage}", f"graph v{state.versions.graph}, policy v{state.versions.policy}, pass {state.versions.pass_no}"]
    lines.append("gates:")
    for name in GATES_IN_ORDER:
        g = state.gates.get(name)
        if g is None:
            lines.append(f"  {name:<12} -")
        else:
            transports = g.receipt.input_revisions.get("evidence_transports", "?") if g.receipt is not None else "-"
            lines.append(
                f"  {name:<12} {g.authority.value:<14} {g.status.value:<5} {g.at}  token={g.token}  evidence={transports}"
            )
    if state.pending_requests:
        lines.append("pending requests: " + ", ".join(state.pending_requests))
    if state.blockers:
        lines.append("BLOCKERS: " + "; ".join(state.blockers))
    return "\n".join(lines)


def dump(state: State) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(state.model_dump_json())
    return data
