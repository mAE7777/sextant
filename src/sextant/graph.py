"""Versioned graph files, diffs, views, and the well-formedness checks the schema cannot express alone.

Every check returns findings instead of raising, so the self-test can prove each check
catches its planted violation and passes its clean twin.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sextant import wire
from sextant.certificates import certificate_state, read_certificates
from sextant.ledger import exclusive
from sextant.rolecards import RoleCard, compute_hash
from sextant.schema import (
    ACYCLIC_EDGE_TYPES,
    TIER_BUDGETS,
    ContributingCall,
    DecisionGraph,
    Edge,
    EdgeType,
    Estimate,
    Node,
    NodeKind,
    Primitive,
    Tier,
    edge_id,
    node_id,
)

GRAPH_FILE_RE = re.compile(r"^graph\.v(\d+)\.json$")


@dataclass
class Finding:
    check: str
    subject: str
    message: str
    severity: str = "FAIL"


@dataclass
class CheckResult:
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.severity == "FAIL" for f in self.findings)


def load_graph(path: Path) -> DecisionGraph:
    return DecisionGraph.model_validate_json(path.read_text())


def validate_graph_dict(data: dict[str, Any]) -> tuple[DecisionGraph | None, list[Finding]]:
    """Schema-level validation reported as findings (check name: schema)."""
    try:
        return DecisionGraph.model_validate(data), []
    except ValidationError as e:
        findings = [
            Finding("schema", ".".join(str(p) for p in err.get("loc", ())) or "graph", str(err.get("msg")))
            for err in e.errors()
        ]
        return None, findings


def latest_version(decision_dir: Path) -> int:
    versions = [int(m.group(1)) for p in decision_dir.glob("graph.v*.json") if (m := GRAPH_FILE_RE.match(p.name))]
    return max(versions) if versions else 0


def graph_path(decision_dir: Path, version: int) -> Path:
    return decision_dir / f"graph.v{version}.json"


def save_graph(graph: DecisionGraph, decision_dir: Path) -> Path:
    """Write graph.v<N>.json (never overwrite), the three views, and for N > 1 the diff to the parent.

    A version above 1 must name its parent_version, and that parent file must exist. The
    graph file is created exclusively under the directory lock, so two writers cannot both be
    told they saved the same version.
    """
    text = graph.model_dump_json(indent=2, exclude_none=True)
    try:
        DecisionGraph.model_validate_json(text)
    except ValidationError as exc:
        raise ValueError(f"graph v{graph.version} would not load back; refusing to save it: {exc.errors()[0].get('msg')}") from exc
    decision_dir.mkdir(parents=True, exist_ok=True)
    with exclusive(decision_dir):
        path = graph_path(decision_dir, graph.version)
        if path.exists():
            raise FileExistsError(f"{path} exists; bump the version instead of overwriting")
        diff_text: str | None = None
        if graph.version > 1:
            if graph.parent_version is None:
                raise ValueError(f"graph v{graph.version} must name its parent_version")
            parent_path = graph_path(decision_dir, graph.parent_version)
            if not parent_path.exists():
                raise FileNotFoundError(f"parent {parent_path} does not exist")
            parent = load_graph(parent_path)
            diff_text = render_diff(diff_graphs(parent, graph))
        elif graph.parent_version is not None:
            raise ValueError("graph v1 cannot have a parent_version")
        with path.open("x", encoding="utf-8") as f:
            f.write(text)
        if diff_text is not None:
            (decision_dir / f"diff.v{graph.parent_version}-v{graph.version}.md").write_text(diff_text)
        (decision_dir / f"world.v{graph.version}.json").write_text(json.dumps(world_view(graph), indent=2, ensure_ascii=False))
        (decision_dir / f"policy.v{graph.version}.json").write_text(json.dumps(policy_view(graph), indent=2, ensure_ascii=False))
        # objective.json is the CURRENT objective view, re-rendered from every saved version
        (decision_dir / "objective.json").write_text(json.dumps(objective_view(graph), indent=2, ensure_ascii=False))
        (decision_dir / f"objective.v{graph.version}.json").write_text(json.dumps(objective_view(graph), indent=2, ensure_ascii=False))
        return path


# ---------------------------------------------------------------- views

def _nodes_of(graph: DecisionGraph, *kinds: NodeKind) -> list[dict[str, Any]]:
    return [n.model_dump(mode="json", exclude_none=True) for n in graph.nodes if n.kind in kinds]


def world_view(graph: DecisionGraph) -> dict[str, Any]:
    nodes = _nodes_of(
        graph,
        NodeKind.factor,
        NodeKind.evidence,
        NodeKind.evidence_link,
        NodeKind.conflict,
        NodeKind.constraint,
    )
    present = {item["id"] for item in nodes}
    outside = sorted({m for item in nodes for m in item.get("members", []) if m not in present})
    return {
        "slug": graph.slug,
        "version": graph.version,
        "nodes": nodes,
        # members named by nodes in this view whose own node kinds are not part of the view
        "members_outside_view": outside,
        "edges": [
            e.model_dump(mode="json", exclude_none=True)
            for e in graph.edges
            if e.type in (
                EdgeType.influences,
                EdgeType.depends_on,
                EdgeType.cites,
                EdgeType.supports,
                EdgeType.conflicts,
                EdgeType.measured_by,
            )
            and e.from_id in present
            and e.to_id in present
        ],
    }


def objective_view(graph: DecisionGraph) -> dict[str, Any]:
    return {
        "slug": graph.slug,
        "mode": graph.metadata.get("objective_mode", "personal"),
        "criteria": _nodes_of(graph, NodeKind.criterion),
        "values": _nodes_of(graph, NodeKind.value),
        "constraints": _nodes_of(graph, NodeKind.constraint),
        "weights": graph.metadata.get("weights", {}),
        "info_value_line": graph.metadata.get("info_value_line"),
    }


def policy_view(graph: DecisionGraph) -> dict[str, Any]:
    return {
        "slug": graph.slug,
        "version": graph.version,
        "options": _nodes_of(graph, NodeKind.option),
        "probes": _nodes_of(graph, NodeKind.probe),
        "ranking": graph.metadata.get("ranking"),
        "tripwires": graph.metadata.get("tripwires", []),
        "prediction": graph.metadata.get("prediction"),
    }


# ---------------------------------------------------------------- checks

def _node_kinds(graph: DecisionGraph) -> dict[str, NodeKind]:
    return {n.id: n.kind for n in graph.nodes}


def _endpoints_present(graph: DecisionGraph, edge: Edge) -> bool:
    kinds = _node_kinds(graph)
    return edge.from_id in kinds and edge.to_id in kinds


def check_endpoints(graph: DecisionGraph) -> list[Finding]:
    """Edges whose endpoints are missing (possible only for a graph changed after validation)."""
    kinds = _node_kinds(graph)
    return [
        Finding("endpoints", e.id, f"endpoint not in graph ({e.from_id} -> {e.to_id})")
        for e in graph.edges
        if e.from_id not in kinds or e.to_id not in kinds
    ]


def check_acyclic(graph: DecisionGraph) -> list[Finding]:
    """Iterative depth-first search, so a long legal chain cannot exhaust the call stack."""
    adj: dict[str, list[str]] = {n.id: [] for n in graph.nodes}
    for e in graph.edges:
        if e.type in ACYCLIC_EDGE_TYPES and e.from_id in adj and e.to_id in adj:
            adj[e.from_id].append(e.to_id)
    WHITE, GREY, BLACK = 0, 1, 2
    color = {k: WHITE for k in adj}
    findings: list[Finding] = []
    for start in adj:
        if color[start] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        path: list[str] = [start]
        color[start] = GREY
        while stack:
            u, i = stack[-1]
            if i < len(adj[u]):
                stack[-1] = (u, i + 1)
                v = adj[u][i]
                if color[v] == GREY:
                    cycle = path[path.index(v):] + [v]
                    findings.append(Finding("acyclic", u, "cycle: " + " -> ".join(cycle)))
                elif color[v] == WHITE:
                    color[v] = GREY
                    stack.append((v, 0))
                    path.append(v)
            else:
                color[u] = BLACK
                stack.pop()
                path.pop()
    return findings


def sibling_sets(graph: DecisionGraph) -> dict[str, list[Node]]:
    by_id = {n.id: n for n in graph.nodes}
    out: dict[str, list[Node]] = {}
    for e in graph.edges:
        if e.type == EdgeType.decomposes_into and e.to_id in by_id:
            out.setdefault(e.from_id, []).append(by_id[e.to_id])
    return out


def check_residuals(graph: DecisionGraph) -> list[Finding]:
    """Every sibling set has exactly one residual, and only decomposition children are residual.

    A residual stands for "whatever this decomposition did not name". More than one per set,
    or a residual flag anywhere else, turns it into a self-declared exemption from evidence.
    """
    findings: list[Finding] = []
    for parent, kids in sibling_sets(graph).items():
        residuals = [k for k in kids if k.residual]
        if not residuals:
            findings.append(Finding("residual", parent, f"sibling set under {parent} has no residual node"))
        if len(residuals) > 1:
            findings.append(
                Finding("residual", parent, f"sibling set under {parent} has {len(residuals)} residual nodes; one names the remainder")
            )
        if residuals and len(kids) == len(residuals):
            findings.append(Finding("residual", parent, f"sibling set under {parent} is only a residual"))
    children = {e.to_id for e in graph.edges if e.type == EdgeType.decomposes_into}
    for n in graph.nodes:
        if n.residual and n.id not in children:
            findings.append(Finding("residual", n.id, "only a decomposition child can be a residual"))
    return findings


def check_connected(graph: DecisionGraph) -> list[Finding]:
    """Every node reaches a question root through some edge; a graph needs a root at all."""
    kinds = _node_kinds(graph)
    decomposed = {e.to_id for e in graph.edges if e.type == EdgeType.decomposes_into}
    roots = [n.id for n in graph.nodes if n.kind == NodeKind.question and n.id not in decomposed]
    if graph.nodes and not roots:
        return [Finding("connected", "graph", "the graph has no question root")]
    neighbours: dict[str, set[str]] = {n: set() for n in kinds}
    for e in graph.edges:
        if e.from_id in kinds and e.to_id in kinds:
            neighbours[e.from_id].add(e.to_id)
            neighbours[e.to_id].add(e.from_id)
    seen = set(roots)
    frontier = list(roots)
    while frontier:
        u = frontier.pop()
        for v in neighbours[u]:
            if v not in seen:
                seen.add(v)
                frontier.append(v)
    return [
        Finding("connected", n, "isolated: not connected to any question root")
        for n in kinds
        if n not in seen
    ]


def check_escapes(graph: DecisionGraph) -> list[Finding]:
    findings: list[Finding] = []
    for n in graph.nodes:
        if n.answer_space and n.answer_space.primitive == Primitive.choice:
            esc = n.answer_space.escape_option
            if not esc or esc not in (n.answer_space.options or {}):
                findings.append(Finding("escape", n.id, "choice answer space without an escape option"))
    return findings


def check_budgets(graph: DecisionGraph) -> list[Finding]:
    """Depth and breadth only. `error_budget_target` is not an executing budget.

    Depth is the longest decomposition path from a root, computed once per node in
    topological order. A walk over every path is exponential in shared children; a
    recursive one exhausts the stack on long chains.
    """
    findings: list[Finding] = []
    if graph.budgets is not None:
        budgets = graph.budgets
    elif graph.frame is not None:
        budgets = TIER_BUDGETS[graph.frame.tier]
    else:
        budgets = TIER_BUDGETS[Tier.T2]
        findings.append(Finding("budget", "graph", "no frame and no budgets on the graph; T2 budgets assumed", severity="WARN"))
    sets_ = sibling_sets(graph)
    for parent, kids in sets_.items():
        n_real = sum(1 for k in kids if not k.residual)
        if n_real > budgets.breadth_max:
            findings.append(Finding("budget", parent, f"{n_real} children exceed breadth_max {budgets.breadth_max}"))
    kinds = _node_kinds(graph)
    children: dict[str, list[str]] = {n: [] for n in kinds}
    indegree: dict[str, int] = {n: 0 for n in kinds}
    for e in graph.edges:
        if e.type == EdgeType.decomposes_into and e.from_id in kinds and e.to_id in kinds:
            children[e.from_id].append(e.to_id)
            indegree[e.to_id] += 1
    order: list[str] = []
    ready = [n for n, d in indegree.items() if d == 0]
    remaining = dict(indegree)
    while ready:
        u = ready.pop()
        order.append(u)
        for v in children[u]:
            remaining[v] -= 1
            if remaining[v] == 0:
                ready.append(v)
    if len(order) < len(kinds):
        findings.append(
            Finding("budget", "graph", "depth not computed: the decomposition edges contain a cycle", severity="WARN")
        )
        return findings
    roots = {n.id for n in graph.nodes if n.kind == NodeKind.question and indegree[n.id] == 0}
    depth: dict[str, int] = {}
    for u in order:
        if u in roots:
            depth[u] = 0
        if u not in depth:
            continue
        for v in children[u]:
            depth[v] = max(depth.get(v, 0), depth[u] + 1)
    for nid, d in depth.items():
        if d > budgets.depth_max:
            findings.append(Finding("budget", nid, f"depth {d} exceeds depth_max {budgets.depth_max}"))
    return findings


def check_value_nodes(graph: DecisionGraph) -> list[Finding]:
    findings: list[Finding] = []
    for n in graph.nodes:
        if n.kind == NodeKind.value and (n.estimate or n.sufficiency or n.state_vector):
            findings.append(Finding("value-node", n.id, "a value node carries a Jev estimate"))
    return findings


def check_reified(graph: DecisionGraph) -> list[Finding]:
    findings: list[Finding] = []
    for n in graph.nodes:
        if n.kind == NodeKind.evidence_link:
            supports = sorted(e.to_id for e in graph.edges if e.type == EdgeType.supports and e.from_id == n.id)
            cites = sorted(e.to_id for e in graph.edges if e.type == EdgeType.cites and e.from_id == n.id)
            targets = sorted(set(supports) | set(cites))
            if len(supports) != 1 or len(cites) != 1 or targets != sorted(set(n.members)):
                findings.append(
                    Finding(
                        "reified",
                        n.id,
                        "evidence_link members do not match its supports and cites edges",
                    )
                )
        if n.kind == NodeKind.conflict:
            targets = sorted(e.to_id for e in graph.edges if e.type == EdgeType.conflicts and e.from_id == n.id)
            if len(set(n.members)) != 2 or targets != sorted(n.members):
                findings.append(Finding("reified", n.id, "conflict edges do not match its two distinct members"))
    return findings


def check_resolution(graph: DecisionGraph) -> list[Finding]:
    """Redundant with the schema validator on purpose: a check must exist as a check."""
    findings: list[Finding] = []
    for n in graph.nodes:
        unresolvable = n.resolution is None or not n.resolution.criteria.strip()
        if n.kind in (NodeKind.factor, NodeKind.probe) and n.estimate is not None and unresolvable:
            findings.append(Finding("resolution", n.id, "estimate without resolution criteria"))
    return findings


VERIFICATION_FIELDS = ("relevance", "concreteness_gain", "skipped_level", "edge_type")
# Which card question each verification field must come from.
VERIFICATION_SOURCES = {
    "relevance": ("R3", "relevance"),
    "concreteness_gain": ("R3", "more_concrete"),
    "skipped_level": ("R3", "skipped_level"),
    "edge_type": ("R3", "edge_type"),
}

CallIndex = dict[str, dict[str, Any]]


def judgment_sites(graph: DecisionGraph) -> list[tuple[str, str, Estimate]]:
    """Every place in a graph that carries a Jev judgment.

    Auditing only `node.estimate` and `node.sufficiency` left `node.state_vector` and
    `edge.verification` unchecked, so a state vector could cite a call that never ran and a
    verified edge could cite another question's answer.
    """
    sites: list[tuple[str, str, Estimate]] = []
    for n in graph.nodes:
        if n.estimate is not None:
            sites.append((n.id, "estimate", n.estimate))
        if n.sufficiency is not None:
            sites.append((n.id, "sufficiency", n.sufficiency))
        for key, est in (n.state_vector or {}).items():
            sites.append((n.id, f"state_vector.{key}", est))
    for e in graph.edges:
        if e.verification is None:
            continue
        for name in VERIFICATION_FIELDS:
            est = getattr(e.verification, name)
            if est is not None:
                sites.append((e.id, f"verification.{name}", est))
    return sites


def runs_of(call_index: CallIndex) -> dict[str, list[dict[str, Any]]]:
    """Every recorded call grouped by run, retracted ones included, in recorded order."""
    runs: dict[str, list[dict[str, Any]]] = {}
    for row in call_index.values():
        if not isinstance(row, dict) or row.get("_ambiguous") or "run_id" not in row:
            continue
        runs.setdefault(str(row["run_id"]), []).append(row)
    for rows in runs.values():
        rows.sort(key=_seq_of)
    return runs


def _seq_of(row: dict[str, Any]) -> int:
    seq = row.get("seq")
    return seq if isinstance(seq, int) else 0


def _keys_of(row: dict[str, Any]) -> list[Any]:
    keys = row.get("question_keys")
    return keys if isinstance(keys, list) else []


def ambiguous_calls(call_index: CallIndex) -> list[str]:
    """Call ids recorded more than once. Their rows cannot be told apart, so their runs cannot be seen."""
    return sorted(cid for cid, row in call_index.items() if isinstance(row, dict) and row.get("_ambiguous"))


def _sdk_of(row: dict[str, Any], key: str) -> dict[str, Any]:
    found = ((row.get("request") or {}).get("questions") or {}).get(key)
    return found if isinstance(found, dict) else {}


def _payload_findings(fail: Any, row: dict[str, Any]) -> list[Finding]:
    """The recorded request must hash to what the row says was sent (`wire.payload_problems`)."""
    return [fail(p) for p in wire.payload_problems(row)]


def _run_problems(run: list[dict[str, Any]], run_id: str) -> list[str]:
    """A run must be whole: one primary, the protocol's re-runs and permutations, no retractions."""
    problems: list[str] = []
    if any(r.get("_retracted") for r in run):
        problems.append(f"run {run_id} contains a retracted call; a corrected run is not evidence")
    primaries = [r for r in run if r.get("call_role") == "primary"]
    if len(primaries) != 1:
        problems.append(f"run {run_id} has {len(primaries)} primary calls")
        return problems
    for field_name in ("card_id", "state_sha", "model_id", "transport", "protocol_version", "input_language"):
        if len({str(r.get(field_name)) for r in run}) != 1:
            problems.append(f"run {run_id} mixes values of {field_name}")
    protocol = primaries[0].get("run_protocol") or {}
    if any((r.get("run_protocol") or {}) != protocol for r in run):
        problems.append(f"run {run_id} changed its re-test settings part way")
    active = [r for r in run if not r.get("_retracted")]
    for key in primaries[0].get("question_keys") or []:
        kind = _sdk_of(primaries[0], key).get("type")
        reruns = sum(1 for r in active if r.get("call_role") == "rerun" and key in (r.get("question_keys") or []))
        perms = sum(1 for r in active if r.get("call_role") == "permutation" and key in (r.get("question_keys") or []))
        if kind == "noul" and reruns != int(protocol.get("reruns", -1)):
            problems.append(f"run {run_id}: {key} has {reruns} re-runs, the protocol requires {protocol.get('reruns')}")
        if kind == "choice":
            if perms != max(0, int(protocol.get("permutations", 0)) - 1):
                problems.append(f"run {run_id}: {key} has {perms} permutations, the protocol requires {int(protocol.get('permutations', 0)) - 1}")
            vectors_rows = [r for r in active if r.get("call_role") in ("primary", "permutation") and key in (r.get("question_keys") or [])]
            owed = wire.derive_question(vectors_rows, key).contested
            if reruns != (int(protocol.get("reruns", 0)) if owed else 0):
                problems.append(f"run {run_id}: {key} has {reruns} stability re-runs, owed {int(protocol.get('reruns', 0)) if owed else 0}")
    return problems


def _run_findings(fail: Any, run: list[dict[str, Any]], est: Estimate) -> list[Finding]:
    return [fail(p) for p in _run_problems(run, est.run_id)]


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return isclose(a, b, abs_tol=1e-9)


def _derivation_findings(fail: Any, run: list[dict[str, Any]], est: Estimate) -> list[Finding]:
    """The judgment must equal what its whole run derives to, field by field."""
    active = [r for r in run if not r.get("_retracted")]
    d = wire.derive_question(active, est.question_key)
    if d.problems:
        return [fail(f"cannot be derived from its run: {d.problems[0]}")]
    findings: list[Finding] = []
    derived_contrib = [(c.call_id, c.role, c.variant, c.rendered_hash, c.contributes_to_value) for c in d.contributions]
    stated_contrib = [(c.call_id, c.role, c.variant, c.rendered_hash, c.contributes_to_value) for c in est.contributions]
    if derived_contrib != stated_contrib:
        findings.append(fail("names calls other than every call of its run, in their roles"))
    elif any(not _close(a.value, b.value) for a, b in zip(d.contributions, est.contributions, strict=True)):
        findings.append(fail("records per-call values its run did not answer"))
    if d.aggregation != est.aggregation:
        findings.append(fail(f"declares aggregation {est.aggregation}; its run aggregates as {d.aggregation}"))
    if d.primitive != est.primitive.value:
        findings.append(fail(f"is a {est.primitive.value} judgment of a {d.primitive} question"))
    if not _close(d.p, est.p) or not _close(d.score, est.score) or d.choice != est.choice:
        findings.append(fail(f"does not match the ledger answers ({d.aggregation} replay gives {d.p if d.primitive == 'noul' else d.choice if d.primitive == 'choice' else d.score})"))
    stored = est.probabilities or {}
    replayed = d.probabilities or {}
    if set(stored) != set(replayed) or any(not _close(stored[o], replayed[o]) for o in stored):
        findings.append(fail("records probabilities its run does not replay to"))
    if d.contested != est.contested:
        findings.append(fail(f"records contested={est.contested}; its run derives contested={d.contested}"))
    st = est.stability
    expected = {
        "rerun_sd": d.rerun_sd, "permutation_sd": d.permutation_sd, "paraphrase_sd": d.paraphrase_sd,
        "negation_gap": wire.negation_gap(active, est.question_key) if d.primitive == "noul" else None,
    }
    for name, value in expected.items():
        if not _close(getattr(st, name), value):
            findings.append(fail(f"records stability {name}={getattr(st, name)}; its run derives {value}"))
    for name, count in (("n_reruns", d.n_reruns), ("n_permutations", d.n_permutations), ("n_paraphrases", d.n_paraphrases)):
        if getattr(st, name) != count:
            findings.append(fail(f"records {name}={getattr(st, name)}; its run made {count}"))
    if est.sufficiency_question_key is not None:
        twin = wire.derive_question(active, est.sufficiency_question_key)
        if twin.problems or not _close(twin.p, est.sufficiency_p):
            findings.append(fail(f"copies sufficiency {est.sufficiency_p} from {est.sufficiency_question_key}, which its run derives as {twin.p}"))
    return findings


def _card_findings(fail: Any, row: dict[str, Any], est: Estimate, cards: dict[str, RoleCard]) -> list[Finding]:
    card = cards.get(est.card_id)
    base = est.question_key.split("[")[0]
    if card is None:
        return [fail(f"names card {est.card_id}, which is not loaded")]
    if base not in card.questions:
        return [fail(f"names question {base}, which card {card.id} does not have")]
    findings: list[Finding] = []
    if compute_hash(card, base) != est.question_hash:
        findings.append(fail(f"was asked under seal {est.question_hash}; the card is now sealed {compute_hash(card, base)}"))
    instructions = str(_sdk_of(row, est.question_key).get("instructions") or "")
    for part, name in ((card.instructions.strip(), "instructions"), (card.purpose.strip(), "purpose")):
        if part not in instructions:
            findings.append(fail(f"was sent without the card's current {name}"))
    if f"Answer language: {est.language}" not in instructions:
        findings.append(fail(f"claims language {est.language}, which the recorded instructions do not carry"))
    if (row.get("run_protocol") or {}) != wire.run_protocol(card):
        findings.append(fail("ran under re-test settings the card no longer has"))
    if est.sufficiency_question_key is not None:
        twin = card.questions.get(est.sufficiency_question_key.split("[")[0])
        if twin is None or twin.role != "sufficiency":
            findings.append(fail(f"copies sufficiency from {est.sufficiency_question_key}, which is not the card's sufficiency question"))
    return findings


def _identity_findings(fail: Any, row: dict[str, Any], est: Estimate, call: ContributingCall) -> list[Finding]:
    """One contributing call must match the judgment on every identity the protocol binds."""
    findings: list[Finding] = []
    recorded = row.get("protocol_version")
    if not recorded:
        return [fail(f"call {call.call_id} has no wire protocol version; it predates the protocol and cannot be evidence")]
    if recorded != est.protocol_version:
        findings.append(fail(f"call {call.call_id} used wire protocol {recorded}, the judgment claims {est.protocol_version}"))
    for field_name, value in (("card_id", est.card_id), ("model_id", est.model_id), ("transport", est.transport),
                              ("state_sha", est.state_sha), ("run_id", est.run_id)):
        if row.get(field_name) != value:
            findings.append(fail(f"call {call.call_id} {field_name} is {row.get(field_name)}, the judgment claims {value}"))
    if row.get("call_role") != call.role or row.get("variant") != call.variant:
        findings.append(fail(f"call {call.call_id} was a {row.get('call_role')} ({row.get('variant')}), the judgment says {call.role} ({call.variant})"))
    if est.question_key not in (row.get("question_keys") or []):
        findings.append(fail(f"call {call.call_id} never asked question key {est.question_key}"))
        return findings
    binding = (row.get("question_bindings") or {}).get(est.question_key) or {}
    if binding.get("template_hash") != est.question_hash:
        findings.append(fail(f"call {call.call_id} sealed question hash mismatch"))
    if binding.get("rendered_hash") != call.rendered_hash:
        findings.append(fail(f"call {call.call_id} sent a different rendered question than the judgment records"))
    return findings


def _other_runs(runs: dict[str, list[dict[str, Any]]], est: Estimate) -> list[str]:
    """Every other run in this ledger that asked the same question of the same input bytes.

    Transport label, model and protocol are deliberately not part of the match: a label is free
    text, and a run under another one is still an answer to the same question about the same
    input. The rule is per ledger; a fresh decision directory is a fresh ledger.
    """
    others: list[str] = []
    for run_id, rows in runs.items():
        if run_id == est.run_id:
            continue
        if any(
            r.get("card_id") == est.card_id and r.get("state_sha") == est.state_sha
            and est.question_key in _keys_of(r)
            for r in rows
        ):
            others.append(run_id)
    return sorted(others)


def audit_judgment(
    subject: str,
    label: str,
    est: Estimate,
    call_index: CallIndex,
    runs: dict[str, list[dict[str, Any]]],
    cards: dict[str, RoleCard] | None,
    calibration_root: Path | None = None,
) -> list[Finding]:
    """Everything a judgment claims, checked against the calls of its run."""

    def fail(message: str) -> Finding:
        return Finding("ledger", subject, f"{label} {message}")

    findings: list[Finding] = []
    if est.protocol_version != wire.WIRE_PROTOCOL_VERSION:
        findings.append(fail(
            f"was recorded under wire protocol {est.protocol_version}; the current protocol is "
            f"{wire.WIRE_PROTOCOL_VERSION}, so it is not current evidence"
        ))
    if not est.contributions:
        return [*findings, fail("has no contributing calls")]
    for call in est.contributions:
        row = call_index.get(call.call_id)
        if row is None:
            findings.append(fail(f"call {call.call_id} not in jev-calls ledger"))
            continue
        if row.get("_ambiguous"):
            findings.append(fail(f"call {call.call_id} is recorded {row['_ambiguous']} times; the evidence is ambiguous"))
            continue
        if row.get("_retracted"):
            findings.append(fail(f"call {call.call_id} was retracted"))
        findings.extend(_identity_findings(fail, row, est, call))
        if est.question_key in (row.get("question_keys") or []):
            findings.extend(_payload_findings(fail, row))
    if any(f.severity == "FAIL" for f in findings):
        return findings
    run = runs.get(est.run_id, [])
    if not run:
        return [*findings, fail(f"run {est.run_id} has no calls in the ledger")]
    malformed = [p for r in run for p in wire.row_shape_problems(r)]
    if malformed:
        return [*findings, fail(f"run {est.run_id} has a malformed call row: {malformed[0]}")]
    findings.extend(_run_findings(fail, run, est))
    findings.extend(_derivation_findings(fail, run, est))
    primary = next((r for r in run if r.get("call_role") == "primary"), None)
    if primary is not None and primary.get("input_language") != est.input_language:
        findings.append(fail(f"claims input language {est.input_language!r}; its run declared {primary.get('input_language')!r}"))
    if cards is not None and primary is not None:
        findings.extend(_card_findings(fail, primary, est, cards))
    if calibration_root is not None and est.threshold_ref is not None:
        ref = est.threshold_ref
        issued = [
            c for c in read_certificates(calibration_root, est.card_id)
            if c.key == ref.key and c.status == "validated" and c.value == ref.value
            and c.experiment_id == ref.experiment_id and c.at == ref.fitted_on
        ]
        if ref.card_id != est.card_id or ref.key != est.question_key.split("[")[0] or not issued:
            findings.append(fail(f"records threshold {ref.value} ({ref.experiment_id}), which no issued certificate for {est.card_id}.{ref.key} carries"))
    if calibration_root is not None and cards is not None and est.calibration_status.state == "calibrated":
        card = cards.get(est.card_id)
        if card is not None:
            verdict = certificate_state(
                card, est.question_key.split("[")[0], model_id=est.model_id, transport=est.transport,
                language=est.language, root=calibration_root, asked_keys=wire.asked_base_keys(run),
                input_language=_input_language_of_run(run),
            )
            if verdict.state != "calibrated":
                findings.append(fail(f"claims calibrated; its certificate is now {verdict.state}: {verdict.reason}"))
    others = _other_runs(runs, est)
    if others:
        findings.append(Finding("ledger", subject, f"{label} is one of {len(others) + 1} runs on this input ({', '.join(others[:3])})", severity="WARN"))
    return findings


def _answer_space_findings(node: Node, runs: dict[str, list[dict[str, Any]]]) -> list[Finding]:
    """A node's judgment must have been asked over the node's own declared answer space."""
    est, space = node.estimate, node.answer_space
    assert est is not None and space is not None

    def fail(message: str) -> Finding:
        return Finding("ledger", node.id, f"estimate {message}")

    if est.primitive != space.primitive:
        return [fail(f"is a {est.primitive.value} judgment on a {space.primitive.value} answer space")]
    primary = next((r for r in runs.get(est.run_id, []) if r.get("call_role") == "primary"), None)
    if primary is None or space.primitive == Primitive.noul:
        return []
    sdk = _sdk_of(primary, est.question_key)
    if space.primitive == Primitive.choice and sorted(wire.declared_options(sdk)) != sorted(space.options or {}):
        return [fail(f"was asked over options {sorted(wire.declared_options(sdk))}, the node declares {sorted(space.options or {})}")]
    if space.primitive == Primitive.score and list(sdk.get("criteria") or []) != list(space.levels or []):
        return [fail("was asked over other levels than the node declares")]
    return []


def check_against_ledger(
    graph: DecisionGraph,
    call_index: CallIndex,
    cards: dict[str, RoleCard] | None = None,
    calibration_root: Path | None = None,
) -> list[Finding]:
    """Re-derive every judgment in the graph from the complete run that produced it."""
    runs = runs_of(call_index)
    findings: list[Finding] = []
    ambiguous = ambiguous_calls(call_index)
    if ambiguous:
        findings.append(Finding(
            "ledger", "ledger",
            f"{len(ambiguous)} call id(s) are recorded more than once (first {ambiguous[0]}); a duplicated run "
            "cannot be told apart, so it could hide from the same-input comparison. Resolve the duplicates first",
        ))
    for subject, label, est in judgment_sites(graph):
        findings.extend(_guarded("ledger", audit_judgment, subject, label, est, call_index, runs, cards, calibration_root))
    for n in graph.nodes:
        if n.estimate is not None and n.answer_space is not None:
            findings.extend(_guarded("ledger", _answer_space_findings, n, runs))
    if cards is None and judgment_sites(graph):
        findings.append(Finding("ledger", "graph", "card binding was not checked: no role cards were supplied", severity="WARN"))
    return findings


def check_ids(graph: DecisionGraph) -> list[Finding]:
    """Content-addressed ids: a node id is the hash of (kind, text_en, definition); an edge id of (type, from, to)."""
    findings: list[Finding] = []
    for n in graph.nodes:
        expected = node_id(n.kind, n.text_en, n.definition)
        if n.id != expected:
            findings.append(Finding("ids", n.id, f"node id does not match its content (expected {expected})"))
    for e in graph.edges:
        expected = edge_id(e.type, e.from_id, e.to_id)
        if e.id != expected:
            findings.append(Finding("ids", e.id, f"edge id does not match its endpoints (expected {expected})"))
    return findings


def check_unique(graph: DecisionGraph) -> list[Finding]:
    """Node and edge ids are unique. Loading refuses duplicates; a graph built in memory may not."""
    findings: list[Finding] = []
    for what, ids in (("node", [n.id for n in graph.nodes]), ("edge", [e.id for e in graph.edges])):
        seen: set[str] = set()
        for item in ids:
            if item in seen:
                findings.append(Finding("unique", item, f"{what} id appears more than once"))
            seen.add(item)
    return findings


STRUCTURAL_CHECKS = {
    "unique": check_unique,
    "endpoints": check_endpoints,
    "ids": check_ids,
    "acyclic": check_acyclic,
    "residual": check_residuals,
    "connected": check_connected,
    "escape": check_escapes,
    "budget": check_budgets,
    "value-node": check_value_nodes,
    "reified": check_reified,
    "resolution": check_resolution,
}


def _guarded(name: str, fn: Any, *args: Any, **kwargs: Any) -> list[Finding]:
    """A check that raises reports a FAIL naming itself; a checker never crashes its caller."""
    try:
        found: list[Finding] = fn(*args, **kwargs)
        return found
    except Exception as exc:  # noqa: BLE001 - any failure of a check is a finding, not a crash
        return [Finding(name, "graph", f"check raised {type(exc).__name__}: {str(exc)[:300]}")]


def check_all(
    graph: DecisionGraph,
    call_index: CallIndex | None = None,
    cards: dict[str, RoleCard] | None = None,
    calibration_root: Path | None = None,
) -> CheckResult:
    result = CheckResult()
    for name, fn in STRUCTURAL_CHECKS.items():
        result.findings.extend(_guarded(name, fn, graph))
    if call_index is not None:
        result.findings.extend(_guarded("ledger", check_against_ledger, graph, call_index, cards, calibration_root))
    return result


# ---------------------------------------------------------------- edge evidence and release

# The only edge type with an evidence protocol in this version: a calibrated R3 `relevance`
# decision ("the child's answer contributes directly to the parent's") bound to the edge's own
# endpoints. A later redefinition of R3 replaces this rule; other edge types have no protocol yet.
EDGE_EVIDENCE_PROTOCOLS: dict[EdgeType, tuple[str, str, str]] = {
    EdgeType.decomposes_into: ("relevance", "R3", "relevance"),
}

EDGE_STATES = (
    "proposed",                        # the graph is not structurally valid, or the edge failed
    "structurally_valid",              # no usable judgment about this edge
    "reviewed",                        # a judgment is recorded but no ledger was supplied
    "reconciled",                      # the judgment replays and is about this edge, but decides nothing
    "direct_contribution_supported",   # a calibrated R3 relevance decision says yes, on every run
    "direct_contribution_rejected",    # a calibrated R3 relevance decision says no
    "externally_tested",               # reserved for external evidence; never produced now
)


def verification_estimates(edge: Edge) -> list[Estimate]:
    if edge.verification is None:
        return []
    found: list[Estimate] = []
    for name in VERIFICATION_FIELDS:
        est = getattr(edge.verification, name)
        if isinstance(est, Estimate):
            found.append(est)
    return found


def _structural_failures(graph: DecisionGraph) -> list[Finding]:
    findings: list[Finding] = []
    for name, fn in STRUCTURAL_CHECKS.items():
        findings.extend(item for item in _guarded(name, fn, graph) if item.severity == "FAIL")
    return findings


def _decision_of_run(
    rows: list[dict[str, Any]],
    key: str,
    card: RoleCard,
    root: Path,
) -> tuple[bool | None, str]:
    """The calibrated decision one run reaches on `key`, or None with the reason it cannot.

    A run that cannot even be read decides nothing, and so blocks release like any other run
    that does not reach the same calibrated decision.
    """
    try:
        return _decide_run(rows, key, card, root)
    except Exception as exc:  # noqa: BLE001 - an unreadable run is a reason, not a crash
        return None, f"a run on this input cannot be read ({type(exc).__name__}: {str(exc)[:120]})"


def _decide_run(
    rows: list[dict[str, Any]],
    key: str,
    card: RoleCard,
    root: Path,
) -> tuple[bool | None, str]:
    if any(r.get("_retracted") for r in rows):
        return None, "a run on this input contains a retracted call"
    active = [r for r in rows if not r.get("_retracted")]
    d = wire.derive_question(active, key)
    if d.problems:
        return None, f"a run on this input is incomplete or invalid: {d.problems[0]}"
    primary = next((r for r in active if r.get("call_role") == "primary"), {})
    verdict = certificate_state(
        card, key, model_id=str(primary.get("model_id")), transport=str(primary.get("transport")),
        language=wire.answer_language(primary, key), root=root, asked_keys=wire.asked_base_keys(active),
        input_language=_input_language_of_run(active),
    )
    if verdict.state != "calibrated" or verdict.certificate is None or verdict.certificate.value is None:
        return None, f"not calibrated ({verdict.state}: {verdict.reason})"
    if d.p is None:
        return None, "the run has no value"
    answer = wire.decision(d.p, verdict.certificate.value, contested=d.contested)
    if answer is None:
        return None, f"p={d.p:.3f} abstains at the certified threshold {verdict.certificate.value}" + (
            " (the run is contested)" if d.contested else ""
        )
    return answer, "decided"


def _agrees(rows: list[dict[str, Any]], run_id: str, key: str, threshold: float) -> tuple[bool | None, str]:
    """What another run on the same input answers at the certified threshold, if it is whole.

    It need not be in the certificate's domain; it only has to be a complete, clean run whose
    answer under the one decision rule can be compared.
    """
    try:
        problems = [p for r in rows for p in wire.row_shape_problems(r)] or _run_problems(rows, run_id)
        if problems:
            return None, problems[0]
        d = wire.derive_question(rows, key)
        if d.problems or d.p is None:
            return None, f"a run on this input is incomplete or invalid: {(d.problems or ['no value'])[0]}"
        answer = wire.decision(d.p, threshold, contested=d.contested)
        if answer is None:
            return None, f"another run on this input abstains (p={d.p:.3f})"
        return answer, "decided"
    except Exception as exc:  # noqa: BLE001 - an unreadable run blocks, it does not crash release
        return None, f"a run on this input cannot be read ({type(exc).__name__})"


def _input_language_of_run(rows: list[dict[str, Any]]) -> str | None:
    """The run's effective input language: its declaration, if its state is in that script."""
    primary = next((r for r in rows if r.get("call_role") == "primary"), None)
    if primary is None:
        return None
    return wire.input_language_of((primary.get("request") or {}).get("state"), primary.get("input_language"))


def edge_evidence(
    graph: DecisionGraph,
    call_index: CallIndex | None = None,
    cards: dict[str, RoleCard] | None = None,
    calibration_root: Path | None = None,
) -> dict[str, tuple[str, str]]:
    """Each edge's usability state with the reason, derived from evidence and never stored."""
    structural = _structural_failures(graph)
    ledger_failures = (
        {item.subject for item in check_against_ledger(graph, call_index, cards, calibration_root) if item.severity == "FAIL"}
        if call_index is not None
        else set()
    )
    runs = runs_of(call_index) if call_index is not None else {}
    by_id = {n.id: n for n in graph.nodes}
    out: dict[str, tuple[str, str]] = {}
    for edge in graph.edges:
        if structural:
            out[edge.id] = ("proposed", "the graph is not structurally valid")
            continue
        if edge.status == "failed":
            out[edge.id] = ("proposed", "the edge is marked failed")
            continue
        if not verification_estimates(edge):
            out[edge.id] = ("structurally_valid", "no judgment about this edge")
            continue
        if call_index is None:
            out[edge.id] = ("reviewed", "a judgment is recorded but no ledger was supplied")
            continue
        if edge.id in ledger_failures:
            out[edge.id] = ("structurally_valid", "its judgment does not replay against the ledger")
            continue
        protocol = EDGE_EVIDENCE_PROTOCOLS.get(edge.type)
        wrong_source = [
            name for name in VERIFICATION_FIELDS
            if (est := getattr(edge.verification, name)) is not None
            and (est.card_id, est.question_key) != VERIFICATION_SOURCES[name]
        ]
        if wrong_source:
            out[edge.id] = ("structurally_valid", f"verification {', '.join(wrong_source)} did not come from its declared R3 question")
            continue
        if protocol is None:
            out[edge.id] = ("reconciled", f"no evidence protocol for {edge.type.value} edges in this version")
            continue
        field_name, card_id, key = protocol
        est = getattr(edge.verification, field_name)
        if est is None:
            out[edge.id] = ("reconciled", f"no {field_name} judgment, which is the evidence protocol for this edge type")
            continue
        parent, child = by_id[edge.from_id], by_id[edge.to_id]
        primary = next((r for r in runs.get(est.run_id, []) if r.get("call_role") == "primary"), {})
        state = (primary.get("request") or {}).get("state") or {}
        if (
            state.get("parent_question") != parent.text_en or state.get("child_question") != child.text_en
            or state.get("parent_kind") != parent.kind.value or state.get("child_kind") != child.kind.value
        ):
            out[edge.id] = ("structurally_valid", "its judgment was asked about other questions than this edge's endpoints")
            continue
        if parent.definition.strip() or child.definition.strip():
            out[edge.id] = ("reconciled", "an endpoint carries a definition, which the R3 pack does not show the model")
            continue
        if cards is None or calibration_root is None or card_id not in cards:
            out[edge.id] = ("reconciled", "cards and a calibration root are needed to read a calibrated decision")
            continue
        card = cards[card_id]
        cited, reason = _decision_of_run(runs[est.run_id], key, card, calibration_root)
        others = _other_runs(runs, est)
        if cited is None:
            out[edge.id] = ("reconciled", reason if not others else f"{len(others) + 1} runs on this input; {reason}")
            continue
        verdict = certificate_state(
            card, key, model_id=est.model_id, transport=est.transport, language=est.language,
            root=calibration_root, asked_keys=wire.asked_base_keys(runs[est.run_id]),
            input_language=_input_language_of_run(runs[est.run_id]),
        )
        if verdict.certificate is None or verdict.certificate.value is None:
            out[edge.id] = ("reconciled", "the certificate that decided the cited run cannot be read again")
            continue
        blocked = None
        for other in others:
            answer, why = _agrees(runs[other], other, key, verdict.certificate.value)
            if answer is None:
                blocked = f"{len(others) + 1} runs on this input; {why}"
                break
            if answer != cited:
                blocked = f"{len(others) + 1} runs on this input disagree"
                break
        if blocked is not None:
            out[edge.id] = ("reconciled", blocked)
        elif cited:
            out[edge.id] = ("direct_contribution_supported", f"calibrated yes; {len(others)} other run(s) agree")
        else:
            out[edge.id] = ("direct_contribution_rejected", f"calibrated no; {len(others)} other run(s) agree")
    return out


def derive_edge_states(
    graph: DecisionGraph,
    call_index: CallIndex | None = None,
    cards: dict[str, RoleCard] | None = None,
    calibration_root: Path | None = None,
) -> dict[str, str]:
    """Usable edge state, derived from evidence. A stored label cannot raise it."""
    try:
        evidence = edge_evidence(graph, call_index, cards, calibration_root)
    except Exception:  # noqa: BLE001 - evidence that cannot be derived supports nothing
        return {e.id: "proposed" for e in graph.edges}
    return {eid: state for eid, (state, _) in evidence.items()}


def _residual_decomposition(graph: DecisionGraph, edge: Edge) -> bool:
    if edge.type != EdgeType.decomposes_into:
        return False
    try:
        return graph.node(edge.to_id).residual
    except KeyError:
        return False


def check_release(
    graph: DecisionGraph,
    call_index: CallIndex | None = None,
    cards: dict[str, RoleCard] | None = None,
    calibration_root: Path | None = None,
) -> list[Finding]:
    """A draft may exist. It may not pass a release check without calibrated evidence per edge."""
    findings: list[Finding] = []
    for missing, what in ((call_index is None, "call ledger"), (cards is None, "role cards"), (calibration_root is None, "calibration root")):
        if missing:
            findings.append(Finding("release", "graph", f"no {what} was supplied; missing evidence cannot be released"))
    if findings:
        return findings
    assert call_index is not None
    ambiguous = ambiguous_calls(call_index)
    if ambiguous:
        findings.append(Finding("release", "ledger", f"{len(ambiguous)} call id(s) are recorded more than once; an ambiguous ledger cannot support release"))
    try:
        evidence = edge_evidence(graph, call_index, cards, calibration_root)
    except Exception as exc:  # noqa: BLE001 - release fails closed with the reason, it never crashes
        return [*findings, Finding("release", "graph", f"edge evidence could not be derived: {type(exc).__name__}: {str(exc)[:200]}")]
    supported = 0
    for edge in graph.edges:
        if _residual_decomposition(graph, edge):
            continue
        if edge.type not in EDGE_EVIDENCE_PROTOCOLS:
            findings.append(Finding("release", edge.id, f"no evidence protocol for this edge type in this version ({edge.type.value})"))
            continue
        state, reason = evidence.get(edge.id, ("proposed", "unknown edge"))
        if state == "direct_contribution_supported":
            supported += 1
            continue
        findings.append(Finding("release", edge.id, f"usability is {state}: {reason}; release needs a calibrated direct-contribution decision"))
    ledger_failures = [item for item in check_against_ledger(graph, call_index, cards, calibration_root) if item.severity == "FAIL"]
    if ledger_failures:
        findings.append(Finding("release", "graph", "a judgment site does not replay against the ledger"))
    if not supported:
        findings.append(Finding("release", "graph", "draft has no edge supported by calibrated evidence"))
    return findings


# ---------------------------------------------------------------- diff

def _dump(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", warnings=False)
    if isinstance(value, dict):
        return {str(key): _dump(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_dump(item) for item in value]
    return value


# Graph-level fields that are identity or bookkeeping of the version itself, not content.
_GRAPH_IDENTITY = ("nodes", "edges", "version", "parent_version", "created", "diff_summary")


def diff_graphs(a: DecisionGraph, b: DecisionGraph) -> dict[str, Any]:
    """Every field of every shared node and edge, every graph-level field and metadata key.

    Listing the fields to compare was how a residual flip, a constraint predicate and a frame
    change went unseen. Comparing whole dumps cannot miss a field added later.
    """
    a_nodes = {n.id: n for n in a.nodes}
    b_nodes = {n.id: n for n in b.nodes}
    a_edges = {e.id: e for e in a.edges}
    b_edges = {e.id: e for e in b.edges}
    changed: list[dict[str, Any]] = []
    for nid in sorted(a_nodes.keys() & b_nodes.keys()):
        before_node, after_node = _dump(a_nodes[nid]), _dump(b_nodes[nid])
        for name in sorted(set(before_node) | set(after_node)):
            if before_node.get(name) != after_node.get(name):
                changed.append({"node": nid, name: {"before": before_node.get(name), "after": after_node.get(name)}})
    for eid in sorted(a_edges.keys() & b_edges.keys()):
        before_edge, after_edge = _dump(a_edges[eid]), _dump(b_edges[eid])
        for name in sorted(set(before_edge) | set(after_edge)):
            if before_edge.get(name) != after_edge.get(name):
                changed.append({"edge": eid, name: {"before": before_edge.get(name), "after": after_edge.get(name)}})
    before_graph = {k: v for k, v in _dump(a).items() if k not in _GRAPH_IDENTITY and k != "metadata"}
    after_graph = {k: v for k, v in _dump(b).items() if k not in _GRAPH_IDENTITY and k != "metadata"}
    for name in sorted(set(before_graph) | set(after_graph)):
        if before_graph.get(name) != after_graph.get(name):
            changed.append({"graph": name, name: {"before": before_graph.get(name), "after": after_graph.get(name)}})
    for key in sorted(set(a.metadata) | set(b.metadata)):
        before, after = a.metadata.get(key), b.metadata.get(key)
        if before != after:
            changed.append({"metadata": key, key: {"before": before, "after": after}})
    return {
        "from": a.version,
        "to": b.version,
        "added_nodes": sorted(b_nodes.keys() - a_nodes.keys()),
        "removed_nodes": sorted(a_nodes.keys() - b_nodes.keys()),
        "added_edges": sorted(b_edges.keys() - a_edges.keys()),
        "removed_edges": sorted(a_edges.keys() - b_edges.keys()),
        "changed": changed,
    }


def render_diff(d: dict[str, Any]) -> str:
    lines = [f"# Diff v{d['from']} -> v{d['to']}", ""]
    for key in ("added_nodes", "removed_nodes", "added_edges", "removed_edges"):
        lines.append(f"- {key.replace('_', ' ')}: {len(d[key])}" + (f" ({', '.join(d[key][:8])}{'…' if len(d[key]) > 8 else ''})" if d[key] else ""))
    lines.append(f"- changed nodes: {len(d['changed'])}")
    for c in d["changed"][:20]:
        lines.append(f"  - {c}")
    return "\n".join(lines) + "\n"
