"""The canonical decision-graph schema (pydantic models plus the exported JSON Schema).

Anchors, each borrowed from an existing convention so that little is invented:
- document shape: nodes[], edges[], metadata{}, resources[] (SADFace)
- reified relations: evidence links and conflicts are nodes with their own estimate (AIF)
- resolution fields on every probability-bearing node (Metaculus): an estimate on a factor
  or probe without resolution criteria is rejected, because it could never be scored
- closed enumerations with a legal-endpoint matrix for edges (DMN discipline)
- decomposition carries method semantics: ordered subtasks plus applicability (HDDL)
- node status uses the Agent2Agent task lifecycle verbatim
- content-addressed ids: a node id is a hash of its immutable fields
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from math import isclose, isfinite
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sextant.ledger import sha_of

# 2.0.0: judgments carry their run, language and contested flag; graphs of 1.x are refused.
SCHEMA_VERSION: Final = "2.0.0"


class NodeKind(StrEnum):
    question = "question"
    option = "option"
    criterion = "criterion"
    factor = "factor"
    constraint = "constraint"
    value = "value"
    probe = "probe"
    evidence = "evidence"
    evidence_link = "evidence_link"
    conflict = "conflict"


class EdgeType(StrEnum):
    decomposes_into = "decomposes_into"
    influences = "influences"
    constrains = "constrains"
    depends_on = "depends_on"
    is_value_for = "is_value_for"
    enables = "enables"
    measured_by = "measured_by"
    cites = "cites"
    supports = "supports"
    conflicts = "conflicts"


# The legal (from-kind, to-kind) pairs per edge type. A validator rejects anything else.
LEGAL_ENDPOINTS: dict[EdgeType, frozenset[tuple[NodeKind, NodeKind]]] = {
    EdgeType.decomposes_into: frozenset(
        (NodeKind.question, k)
        for k in (NodeKind.question, NodeKind.factor, NodeKind.criterion, NodeKind.constraint, NodeKind.option)
    ),
    EdgeType.influences: frozenset(
        {
            (NodeKind.factor, NodeKind.factor),
            (NodeKind.factor, NodeKind.criterion),
            (NodeKind.option, NodeKind.factor),
            (NodeKind.option, NodeKind.criterion),
        }
    ),
    EdgeType.constrains: frozenset(
        {(NodeKind.constraint, NodeKind.option), (NodeKind.constraint, NodeKind.question)}
    ),
    EdgeType.depends_on: frozenset(
        {(NodeKind.factor, NodeKind.factor), (NodeKind.probe, NodeKind.probe), (NodeKind.question, NodeKind.question)}
    ),
    EdgeType.is_value_for: frozenset({(NodeKind.value, NodeKind.criterion)}),
    EdgeType.enables: frozenset({(NodeKind.option, NodeKind.factor), (NodeKind.probe, NodeKind.factor)}),
    EdgeType.measured_by: frozenset(
        {(NodeKind.criterion, NodeKind.factor), (NodeKind.factor, NodeKind.probe), (NodeKind.factor, NodeKind.evidence)}
    ),
    EdgeType.cites: frozenset({(NodeKind.evidence_link, NodeKind.evidence), (NodeKind.conflict, NodeKind.evidence)}),
    EdgeType.supports: frozenset(
        {
            (NodeKind.evidence_link, NodeKind.factor),
            (NodeKind.evidence_link, NodeKind.criterion),
            (NodeKind.evidence_link, NodeKind.option),
            (NodeKind.evidence_link, NodeKind.constraint),
        }
    ),
    EdgeType.conflicts: frozenset(
        (NodeKind.conflict, k)
        for k in (NodeKind.factor, NodeKind.evidence, NodeKind.constraint, NodeKind.option, NodeKind.criterion)
    ),
}

# Edge types whose union must stay acyclic.
ACYCLIC_EDGE_TYPES = frozenset({EdgeType.decomposes_into, EdgeType.depends_on})

# Node kinds that carry a world estimate and therefore need resolution fields.
RESOLVABLE_KINDS = frozenset({NodeKind.factor, NodeKind.probe})


class Status(StrEnum):
    submitted = "submitted"
    working = "working"
    input_required = "input_required"
    auth_required = "auth_required"
    completed = "completed"
    failed = "failed"
    canceled = "canceled"
    rejected = "rejected"


class Primitive(StrEnum):
    noul = "noul"
    choice = "choice"
    score = "score"
    none = "none"


class SourceTier(StrEnum):
    verified = "verified"
    corroborated = "corroborated"
    single_source = "single-source"
    inferred = "inferred"
    jev_judged = "jev-judged"
    user_stated = "user-stated"


class Rung(StrEnum):
    code = "code"
    data = "data"
    jev = "jev"
    llm_isolated = "llm-isolated"
    human = "human"


class Tier(StrEnum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class AnswerSpace(Strict):
    primitive: Primitive
    options: dict[str, str | None] | None = None
    levels: list[str] | None = None
    escape_option: str | None = None

    @model_validator(mode="after")
    def _check(self) -> AnswerSpace:
        if self.primitive == Primitive.choice:
            if not self.options or len(self.options) < 2:
                raise ValueError("a choice answer space needs at least two options")
            if len(self.options) > 32:
                raise ValueError("a choice answer space is capped at 32 options; bucket beyond")
            if not self.escape_option or self.escape_option not in self.options:
                raise ValueError("a choice answer space needs an escape option that is one of its options")
        if self.primitive == Primitive.score and (not self.levels or not (2 <= len(self.levels) <= 10)):
            raise ValueError("a score answer space needs 2 to 10 levels")
        return self


class Stability(Strict):
    rerun_sd: float | None = None
    paraphrase_sd: float | None = None
    permutation_sd: float | None = None
    negation_gap: float | None = None
    n_reruns: int = 0
    n_paraphrases: int = 0
    n_permutations: int = 0


class CalibrationStatus(Strict):
    """How much authority this judgment's threshold has, kept apart from the judgment itself.

    `provisional` is the state that used to be missing: a fit the holdout could not confirm,
    only fail to reject. `invalidated` covers a threshold whose certificate cannot be checked
    at all, including one written before thresholds carried a certificate.
    """

    kind: Literal["agreement", "outcome"]
    state: Literal[
        "pending", "provisional", "calibrated", "degraded", "invalidated", "out_of_domain"
    ] = "pending"
    n: int = 0
    bar: int | None = None
    reason: str | None = None


class ThresholdRef(Strict):
    """The certificate a calibrated judgment was decided against (see `certificates.py`)."""

    card_id: str
    key: str
    value: float
    fitted_on: str | None = None
    n: int = 0
    status: Literal["validated", "provisional", "unfit", "revoked", "unverified"] = "unverified"
    experiment_id: str | None = None
    dataset_version: str | None = None
    measurement_conditions: str | None = None


class Range(Strict):
    low: float
    high: float

    @model_validator(mode="after")
    def _check(self) -> Range:
        if not (isfinite(self.low) and isfinite(self.high)):
            raise ValueError("range bounds must be finite")
        if self.low > self.high:
            raise ValueError("range low must not exceed high")
        return self


class ContributingCall(Strict):
    """One Jev call that actually fed one question's judgment.

    `rendered_hash` is per contribution because a permutation shuffles the options and a
    paraphrase replaces the wording: those calls legitimately sent a different payload for
    the same sealed question, and only the primary call sent the estimate's own payload.
    """

    call_id: str
    variant: str
    role: Literal["primary", "permutation", "rerun", "paraphrase"]
    rendered_hash: str
    value: float | None = None
    contributes_to_value: bool = True


class Estimate(Strict):
    """One judgment, bound to the exact calls, payload, input and protocol that produced it.

    The provenance fields carry no defaults on purpose. An estimate stored before the wire
    protocol existed cannot be loaded as evidence, because nothing recorded which bytes the
    model saw; it fails validation instead of being silently accepted.
    """

    primitive: Primitive
    p: float | None = Field(default=None, ge=0.0, le=1.0)
    probabilities: dict[str, float] | None = None
    choice: str | None = None
    score: float | None = None
    card_id: str
    question_key: str
    question_hash: str
    rendered_hash: str
    state_sha: str
    protocol_version: str
    run_id: str
    # the answer language the question was asked in
    language: str
    # the caller's declaration of the language the state is written in; None when undeclared,
    # which no certificate covers
    input_language: str | None = None
    aggregation: Literal["single", "mean_of_runs", "permutation_mean"]
    contested: bool
    contributions: list[ContributingCall] = Field(min_length=1)
    model_id: str
    transport: str
    source_tier: SourceTier
    stability: Stability = Field(default_factory=Stability)
    sufficiency_p: float | None = Field(default=None, ge=0.0, le=1.0)
    sufficiency_question_key: str | None = None
    jev_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    threshold_ref: ThresholdRef | None = None
    calibration_status: CalibrationStatus
    range: Range | None = None
    at: str

    @property
    def call_ids(self) -> list[str]:
        """Derived, never stored: the contributing calls are the single source of truth."""
        return [c.call_id for c in self.contributions]

    @model_validator(mode="after")
    def _check(self) -> Estimate:
        if self.primitive == Primitive.noul and self.p is None:
            raise ValueError("a noul estimate needs p")
        if self.primitive == Primitive.choice and not self.probabilities:
            raise ValueError("a choice estimate needs probabilities")
        if self.primitive == Primitive.score and (self.score is None or not self.probabilities):
            raise ValueError("a score estimate needs score and per-level probabilities")
        if self.score is not None and not isfinite(self.score):
            raise ValueError("a score must be a finite number")
        if self.probabilities is not None and not all(isfinite(v) for v in self.probabilities.values()):
            raise ValueError("probabilities must be finite")
        if any(c.value is not None and not isfinite(c.value) for c in self.contributions):
            raise ValueError("a contributing value must be finite")
        primaries = [c for c in self.contributions if c.role == "primary"]
        if len(primaries) != 1:
            raise ValueError("an estimate names exactly one primary call")
        if primaries[0].rendered_hash != self.rendered_hash:
            raise ValueError("the estimate's rendered question must be the one its primary call sent")
        in_value = [c for c in self.contributions if c.contributes_to_value]
        if not in_value:
            raise ValueError("an estimate needs at least one call that feeds its value")
        if self.aggregation == "single" and len(in_value) != 1:
            raise ValueError("a single-call aggregation cannot have several contributing calls")
        if (self.sufficiency_p is None) != (self.sufficiency_question_key is None):
            raise ValueError("a copied sufficiency value names the question it came from")
        if self.probabilities is not None:
            if any(value < 0.0 for value in self.probabilities.values()):
                raise ValueError("probabilities must be non-negative")
            total = sum(self.probabilities.values())
            if not isclose(total, 1.0, abs_tol=1e-6):
                raise ValueError("probabilities must sum to 1")
            if self.choice is not None and self.choice not in self.probabilities:
                raise ValueError("choice is not in the probability vector")
        if self.calibration_status.state == "calibrated":
            # A label cannot outrank its evidence: `calibrated` is a claim about a threshold
            # certificate, so the certificate has to be here and has to say `validated`.
            if self.threshold_ref is None:
                raise ValueError("a calibrated estimate needs the threshold it was calibrated against")
            if self.threshold_ref.status != "validated":
                raise ValueError(
                    f"a {self.threshold_ref.status} threshold cannot make an estimate calibrated"
                )
        return self


class Resolution(Strict):
    criteria: str = ""
    fine_print: str = ""
    source: str = ""
    resolve_by: str | None = None
    resolution: Literal["yes", "no", "ambiguous", "annulled", "unresolved"] = "unresolved"


class Provenance(Strict):
    request_id: str | None = None
    provider: str
    family: str
    isolation: Literal["same-session", "isolated", "n/a"] = "n/a"


class Evaluator(Strict):
    rung: Rung
    card_id: str | None = None


class EvidenceRequirement(Strict):
    settles_when: str
    cheapest_probe_id: str | None = None
    cost: str | None = None


class Readiness(Strict):
    is_leaf: bool = False
    bounded_p: float | None = None
    criteria_p: float | None = None
    obtainable_p: float | None = None
    asks_preference_p: float | None = None


class Node(Strict):
    id: str
    kind: NodeKind
    residual: bool = False
    text_original: str
    text_en: str
    definition: str = ""
    answer_space: AnswerSpace | None = None
    evaluator: Evaluator | None = None
    state_vector: dict[str, Estimate] | None = None
    estimate: Estimate | None = None
    sufficiency: Estimate | None = None
    readiness: Readiness | None = None
    evidence_requirement: EvidenceRequirement | None = None
    resolution: Resolution | None = None
    freshness_window_days: int | None = None
    verification_cost: str | None = None
    generation_cost: str | None = None
    predicate: str | None = None
    members: list[str] = Field(default_factory=list)
    direction: Literal["+", "-", "?"] | None = None
    status: Status = Status.submitted
    created_by: Provenance | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> Node:
        unresolvable = self.resolution is None or not self.resolution.criteria.strip()
        if self.kind in RESOLVABLE_KINDS and self.estimate is not None and unresolvable:
            raise ValueError(
                f"node {self.id}: an estimate on a {self.kind.value} needs resolution criteria (it must be scoreable)"
            )
        if self.kind == NodeKind.value and (self.estimate is not None or self.sufficiency is not None or self.state_vector):
            raise ValueError(f"node {self.id}: a value node never carries a Jev estimate, sufficiency or state vector")
        if self.kind in (NodeKind.evidence_link, NodeKind.conflict) and len(self.members) < 1:
            raise ValueError(f"node {self.id}: a {self.kind.value} node names its members")
        if self.kind == NodeKind.conflict and (len(self.members) != 2 or len(set(self.members)) != 2):
            raise ValueError(f"node {self.id}: a conflict names exactly two distinct members")
        options = self.answer_space.options if self.answer_space is not None else None
        if (
            self.estimate is not None
            and self.estimate.choice
            and options is not None
            and self.estimate.choice not in options
        ):
            raise ValueError(
                f"node {self.id}: choice {self.estimate.choice} is not in the answer space"
            )
        return self


class Method(Strict):
    ordered: bool = False
    subtasks: list[str] = Field(default_factory=list)
    applicability: str = ""


class Verification(Strict):
    relevance: Estimate | None = None
    concreteness_gain: Estimate | None = None
    skipped_level: Estimate | None = None
    edge_type: Estimate | None = None


class Strength(Strict):
    direction: Literal["+", "-", "?"] = "?"
    prior_source: Literal["llm", "data", "user"] = "llm"


class Edge(Strict):
    id: str
    type: EdgeType
    from_id: str
    to_id: str
    method: Method | None = None
    verification: Verification | None = None
    strength: Strength | None = None
    status: Literal["proposed", "failed", "repaired"] = "proposed"

    @model_validator(mode="after")
    def _check(self) -> Edge:
        if self.type == EdgeType.decomposes_into and self.method is None:
            raise ValueError(f"edge {self.id}: decomposes_into carries a method payload")
        return self


class Frame(Strict):
    x: str
    y: str
    decision_type: Literal["choose", "evaluate", "predict", "diagnose", "design_path"]
    horizon: str = ""
    stakes: Literal["low", "medium", "high"] = "medium"
    reversibility: Literal["reversible", "partly", "irreversible"] = "partly"
    tier: Tier = Tier.T2
    tier_override_by: str | None = None
    options_open: bool = False


class Budgets(Strict):
    depth_max: int
    breadth_max: int
    error_budget_target: float


TIER_BUDGETS: dict[Tier, Budgets] = {
    Tier.T1: Budgets(depth_max=2, breadth_max=5, error_budget_target=0.60),
    Tier.T2: Budgets(depth_max=4, breadth_max=7, error_budget_target=0.75),
    Tier.T3: Budgets(depth_max=5, breadth_max=7, error_budget_target=0.85),
}


def error_budget_constrains_run(budgets: Budgets | None = None) -> bool:
    """`error_budget_target` is stored on the tier and does not constrain a run.

    Depth and breadth are the checks that actually run. This stays false until a
    run compares an observed error to the recorded target and can refuse on it.
    """
    del budgets
    return False


class GateAuthority(StrEnum):
    checker = "checker"
    user_judgment = "user_judgment"


class GateReceipt(Strict):
    """What a checker binds when it issues a gate. A free-form token is not a receipt."""

    protocol_version: str = Field(min_length=1)
    checker: str = Field(min_length=1)
    input_revisions: dict[str, str]
    artifact_hash: str = Field(min_length=1)
    prerequisites: list[str] = Field(default_factory=list)
    result: Literal["PASS", "FAIL"]


def receipt_token(receipt: GateReceipt) -> str:
    """The only token a checker gate may carry: the hash of the receipt body."""
    body = {
        "protocol_version": receipt.protocol_version,
        "checker": receipt.checker,
        "input_revisions": receipt.input_revisions,
        "artifact_hash": receipt.artifact_hash,
        "prerequisites": list(receipt.prerequisites),
        "result": receipt.result,
    }
    return "gr_" + sha_of(body)


class Gate(Strict):
    """A gate is either a checker receipt or a user judgment. It is never both.

    `PASS` exists only on a checker receipt whose token is `receipt_token`. A human
    approval is `user_judgment` and cannot use `PASS`.
    """

    authority: GateAuthority
    status: GateStatus
    at: str
    token: str = Field(min_length=1)
    receipt: GateReceipt | None = None
    note: str | None = None

    @model_validator(mode="after")
    def _bound(self) -> Gate:
        if not self.token.strip():
            raise ValueError("a gate token must be non-empty")
        if self.authority == GateAuthority.user_judgment:
            if self.status == GateStatus.PASS:
                raise ValueError("a human approval is a user judgment, never a checker PASS")
            if self.receipt is not None:
                raise ValueError("a user judgment does not carry a checker receipt")
            return self
        if self.receipt is None:
            raise ValueError("a checker gate needs the receipt issued by the checker that ran")
        if self.status.value != self.receipt.result:
            raise ValueError("gate status does not match the checker receipt")
        if self.token != receipt_token(self.receipt):
            raise ValueError("gate token does not match its receipt")
        return self


class GraphProvenance(Strict):
    generator_families: list[str] = Field(default_factory=list)
    challenger_families: list[str] = Field(default_factory=list)
    jev_model: str | None = None
    transport: str | None = None


class Language(Strict):
    original: str = "en"
    canonical: str = "en"


class Question(Strict):
    text_original: str
    text_en: str
    hash: str


class DecisionGraph(Strict):
    schema_version: Literal["2.0.0"] = SCHEMA_VERSION
    slug: str
    version: int = Field(default=1, ge=1)
    parent_version: int | None = Field(default=None, ge=1)
    created: str
    language: Language = Field(default_factory=Language)
    question: Question
    frame: Frame | None = None
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    budgets: Budgets | None = None
    provenance: GraphProvenance = Field(default_factory=GraphProvenance)
    gates: dict[str, Gate] = Field(default_factory=dict)
    diff_summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    resources: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> DecisionGraph:
        if self.question.hash != question_hash(self.question.text_en):
            raise ValueError("question hash does not match its text")
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate node ids: {dupes}")
        kinds = {n.id: n.kind for n in self.nodes}
        eids = [e.id for e in self.edges]
        if len(eids) != len(set(eids)):
            raise ValueError("duplicate edge ids")
        for e in self.edges:
            if e.from_id not in kinds or e.to_id not in kinds:
                raise ValueError(f"edge {e.id}: endpoint not in graph ({e.from_id} -> {e.to_id})")
            pair = (kinds[e.from_id], kinds[e.to_id])
            if pair not in LEGAL_ENDPOINTS[e.type]:
                raise ValueError(
                    f"edge {e.id}: {e.type.value} is not legal from {pair[0].value} to {pair[1].value}"
                )
        for n in self.nodes:
            for m in n.members:
                if m not in kinds:
                    raise ValueError(f"node {n.id}: member {m} not in graph")
        return self

    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def children(self, parent_id: str, edge_type: EdgeType = EdgeType.decomposes_into) -> list[Node]:
        return [self.node(e.to_id) for e in self.edges if e.type == edge_type and e.from_id == parent_id]


def _digest(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def node_id(kind: NodeKind | str, text_en: str, definition: str = "") -> str:
    """Content-addressed node id: a hash of the immutable fields."""
    k = kind.value if isinstance(kind, NodeKind) else str(kind)
    return "n_" + _digest(k, text_en.strip(), definition.strip())[:12]


def edge_id(edge_type: EdgeType | str, from_id: str, to_id: str) -> str:
    t = edge_type.value if isinstance(edge_type, EdgeType) else str(edge_type)
    return "e_" + _digest(t, from_id, to_id)[:12]


def question_hash(text_en: str) -> str:
    return "q_" + _digest(text_en.strip())[:16]


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def export_json_schema() -> dict[str, Any]:
    schema = DecisionGraph.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://sextant.local/schema/decision-graph/{SCHEMA_VERSION}"
    schema["title"] = "sextant decision graph"
    schema["x-legal-endpoints"] = {
        t.value: sorted(f"{a.value}->{b.value}" for a, b in pairs) for t, pairs in LEGAL_ENDPOINTS.items()
    }
    schema["x-acyclic-edge-types"] = sorted(t.value for t in ACYCLIC_EDGE_TYPES)
    schema["x-resolvable-kinds"] = sorted(k.value for k in RESOLVABLE_KINDS)
    return schema
