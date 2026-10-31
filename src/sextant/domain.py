"""V1 public contracts. Pure data and validation; no I/O or model authority."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical(value: Any) -> str:
    def model_data(item: Any) -> Any:
        if isinstance(item, BaseModel):
            return item.model_dump(mode="json")
        raise TypeError(f"not canonical JSON data: {type(item).__name__}")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=model_data)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def identity(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def utcnow() -> str:
    return datetime.now(UTC).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: Literal[1] = 1


class Scope(Contract):
    subject: str = "unknown"
    domain: str = "unknown"
    environment: str = "unknown"
    population: str = "unknown"
    valid_from: str | None = None
    valid_until: str | None = None
    units: dict[str, str] = Field(default_factory=dict)
    language: str = "unknown"
    representation_version: str = "1"


class ObjectRef(Contract):
    object_id: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,127}$")
    revision_id: str = Field(pattern=r"^[a-f0-9]{64}$")
    object_kind: str


class ArtifactRef(Contract):
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    size: int = Field(ge=0)
    media_type: str
    source: str
    locator: str = "whole"
    read_range: tuple[int, int] | None = None
    access_class: str = "mission-private"


class Revision(Contract):
    ref: ObjectRef
    mission_id: str
    scope: Scope
    payload: dict[str, Any]
    dependencies: list[ObjectRef] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    # Historical lineage is hash-bound but is not a semantic dependency: a new
    # correction must remain usable after its predecessor is invalidated.
    predecessor: ObjectRef | None = None


class AuthorityContext(Contract):
    """Host supplied, never deserialized from a model's proposal or command payload."""

    actor: str
    source: str
    can_authorize: bool = False


class Authorization(Contract):
    actor: str
    source: str
    mission_id: str
    read_paths: list[str] = Field(default_factory=list)
    # Object IDs (or explicit '*') separately authorized for each destination.
    send_to: dict[str, list[str]] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)
    valid_until: str | None = None
    max_uses: int | None = Field(default=None, ge=1)
    status: Literal["active", "revoked"] = "active"


class Mission(Contract):
    goal: str = Field(min_length=1)
    goal_source: str
    hard_constraints: list[dict[str, Any]] = Field(default_factory=list)
    preference_order: list[str] = Field(default_factory=list)
    utility_mapping: dict[str, Any] | None = None
    objective_metrics: list[dict[str, Any]] = Field(default_factory=list)
    material_scope: list[str] = Field(default_factory=list)
    exit_conditions: list[str] = Field(default_factory=list)
    waiting_conditions: list[str] = Field(default_factory=list)
    scope: Scope = Field(default_factory=Scope)


class Observation(Contract):
    content: Any
    source: str
    locator: str = "whole"
    source_group: str
    external_observation_id: str | None = None
    source_revision: str | None = None
    read_range: tuple[int, int] | None = None
    total_bytes: int | None = Field(default=None, ge=0)
    environment_scope: dict[str, Any] | None = None
    occurred_at: str | None = None
    observed_at: str
    measurement_protocol: str
    completeness: Literal["complete", "partial", "unknown"] = "unknown"
    provenance: Literal["declared", "local-file", "constructed", "environment"] = "declared"
    supports_scope: str
    status: Literal["active", "withdrawn"] = "active"
    supersedes: list[ObjectRef] = Field(default_factory=list)
    correction_reason: str | None = None


class Claim(Contract):
    proposition: str
    state: Literal["hypothesis", "supported", "refuted", "withdrawn"] = "hypothesis"
    support: list[ObjectRef] = Field(default_factory=list)
    unresolved_support: list[str] = Field(default_factory=list)
    counterevidence: list[ObjectRef] = Field(default_factory=list)
    derivation_rule: str = "model-proposal"


class ContextBundle(Contract):
    selected: list[ObjectRef]
    joint_dependencies: list[list[ObjectRef]] = Field(default_factory=list)
    counterevidence: list[ObjectRef] = Field(default_factory=list)
    alternative_explanations: list[str] = Field(default_factory=list)
    source_groups: dict[str, list[ObjectRef]] = Field(default_factory=dict)
    omissions: list[str] = Field(default_factory=list)
    loss: list[str] = Field(default_factory=list)


class Representation(Contract):
    kind: Literal["argument_graph", "table", "program", "conditional_policy", "unsupported"]
    version: str = "1"
    body: dict[str, Any]
    input_interface: dict[str, Any] = Field(default_factory=dict)
    output_interface: dict[str, Any] = Field(default_factory=dict)
    composition_rule: str
    interactions: list[str] = Field(default_factory=list)
    residual: list[str] = Field(default_factory=list)
    loss: list[str] = Field(default_factory=list)


class Experiment(Contract):
    key: str
    adapter: Literal["sandbox.python", "environment", "source.file"]
    action: str
    inputs: dict[str, Any]
    purpose: str
    distinguishes: list[str] = Field(default_factory=list)
    changes_world: bool = False
    expected_observations: list[str] = Field(default_factory=list)
    independent_group: str | None = None


class PredictionSpec(Contract):
    """A proposed question/outcome contract; only the host supplies its freeze time."""

    question: str
    outcome_domain: list[Any]
    horizon: str
    resolver: Literal["field-equals/1", "numeric-threshold/1", "declared/1"]
    resolver_spec: dict[str, Any] = Field(description=(
        "Deterministic resolvers bind measurement_protocol/source/supports_scope and an explicit field array "
        "of JSON string keys/nonnegative integer indices ([] selects the whole value). field_path is an "
        "accepted alias; both must agree if supplied. field-equals requires equals; numeric-threshold requires "
        "finite threshold, comparison le/ge, explicit unit string and unit_field array. declared/1 stays pending."
    ))
    sampling_rule: str
    censoring_rule: str


class Candidate(Contract):
    key: str
    mechanism: str
    assumptions: list[str]
    representation: Representation
    # The observations this branch relies on (same reference forms as evidence_links).
    # When every one resolves, the branch depends on the goal and these alone, so a
    # change to other evidence leaves it intact; otherwise it depends on the whole context.
    evidence: list[str] = Field(default_factory=list)
    predictions: list[PredictionSpec] = Field(default_factory=list)
    experiments: list[Experiment] = Field(default_factory=list)
    lineage: list[str] = Field(default_factory=list)
    temporary_disadvantage: str | None = None
    combination_dependencies: list[str] = Field(default_factory=list)
    reopen_trigger: str = "new contrary evidence"
    scenario_metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ResolutionCorrection(Contract):
    """Computed retrospective provenance; never a new confirming sample."""

    supersedes_prediction_ref: ObjectRef
    original_status: Literal["resolved"]
    original_outcome: Any
    original_frozen_at: str
    observation_ref: ObjectRef
    correction_reason: str
    computed_at: str
    protocol_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    confirming_sample: Literal[False] = False
    disposition: Literal["recomputed", "invalid-contract"]
    contract_error_code: str | None = None
    contract_error: str | None = None


class Prediction(PredictionSpec):
    frozen_at: str
    status: Literal["pending", "resolved", "censored", "invalid"] = "pending"
    outcome: Any = None
    resolution_refs: list[ObjectRef] = Field(default_factory=list)
    resolution_correction: ResolutionCorrection | None = None


class Policy(Contract):
    summary: str
    branches: list[dict[str, Any]]
    hard_constraints: list[dict[str, Any]]
    unresolved: list[str]
    reversal_triggers: list[str]
    ordering: Literal["conditional-partial-order", "authorized-cardinal", "deterministic"]
    basis: str
    next_experiments: list[Experiment] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    decision_analysis: dict[str, Any] = Field(default_factory=dict)


class Proposal(Contract):
    """Generator output: assertions remain proposals until checked against evidence."""

    understanding: str
    candidates: list[Candidate] = Field(min_length=1)
    policy: Policy
    unknowns: list[str]
    evidence_links: dict[str, list[str]] = Field(default_factory=dict)
    direct_result: Any = None


class Receipt(Contract):
    executor: str
    requested_model: str | None = None
    observed_model: str | None = None
    requested_effort: str | None = None
    observed_effort: str | None = None
    session: str | None = None
    auth_route: str = "unknown"
    auth_evidence: str = "unknown"
    started_at: str
    ended_at: str
    request_hash: str
    output_hash: str | None = None
    status: Literal["completed", "failed", "unknown", "cancelled"]
    exit_code: int | None = None
    timed_out: bool = False
    usage: dict[str, Any] = Field(default_factory=dict)
    isolation: dict[str, Any] = Field(default_factory=dict)
    residual_processes: str = "unknown"
    tool_state: str = "unknown"
    external_id: str | None = None
    errors: list[str] = Field(default_factory=list)


CommandKind = Literal[
    "start", "inspect", "run", "pause", "resume", "cancel", "observe", "withdraw",
    "resolve", "revise", "export", "import", "authorize", "replay", "reconcile",
]


class Command(Contract):
    protocol: Literal["sextant-agent/1"] = "sextant-agent/1"
    command_id: str = Field(min_length=1, max_length=128)
    mission_id: str | None = None
    expected_revision: str | None = None
    kind: CommandKind
    payload: dict[str, Any] = Field(default_factory=dict)
    authorization_ref: ObjectRef | None = None

    @model_validator(mode="after")
    def finite_payload(self) -> Command:
        canonical(self.payload)
        return self


class CommandResult(Contract):
    protocol: Literal["sextant-agent/1"] = "sextant-agent/1"
    command_id: str
    mission_id: str | None = None
    status: str
    current_revision: str | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[dict[str, str]] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class DomainError(RuntimeError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
