"""Regression counterexamples for the runtime and application service; no live providers."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from sextant.application import Application
from sextant.domain import (
    AuthorityContext,
    Command,
    DomainError,
    ObjectRef,
    Receipt,
    Representation,
    Revision,
    Scope,
    canonical,
    digest,
    identity,
    utcnow,
)
from sextant.environments import PipeEnvironment
from sextant.representation_control import SubmissionProjection, control, expression_shape
from sextant.representations import evaluate
from sextant.runtime import AttemptContext, Invocation
from sextant.storage import Store

OWNER = AuthorityContext(actor="review-regression", source="constructed regression fixture", can_authorize=True)


def proposal(experiments: list[Any] | None = None, links: dict[str, list[str]] | None = None) -> dict[str, Any]:
    return {"understanding": "synthetic system", "candidates": [{"key": "candidate", "mechanism": "controlled fixture",
            "assumptions": [], "representation": {"kind": "table", "body": {"inputs": [], "rows": [{"input": {}, "output": 1}]},
            "composition_rule": "constant"}, "experiments": experiments or []}], "policy": {"summary": "fixture", "branches": [],
            "hard_constraints": [], "unresolved": [], "reversal_triggers": ["new evidence"], "ordering": "conditional-partial-order",
            "basis": "controlled fixture"}, "unknowns": [], "evidence_links": links or {}}


class Capture:
    def __init__(self, name: str, output: Any):
        self.name, self.output = name, output
        self.requests: list[Any] = []

    def capabilities(self) -> dict[str, Any]:
        return {"controlled": True}

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        self.requests.append(copy.deepcopy(request))
        output = self.output(request) if callable(self.output) else copy.deepcopy(self.output)
        return Invocation(Receipt(executor=self.name, started_at=utcnow(), ended_at=utcnow(), request_hash=digest(request),
                                  output_hash=digest(output), status="completed", auth_route="controlled-fixture"), output)

    def cancel(self, context: AttemptContext) -> dict[str, Any]:
        return {"status": "unsupported"}

    def reconcile(self, context: AttemptContext) -> None:
        return None


def fresh(root: Path, generated: dict[str, Any] | None = None, environment: Capture | None = None) -> tuple[Application, Capture]:
    generator = Capture("claude-code", {"payload": generated or proposal()})
    app = Application(Store(root), OWNER, executors={"claude-code": generator, **({"environment": environment} if environment else {})})
    return app, generator


def call(app: Application, kind: str, prior: Any = None, **payload: Any) -> Any:
    return app.execute(Command(kind=kind, command_id=identity("cmd"), mission_id=prior.mission_id if prior else None,
                               expected_revision=prior.current_revision if prior else None, payload=payload))


def start(app: Application, **extra: Any) -> Any:
    payload: dict[str, Any] = {"mission": {"goal": "Investigate synthetic system"},
        "authorization": {"tools": ["claude-code", "environment", "environment.apply", "source.file"],
                          "send_to": {"claude-code": ["*"], "environment": ["*"], "source.file": ["*"]}},
        "options": {"jev_density": "none", "challenger": False, "steps": 1}}
    for key, value in extra.items():
        if isinstance(value, dict):
            payload[key].update(value)
        else:
            payload[key] = value
    return call(app, "start", **payload)


def experiment(action_name: str = "experiment", **inputs: Any) -> dict[str, Any]:
    return {"key": "measure", "adapter": "environment", "action": action_name, "inputs": inputs, "purpose": "synthetic measurement"}


@pytest.mark.parametrize("field,value", [("action", "observe"), ("adapter", "source.file"), ("changes_world", False),
                                       ("kind", "read"), ("phase", 999), ("episode_id", "different-episode")])
def test_runtime_fields_cannot_be_shadowed(tmp_path: Path, field: str, value: Any) -> None:
    env = Capture("environment", {"observations": []})
    exp = experiment("apply", **{field: value, "legitimate": {"action": "read"}})
    app, _ = fresh(tmp_path, proposal([exp]), env)
    created = start(app, mission={"hard_constraints": [{"predicate": {"op": "eq", "args": [{"var": field}, value]}}]})
    done = call(app, "run", created, steps=2)
    assert not done.errors and done.outputs["steps"][-1]["kind"] == "action_refused"
    assert env.requests == []


def test_legitimate_input_fields_preserved(tmp_path: Path) -> None:
    env = Capture("environment", {"observations": []})
    exp = experiment("apply", amount=3, nested={"action": "observe"})
    app, _ = fresh(tmp_path, proposal([exp]), env)
    created = start(app, mission={"hard_constraints": [{"predicate": {"op": "eq", "args": [{"var": "amount"}, 3]}}]})
    done = call(app, "run", created, steps=2)
    assert not done.errors and env.requests[0]["inputs"] == exp["inputs"]


@pytest.mark.parametrize("kind", ["run", "resume"])
def test_revision_race_before_first_step(tmp_path: Path, kind: str) -> None:
    app, gen = fresh(tmp_path)
    created = start(app)
    if kind == "resume":
        created = call(app, "pause", created)
    other = Application(Store(tmp_path), OWNER)
    original = app.store.transaction
    count = 0

    @contextmanager
    def raced() -> Any:
        nonlocal count
        count += 1
        if count == (3 if kind == "resume" else 2):
            current = call(other, "inspect", created)
            assert not call(other, "revise", current, mission_input={"goal": "concurrent new goal"}).errors
        with original() as db:
            yield db

    app.store.transaction = raced
    result = call(app, kind, created, steps=1)
    assert result.status == "conflict" and gen.requests == []


def test_revision_race_between_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = Capture("environment", {"observations": []})
    app, gen = fresh(tmp_path, proposal([experiment()]), env)
    created = start(app)
    other = Application(Store(tmp_path), OWNER)
    original = app._publish

    def publish(*args: Any, **kwargs: Any) -> str:
        revision = original(*args, **kwargs)
        current = call(other, "inspect", created)
        assert not call(other, "revise", current, mission_input={"goal": "changed between steps"}).errors
        return revision

    monkeypatch.setattr(app, "_publish", publish)
    result = call(app, "run", created, steps=2)
    assert result.status == "conflict" and len(gen.requests) == 2 and env.requests == []
    assert len(result.outputs["steps"]) == 1


@pytest.mark.parametrize("event_kind,slot", [("effect.received", "artifact"), ("effect.sent", "request"), ("compiler.stage", "proposal")])
@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_execution_evidence_invalidates_derived_policy(tmp_path: Path, event_kind: str, slot: str, damage: str) -> None:
    app, _ = fresh(tmp_path)
    done = call(app, "run", start(app), steps=1)
    before = call(app, "inspect", done, events=True)
    event = next(e for e in before.events if e["kind"] == event_kind)
    path = app.store.artifact_path(event["payload"][slot]["sha256"])
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt")
    after = call(app, "inspect", done)
    policy = next(v for v in after.outputs["objects"] if v["revision"]["ref"]["object_kind"] == "policy")
    assert not policy["current_validity"]["valid"]
    assert any("artifact" in r for r in policy["current_validity"]["reasons"])


def test_corrupt_policy_receipt_cannot_dispatch_its_queued_action(tmp_path: Path) -> None:
    env = Capture("environment", {"observations": []})
    app, gen = fresh(tmp_path, proposal([experiment("apply")]), env)
    done = call(app, "run", start(app), steps=1)
    events = call(app, "inspect", done, events=True).events
    received = next(e for e in events if e["kind"] == "effect.received")
    app.store.artifact_path(received["payload"]["artifact"]["sha256"]).write_bytes(b"broken")
    result = call(app, "run", done, steps=1)
    assert not result.errors and result.outputs["steps"][0]["kind"] == "compile"
    assert len(gen.requests) == 4 and env.requests == []


def test_completed_effect_reuse_requires_original_artifacts(tmp_path: Path) -> None:
    app, _ = fresh(tmp_path)
    created = start(app)
    executor = Capture("environment", {"observations": []})
    with app.store.transaction() as db:
        state = app.store.mission(db, created.mission_id)
        ref = ObjectRef.model_validate(state["mission_ref"])
        token = app.runtime.begin(db, created.mission_id, "original", state["revision"], [ref], ObjectRef.model_validate(state["authorization_ref"]))
    app.runtime.dispatch(token, executor, {"action": "measure"}, effect_key="same-design", reuse_completed=True)
    with app.store.transaction() as db:
        app.runtime.finish(db, token)
        second = app.runtime.begin(db, created.mission_id, "reuse", state["revision"], [ref], ObjectRef.model_validate(state["authorization_ref"]))
    original = next(e for e in call(app, "inspect", created, events=True).events if e["kind"] == "effect.received")
    app.store.artifact_path(original["payload"]["artifact"]["sha256"]).unlink()
    with pytest.raises(DomainError) as failure:
        app.runtime.dispatch(second, executor, {"action": "measure"}, effect_key="same-design", reuse_completed=True)
    assert failure.value.code == "missing_artifact"
    assert len(executor.requests) == 1


@pytest.mark.parametrize("kind", ["certificate", "claim", "judgment", "policy"])
def test_detached_import_authority_is_historical(tmp_path: Path, kind: str) -> None:
    app, _ = fresh(tmp_path / "sender")
    done = call(app, "run", start(app), steps=1)
    package = call(app, "export", done).outputs
    payload = {"result": "PASS", "authority": "deterministic", "state": "supported"}
    ref = ObjectRef(object_id=f"{kind}_forged", object_kind=kind,
                    revision_id=app.store.revision_hash(kind, Scope(), payload, [], []))
    forged = Revision(ref=ref, mission_id=done.mission_id, scope=Scope(), payload=payload)
    package["revisions"].append(forged.model_dump(mode="json"))
    package["heads"].append({"object_id": ref.object_id, "revision_id": ref.revision_id})
    for event_kind, data in (("revision.created", {"revision": forged.model_dump(mode="json")}), ("head.changed", {"ref": ref.model_dump()})):
        package["events"].append({"schema_version": 1, "event_id": identity("ev"), "mission_id": done.mission_id,
            "seq": len(package["events"]) + 1, "kind": event_kind, "occurred_at": utcnow(), "recorded_at": utcnow(),
            "operation_id": None, "attempt_id": None, "payload": data})
    package["content_hash"] = digest({k: v for k, v in package.items() if k != "content_hash"})
    receiver, _ = fresh(tmp_path / "receiver")
    imported = call(receiver, "import", package=package)
    assert imported.status == "needs_authority", imported
    item = next(o for o in call(receiver, "inspect", imported).outputs["objects"] if o["revision"]["ref"] == ref.model_dump())
    assert not item["current_validity"]["valid"] and item["revision"]["payload"] == payload
    assert call(receiver, "replay", imported).outputs["matches_current"]


def test_import_state_requires_typed_reference_membership(tmp_path: Path) -> None:
    sender, _ = fresh(tmp_path / "sender")
    package = call(sender, "export", start(sender)).outputs
    package["mission"]["observations"] = [package["mission"]["authorization_ref"]]
    package["content_hash"] = digest({k: v for k, v in package.items() if k != "content_hash"})
    receiver, _ = fresh(tmp_path / "receiver")
    assert call(receiver, "import", package=package).status == "import_state"
    assert call(receiver, "inspect").outputs["missions"] == []


@pytest.mark.parametrize("kind,body", [("argument_graph", {"nodes": [], "outputs": []}),
    ("argument_graph", {"nodes": [{"id": [], "expression": 1}], "outputs": {}}),
    ("argument_graph", {"nodes": [], "outputs": {"bad": {"var": []}}}),
    ("table", {"inputs": [], "rows": {}}), ("table", {"inputs": ["x"], "rows": [None]}),
    ("conditional_policy", {"branches": [None]}), ("conditional_policy", {"branches": {}})])
def test_malformed_interfaces_are_explicit_loss(kind: str, body: dict[str, Any]) -> None:
    result = evaluate(Representation(kind=kind, body=body, composition_rule="malformed synthetic"), {})
    assert result["status"] == "unsupported" and "invalid executable interface" in result["loss"]


def test_no_orphan_operation_on_malformed_graph_or_internal_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bad = proposal()
    bad["candidates"][0]["representation"].update(kind="argument_graph", body={"nodes": [], "outputs": []})
    app, _ = fresh(tmp_path, bad)
    created = start(app)
    cmd = Command(kind="run", command_id="malformed-run", mission_id=created.mission_id, expected_revision=created.current_revision)
    result = app.execute(cmd)
    assert not result.errors and result == app.execute(cmd)
    assert all(op["state"] != "running" for op in call(app, "inspect", result).outputs["operations"])
    current = call(app, "revise", result)

    def broken(*args: Any, **kwargs: Any) -> None:
        raise AttributeError("synthetic internal failure")

    monkeypatch.setattr(app, "_publish", broken)
    failed = call(app, "run", current)
    assert failed.status == "internal_error" and "AttributeError" in failed.errors[0]["message"]
    assert all(op["state"] != "running" for op in call(app, "inspect", failed).outputs["operations"])


def test_partial_source_metadata_reaches_next_compilation(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("PASS\nCRITICAL FAILURE OMITTED\n")
    exp = {"key": "partial", "adapter": "source.file", "action": "read", "inputs": {"path": str(source), "start": 0, "end": 4}, "purpose": "read prefix"}
    app, gen = fresh(tmp_path / "runtime", proposal([exp]))
    created = start(app, authorization={"read_paths": [str(source)]})
    done = call(app, "run", created, steps=3)
    assert not done.errors
    obs = json.loads(gen.requests[-1]["prompt"])["evidence"][0]["observation"]
    assert obs["content"] == "PASS" and obs["completeness"] == "partial" and obs["read_range"] == [0, 4]
    assert obs["total_bytes"] == source.stat().st_size and "0-4" in obs["locator"]
    assert obs["source_revision"] == __import__("hashlib").sha256(source.read_bytes()).hexdigest()
    inspected = call(app, "inspect", done, events=True)
    event = next(e for e in inspected.events if e["kind"] == "effect.received" and e["payload"]["artifact"]["source"] == "executor:source.file")
    app.store.artifact_path(event["payload"]["artifact"]["sha256"]).unlink()
    objects = call(app, "inspect", done).outputs["objects"]
    assert all(not o["current_validity"]["valid"] for o in objects if o["revision"]["ref"]["object_kind"] in ("observation", "policy"))


@pytest.mark.parametrize("drain", [False, True])
def test_stalled_large_pipe_write_is_bounded_and_bridge_cannot_reuse_partial_frame(drain: bool) -> None:
    # A watchdog prevents a future blocking regression from hanging the suite.
    code = f'''
import os,time,threading
from sextant.environments import PipeEnvironment
from sextant.runtime import AttemptContext
rr,rw=os.pipe(); sr,sw=os.pipe()
port=PipeEnvironment(sr,rw,timeout=.15)
if {drain!r}:
    reader=threading.Thread(target=lambda: os.read(rr,1024)); reader.start()
t=time.monotonic()
try:
    port.invoke({{"episode_id":"test","inputs":{{"text":"x"*2000000}}}},AttemptContext("o","a","m",1,"e"))
    raise AssertionError("unexpected completion")
except TimeoutError:
    assert time.monotonic()-t < 1
try:
    port.invoke({{"episode_id":"test"}},AttemptContext("o","b","m",1,"f"))
    raise AssertionError("partial frame reused")
except ConnectionError:
    pass
print("bounded")
'''
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3, check=False)
    assert result.returncode == 0 and result.stdout.strip() == "bounded", result.stderr


def test_normal_pipe_exchange_and_unknown_runtime_failure(tmp_path: Path) -> None:
    rr, rw = os.pipe()
    sr, sw = os.pipe()
    try:
        port = PipeEnvironment(sr, rw, timeout=1)

        def broker() -> None:
            with os.fdopen(os.dup(rr), "rb") as stream:
                request = json.loads(stream.readline())
            response = {"schema_version": 1, "protocol": "sextant-eval-env/1", "episode_id": "test",
                        "request_id": request["request_id"], "status": "completed", "observations": []}
            response["receipt"] = {"input_hash": digest(request), "output_hash": digest(response)}
            os.write(sw, (canonical(response) + "\n").encode())

        worker = threading.Thread(target=broker)
        worker.start()
        assert port.invoke({"episode_id": "test"}, AttemptContext("o", "a", "m", 1, "e")).receipt.status == "completed"
        worker.join(timeout=2)
        assert not worker.is_alive()
        port.timeout = .05
        app, _ = fresh(tmp_path, proposal([experiment("experiment", data="x" * 200000)]))
        app.executors["environment"] = port
        result = call(app, "run", start(app), steps=2)
        assert result.status == "reconciliation_required"
        receipt = call(app, "inspect", result).outputs["saved_receipts"][-1]
        assert receipt["receipt"]["status"] == "unknown"
    finally:
        for fd in (rr, rw, sr, sw):
            os.close(fd)


def test_external_links_resolve_and_unresolved_visible(tmp_path: Path) -> None:
    env = Capture("environment", {"observations": [{"observation_id": "external-1", "source_id": "sensor", "content": 4,
        "observed_at": utcnow(), "measurement_protocol": "fixture/1", "supports_scope": {"phase": 0}}]})
    app, gen = fresh(tmp_path, proposal([experiment()]), env)
    observed = call(app, "run", start(app), steps=2)
    gen.output = {"payload": proposal(links={"Measured four": ["external-1"], "Unknown": ["missing-id"]})}
    compiled = call(app, "run", observed, steps=1)
    inspected = call(app, "inspect", compiled).outputs
    claims = {o["revision"]["payload"]["proposition"]: o["revision"]["payload"] for o in inspected["objects"] if o["revision"]["ref"]["object_kind"] == "claim"}
    assert len(claims["Measured four"]["support"]) == 1 and claims["Measured four"]["state"] == "hypothesis"
    assert claims["Unknown"]["unresolved_support"] == ["missing-id"] and not claims["Unknown"]["support"]
    policy = next(o["revision"]["payload"] for o in inspected["objects"] if o["revision"]["ref"] == inspected["state"]["policy_ref"])
    assert any("missing-id" in reason for reason in policy["unresolved"])


def test_alias_collision_requires_explicit_namespace(tmp_path: Path) -> None:
    app, gen = fresh(tmp_path)
    created = start(app, observations=[{"content": "first source", "source": "manual"}])
    first = call(app, "inspect", created).outputs["state"]["observations"][0]
    env = Capture("environment", {"observations": [{"observation_id": first["object_id"], "source_id": "sensor", "content": 4,
        "observed_at": utcnow(), "measurement_protocol": "fixture/1", "supports_scope": {"phase": 0}}]})
    app.executors["environment"] = env
    gen.output = {"payload": proposal([experiment()])}
    observed = call(app, "run", created, steps=2)
    alias = first["object_id"]
    gen.output = {"payload": proposal(links={"ambiguous": [alias], "internal": [f"object:{alias}"], "external": [f"external:{alias}"]})}
    compiled = call(app, "run", observed)
    claims = {o["revision"]["payload"]["proposition"]: o["revision"]["payload"] for o in call(app, "inspect", compiled).outputs["objects"]
              if o["revision"]["ref"]["object_kind"] == "claim"}
    assert claims["ambiguous"]["unresolved_support"] == [alias] and claims["ambiguous"]["support"] == []
    assert claims["internal"]["support"] == [first]
    assert claims["external"]["support"] and claims["external"]["support"] != [first]


@pytest.mark.parametrize("payload", [{"mission": None}, {"observations": [None]}, {"authorization": []}, {"options": False}])
def test_nested_invalid_input_is_durable_and_single_json(tmp_path: Path, payload: dict[str, Any]) -> None:
    raw = canonical({"kind": "start", "command_id": "malformed", "payload": payload})
    argv = [sys.executable, "-m", "sextant", "--root", str(tmp_path), "agent"]
    first = subprocess.run(argv, input=raw, capture_output=True, text=True, timeout=10, check=False)
    second = subprocess.run(argv, input=raw, capture_output=True, text=True, timeout=10, check=False)
    assert first.returncode == 1 and not first.stderr and first.stdout == second.stdout
    assert json.loads(first.stdout)["status"] == "invalid_input"
    assert call(Application(Store(tmp_path), OWNER), "inspect").outputs["missions"] == []


@pytest.mark.parametrize("field,value", [("package", []), ("mission", []), ("revisions", None), ("heads", [None]), ("artifacts", [])])
def test_malformed_import_containers_are_structured_errors(tmp_path: Path, field: str, value: Any) -> None:
    app, _ = fresh(tmp_path)
    package = {"format": "sextant-export/1", "mission": {}, "revisions": [], "heads": [], "invalidations": [], "events": [], "artifacts": {}}
    if field == "package":
        package = value
    else:
        package[field] = value
    cmd = Command(kind="import", command_id="bad-import", payload={"package": package})
    result = app.execute(cmd)
    assert result.status == "invalid_input" and result == app.execute(cmd)
    assert call(app, "inspect").outputs["missions"] == []


def projection() -> SubmissionProjection:
    return SubmissionProjection(path=["prediction_model"], model_schema={
        "$defs": {"expr": {"oneOf": [{"type": "integer"}, {"type": "boolean"},
            {"type": "object", "required": ["var"], "additionalProperties": False, "properties": {"var": {"type": "string"}}},
            {"type": "object", "required": ["op", "args"], "additionalProperties": False,
             "properties": {"op": {"enum": ["xor", "and", "add", "mul"]}, "args": {"type": "array", "minItems": 2, "maxItems": 2,
                                                                                     "items": {"$ref": "#/$defs/expr"}}}}]}},
        "oneOf": [
            {"type": "object", "required": ["kind", "payload"], "additionalProperties": False,
             "properties": {"kind": {"const": "expression"}, "payload": {"$ref": "#/$defs/expr"}}},
            {"type": "object", "required": ["kind", "payload"], "additionalProperties": False,
             "properties": {"kind": {"const": "table"}, "payload": {"type": "object", "required": ["variables", "rows"], "additionalProperties": False,
                "properties": {"variables": {"type": "array", "items": {"type": "string"}}, "rows": {"type": "array", "items": {
                    "type": "object", "required": ["inputs", "output"], "additionalProperties": False,
                    "properties": {"inputs": {"type": "object"}, "output": {"type": "integer"}}}}}}}},
        ]})


def table_model(rows: list[Any] | None = None) -> dict[str, Any]:
    return {"kind": "table", "payload": {"variables": ["x", "y"], "rows": rows or []}}


def expression_model(op: str = "mul", leaf: Any = 2) -> dict[str, Any]:
    return {"kind": "expression", "payload": {"op": op, "args": [{"var": "x"}, leaf]}}


@pytest.mark.parametrize("enabled", [False, True])
def test_representation_control_through_application_parameters_actions_and_structure(tmp_path: Path, enabled: bool) -> None:
    initial = proposal()
    initial["direct_result"] = {"prediction_model": table_model(), "policy": {"config": 1}, "evidence_refs": ["old"]}
    app, gen = fresh(tmp_path, initial)
    created = start(app, options={"rerepresentation": enabled, "submission_projection": projection().model_dump()})
    first = call(app, "run", created)
    assert not first.errors
    initial_state = call(app, "inspect", first).outputs["state"]["representation_state"]
    changed = call(app, "observe", first, content="new measurement", source="fixture")
    learned = copy.deepcopy(initial)
    learned["candidates"][0]["representation"]["body"]["rows"][0]["output"] = 9
    learned["direct_result"] = {"prediction_model": table_model([{"inputs": {"x": 0, "y": 1}, "output": 1}]),
                                "policy": {"config": 2}, "evidence_refs": ["current"]}
    gen.output = {"payload": learned}
    second = call(app, "run", changed)
    assert not second.errors and second.outputs["steps"][0]["direct_result"] == learned["direct_result"]
    if not enabled:
        assert call(app, "inspect", second).outputs["state"]["representation_state"] == initial_state
    changed = call(app, "observe", second, content="different mechanism", source="fixture")
    learned["direct_result"]["prediction_model"] = expression_model()
    gen.output = {"payload": learned}
    third = call(app, "run", changed)
    assert not third.errors if enabled else third.status == "representation_change"
    inspected = call(app, "inspect", third, events=True)
    assert any(e["kind"] == ("representation.checked" if enabled else "representation.rejected") for e in inspected.events)
    assert all(o["state"] != "running" for o in inspected.outputs["operations"])


@pytest.mark.parametrize("mutate", ["operator", "variable", "arity", "type", "remove", "relocate", "extra", "rename_candidate"])
def test_fixed_expression_rejects_structural_escape_but_learns_scalars(mutate: str) -> None:
    from sextant.domain import Proposal
    raw = proposal()
    raw["direct_result"] = {"prediction_model": expression_model()}
    first = Proposal.model_validate(raw)
    frozen, _ = control(first, enabled=False, projection=projection())
    changed = first.model_copy(deep=True)
    changed.direct_result["prediction_model"] = expression_model(leaf=7)
    assert control(changed, enabled=False, projection=projection(), frozen=frozen)[0] == frozen
    model = changed.direct_result["prediction_model"]
    if mutate == "operator":
        model["payload"]["op"] = "add"
    elif mutate == "variable":
        model["payload"]["args"][0]["var"] = "y"
    elif mutate == "arity":
        model["payload"]["args"].append(1)
    elif mutate == "type":
        model["payload"]["args"][1] = True
    elif mutate == "remove":
        changed.direct_result["prediction_model"] = None
    elif mutate == "relocate":
        changed.direct_result["new_model"] = changed.direct_result.pop("prediction_model")
    elif mutate == "extra":
        model["unmeasured_override"] = {"op": "add"}
    else:
        changed.candidates[0].key = "new-key"
    with pytest.raises(DomainError) as failure:
        control(changed, enabled=False, projection=projection(), frozen=frozen)
    assert failure.value.code == "representation_change"


@pytest.mark.parametrize("initial_model", [None, {"kind": "table", "payload": []}, table_model([
    {"inputs": {"x": 0, "y": 0}, "output": 0}, {"inputs": {"x": 0, "y": 0}, "output": 1}])])
def test_null_or_invalid_submission_cannot_poison_first_structural_freeze(initial_model: Any) -> None:
    from sextant.domain import Proposal
    raw = proposal()
    raw["direct_result"] = {"prediction_model": initial_model}
    candidate = Proposal.model_validate(raw)
    empty, _ = control(candidate, enabled=False, projection=projection())
    assert empty["external"] is None
    candidate.direct_result["prediction_model"] = expression_model()
    frozen, _ = control(candidate, enabled=False, projection=projection(), frozen=empty)
    assert frozen["external"]["kind"] == "expression"


def test_internal_structure_freezes_operators_not_coefficients() -> None:
    from sextant.domain import Proposal
    raw = proposal()
    raw["candidates"][0]["representation"] = {"kind": "argument_graph", "body": {"nodes": [], "outputs": {"v": {"op": "mul", "args": [{"var": "x"}, 2]}}},
                                                "composition_rule": "finite expression"}
    candidate = Proposal.model_validate(raw)
    frozen, _ = control(candidate, enabled=False, projection=None)
    candidate.candidates[0].representation.body["outputs"]["v"]["args"][1] = 5
    assert control(candidate, enabled=False, projection=None, frozen=frozen)[0] == frozen
    candidate.candidates[0].representation.body["outputs"]["v"]["op"] = "add"
    with pytest.raises(DomainError, match="frozen structure"):
        control(candidate, enabled=False, projection=None, frozen=frozen)


def test_projection_declaration_cannot_change_silently_and_remote_schema_is_refused() -> None:
    from sextant.domain import Proposal
    first = Proposal.model_validate(proposal())
    frozen, _ = control(first, enabled=False, projection=projection())
    changed = projection().model_copy(update={"path": ["new_model"]})
    with pytest.raises(DomainError) as failure:
        control(first, enabled=False, projection=changed, frozen=frozen)
    assert failure.value.code == "representation_treatment_changed"
    with pytest.raises(ValueError, match="only its own document"):
        SubmissionProjection(path=["model"], model_schema={"$ref": "https://example.invalid/schema"})
    node: Any = 1
    for _ in range(34):
        node = {"op": "add", "args": [node, 1]}
    with pytest.raises(DomainError, match="depth"):
        expression_shape(node)
