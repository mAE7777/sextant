"""Constructed CLI/render regression checks. No provider process or model call."""
from __future__ import annotations

import json

import pytest

from sextant import cli
from sextant.domain import CommandResult, canonical
from sextant.render import chinese


def invoke(capsys, root, *args):
    code = cli.main(["--root", str(root), *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def start(capsys, root, *, large=False, native=False):
    payload = {"mission": {"goal": "解释待检验的机制" if native else "计算 7 * (3 + 2)"},
               "options": {"jev_density": "none", "challenger": False}}
    if large:
        payload["observations"] = [{"content": {"blob": "x" * 1_050_000}, "source": "constructed size control"}]
    if native:
        payload["options"]["generator"] = "claude-code"
        payload["authorization"] = {"tools": ["claude-code"], "send_to": {"claude-code": ["*"]}}
    code, out, err = invoke(capsys, root, "start", "--payload", json.dumps(payload), "--json")
    assert code == 0 and not err
    return json.loads(out)


def query(capsys, root, mission):
    code, out, err = invoke(capsys, root, "inspect", mission, "--json")
    assert code == 0 and not err
    return json.loads(out)


def view(*, deterministic=False, certificate_valid=True, outcome=False):
    def ref(kind, name):
        return {"object_kind": kind, "object_id": name, "revision_id": "rev-" + name, "schema_version": 1}
    policy = ref("policy", "policy")
    proposal = ref("proposal", "proposal")
    experiment = {"key": "distinguish-cache", "adapter": "environment", "action": "experiment",
                  "inputs": {"workload": "mixed-regions"}, "purpose": "区分缓存键与过期数据",
                  "distinguishes": ["cache-key", "stale-data"], "changes_world": False,
                  "expected_observations": ["两个区域返回不同价格"]}
    objects = []
    def add(kind, name, payload, valid=True):
        objects.append({"revision": {"ref": ref(kind, name), "payload": payload},
                        "current_validity": {"valid": valid, "reasons": [] if valid else ["mission changed"]}})
    add("proposal", "proposal", {"understanding": "先区分两种缓存机制",
        "unknowns": ["尚不知道更新信号是否可靠"],
        "candidates": [{"key": "cache-key", "mechanism": "区域未进入缓存键", "assumptions": ["区域价格不同"]},
                       {"key": "stale-data", "mechanism": "价格更新后缓存未失效", "assumptions": ["存在更新事件"]}]})
    add("policy", "policy", {"summary": "先实验再选择修复", "branches": [{"condition": "缓存键有误", "action": "加入区域"}],
        "unresolved": ["更新事件来源未核验"], "reversal_triggers": ["更新信号缺失"],
        "ordering": "deterministic" if deterministic else "conditional-partial-order", "basis": "构造输入中的条件比较",
        "assumptions": ["测量覆盖两个区域"], "next_experiments": [experiment]})
    add("prediction", "prediction", {"status": "resolved", "question": "混合区域请求是否全部正确",
        "horizon": "2030-01-01T00:00:00Z", "resolver": "field-equals/1", "outcome": outcome,
        "outcome_domain": [True, False], "resolution_refs": [ref("observation", "measurement")]})
    add("judgment", "judgment", {"authority": "advisory", "card_id": "R6", "derived": {"stop": "unknown"}})
    if deterministic:
        add("certificate", "certificate", {"authority": "deterministic", "domain": "explicit finite arithmetic only",
            "protocol": "arithmetic/1", "result": {"result": 35}, "identity_attestation": False}, certificate_valid)
    return CommandResult(command_id="constructed-view", mission_id="mission", status="ok", outputs={
        "state": {"status": "waiting", "policy_ref": policy, "proposal_ref": proposal,
                  "history": [{"reason": "new contrary observation", "changed": True}],
                  "pending_experiments": [{"branch": "cache-key", "experiment": experiment}]},
        "objects": objects, "effects_executed": 0})


def test_human_start_is_readable(capsys, tmp_path):
    code, out, err = invoke(capsys, tmp_path / "runtime", "start", "--goal", "计算 7 * (3 + 2)")
    assert code == 0 and not err
    assert "目标：计算 7 * (3 + 2)" in out
    assert '"authorization_ref"' not in out


def test_mission_list_does_not_dump_state(capsys, tmp_path):
    result = start(capsys, tmp_path / "runtime")
    code, out, err = invoke(capsys, tmp_path / "runtime", "inspect")
    assert code == 0 and not err
    assert result["mission_id"] in out
    assert '"observations"' not in out and '"options"' not in out


def test_human_export_large_package_stays_readable(capsys, tmp_path):
    root = tmp_path / "runtime"
    created = start(capsys, root, large=True)
    file = tmp_path / "export.json"
    code, out, err = invoke(capsys, root, "export", created["mission_id"], "--output", str(file))
    assert code == 0 and not err
    assert file.stat().st_size > 1_000_000
    assert len(out) < 5000
    assert str(file) in out and "SHA-256" in out
    code, raw, err = invoke(capsys, root, "export", created["mission_id"], "--json")
    assert code == 0 and not err
    assert json.loads(raw)["outputs"] == json.loads(file.read_text())


def test_human_export_without_file_explains_how_to_save(capsys, tmp_path):
    root = tmp_path / "runtime"
    created = start(capsys, root)
    code, out, err = invoke(capsys, root, "export", created["mission_id"])
    assert code == 0 and not err
    assert "--output" in out and "--json" in out and "尚未写入文件" in out
    assert '"artifacts"' not in out


def test_missing_revision_is_not_presented_as_stale_conflict(capsys, tmp_path):
    root = tmp_path / "runtime"
    created = start(capsys, root)
    code, out, err = invoke(capsys, root, "pause", created["mission_id"])
    assert code == 1
    assert "expected_revision_required" in out + err
    assert "--expected-revision" in out + err and "inspect" in out + err
    assert "conflict" not in out + err
    assert query(capsys, root, created["mission_id"])["current_revision"] == created["current_revision"]


def test_stale_revision_has_actionable_recovery(capsys, tmp_path):
    root = tmp_path / "runtime"
    created = start(capsys, root)
    code, out, err = invoke(capsys, root, "pause", created["mission_id"], "--expected-revision", "stale")
    assert code == 1
    assert "conflict" in out + err
    assert "--expected-revision" in out + err and "--root" in out + err and "新" in out + err
    assert query(capsys, root, created["mission_id"])["current_revision"] == created["current_revision"]


def test_report_exposes_reasoning_and_queued_experiment():
    out = chinese(view())
    for fact in ["先区分两种缓存机制", "尚不知道更新信号是否可靠", "区域未进入缓存键",
                 "价格更新后缓存未失效", "测量覆盖两个区域", "区分缓存键与过期数据",
                 "mixed-regions", "cache-key", "stale-data", "new contrary observation"]:
        assert fact in out
    assert "待执行实验" in out and "建议" in out


@pytest.mark.parametrize("value", [False, 0])
def test_prediction_outcome_is_not_lost_when_false_or_zero(value):
    out = chinese(view(outcome=value))
    assert "结果：" + canonical(value) in out
    assert "measurement" in out


def test_deterministic_certificate_is_distinct_from_advisory_judgment():
    out = chinese(view(deterministic=True))
    assert "确定性核验" in out and "explicit finite arithmetic only" in out and "35" in out
    assert "R6" in out and "advisory" in out and "建议" in out


def test_invalid_certificate_does_not_claim_current_authority():
    out = chinese(view(deterministic=True, certificate_valid=False))
    assert "历史确定性记录" in out and "已失效" in out
    assert "确定性核验（当前有效）" not in out


def test_arithmetic_actual_cli_has_reported_finite_result(capsys, tmp_path):
    root = tmp_path / "runtime"
    created = start(capsys, root)
    code, out, err = invoke(capsys, root, "run", created["mission_id"],
                            "--expected-revision", created["current_revision"])
    assert code == 0 and not err
    assert "确定性核验" in out and "35" in out and "explicit finite arithmetic only" in out
    assert '"pending_experiments"' not in out
    snapshot = query(capsys, root, created["mission_id"])
    assert snapshot["outputs"]["state"]["status"] == "complete"


@pytest.mark.parametrize("mode", ["json", "agent"])
def test_machine_result_is_exactly_the_application_protocol(capsys, tmp_path, monkeypatch, mode):
    fixed = CommandResult(command_id="stable", mission_id="mission", status="conflict",
                          outputs={"nested": {"opaque": "x" * 2048}},
                          errors=[{"code": "conflict", "message": "stable message"}], next_actions=["inspect"])
    seen = []
    class App:
        def __init__(self, *args, **kwargs):
            pass
        def execute(self, command):
            seen.append(command)
            return fixed
    cli.Store(tmp_path / "runtime")
    monkeypatch.setattr(cli, "Application", App)
    if mode == "agent":
        from io import StringIO
        monkeypatch.setattr(cli.sys, "stdin", StringIO('{"command_id":"stable","kind":"inspect","mission_id":"mission"}'))
        code, out, err = invoke(capsys, tmp_path / "runtime", "agent")
    else:
        code, out, err = invoke(capsys, tmp_path / "runtime", "inspect", "mission", "--command-id", "stable", "--json")
    assert code == 1 and not err
    assert out == canonical(fixed) + "\n" and len(seen) == 1


def test_backup_and_examples_are_readable_and_json_is_explicit(capsys, tmp_path):
    root = tmp_path / "runtime"
    start(capsys, root)
    code, out, err = invoke(capsys, root, "backup", str(tmp_path / "backup"))
    assert code == 0 and not err and "备份" in out and not out.startswith("{")
    code, out, err = invoke(capsys, root, "examples", str(tmp_path / "examples"))
    assert code == 0 and not err and "示例" in out and not out.startswith("{")
    code, out, err = invoke(capsys, root, "backup", str(tmp_path / "backup-json"), "--json")
    assert code == 0 and not err and json.loads(out)["status"] == "backed_up"


def test_missing_native_binary_is_actionable_without_native_calls(capsys, tmp_path, monkeypatch):
    from sextant.providers import native
    monkeypatch.setattr(native.shutil, "which", lambda name: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("No native/auth/help/model subprocess is permitted")
    monkeypatch.setattr(native.subprocess, "run", forbidden)
    root = tmp_path / "runtime"
    created = start(capsys, root, native=True)
    code, out, err = invoke(capsys, root, "run", created["mission_id"],
                            "--expected-revision", created["current_revision"])
    assert code == 1 and "runtime_unavailable" in out + err
    assert "PATH" in out + err and "guide runtime" in out + err
    snapshot = query(capsys, root, created["mission_id"])
    assert snapshot["outputs"]["state"]["status"] == "blocked"



def test_json_missing_revision_retains_application_error(capsys, tmp_path):
    root = tmp_path / "runtime"
    created = start(capsys, root)
    code, out, err = invoke(capsys, root, "pause", created["mission_id"], "--json")
    assert code == 1 and not err
    result = json.loads(out)
    assert result["status"] == "revision_required" and result["errors"][0]["code"] == "revision_required"


def test_runtime_hint_preserves_the_recorded_backend():
    result = CommandResult(command_id="codex-missing", status="runtime_unavailable",
        errors=[{"code":"runtime_unavailable","message":"Codex executable not found; configured binary unavailable"}])
    out = chinese(result)
    assert "Codex executable not found" in out and "PATH" in out and "guide runtime" in out


def test_broker_registers_codex_without_invoking_it(capsys, tmp_path, monkeypatch):
    services = {}
    class Pipe:
        def __init__(self, *args, **kwargs):
            pass
    class Broker:
        def __init__(self, name, pipe, episode_id, *, phase):
            self.name, self.episode_id, self.phase = name, episode_id, phase
        def invoke(self, request):
            pytest.fail("registration must not invoke a service")
    class App:
        def __init__(self, *args, executors, **kwargs):
            services.update(executors)
        def execute(self, command):
            return CommandResult(command_id=command.command_id, status="ok")
    monkeypatch.setattr(cli, "PipeEnvironment", Pipe)
    monkeypatch.setattr(cli, "BrokerExecutor", Broker)
    monkeypatch.setattr(cli, "Application", App)
    code, out, err = invoke(capsys, tmp_path / "runtime", "--broker-read-fd", "99",
        "--broker-write-fd", "100", "--broker-services", "--episode-id", "public-episode",
        "--phase", "2", "start", "--goal", "constructed registration", "--json")
    assert code == 0 and not err and json.loads(out)["status"] == "ok"
    assert set(services) == {"environment", "codex", "claude-code", "typesafe", "sandbox.python"}
    assert services["codex"].episode_id == "public-episode" and services["codex"].phase == 2


def test_human_mutation_does_not_merge_a_racing_new_revision(capsys, tmp_path, monkeypatch):
    commands = []
    class App:
        def __init__(self, *args, **kwargs):
            pass
        def execute(self, command):
            commands.append(command.kind)
            if command.kind == "start":
                return CommandResult(command_id=command.command_id, mission_id="mission", status="created",
                                     current_revision="revision-A", outputs={"state": {"status": "active"}})
            assert command.kind == "inspect"
            return CommandResult(command_id=command.command_id, mission_id="mission", status="ok",
                current_revision="revision-B", outputs={"objects": [{"revision": {
                    "ref": {"object_kind": "mission"}, "payload": {"goal": "NEWER GOAL", "goal_source": "newer"}},
                    "current_validity": {"valid": True}}]})
    monkeypatch.setattr(cli, "Application", App)
    code, out, err = invoke(capsys, tmp_path / "runtime", "start", "--goal", "older goal")
    assert code == 0 and not err
    assert commands == ["start", "inspect"]
    assert "revision-A" in out and "NEWER GOAL" not in out and "inspect" in out


@pytest.mark.parametrize(('outcome', 'original', 'disposition'), [
    (True, False, 'recomputed'), (False, True, 'recomputed'),
    (0, 1, 'recomputed'), (None, False, 'invalid-contract'),
])
def test_human_report_exposes_retrospective_provenance_without_mutating_json(outcome, original, disposition):
    result = view(outcome=outcome)
    prediction = next(item['revision']['payload'] for item in result.outputs['objects']
                      if item['revision']['ref']['object_kind'] == 'prediction')
    prediction['status'] = 'invalid' if disposition == 'invalid-contract' else 'resolved'
    prediction['resolution_correction'] = {
        'disposition': disposition, 'confirming_sample': False, 'original_outcome': original,
        'supersedes_prediction_ref': {'object_id': 'original_prediction', 'revision_id': 'a' * 64,
                                     'object_kind': 'prediction', 'schema_version': 1},
        'correction_reason': '保留同一测量与原问题，修正保存的派生计算',
        'original_frozen_at': '2026-09-01T00:00:00+00:00',
        'contract_error': 'unsupported original frozen string selector' if disposition == 'invalid-contract' else None,
    }
    before = canonical(result)
    out = chinese(result)
    assert '保存的事后更正：' + disposition in out
    assert '新增确认样本标记：false' in out
    assert '原结果：' + canonical(original) in out and 'original_prediction' in out
    assert '保留同一测量与原问题' in out and '2026-09-01T00:00:00+00:00' in out
    if disposition == 'invalid-contract':
        assert '原结算规范不可用，旧结果仅供追溯' in out
        assert 'unsupported original frozen string selector' in out
    else:
        assert '结果：' + canonical(outcome) in out
    assert canonical(result) == before
