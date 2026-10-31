"""Offline transport controls; fake auth is never a subscription certificate."""
import inspect
import json
import tomllib
from pathlib import Path

import pytest

from sextant.isolation import ProcessResult
from sextant.providers import codex_native as native
from sextant.providers.codex_transport import encode_request
from sextant.runtime import AttemptContext, NotSent

REQUEST = {"model": "gpt-6-astra", "effort": "max", "system": "Return the answer.",
           "prompt": "Public fixture", "schema": {"type": "object", "required": ["ready"],
           "properties": {"ready": {"const": True}}, "additionalProperties": False}}
CONTEXT = AttemptContext("op", "attempt", "mission", 1, "primary")


@pytest.fixture
def provider(tmp_path, monkeypatch):
    binary = tmp_path / "codex"
    host = tmp_path / "codex-code-mode-host"
    binary.write_text("fixture native binary")
    host.write_text("fixture host binary")
    auth = tmp_path / "auth.json"
    auth.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "fixture-only"},
                                "OPENAI_API_KEY": None, "parent_settings": "must not import"}))
    p = native.CodexHeadless(binary=str(binary), host_binary=str(host), auth_path=auth, temp_root=tmp_path,
                             evidence_root=tmp_path / "evidence")
    # Offline transport fixtures explicitly bypass the unconditional production
    # guard. These tests cannot qualify the native permissions boundary.
    monkeypatch.setattr(p, "_require_boundary", lambda fields: None)
    monkeypatch.setattr(p, "_installed_capabilities", lambda: {"available": True, "fixture_only": True})
    def local(args,home,work):
        config = tomllib.loads((home/'config.toml').read_text())
        stdout = json.dumps({'models':[{'slug':config['model'],'shell_type':'disabled','node_repl_disabled':True,
                                       'experimental_supported_tools':[]}]}) if args == ['debug','models'] else ''
        return type('Result',(),{'returncode':0,'stdout':stdout,'stderr':'Logged in using ChatGPT'})()
    monkeypatch.setattr(p, "_local", local)
    def catalog(home,work,requested_model):
        path=work/'model-catalog.json'
        path.write_text(json.dumps({'models':[{'slug':requested_model,'shell_type':'disabled','node_repl_disabled':True}]}))
        return path, {'fixture_only':True,'tool_exposure_attested':False}
    monkeypatch.setattr(p,"_tool_catalog",catalog)
    # Fake native success writes both CLI events and the fresh rollout. Disable
    # this fixture behavior for missing/unreadable-log counterexamples.
    p._fixture_write_rollout = True
    receipt = p._receipt
    def capture(*args, **kwargs):
        fields = inspect.signature(receipt).bind(*args, **kwargs).arguments
        records = native._messages(fields["result"].stdout)
        ids = [e.get("thread_id") for e in records if e.get("type") == "thread.started"]
        home = fields["home"]
        if p._fixture_write_rollout and len(ids) == 1 and not list((home / "sessions").glob("**/*.jsonl")):
            sessions = home / "sessions"
            sessions.mkdir()
            final = [e["item"]["text"] for e in records if e.get("type") == "item.completed"
                     and e.get("item", {}).get("type") == "agent_message"]
            (sessions / ("rollout-" + ids[0] + ".jsonl")).write_text(events(*rollout_events(
                fields["model"], fields["effort"], thread=ids[0], prompt=(fields["transport_request"] or fields["request"])["prompt"],
                final=final[-1] if final else "")) + "\n")
        return receipt(*args, **kwargs)
    monkeypatch.setattr(p, "_receipt", capture)
    return p


def events(*items):
    return "\n".join(json.dumps(e) for e in items)


def rollout_events(model, effort, *, thread="fresh-id", prompt="Public fixture", final='{"ready":true}'):
    return [
        {"type": "session_meta", "payload": {"id": thread}},
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "public-turn"}},
        {"type": "turn_context", "payload": {"model": model, "effort": effort, "turn_id": "public-turn"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
         "content": [{"type": "input_text", "text": prompt}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": final}]}},
        {"type": "event_msg", "payload": {"type": "task_complete", "turn_id": "public-turn"}},
    ]


def success(text='{"ready":true}'):
    text = json.dumps({"json_payload": text})
    return ProcessResult(0, events({"type": "thread.started", "thread_id": "fresh-id"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 4}}), "", False, True)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_actual_effective_client_drift_cannot_release_or_become_zero_send(provider, monkeypatch, phase):
    local = provider._local
    catalog_count = 0
    sends = []
    def drifting(args,home,work):
        nonlocal catalog_count
        if args == ['debug','models']:
            catalog_count += 1
            if catalog_count == (1 if phase == 'before' else 2):
                return type('Result',(),{'returncode':0,'stdout':'{"models":[{"slug":"other-model"}]}','stderr':''})()
        return local(args,home,work)
    monkeypatch.setattr(provider,'_local',drifting)
    monkeypatch.setattr(native,'run_isolated',lambda *a,**kw: (sends.append(kw['stdin']) or success()))
    if phase == 'before':
        with pytest.raises(NotSent,match='effective client tools/model differ'):
            provider.invoke(REQUEST,CONTEXT)
        assert sends == []
    else:
        result = provider.invoke(REQUEST,CONTEXT)
        assert len(sends) == 1 and result.receipt.status == 'failed' and result.output is None
        assert result.receipt.isolation['request_state'] == 'attempted'
        assert result.receipt.isolation['raw_artifacts']['client-catalog-after.json']
        assert any('post-dispatch' in e for e in result.receipt.errors)


@pytest.mark.parametrize("logical", ['{"free":1e999}', r'{"free":"\ud800"}'])
def test_noncanonical_open_logical_json_returns_attempted_receipt_with_raw_evidence(provider, monkeypatch, logical):
    monkeypatch.setattr(native, 'run_isolated', lambda *a, **kw: success(logical))
    invocation = provider.invoke({**REQUEST, 'schema': {'type': 'object'}}, CONTEXT)
    assert invocation.output is None and invocation.receipt.status == 'failed'
    assert invocation.receipt.isolation['request_state'] == 'attempted'
    artifacts = invocation.receipt.isolation['raw_artifacts']
    assert Path(artifacts['stdout.jsonl']['path']).is_file()
    assert Path(artifacts['rollout-0.jsonl']['path']).is_file()
    assert (Path(artifacts['stdout.jsonl']['path']).parent / 'receipt.json').is_file()


@pytest.mark.parametrize("field", ["prompt", "system"])
def test_non_utf8_request_is_refused_before_auth_or_native_entry(provider, monkeypatch, field):
    monkeypatch.setattr(provider, '_copy_auth', lambda *a: pytest.fail('invalid request must not copy auth'))
    monkeypatch.setattr(native, 'run_isolated', lambda *a, **kw: pytest.fail('invalid request must not launch'))
    with pytest.raises(NotSent) as exc:
        provider.invoke({**REQUEST, field: '\ud800'}, CONTEXT)
    assert exc.value.code == 'provider_request_refused'


def test_persistent_raw_destination_is_checked_before_auth_or_send(provider,monkeypatch,tmp_path):
    path=tmp_path/'not-a-directory';path.write_text('public obstruction')
    provider.evidence_root=path
    monkeypatch.setattr(provider,'_copy_auth',lambda *a:pytest.fail('no auth without evidence'))
    monkeypatch.setattr(native,'run_isolated',lambda *a,**kw:pytest.fail('no send without evidence'))
    with pytest.raises(NotSent) as exc:
        provider.invoke(REQUEST,CONTEXT)
    assert exc.value.code=='native_evidence_unavailable'


def test_environment_drops_parent_and_provider_overrides(tmp_path, monkeypatch):
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_THREAD_ID", "HTTP_PROXY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, "do-not-copy")
    env = native.codex_environment(tmp_path / "home", tmp_path / "work", tmp_path / "codex", tmp_path / "host")
    assert not set(env) & {"OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_THREAD_ID", "HTTP_PROXY", "ANTHROPIC_API_KEY"}
    assert env["CODEX_HOME"] == str(tmp_path / "home")
    assert env["HOME"] == env["CODEX_HOME"]


@pytest.mark.parametrize("auth", [None, {}, {"auth_mode": "api", "OPENAI_API_KEY": "fixture-api"},
    {"auth_mode": "chatgpt", "OPENAI_API_KEY": "fixture-api", "tokens": {"access_token": "fixture"}}])
def test_auth_boundary_refuses_without_launch(provider, monkeypatch, auth):
    provider.auth_path.write_text(json.dumps(auth))
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: pytest.fail("no native launch permitted"))
    with pytest.raises(NotSent) as exc:
        provider.invoke(REQUEST, CONTEXT)
    assert exc.value.code == "runtime_unavailable"
    assert exc.value.receipt_fields["requested_model"] == "gpt-6-astra"
    assert not list(provider.temp_root.glob("sextant-codex-*"))


@pytest.mark.parametrize("field,value", [("model", "opus"), ("effort", "high"), ("schema", {"type": "nonsense"})])
def test_explicit_wrong_route_or_contract_is_refused(provider, monkeypatch, field, value):
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(NotSent) as exc:
        provider.invoke({**REQUEST, field: value}, CONTEXT)
    assert exc.value.kind == "refused"


def test_fresh_home_config_auth_scope_and_cleanup_each_invoke(provider, monkeypatch):
    homes = []
    def run(command, **kw):
        home = Path(kw["environment"]["CODEX_HOME"])
        homes.append(home)
        auth = json.loads((home / "auth.json").read_text())
        assert "parent_settings" not in auth and "OPENAI_API_KEY" not in auth
        assert (home / "auth.json").stat().st_mode & 0o777 == 0o600
        config = tomllib.loads((home / "config.toml").read_text())
        assert config["model_catalog_json"] and config["forced_login_method"] == "chatgpt"
        assert (home / "config.toml").stat().st_mode & 0o777 == 0o600
        assert "resume" not in command and "--ignore-user-config" not in command
        assert 'forced_login_method="chatgpt"' in command and "project_doc_max_bytes=0" in command
        assert 'model_provider="openai"' in command and "--output-schema" in command
        assert "--disable" in command and "shell_tool" in command
        assert json.loads(kw["stdin"])["task_packet"] == REQUEST["prompt"]
        assert json.loads((home.parent / "work/schema.json").read_text()) == encode_request(REQUEST)["schema"]
        assert str(provider.auth_path) not in kw["profile"]
        return success()
    monkeypatch.setattr(native, "run_isolated", run)
    first = provider.invoke(REQUEST, CONTEXT)
    second = provider.invoke(REQUEST, CONTEXT)
    assert first.receipt.status == second.receipt.status == "completed"
    assert homes[0] != homes[1] and all(not h.exists() for h in homes)
    assert first.receipt.observed_model is None and first.receipt.observed_effort is None
    assert first.receipt.auth_route == "native-codex-chatgpt-subscription"


def test_sandbox_refusal_is_not_sent(provider, monkeypatch):
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: ProcessResult(
        71, "", "sandbox-exec: sandbox_apply: Operation not permitted\n", False, True))
    with pytest.raises(NotSent) as exc:
        provider.invoke(REQUEST, CONTEXT)
    assert exc.value.code == "isolation_unavailable"
    assert exc.value.receipt_fields["exit_code"] == 71


def test_quota_is_a_real_attempted_rejection_not_not_sent(provider, monkeypatch):
    stream = events({"type": "thread.started", "thread_id": "fresh-id"},
                    {"type": "turn.failed", "error": {"message": "You've hit your usage limit"}})
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: ProcessResult(1, stream, "", False, True))
    response = provider.invoke(REQUEST, CONTEXT)
    assert response.receipt.status == "failed" and response.output is None
    assert response.receipt.isolation["request_state"] == "attempted_rejected"
    assert response.receipt.isolation["availability_failure"] == "quota"


def test_success_quoting_quota_is_not_reclassified(provider, monkeypatch):
    request = {**REQUEST, "schema": {"type": "string"}}
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: success(json.dumps("You've hit your usage limit")))
    response = provider.invoke(request, CONTEXT)
    assert response.receipt.status == "completed"
    assert response.receipt.isolation["availability_failure"] is None


@pytest.mark.parametrize("result", [ProcessResult(1, "", "connection closed", False, True),
                                    ProcessResult(-9, "", "", True, True)])
def test_no_terminal_or_timeout_is_unknown(provider, monkeypatch, result):
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: result)
    response = provider.invoke(REQUEST, CONTEXT)
    assert response.receipt.status == "unknown" and response.output is None


def test_tool_activity_is_not_accepted(provider, monkeypatch):
    result = success()
    result = ProcessResult(0, result.stdout + "\n" + events({"type": "item.completed",
        "item": {"type": "command_execution", "command": "fixture"}}), "", False, True)
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: result)
    response = provider.invoke(REQUEST, CONTEXT)
    assert response.receipt.status == "failed" and response.output is None


def test_schema_rejects_null(provider, monkeypatch):
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: success("null"))
    assert provider.invoke(REQUEST, CONTEXT).receipt.status == "failed"


def test_profile_has_no_parent_home_or_checkout_mount(tmp_path):
    work, home = tmp_path / "work", tmp_path / "home"
    profile = native.codex_profile(work, home, Path('/fixture/native'), Path('/fixture/host'))
    assert '(deny default)' in profile
    assert '(allow process-exec (literal "/fixture/native"))' in profile
    assert '(allow process-fork)' not in profile
    assert '(allow network-bind' not in profile
    assert '(allow network-outbound (remote ip "localhost:*"))' not in profile
    assert '(deny file-write* (literal ' + json.dumps(str(home / 'auth.json')) + '))' in profile
    assert '(deny file-write-unlink (literal ' + json.dumps(str(home)) + '))' in profile
    assert '(deny file-write-unlink (literal ' + json.dumps(str(work)) + '))' in profile
    assert '(subpath ' + json.dumps(str(Path.home())) + ')' not in profile
    assert 'AGENTS.md' in profile and 'CLAUDE.md' in profile
    assert 'remote tcp "*:443"' in profile


@pytest.mark.parametrize("model,effort", [("gpt-6-astra", "max"), ("gpt-6.1-sol", "xhigh")])
@pytest.mark.parametrize("explicit", [False, True])
def test_production_route_blocks_before_auth_capabilities_or_launch(tmp_path, monkeypatch, model, effort, explicit):
    p = native.CodexHeadless(model=model, effort=effort, binary=str(tmp_path / "unknown-client"),
                             host_binary=str(tmp_path / "unknown-host"), auth_path=tmp_path / "must-not-read", temp_root=tmp_path)
    for method in ("_local", "_copy_auth", "_installed_capabilities"):
        monkeypatch.setattr(p, method, lambda *a: pytest.fail("blocked boundary must do no native/auth work"))
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: pytest.fail("must not launch"))
    for method in ("run", "Popen"):
        monkeypatch.setattr(native.subprocess, method, lambda *a, **k: pytest.fail("no subprocess before guard"))
    for method in ("TemporaryDirectory", "mkdtemp"):
        monkeypatch.setattr(native.tempfile, method, lambda *a, **k: pytest.fail("no temporary home before guard"))
    monkeypatch.setenv("SEXTANT_CODEX_BOUNDARY_QUALIFIED", "true")
    assert p.capabilities()["available"] is False
    request = {k: v for k, v in REQUEST.items() if k not in ("model", "effort")}
    if explicit:
        request.update(model=model, effort=effort)
    with pytest.raises(NotSent) as exc:
        p.invoke({**request, "boundary_qualified": True}, CONTEXT)
    assert exc.value.code == "native_boundary_unqualified"
    assert (exc.value.receipt_fields["requested_model"], exc.value.receipt_fields["requested_effort"]) == (model, effort)
    assert exc.value.receipt_fields["isolation"]["actual_native_sends"] == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("requested_model,requested_effort,model,effort,status", [
    ("gpt-6-astra", "max", "gpt-6-astra", "max", "completed"),
    ("gpt-6.1-sol", "xhigh", "gpt-6.1-sol", "xhigh", "completed"),
    ("gpt-6.1-sol", "xhigh", "gpt-6-astra", "max", "failed"),
    ("gpt-6-astra", "max", "gpt-6.1-sol", "xhigh", "failed"),
    ("gpt-6.1-sol", "xhigh", "gpt-6.1-sol", "max", "failed"),
    ("gpt-6-astra", "max", "other-model", "max", "failed"),
    ("gpt-6-astra", "max", "gpt-6-astra", "high", "failed")])
def test_observations_use_exact_native_artifact_not_requested(provider, monkeypatch, requested_model, requested_effort, model, effort, status):
    provider.evidence_root = provider.temp_root / "evidence"
    def run(command, **kw):
        sessions = Path(kw["environment"]["CODEX_HOME"]) / "sessions"
        sessions.mkdir()
        (sessions / "rollout-fresh-id.jsonl").write_text(events(*rollout_events(
            model, effort, prompt=kw["stdin"], final=json.dumps({"json_payload": '{"ready":true}'}))) + "\n")
        return success()
    monkeypatch.setattr(native, "run_isolated", run)
    receipt = provider.invoke({**REQUEST, "model": requested_model, "effort": requested_effort}, CONTEXT).receipt
    observation = receipt.isolation["client_configuration_observation"]
    path = Path(observation["path"])
    assert native._sha(path.read_bytes()) == observation["sha256"]
    payload = json.loads(path.read_text())["payload"]
    assert {key: payload[key] for key in ("model", "effort")} == {"model": model, "effort": effort}
    assert payload["turn_id"] == "public-turn"
    assert receipt.observed_model is None and receipt.observed_effort is None
    assert observation["client_model"] == model and observation["client_effort"] == effort
    assert observation["json_pointer"] == "/payload/model"
    assert receipt.status == status


def test_code_mode_error_does_not_become_false_ok(provider, monkeypatch):
    result = success()
    stream = result.stdout + "\n" + events({"type": "item.completed", "item": {
        "type": "error", "message": "Code mode is unavailable: missing code-mode host"}})
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: ProcessResult(0, stream, "", False, True))
    result = provider.invoke(REQUEST, CONTEXT)
    assert result.receipt.status == "failed" and result.output is None
    assert result.receipt.isolation["availability_failure"] == "tool_host_unavailable"


def test_started_tool_and_unreaped_group_each_prevent_output(provider, monkeypatch):
    stream = success().stdout + "\n" + events({"type": "item.started", "item": {"type": "command_execution"}})
    for result in (ProcessResult(0, stream, "", False, True),
                   ProcessResult(0, success().stdout, "", False, False)):
        monkeypatch.setattr(native, "run_isolated", lambda *a, result=result, **k: result)
        invocation = provider.invoke(REQUEST, CONTEXT)
        assert invocation.receipt.status == "failed" and invocation.output is None


def test_new_default_pair_reaches_guard_with_null_backend_fields(tmp_path):
    p = native.CodexHeadless(binary=str(tmp_path / "unknown-client"), host_binary=str(tmp_path / "unknown-host"),
                             auth_path=tmp_path / "must-not-read", temp_root=tmp_path)
    assert (p.model, p.effort) == ("gpt-6.1-sol", "xhigh")
    request = {k: v for k, v in REQUEST.items() if k not in ("model", "effort")}
    with pytest.raises(NotSent) as exc:
        p.invoke(request, CONTEXT)
    assert exc.value.code == "native_boundary_unqualified"
    assert (exc.value.receipt_fields["requested_model"], exc.value.receipt_fields["requested_effort"]) == (p.model, p.effort)


@pytest.mark.parametrize("model,effort", [("gpt-6-astra", "xhigh"), ("gpt-6.1-sol", "max"),
    ("gpt-6.1-sol-preview", "xhigh"), ("unknown", "max")])
def test_unauthorized_cross_pair_or_alias_is_refused_before_guard(provider, monkeypatch, model, effort):
    monkeypatch.setattr(provider, "_require_boundary", lambda *a: pytest.fail("bad pair must be refused first"))
    with pytest.raises(NotSent) as exc:
        provider.invoke({**REQUEST, "model": model, "effort": effort}, CONTEXT)
    assert exc.value.code == "provider_request_refused" and exc.value.kind == "refused"


@pytest.mark.parametrize("model,effort", [("gpt-6-astra", "max"), ("gpt-6.1-sol", "xhigh")])
def test_explicit_pair_drives_catalog_and_command_not_instance_default(provider, monkeypatch, model, effort):
    calls = []
    def catalog(home, work, requested_model):
        calls.append(requested_model)
        path = work / "catalog.json"
        path.write_text('{}')
        return path, {"fixture_only": True, "tool_exposure_attested": False}
    monkeypatch.setattr(provider, "_tool_catalog", catalog)
    def run(command, **kwargs):
        assert command[command.index("-m") + 1] == model
        assert 'model_reasoning_effort="' + effort + '"' in command
        return success()
    monkeypatch.setattr(native, "run_isolated", run)
    response = provider.invoke({**REQUEST, "model": model, "effort": effort}, CONTEXT)
    assert calls == [model] and response.receipt.status == "completed"
    assert response.receipt.requested_model == model and response.receipt.requested_effort == effort
    assert response.receipt.observed_model is response.receipt.observed_effort is None


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-6.1-sol"])
@pytest.mark.parametrize("envelope", [False, True])
def test_catalog_selects_exact_requested_slug_from_offline_fixture(provider, monkeypatch, tmp_path, model, envelope):
    records = [{"slug": "gpt-6.1-sol-preview"}, {"slug": "gpt-6.1-sol", "marker": "sol"},
               {"slug": "gpt-6-astra", "marker": "astra"}]
    for record in records:
        record.update(tool_mode="code_mode_only", node_repl_disabled=False)
    payload = json.dumps({"models": records} if envelope else records)
    def local(args, home, work):
        assert args == ["debug", "models", "--bundled"]
        return type("Result", (), {"returncode": 0, "stdout": payload})()
    monkeypatch.setattr(provider, "_local", local)
    path, evidence = native.CodexHeadless._tool_catalog(provider, tmp_path, tmp_path, model)
    selected = json.loads(path.read_text())
    selected = selected["models"] if envelope else selected
    assert len(selected) == 1 and selected[0]["slug"] == model
    assert selected[0]["marker"] == ("sol" if model == "gpt-6.1-sol" else "astra")
    assert selected[0]["shell_type"] == "disabled"
    assert selected[0]["node_repl_disabled"] is True and "tool_mode" not in selected[0]
    assert evidence["requested_model"] == evidence["selected_slug"] == model
    assert evidence["tool_exposure_attested"] is False


@pytest.mark.parametrize("records,returncode", [
    ([{"slug": "gpt-6-astra"}, {"slug": "gpt-6.1-sol-preview"}], 0),
    ([{"slug": "gpt-6.1-sol"}, {"slug": "gpt-6.1-sol"}], 0),
    ([{"slug": "gpt-6.1-sol"}], 1), (None, 0)])
def test_missing_ambiguous_or_failed_catalog_refuses_without_fallback(provider, monkeypatch, tmp_path, records, returncode):
    monkeypatch.setattr(provider, "_local", lambda *a: type("Result", (), {
        "returncode": returncode, "stdout": json.dumps({"models": records})})())
    with pytest.raises(NotSent) as exc:
        native.CodexHeadless._tool_catalog(provider, tmp_path, tmp_path, "gpt-6.1-sol")
    assert exc.value.code == "runtime_unavailable"
    assert not (tmp_path / "model-catalog.json").exists()


@pytest.mark.parametrize("model,effort", [("gpt-6-astra", "max"), ("gpt-6.1-sol", "xhigh")])
def test_known_fresh_startup_config_bytes_and_catalog(provider, tmp_path, model, effort):
    home = tmp_path / "public-known-home"
    home.mkdir()
    catalog = tmp_path / "public-catalog.json"
    catalog.write_text('{"models":[]}')
    path, config, evidence = provider._write_configuration(home, catalog, model, effort, "Only this public packet.")
    parsed = tomllib.loads(path.read_text())
    assert (parsed["model"], parsed["model_reasoning_effort"]) == (model, effort)
    assert parsed["model_catalog_json"] == str(catalog)
    assert parsed["model_provider"] == "openai" and parsed["model_providers"] == {}
    assert parsed["forced_login_method"] == "chatgpt" and parsed["project_doc_max_bytes"] == 0
    assert parsed["mcp_servers"] == {} and parsed["web_search"] == "disabled"
    assert all(parsed["features"][feature] is False for feature in native.DISABLED_FEATURES)
    assert "tools" not in parsed and "tools.view_image" not in config
    assert "Only this public packet." in parsed["developer_instructions"]
    assert evidence["sha256"] == native._sha(path.read_bytes()) and path.stat().st_mode & 0o777 == 0o600
    assert not (home / "auth.json").exists()


def install_rollout_result(provider, monkeypatch, payloads, *, model="gpt-6.1-sol", effort="xhigh", headers=1, contexts=1, files=1):
    provider._fixture_write_rollout = False
    def run(command, **kwargs):
        sessions = Path(kwargs["environment"]["CODEX_HOME"]) / "sessions"
        sessions.mkdir()
        normal = rollout_events(model, effort, prompt=kwargs["stdin"],
                                final=json.dumps({"json_payload": '{"ready":true}'}))
        records = ([normal[0]] * headers + [normal[1]] + [normal[2]] * contexts + [normal[3]]
                   + payloads)
        if not any(p.get("type") == "response_item" and p.get("payload", {}).get("role") == "assistant" for p in payloads):
            records.append(normal[4])
        records.append(normal[5])
        for index in range(files):
            (sessions / ("rollout-" + str(index) + "-fresh-id.jsonl")).write_text(events(*records) + "\n")
        return success()
    monkeypatch.setattr(native, "run_isolated", run)


@pytest.mark.parametrize("kind", ["custom_tool_call", "custom_tool_call_output", "function_call", "function_call_output",
    "mcp_tool_call", "dynamic_tool_call", "web_search_call", "file_search_call", "code_interpreter_call",
    "computer_call", "local_shell_call", "image_generation_call", "unknown_future_call", "unknown_future_call_output"])
def test_full_rollout_tool_types_reject_even_with_successful_cli_final(provider, monkeypatch, kind):
    install_rollout_result(provider, monkeypatch, [{"type": "response_item", "payload": {
        "type": kind, "name": "exec", "call_id": "public-tool-fixture", "output": "unsupported custom tool call: exec"}}])
    response = provider.invoke({**REQUEST, "model": "gpt-6.1-sol", "effort": "xhigh"}, CONTEXT)
    assert response.receipt.status == "failed" and response.output is None
    assert response.receipt.isolation["tools_observed"]
    rollout = response.receipt.isolation["matching_fresh_rollouts"][0]
    saved = Path(rollout["path"])
    assert saved.is_file() and native._sha(saved.read_bytes()) == rollout["sha256"]
    assert saved.stat().st_mode & 0o777 == 0o600
    assert response.receipt.observed_model is response.receipt.observed_effort is None


@pytest.mark.parametrize("kind", ["exec_command_begin", "exec_command_end", "patch_apply_begin", "mcp_tool_call_begin"])
def test_full_rollout_eventlog_attempts_reject_without_stdout_tool_records(provider, monkeypatch, kind):
    install_rollout_result(provider, monkeypatch, [{"type": "event_msg", "payload": {"type": kind}}])
    response = provider.invoke({**REQUEST, "model": "gpt-6.1-sol", "effort": "xhigh"}, CONTEXT)
    assert response.receipt.status == "failed" and response.output is None


@pytest.mark.parametrize("change", [{"contexts": 0}, {"contexts": 2}, {"model": None}, {"effort": None},
    {"model": "gpt-6-astra"}, {"effort": "max"}, {"headers": 0}, {"headers": 2}, {"files": 0}, {"files": 2}])
def test_absent_nonunique_or_mismatching_rollout_metadata_withholds_output(provider, monkeypatch, change):
    install_rollout_result(provider, monkeypatch, [], **change)
    response = provider.invoke({**REQUEST, "model": "gpt-6.1-sol", "effort": "xhigh"}, CONTEXT)
    assert response.receipt.status == "failed" and response.output is None
    assert response.receipt.observed_model is response.receipt.observed_effort is None


def test_unreadable_matching_rollout_withholds_output(provider, monkeypatch):
    install_rollout_result(provider, monkeypatch, [])
    read = Path.read_bytes
    def unreadable(path):
        if path.parent.name == "sessions":
            raise PermissionError("public unreadable rollout fixture")
        return read(path)
    monkeypatch.setattr(Path, "read_bytes", unreadable)
    response = provider.invoke({**REQUEST, "model": "gpt-6.1-sol", "effort": "xhigh"}, CONTEXT)
    assert response.receipt.status == "failed" and response.output is None
    assert "unestablished" in response.receipt.tool_state


def test_captured_complete_rollout_survives_private_home_cleanup_without_backend_promotion(provider, monkeypatch):
    install_rollout_result(provider, monkeypatch, [{"type": "response_item", "payload": {
        "type": "reasoning", "summary": [{"type": "summary_text", "text": "Quoted custom_tool_call:exec"}]}}])
    response = provider.invoke({**REQUEST, "model": "gpt-6.1-sol", "effort": "xhigh"}, CONTEXT)
    assert response.receipt.status == "completed" and response.output == {"ready": True}
    assert response.receipt.isolation["full_tool_activity_inspection_complete"] is True
    assert not Path(response.receipt.isolation["fresh_private_home"]).exists()
    rollout = response.receipt.isolation["matching_fresh_rollouts"][0]
    assert Path(rollout["path"]).exists() and native._sha(Path(rollout["path"]).read_bytes()) == rollout["sha256"]
    assert response.receipt.observed_model is response.receipt.observed_effort is None


def test_unknown_raw_action_without_call_suffix_cannot_pass_pure_generation(provider, monkeypatch):
    install_rollout_result(provider, monkeypatch, [{"type": "response_item", "payload": {
        "type": "future_native_action", "target": "private-home"}}])
    result = provider.invoke({**REQUEST, "model": "gpt-6.1-sol", "effort": "xhigh"}, CONTEXT)
    assert result.receipt.status == "failed" and result.output is None


def test_contradictory_raw_final_and_cli_final_cannot_release_output(provider, monkeypatch):
    install_rollout_result(provider, monkeypatch, [{"type": "response_item", "payload": {
        "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": '{"ready":false}'}]}}])
    result = provider.invoke({**REQUEST, "model": "gpt-6.1-sol", "effort": "xhigh"}, CONTEXT)
    assert result.receipt.status == "failed" and result.output is None


def test_extra_unmatched_session_cannot_escape_native_audit(provider, monkeypatch):
    provider._fixture_write_rollout = False
    def run(command, **kwargs):
        sessions = Path(kwargs["environment"]["CODEX_HOME"]) / "sessions"
        sessions.mkdir()
        for thread in ("fresh-id", "unexpected-id"):
            records = rollout_events("gpt-6.1-sol", "xhigh", prompt=kwargs["stdin"],
                                     final=json.dumps({"json_payload": '{"ready":true}'}))
            records[0]["payload"]["id"] = thread
            (sessions / ("rollout-" + thread + ".jsonl")).write_text(events(*records) + "\n")
        return success()
    monkeypatch.setattr(native, "run_isolated", run)
    result = provider.invoke({**REQUEST, "model": "gpt-6.1-sol", "effort": "xhigh"}, CONTEXT)
    assert result.receipt.status == "failed" and result.output is None
    assert any("additional native session" in error for error in result.receipt.errors)
    assert result.receipt.isolation["matching_fresh_rollouts"][0]["transcript_admission_errors"] == []


@pytest.mark.parametrize("extra", ["not JSON", "[]", '{"type":"future_native_event"}',
    '{"type":"turn.completed"}',
    '{"type":"item.completed","item":{"type":"agent_message","text":"{\\\"ready\\\":true}"}}'])
def test_malformed_unknown_or_duplicate_cli_events_withhold_output(provider, monkeypatch, extra):
    original = success()
    result = ProcessResult(0, original.stdout + "\n" + extra, "", False, True)
    monkeypatch.setattr(native, "run_isolated", lambda *a, **k: result)
    response = provider.invoke(REQUEST, CONTEXT)
    assert response.receipt.status == "failed" and response.output is None
