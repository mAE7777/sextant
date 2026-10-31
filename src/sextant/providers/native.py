"""Native Claude subscription headless, isolated from project and user materials.

Only this trusted adapter can invoke the native client. It returns a proposal and
execution metadata; the application alone controls current state and authorization.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from sextant.domain import Receipt, digest, utcnow
from sextant.isolation import run_isolated, seatbelt_profile
from sextant.jev_transport import redact
from sextant.runtime import AttemptContext, Invocation, NotSent


def native_environment(work: Path) -> dict[str, str]:
    # API keys, alternate endpoints/providers, SDK entrypoints, plugins and inherited
    # model credentials cannot silently replace the selected native subscription route.
    env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG") if k in os.environ}
    env.update({"TMPDIR": str(work), "CLAUDE_CODE_TMPDIR": str(work), "CLAUDE_CODE_SAFE_MODE": "1",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "DISABLE_AUTOUPDATER": "1"})
    return env


class ClaudeHeadless:
    name = "claude-code"

    def __init__(self, *, model: str = "opus", effort: str = "high", timeout: float = 600,
                 binary: str | None = None):
        path = binary or shutil.which("claude")
        self.binary = Path(path).resolve() if path else None
        self.model, self.effort, self.timeout = model, effort, timeout

    def capabilities(self) -> dict[str, Any]:
        if self.binary is None:
            return {"available": False, "reason": "native Claude Code is not installed"}
        with tempfile.TemporaryDirectory(prefix="sextant-cli-help-") as directory:
            env = native_environment(Path(directory))
            result = subprocess.run([str(self.binary), "--help"], capture_output=True, text=True,
                                    env=env, cwd=directory, timeout=20, check=False)
        required = ("--safe-mode", "--tools", "--strict-mcp-config", "--json-schema",
                    "--no-session-persistence", "--effort", "--restricted")
        missing = [flag for flag in required if flag not in result.stdout]
        return {"available": result.returncode == 0 and not missing, "missing_flags": missing,
                "help_exit_code": result.returncode,
                "auth_route": "native-claude.ai-subscription", "cancel": "process-group-on-timeout",
                "reconcile": "unsupported-after-host-crash", "idempotent": False,
                "tool_mode": "disabled", "isolation": "Seatbelt deny-default plus safe-mode"}

    def auth_status(self) -> dict[str, Any]:
        if self.binary is None:
            return {"loggedIn": False}
        with tempfile.TemporaryDirectory(prefix="sextant-auth-") as directory:
            process = subprocess.run([str(self.binary), "--safe-mode", "auth", "status"],
                                     cwd=directory, env=native_environment(Path(directory)),
                                     capture_output=True, text=True, timeout=20, check=False)
        try:
            data = json.loads(process.stdout)
        except ValueError:
            return {"loggedIn": False, "reason": "native auth metadata unavailable"}
        # Never persist account identity, tokens, or the full native configuration.
        selected = {k: data.get(k) for k in ("loggedIn", "authMethod", "apiProvider", "subscriptionType")}
        configuration = Path.home() / ".claude.json"
        try:
            settings = json.loads(configuration.read_text())
            extra = settings.get("oauthAccount", {}).get("hasExtraUsageEnabled")
        except (OSError, ValueError):
            extra = None
        selected["local_extra_usage_enabled"] = extra
        selected["extra_usage_evidence"] = "native cached account setting; not billing attestation"
        return selected

    def _profile(self, work: Path) -> str:
        assert self.binary is not None
        home = Path.home()
        reads = [work, self.binary, home / ".claude.json", home / ".claude" / ".credentials.json",
                 home / "Library" / "Keychains", home / "Library" / "Preferences" / "com.apple.security.plist"]
        # Native credential access is retained; no project/history/skills/config directory is granted.
        return seatbelt_profile(reads=reads, writes=[work], network=True, native_auth=True)

    def preflight(self) -> tuple[list[str], dict[str, Any]]:
        """Every reason the native route cannot be confirmed, before anything is sent."""
        if self.binary is None:
            missing = ("native Claude Code CLI not found: no `claude` executable on PATH; "
                       "install Claude Code or give Sextant its binary path")
            return [missing], {"loggedIn": False}
        problems: list[str] = []
        try:
            capabilities = self.capabilities()
            auth = self.auth_status()
        except (OSError, subprocess.SubprocessError) as exc:
            return [f"native Claude Code CLI at {self.binary} could not be started: {type(exc).__name__}: {exc}"], {"loggedIn": False}
        if capabilities.get("help_exit_code") not in (None, 0):
            problems.append(f"native Claude Code CLI `--help` exited {capabilities['help_exit_code']}")
        if capabilities.get("missing_flags"):
            problems.append("native Claude Code CLI lacks required isolation flags: " + ", ".join(capabilities["missing_flags"]))
        if not auth.get("loggedIn") or auth.get("authMethod") != "claude.ai" or auth.get("apiProvider") != "firstParty" or not auth.get("subscriptionType"):
            problems.append("native Claude subscription login unavailable in this execution context")
        elif auth.get("local_extra_usage_enabled") is not False:
            problems.append("extra usage is enabled or unknown; subscription-only execution cannot be established")
        return problems, auth

    def invoke(self, request: dict[str, Any], context: AttemptContext) -> Invocation:
        started = utcnow()
        model = str(request.get("model", self.model))
        effort = str(request.get("effort", self.effort))
        problems, auth = self.preflight()
        if problems:
            raise NotSent("runtime_unavailable", "; ".join(problems),
                          receipt_fields={"requested_model": model, "requested_effort": effort,
                                          "auth_route": "native-claude.ai-subscription", "auth_evidence": json.dumps(auth)})
        assert self.binary is not None
        schema = request["schema"]
        Draft202012Validator.check_schema(schema)
        with tempfile.TemporaryDirectory(prefix="sextant-generator-") as directory:
            work = Path(directory).resolve()
            profile = self._profile(work)
            command = [str(self.binary), "--safe-mode", "--restricted", "--print", "--tools", "",
                       "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                       "--setting-sources", "", "--permission-mode", "dontAsk",
                       "--permission-prompts", "none", "--disable-slash-commands", "--no-chrome",
                       "--no-session-persistence", "--output-format", "stream-json", "--verbose",
                       "--model", model, "--effort", effort,
                       "--json-schema", json.dumps(schema), "--system-prompt", str(request["system"])]
            result = run_isolated(command, cwd=work, profile=profile, stdin=str(request["prompt"]),
                                  environment=native_environment(work), timeout=self.timeout)
        messages: list[dict[str, Any]] = []
        for line in result.stdout.splitlines():
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    messages.append(parsed)
            except ValueError:
                continue
        initial = next((m for m in messages if m.get("type") == "system" and m.get("subtype") == "init"), {})
        final = next((m for m in reversed(messages) if m.get("type") == "result"), {})
        output = final.get("structured_output")
        errors = []
        tools = initial.get("tools")
        # StructuredOutput is a native response formatter, not file/network/process authority.
        unexpected_tools = [t for t in tools or [] if t not in ("StructuredOutput",)]
        unexpected_plugins = [p for p in initial.get("plugins", [])
                              if p.get("path") != "builtin" or p.get("name") != "agents-md"]
        if not initial or tools is None or unexpected_tools or initial.get("mcp_servers") or unexpected_plugins:
            errors.append("native client did not demonstrate the required empty tool/MCP/plugin surface")
        if result.returncode != 0 or final.get("is_error") or not final:
            errors.append("native client failed: " + str(final.get("subtype", "no-result")))
            if final.get("result"):
                errors.append("native result diagnostic: " + redact(str(final["result"])[:1500]))
            if result.stderr:
                errors.append("native diagnostic: " + redact(result.stderr[-1500:]))
        if output is None and final.get("result"):
            try:
                output = json.loads(final["result"])
            except (TypeError, ValueError):
                errors.append("native result is not structured JSON")
        validation = list(Draft202012Validator(schema).iter_errors(output))
        if validation:
            errors.append("output schema: " + "; ".join(e.message[:200] for e in validation[:3]))
        observed = initial.get("model")
        usage = {"usage": final.get("usage", {}), "model_usage": final.get("modelUsage", {}),
                 "num_turns": final.get("num_turns"), "duration_ms": final.get("duration_ms")}
        receipt = Receipt(executor=self.name, requested_model=model, observed_model=observed,
                          requested_effort=effort, observed_effort=initial.get("effort"),
                          session=final.get("session_id", initial.get("session_id")),
                          auth_route="native-claude.ai-subscription", auth_evidence=json.dumps(auth),
                          started_at=started, ended_at=utcnow(), request_hash=digest(request),
                          output_hash=digest(output), status="unknown" if result.timed_out else ("failed" if errors else "completed"),
                          exit_code=result.returncode, timed_out=result.timed_out, usage=usage,
                          isolation={"mechanism": "Seatbelt deny-default", "profile_hash": digest(profile),
                                     "safe_mode": True, "tools": tools, "mcp_servers": initial.get("mcp_servers"),
                                     "plugins": initial.get("plugins"), "project_context": "denied",
                                     "retry_events": [{k: m.get(k) for k in ("attempt", "error", "error_status", "no_response")}
                                                      for m in messages if m.get("subtype") == "api_retry"],
                                     "event_shapes": [{"type": m.get("type"), "subtype": m.get("subtype"),
                                                       "is_error": m.get("is_error"),
                                                       "block_types": [b.get("type") for b in m.get("message", {}).get("content", []) if isinstance(b, dict)]}
                                                      for m in messages],
                                     "stdout_hash": digest(result.stdout), "stderr_hash": digest(result.stderr)},
                          residual_processes="process group killed and leader reaped" if result.process_group_reaped else "unknown",
                          tool_state="no executable tools" if not unexpected_tools and tools is not None else "unverified",
                          errors=errors + (["process timed out; remote outcome unknown"] if result.timed_out else []))
        return Invocation(receipt, output)

    def cancel(self, context: AttemptContext) -> dict[str, Any]:
        return {"status": "unsupported", "reason": "caller may stop the process; remote cancellation is not guaranteed"}

    def reconcile(self, context: AttemptContext) -> Invocation | None:
        return None
