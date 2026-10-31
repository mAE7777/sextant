"""Provider registry: name -> implementation, built from config."""

from __future__ import annotations

from typing import Any

from sextant.providers.base import Provider
from sextant.providers.claude_code import ClaudeCodeProvider, ClaudeCodeSubagentProvider
from sextant.providers.codex import CodexProvider
from sextant.providers.openai_compat import PRESETS, OpenAICompatProvider

PROVIDER_NAMES = ("claude-code", "claude-code-subagent", "codex", *PRESETS.keys())


def get_provider(name: str, config: dict[str, Any]) -> Provider:
    pconf = config.get("providers", {})
    if name == "claude-code":
        return ClaudeCodeProvider()
    if name == "claude-code-subagent":
        return ClaudeCodeSubagentProvider()
    if name == "codex":
        return CodexProvider(timeout_s=int(pconf.get("codex_timeout_s", 600)), model=pconf.get("codex_model") or None)
    if name in PRESETS:
        return OpenAICompatProvider(name, model=pconf.get(f"{name}_model") or None)
    raise ValueError(f"unknown provider {name}; known: {', '.join(PROVIDER_NAMES)}")


def probe_all(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in PROVIDER_NAMES:
        try:
            out[name] = get_provider(name, config).probe()
        except Exception as e:  # noqa: BLE001
            out[name] = {"available": False, "reason": f"{type(e).__name__}: {e}"}
    return out
