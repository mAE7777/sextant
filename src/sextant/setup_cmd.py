"""`setup`: probe the environment, record the results in config.toml, name the blockers.

Never prints a secret value, only whether a key name is set. It asks nothing itself: the caller
asks the user one plain question per role only where two or more providers are viable.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from sextant.config import load_config, load_raw_config, write_config
from sextant.jev_transport import redact, smoke
from sextant.providers import probe_all

KEY_NAMES = ("TYPESAFE_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
ROLES = ("generator", "challenger", "grader", "paraphraser", "translator", "renderer", "mapper")


def probe(config: dict[str, Any], with_smoke: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "date": datetime.now(UTC).date().isoformat(),
        "keys_present": {k: bool(os.environ.get(k)) for k in KEY_NAMES},
        "providers": probe_all(config),
    }
    result["jev"] = smoke() if with_smoke else {"available": None, "reason": "smoke skipped"}
    return result


def viable_providers(probe_result: dict[str, Any]) -> list[str]:
    return [name for name, p in probe_result["providers"].items() if p.get("available")]


def blockers(config: dict[str, Any], probe_result: dict[str, Any]) -> list[str]:
    out: list[str] = []
    jev = probe_result.get("jev", {})
    if jev.get("available") is False:
        out.append(f"Jev transport unavailable: {jev.get('reason')}. Fix: export TYPESAFE_API_KEY in ~/.zshrc and open a new shell.")
    viable = set(viable_providers(probe_result))
    for role in ROLES:
        chosen = config["providers"].get(role)
        if chosen and chosen not in viable:
            out.append(f"role {role} is configured for provider {chosen}, which is not available now.")
    return out


def render_table(config: dict[str, Any], probe_result: dict[str, Any]) -> str:
    lines = [f"probe date: {probe_result['date']}", "", "keys (set / unset, values never shown):"]
    for k, present in probe_result["keys_present"].items():
        lines.append(f"  {k:<20} {'set' if present else 'unset'}")
    lines.append("")
    lines.append("providers:")
    for name, p in probe_result["providers"].items():
        flag = "available" if p.get("available") else "unavailable"
        extra = p.get("version") or p.get("reason") or p.get("note") or ""
        if p.get("models_count") is not None:
            extra = f"{p['models_count']} models listed; configured: {p.get('configured_model')}"
        lines.append(f"  {name:<22} {flag:<12} {extra}")
    lines.append("")
    jev = probe_result.get("jev", {})
    if jev.get("available"):
        lines.append(f"jev: available, model {jev.get('model_id')}, {jev.get('latency_ms')} ms, smoke answer {jev.get('answer')}")
    else:
        lines.append(f"jev: {jev.get('reason') or 'not probed'}")
    lines.append("")
    lines.append("roles -> providers (from config):")
    viable = set(viable_providers(probe_result))
    for role in ROLES:
        chosen = config["providers"].get(role)
        mark = "ok" if chosen in viable else "NOT AVAILABLE"
        lines.append(f"  {role:<12} {chosen:<22} {mark}")
    bl = blockers(config, probe_result)
    if bl:
        lines.append("")
        lines.append("BLOCKERS:")
        for b in bl:
            lines.append(f"  - {b}")
    # never let a secret value reach stdout, even inside a provider's error text
    return redact("\n".join(lines))


def run_setup(probe_only: bool = False, with_smoke: bool = True) -> tuple[str, dict[str, Any]]:
    config = load_config()
    result = probe(config, with_smoke=with_smoke)
    text = render_table(config, result)
    if not probe_only:
        raw = load_raw_config()
        raw["probe"] = {
            "date": result["date"],
            "jev_available": bool(result["jev"].get("available")),
            "jev_model": str(result["jev"].get("model_id") or ""),
            "viable_providers": viable_providers(result),
            "keys_present": [k for k, v in result["keys_present"].items() if v],
        }
        written = write_config(raw)
        text += f"\n\nconfig written: {written}"
    return text, result
