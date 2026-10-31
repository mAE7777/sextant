"""The Claude Code session (or an isolated subagent) as a provider.

The runtime cannot call the session; it writes the request file, prints AWAITING and
exits. The host session reads the request, writes the answer file, and runs `answer <id>`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from sextant.providers.base import Answer, Request, request_path


class ClaudeCodeProvider:
    name = "claude-code"
    family = "anthropic"
    isolation: Literal["same-session", "isolated", "n/a"] = "same-session"

    def fulfil(self, request: Request, decision_dir: Path) -> Answer | None:
        path = request_path(decision_dir, request.id)
        print(f"AWAITING {request.id} {path}")
        print(
            "  Read the request, write the answer to "
            f"{path.with_name(request.id + '.answer.json')} "
            "as {request_id, answer, provider, family, isolation}, then run: answer "
            f"{request.id}"
        )
        return None

    def probe(self) -> dict[str, Any]:
        return {"available": True, "note": "this session is the provider; no key needed"}


class ClaudeCodeSubagentProvider(ClaudeCodeProvider):
    name = "claude-code-subagent"
    isolation: Literal["same-session", "isolated", "n/a"] = "isolated"

    def fulfil(self, request: Request, decision_dir: Path) -> Answer | None:
        path = request_path(decision_dir, request.id)
        print(f"AWAITING {request.id} {path}")
        print(
            "  Spawn an isolated subagent with ONLY this request file; it writes "
            f"{path.with_name(request.id + '.answer.json')} with isolation=isolated; then run: answer {request.id}"
        )
        return None

    def probe(self) -> dict[str, Any]:
        return {"available": True, "note": "isolated subagent of the same family; self-preference reduced, not removed"}
