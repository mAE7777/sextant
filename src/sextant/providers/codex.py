"""Headless Codex as the second model family, on subscription.

    codex exec --skip-git-repo-check --ephemeral -s read-only \
        --output-schema <schema.json> -o <out.json> -   (prompt on stdin)

The final message is constrained to the request's JSON Schema and lands in <out.json>.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from sextant.providers.base import Answer, Request, requests_dir, schema_for


class CodexProvider:
    name = "codex"
    family = "openai"
    isolation: Literal["same-session", "isolated", "n/a"] = "isolated"

    def __init__(self, timeout_s: int = 600, binary: str | None = None, model: str | None = None):
        self.timeout_s = timeout_s
        self.binary = binary or os.environ.get("SEXTANT_CODEX_BIN", "codex")
        self.model = model

    def command(self, schema_path: Path, out_path: Path) -> list[str]:
        cmd = [
            self.binary,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "-s",
            "read-only",
            "--output-schema",
            str(schema_path),
            "-o",
            str(out_path),
        ]
        if self.model:
            cmd += ["-m", self.model]
        cmd.append("-")
        return cmd

    def fulfil(self, request: Request, decision_dir: Path) -> Answer | None:
        d = requests_dir(decision_dir, create=True)
        schema_path = d / f"{request.id}.schema.json"
        out_path = d / f"{request.id}.codex.json"
        schema_path.write_text(json.dumps(schema_for(request.answer_schema), indent=2))
        if out_path.exists():
            out_path.unlink()
        prompt = (
            request.prompt
            + "\n\nRespond with a single JSON object that conforms to the provided output schema. "
            "Do not read or modify any files. Do not run commands."
        )
        proc = subprocess.run(
            self.command(schema_path, out_path),
            input=prompt,
            text=True,
            capture_output=True,
            timeout=self.timeout_s,
            cwd=str(decision_dir),
            check=False,
        )
        if proc.returncode != 0 or not out_path.exists():
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise RuntimeError(f"codex exec failed (exit {proc.returncode}): {tail}")
        raw = out_path.read_text().strip()
        try:
            answer_obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"codex output is not JSON: {e}: {raw[:500]}") from e
        return Answer(
            request_id=request.id,
            answer=answer_obj,
            provider=self.name,
            family=self.family,
            isolation=self.isolation,
            model=self.model,
        )

    def probe(self) -> dict[str, Any]:
        path = shutil.which(self.binary)
        if not path:
            return {"available": False, "reason": f"{self.binary} not on PATH"}
        try:
            proc = subprocess.run([self.binary, "--version"], capture_output=True, text=True, timeout=20, check=False)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {"available": False, "reason": f"{type(e).__name__}"}
        version = (proc.stdout or proc.stderr).strip()
        return {"available": proc.returncode == 0, "version": version, "path": path}
