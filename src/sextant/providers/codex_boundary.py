"""Package-bound qualification for the reviewed configured native client route.

This is trusted package integrity and finite route qualification, not a sandbox
for a malicious host, backend identity, or a caller-issued enable token.
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path
from typing import Any

QUALIFICATION_SHA256 = "efc0127b76141ca7dc6f7e7f9f5e380ccea7e8b2ffd154fc12766123363012e6"


def boundary_status(binary: Path | None, host: Path | None, model: str, effort: str) -> dict[str, Any]:
    try:
        if platform.system() != "Darwin":
            raise ValueError("native route requires the qualified macOS boundary")
        package = Path(__file__).resolve().parent.parent
        path = package / "resources/native-codex-qualification.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != QUALIFICATION_SHA256:
            raise ValueError("installed qualification record changed or is unavailable")
        record = json.loads(raw)
        if (record["schema_version"] != 1 or record["scope"] != "configured-client-pure-proposals"
                or record["auth_route"] != "native-codex-chatgpt-subscription"
                or record["pairs"] != [{"model": "gpt-6.1-sol", "effort": "xhigh"}]
                or (model, effort) != ("gpt-6.1-sol", "xhigh")):
            raise ValueError("requested route is outside qualified Sol/xhigh scope")
        if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                   for value in record["evidence"].values()):
            raise ValueError("qualification evidence is incomplete")
        if set(record["source_hashes"]) != {"providers/codex_native.py", "providers/codex_transport.py", "isolation.py", "domain.py", "jev_transport.py"}:
            raise ValueError("complete native source boundary binding is required")
        for name, expected in record["source_hashes"].items():
            source = (package / name).resolve(strict=True)
            if not source.is_relative_to(package) or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError("native route source bytes changed")
        if binary is None or host is None:
            raise ValueError("qualified native client/host binaries unavailable")
        if (hashlib.sha256(binary.read_bytes()).hexdigest() != record["binary_sha256"]
                or hashlib.sha256(host.read_bytes()).hexdigest() != record["host_sha256"]):
            raise ValueError("native client/host version changed; requalification required")
        return {"qualified": True, "scope": record["scope"], "qualification_sha256": QUALIFICATION_SHA256,
                "model": model, "effort": effort, "backend_identity": "unknown",
                "trust_assumptions": record["trust_assumptions"]}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {"qualified": False, "reason": "SKIP: " + str(exc), "backend_identity": "unknown"}
