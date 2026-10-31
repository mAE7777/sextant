"""Run an explicit example through the installed CLI and retain every result.

Example: python workflow.py service-cache-payload.json --output ./cache-run --live
The optional feedback file is an owner declaration, never an executed measurement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("payload", type=Path, help="start payload JSON")
    parser.add_argument("--output", required=True, type=Path, help="new output directory")
    parser.add_argument("--feedback", type=Path, help="optional observe payload, followed by another run")
    parser.add_argument("--live", action="store_true", help="execute the payload's explicit provider/tool authorization")
    args = parser.parse_args()
    payload = json.loads(args.payload.read_text())
    if payload.get("authorization", {}).get("tools") and not args.live:
        parser.error("this payload authorizes external execution; supply --live to run it")
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.output.resolve() / "runtime"
    mission_id, revision = None, None
    sequence = 0
    failed = False

    def command(kind: str, data: dict | None = None) -> dict:
        nonlocal mission_id, revision, sequence, failed
        request = {"protocol": "sextant-agent/1", "schema_version": 1,
                   "command_id": "example-" + uuid.uuid4().hex, "kind": kind,
                   "mission_id": mission_id, "expected_revision": revision, "payload": data or {}}
        sequence += 1
        (args.output / f"{sequence:02d}-{kind}-request.json").write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n")
        result = subprocess.run([sys.executable, "-m", "sextant.cli", "--root", str(root), "agent"],
                                input=json.dumps(request), text=True, capture_output=True, check=False)
        (args.output / f"{sequence:02d}-{kind}-result.json").write_text(result.stdout)
        if result.stderr:
            (args.output / f"{sequence:02d}-{kind}-stderr.txt").write_text(result.stderr)
        response = json.loads(result.stdout)
        failed = failed or bool(response["errors"]) or result.returncode != 0
        if response.get("mission_id"):
            mission_id = response["mission_id"]
        if response.get("current_revision"):
            revision = response["current_revision"]
        print(f"{kind}: {response['status']} · {mission_id}", flush=True)
        return response

    if command("start", payload)["errors"]:
        return 1
    first = command("run")
    command("inspect", {"events": True})
    if args.feedback and not first["errors"]:
        if not command("observe", json.loads(args.feedback.read_text()))["errors"]:
            command("run")
        command("inspect", {"events": True})
    command("replay")
    command("export")
    print(f"Saved runtime and exact commands/results: {args.output.resolve()}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
