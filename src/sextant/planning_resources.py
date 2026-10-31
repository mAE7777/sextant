"""Host planning context for the next model call.

This is a pre-reservation view of experimental controls, never admission,
authorization, measured usage, or a guarantee that the next call can run.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from sextant.domain import DomainError, canonical, finite_number

PlanningCallback = Callable[[str, dict[str, Any]], dict[str, Any]]
POOL_NAMES = frozenset({"native", "assistant", "experiment", "read", "apply", "command", "native_seconds", "compute", "typed"})


def planning_material(stage: str, state: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate host counters and preserve every queue occurrence without pruning."""
    phase = state["options"].get("environment", {}).get("phase")
    if (not isinstance(snapshot, dict) or set(snapshot) != {"schema_version", "phase", "pools", "deadline_monotonic", "observed_monotonic", "remaining_seconds"}
            or type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != 1
            or snapshot["phase"] != phase or not isinstance(snapshot["pools"], dict)
            or set(snapshot["pools"]) != POOL_NAMES):
        raise DomainError("planning_resources", "host resource snapshot must match the current phase and complete pools")
    deadline, observed, remaining = (snapshot[k] for k in ("deadline_monotonic", "observed_monotonic", "remaining_seconds"))
    if (any(not finite_number(v) or v < 0 for v in (deadline, observed, remaining))
            or deadline < observed or not math.isclose(remaining, deadline - observed, rel_tol=0, abs_tol=1e-6)):
        raise DomainError("planning_resources", "host watchdog timing must be finite and internally consistent")
    for name, pool in snapshot["pools"].items():
        if not isinstance(pool, dict) or set(pool) != {"capacity", "used", "reserved", "remaining", "unit"}:
            raise DomainError("planning_resources", "resource pool must declare exact counters and unit")
        values = [pool[key] for key in ("capacity", "used", "reserved", "remaining")]
        seconds = name == "native_seconds"
        if (pool["unit"] != ("allowance_seconds" if seconds else "opportunities")
                or any((not finite_number(v) if seconds else type(v) is not int) or v < 0 for v in values)
                or pool["remaining"] != pool["capacity"] - pool["used"] - pool["reserved"]):
            raise DomainError("planning_resources", "invalid, exhausted, or inconsistent host resource counters")
    # Validate strict finite JSON before copying it into the actual request.
    canonical(snapshot)
    queue: list[dict[str, Any]] = []
    signatures: dict[str, str] = {}
    for bucket in ("pending_experiments", "research_queue"):
        for entry in state.get(bucket, []):
            signature = entry.get("signature")
            design = entry.get("experiment")
            if not isinstance(signature, str) or not signature or not isinstance(design, dict):
                raise DomainError("planning_resources", "retained queue requires exact signature and design")
            body = canonical(design)
            if signature in signatures and signatures[signature] != body:
                raise DomainError("planning_resources", "one retained signature identifies contradictory designs")
            signatures[signature] = body
            item = {"bucket": bucket, "signature": signature,
                    "origin_proposal_ref": deepcopy(entry.get("origin_proposal_ref")),
                    "protocol_fingerprint": entry.get("protocol_fingerprint")}
            if stage != "challenger":
                item.update(branch=entry.get("branch"), experiment=deepcopy(design))
            queue.append(item)
    return {"schema_version": 1, "phase": phase, "mission_id": state["mission_id"],
            "mission_revision": state["revision"], "stage": stage,
            "authority": "host planning context; not execution admission or authorization",
            "timing": "immediately before this generation opportunity is reserved",
            "basis": "immediately_before_current_native_reservation",
            "deadline_monotonic": deadline, "observed_monotonic": observed, "remaining_seconds": remaining,
            "pools": deepcopy(snapshot["pools"]), "queue": queue,
            "queue_occurrences": len(queue), "unique_designs": len(signatures),
            "visibility": "opaque queue identities" if stage == "challenger" else "complete recorded designs"}
