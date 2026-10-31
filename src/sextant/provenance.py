"""Protocol byte identity; not an authenticated signature or correctness proof."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path

from sextant.domain import digest


def protocol_fingerprint() -> str:
    root = Path(str(files("sextant")))
    paths = sorted(root.glob("*.py")) + sorted((root / "providers").glob("*.py")) + sorted((root / "resources" / "v1-role-cards").glob("*.yaml"))
    return digest({str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
