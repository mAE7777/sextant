"""What this package can load, and what is still partial.

Offline results from this package are not live Jev or generative-model verification.
"""

from __future__ import annotations

from typing import Any

from sextant import __version__
from sextant.config import packaged_rolecards_dir


def describe_package() -> dict[str, Any]:
    cards = packaged_rolecards_dir()
    names = sorted(path.name for path in cards.glob("*.yaml") if not path.name.startswith("_"))
    return {
        "compiled_product": False,
        "release_accepted": False,
        "distribution_stage": f"{__version__} candidate; not an accepted release",
        "offline_is_live_verification": False,
        "loadable": [
            "sextant-agent/1 and installed CLI share one application service",
            "SQLite revisions, artifacts, fenced dispatch, portable migration and recovery",
            "native Codex subscription proposals and advisory Jev diagnostics; explicit historical Claude compatibility",
            "competing representations, conditional policy, controlled experiment and revision loop",
            "typed graph checks",
            "single-writer ledgers and state files",
            "request/answer handshake with no paid API fallback",
            "offline calibration tools",
            "packaged role-card texts",
            "offline doctor integrity checks",
        ],
        "partial": [
            "role cards are question texts, not a calibrated product",
            "formal comparative capability confirmation and independent candidate acceptance",
            "automatic semantic authority remains disabled",
            "long-horizon real-world outcomes remain pending until observed",
        ],
        "rolecard_files": names,
    }
