"""config.toml: every constant the runtime uses, printed by every report that uses it.

Defaults live here; `setup` writes the file with the probe results and date. Nothing in
the code path reads a constant that is not also in the loaded config.

The file is found through `SEXTANT_CONFIG`, else `config.toml` in the working directory.
There is no silent fallback into a development checkout: an editable install run from another
directory would otherwise write its data into the checkout. Relative paths in the file resolve
against the file's own directory; without a file, against the working directory, and commands
that write data refuse to run until a config file exists.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

# The source checkout, for development tooling only (doctor's checkout-only rungs, tests).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    pass


def config_path() -> Path | None:
    """The config file in force, or None when there is none."""
    env = os.environ.get("SEXTANT_CONFIG")
    if env:
        return Path(env)
    here = Path.cwd() / "config.toml"
    return here if here.exists() else None


def require_config_file() -> Path:
    """Data-writing commands call this: no file means no agreed place to write."""
    found = config_path()
    if found is None or not found.exists():
        raise ConfigError(
            "no config.toml in the working directory and SEXTANT_CONFIG is not set; "
            "run `python -m sextant config --write-defaults` where the data should live"
        )
    return found


def packaged_rolecards_dir() -> Path:
    """Role-card texts shipped inside this package. Not a calibrated product set."""
    return Path(__file__).resolve().parent / "resources" / "role-cards"


def resolve_rolecards_dir(raw: str) -> str:
    if raw in ("", "package"):
        return str(packaged_rolecards_dir())
    return raw


DEFAULTS: dict[str, Any] = {
    "paths": {
        "decisions_dir": "decisions-v2",
        "calibration_dir": "calibration-v2",
        "rolecards_dir": "package",
        "skill_dir": "not-packaged",
    },
    "providers": {
        "generator": "claude-code",
        "challenger": "codex",
        "grader": "codex",
        "paraphraser": "claude-code-subagent",
        "translator": "claude-code",
        "renderer": "claude-code",
        "mapper": "claude-code",
        "auto_fulfil": [],
        "codex_timeout_s": 600,
        "deepseek_model": "",
        "gemini_model": "",
        "openai_model": "",
    },
    "jev": {
        "transport": "typesafe",
        "model_id": "jev-1.13.0",
        "permutations": 4,
        "reruns": 3,
        "contested_margin": 0.10,
        "max_state_tokens": 30000,
        "max_request_tokens": 60000,
        "requests_per_minute": 1000,
        "canonical_language": "en",
    },
    "tiers": {
        "T1": {"depth_max": 2, "breadth_max": 5, "error_budget_target": 0.60, "k_decompositions": 2},
        "T2": {"depth_max": 4, "breadth_max": 7, "error_budget_target": 0.75, "k_decompositions": 3},
        "T3": {"depth_max": 5, "breadth_max": 7, "error_budget_target": 0.85, "k_decompositions": 3},
    },
    "calibration": {
        "target_agreement_T2": 0.90,
        "degrade_agreement_drop": 0.05,
        "degrade_auroc_floor": 0.70,
        "degrade_flip_rate": 0.15,
        "outcome_bar": 30,
        "min_items_per_bin": 100,
        "canary_band": 0.05,
    },
    "aggregate": {
        "box_sd_multiplier": 2.0,
        "sufficiency_widening": 0.5,
        "mc_samples": 4000,
        "mc_seed": 20260918,
        "weight_swing": 0.25,
    },
    "probe": {},
}


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_raw_config(path: Path | None = None) -> dict[str, Any]:
    """Defaults merged with the file, paths exactly as written (for writing config back)."""
    p = path if path is not None else config_path()
    if p is not None and p.exists():
        with p.open("rb") as f:
            data = tomllib.load(f)
        return _merge(DEFAULTS, data)
    return _merge(DEFAULTS, {})


def load_config(path: Path | None = None) -> dict[str, Any]:
    """The effective config: relative data paths resolved against the config file's directory."""
    p = path if path is not None else config_path()
    merged = load_raw_config(p)
    base = p.parent if p is not None and p.exists() else Path.cwd()
    paths = dict(merged["paths"])
    for name in ("decisions_dir", "calibration_dir"):
        raw = Path(str(paths[name])).expanduser()
        paths[name] = str(raw if raw.is_absolute() else (base / raw).resolve())
    paths["rolecards_dir"] = resolve_rolecards_dir(str(paths["rolecards_dir"]))
    merged["paths"] = paths
    return merged


def _toml_scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_scalar(x) for x in v) + "]"
    if v is None:
        return '""'
    raise TypeError(f"cannot serialize {type(v).__name__} to TOML")


def dump_toml(data: dict[str, Any], prefix: str = "") -> str:
    """Minimal TOML writer for nested dicts of scalars and lists (all we need)."""
    lines: list[str] = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict)}
    if prefix and (scalars or not tables):
        lines.append(f"[{prefix}]")
    for k, v in scalars.items():
        lines.append(f"{k} = {_toml_scalar(v)}")
    if scalars or (prefix and not tables):
        lines.append("")
    for k, v in tables.items():
        lines.append(dump_toml(v, f"{prefix}.{k}" if prefix else k).rstrip("\n"))
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def write_config(data: dict[str, Any], path: Path | None = None) -> Path:
    """Write raw values (pass `load_raw_config()`, never a resolved config)."""
    p = path or config_path() or (Path.cwd() / "config.toml")
    p.write_text(dump_toml(data))
    return p
