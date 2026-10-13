"""Configuration loading.

All tunable choices live in ``configs/research_config.yaml``. Code reads them
through :func:`load_config` so that no number is hidden inside a function.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "research_config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML research configuration.

    Parameters
    ----------
    path:
        Optional path to a YAML file. Defaults to ``configs/research_config.yaml``.

    Returns
    -------
    dict
        Parsed configuration. ``.inf`` values in YAML become ``float('inf')``.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def resolve_path(relative: str | Path) -> Path:
    """Turn a project-relative path from the config into an absolute path."""
    p = Path(relative)
    return p if p.is_absolute() else PROJECT_ROOT / p


def ensure_dirs(cfg: dict[str, Any]) -> None:
    """Create every directory listed under ``paths`` if it does not exist."""
    for key, rel in cfg["paths"].items():
        target = resolve_path(rel)
        if key == "report":
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
