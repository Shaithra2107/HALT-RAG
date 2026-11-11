"""Load config.yaml and build full file paths, on a laptop or on Colab."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def on_colab() -> bool:
    return "google.colab" in sys.modules or os.path.exists("/content")


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Read config.yaml (from the repo root unless a path is given)."""
    path = Path(path) if path else REPO_ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir(cfg: dict) -> str:
    """The data folder: Google Drive on Colab, <repo>/data on a laptop."""
    if on_colab():
        d = cfg["paths"]["drive_dir"]
    else:
        d = cfg["paths"]["data_dir"]
        if not os.path.isabs(d):
            d = str(REPO_ROOT / d)
    os.makedirs(d, exist_ok=True)
    return d


def data_path(cfg: dict, key_or_name: str) -> str:
    """Full path of a data file: a key under `paths:` (e.g. 'signals_v5') or a plain file name."""
    name = cfg["paths"].get(key_or_name, key_or_name)
    return os.path.join(data_dir(cfg), name)


drive_path = data_path   # old name, kept so earlier code still works
