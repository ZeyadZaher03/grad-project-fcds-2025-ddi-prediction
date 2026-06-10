# ml/config.py
"""Load config.yaml into a plain dict (access keys with cfg["..."])."""
from __future__ import annotations
import os
import yaml

_DEFAULT = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(path: str | None = None) -> dict:
    with open(path or _DEFAULT) as f:
        return yaml.safe_load(f)
