"""Minimal YAML config loading with dotted-key CLI overrides (P3)."""
import copy
import os
from typing import Any, Dict, List

import yaml


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML; supports `_base_: <relative-path>` for single-level inheritance."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = base_ref if os.path.isabs(base_ref) \
            else os.path.join(os.path.dirname(os.path.abspath(path)), base_ref)
        cfg = _deep_merge(load_config(base_path), cfg)
    return cfg


def _coerce(v: str) -> Any:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """overrides like ['preprocess.suite=libero_10', 'device=mps']."""
    cfg = copy.deepcopy(cfg)
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg
