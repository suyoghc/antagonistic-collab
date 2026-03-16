"""Configuration loading for antagonistic_collab.

Loads defaults from default_config.yaml, optionally overlays a user config
file, and lets CLI flags override everything.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file using only the standard library.

    Supports the subset of YAML used by our config: scalars, simple
    key: value pairs, no nesting.  Falls back to PyYAML if available.
    """
    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass

    # Minimal parser for flat key: value YAML (no nesting, no lists)
    result: dict[str, Any] = {}
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                continue
            key, _, raw_value = stripped.partition(":")
            key = key.strip()
            raw_value = raw_value.strip()

            # Strip inline comments
            if " #" in raw_value:
                raw_value = raw_value[: raw_value.index(" #")].strip()

            # Parse value types
            if raw_value in ("true", "True"):
                result[key] = True
            elif raw_value in ("false", "False"):
                result[key] = False
            elif raw_value in ("null", "None", "~", ""):
                result[key] = None
            elif raw_value.startswith('"') and raw_value.endswith('"'):
                result[key] = raw_value[1:-1]
            elif raw_value.startswith("'") and raw_value.endswith("'"):
                result[key] = raw_value[1:-1]
            else:
                # Try int, then float, then string
                try:
                    result[key] = int(raw_value)
                except ValueError:
                    try:
                        result[key] = float(raw_value)
                    except ValueError:
                        result[key] = raw_value
    return result


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration with layered precedence.

    1. Built-in defaults (default_config.yaml in this package)
    2. User config file (if --config specified or config.yaml exists in cwd)

    CLI flags are applied separately by the caller (they override everything).

    Returns:
        Merged config dict.
    """
    # Layer 1: built-in defaults
    default_path = Path(__file__).parent / "default_config.yaml"
    config = _load_yaml(default_path)

    # Layer 2: user config file
    if config_path is not None:
        user_path = Path(config_path)
        if not user_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        user_config = _load_yaml(user_path)
        config.update(user_config)
    else:
        # Auto-detect config.yaml in current working directory
        cwd_config = Path(os.getcwd()) / "config.yaml"
        if cwd_config.exists():
            user_config = _load_yaml(cwd_config)
            config.update(user_config)

    return config


def apply_config_defaults(parser, config: dict[str, Any]):
    """Set argparse defaults from config dict.

    Maps config keys (snake_case) to argparse destinations. CLI flags
    that are explicitly provided will still override these defaults.
    """
    # Map config keys to argparse dest names
    key_map = {
        "true_model": "true_model",
        "learning_rate": "learning_rate",
        "no_tempering": "no_tempering",
        "no_arbiter": "no_arbiter",
        "no_claim_responsive": "no_claim_responsive",
        "no_richer_design_space": "no_richer_design_space",
        "design_space": "design_space",
        "n_continuous_samples": "n_continuous_samples",
        "critique_rounds": "critique_rounds",
        "hitl_checkpoints": "hitl_checkpoints",
        "output_dir": "output_dir",
    }

    defaults = {}
    for config_key, dest in key_map.items():
        if config_key in config and config[config_key] is not None:
            defaults[dest] = config[config_key]

    # Direct mappings (config key matches argparse dest)
    for key in (
        "cycles",
        "backend",
        "model",
        "selection",
        "mode",
        "batch",
        "selection_strategy",
        "crux_weight",
    ):
        if key in config and config[key] is not None:
            defaults[key] = config[key]

    if defaults:
        parser.set_defaults(**defaults)
