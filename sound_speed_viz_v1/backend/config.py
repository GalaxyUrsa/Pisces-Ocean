"""Per-variable display config persistence."""

import json
import os
import sys


def _app_dir() -> str:
    """Return the directory next to the exe (frozen) or the project root (dev)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    # dev: backend/config.py → go up one level to project root
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


CONFIG_PATH = os.path.join(_app_dir(), "temp", "viz_config.json")

DEFAULTS: dict = {
    "ss":   {"min": 1480, "max": 1560, "colorscale": "Viridis",  "color_min": None, "color_max": None},
    "temp": {"min": 0,    "max": 35,   "colorscale": "RdYlBu_r", "color_min": None, "color_max": None},
    "salt": {"min": 30,   "max": 40,   "colorscale": "Blues",    "color_min": None, "color_max": None},
}


def load_config() -> dict:
    """Return saved config merged with DEFAULTS. Falls back to DEFAULTS on any error."""
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            saved = json.load(f)
        result = {}
        for var, defaults in DEFAULTS.items():
            entry = saved.get(var, {})
            result[var] = {**defaults, **{k: v for k, v in entry.items() if k in defaults}}
        return result
    except Exception:
        return {v: dict(d) for v, d in DEFAULTS.items()}


def save_config(cfg: dict) -> None:
    """Write cfg to CONFIG_PATH, creating ./temp/ if needed."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
