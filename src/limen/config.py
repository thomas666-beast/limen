"""Typed configuration for limen.

Loads YAML from disk, merges with defaults, and validates the result.
Everything else in the package depends on this module — keep it small.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------- schema ----------

@dataclass(frozen=True)
class WorkspaceConfig:
    capacity: int = 3


@dataclass(frozen=True)
class SalienceWeights:
    novelty: float = 0.4
    emotion: float = 0.4
    relevance: float = 0.2

    def total(self) -> float:
        return self.novelty + self.emotion + self.relevance


@dataclass(frozen=True)
class SalienceConfig:
    weights: SalienceWeights = field(default_factory=SalienceWeights)


@dataclass(frozen=True)
class LoggingConfig:
    db_path: str = "data/logs.db"


@dataclass(frozen=True)
class Config:
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


# ---------- loading ----------

def _merge(default: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` into `default` (override wins)."""
    out = dict(default)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _defaults_as_dict() -> dict[str, Any]:
    """Turn the dataclass defaults into a plain dict for merging."""
    return {
        "workspace": {"capacity": WorkspaceConfig().capacity},
        "salience": {
            "weights": {
                "novelty": SalienceWeights().novelty,
                "emotion": SalienceWeights().emotion,
                "relevance": SalienceWeights().relevance,
            }
        },
        "logging": {"db_path": LoggingConfig().db_path},
    }


def _validate(cfg: Config) -> None:
    if cfg.workspace.capacity <= 0:
        raise ValueError(
            f"workspace.capacity must be positive, got {cfg.workspace.capacity}"
        )

    total = cfg.salience.weights.total()
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"salience.weights must sum to 1.0, got {total}"
        )


def load_config(path: str | Path | None = None) -> Config:
    """Load config from `path` (YAML). Missing file → defaults.

    Partial files merge on top of defaults. Invalid values raise ValueError.
    """
    merged = _defaults_as_dict()

    if path is not None:
        p = Path(path)
        if p.exists():
            raw = yaml.safe_load(p.read_text()) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"config root must be a mapping, got {type(raw).__name__}")
            merged = _merge(merged, raw)

    cfg = Config(
        workspace=WorkspaceConfig(capacity=merged["workspace"]["capacity"]),
        salience=SalienceConfig(
            weights=SalienceWeights(
                novelty=merged["salience"]["weights"]["novelty"],
                emotion=merged["salience"]["weights"]["emotion"],
                relevance=merged["salience"]["weights"]["relevance"],
            )
        ),
        logging=LoggingConfig(db_path=merged["logging"]["db_path"]),
    )

    _validate(cfg)
    return cfg
