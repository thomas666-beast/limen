"""Analysis helpers for limen experiment output.

Keeps the notebook thin: data loading, grouping, and summary generation
all live here, so they can be unit-tested without a running Jupyter kernel.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_trials(path: str | Path) -> pd.DataFrame:
    """Load a priming JSON output and return a tidy DataFrame.

    Expects the structure written by `experiments.priming.run_experiment`:
        {"meta": {...}, "trials": [ {trial...}, ... ]}
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no experiment output at {p}")

    raw = json.loads(p.read_text())
    trials = raw.get("trials", [])
    df = pd.DataFrame(trials)

    # ensure a stable column order for readability
    preferred = [
        "pair_id", "condition", "prime", "target",
        "target_novelty", "target_similarity", "target_best_match",
        "top_source", "top_salience",
    ]
    cols = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    return df[cols + rest]


def summarize(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Group by condition and compute descriptive stats.

    For each condition we report:
    - n (trial count)
    - target_novelty_mean / std
    - target_similarity_mean / std
    - top_salience_mean / std
    - top_source_counts: dict of source -> count
    """
    if df.empty:
        return {}

    out: dict[str, dict[str, Any]] = {}
    for condition, group in df.groupby("condition"):
        out[str(condition)] = {
            "n": int(len(group)),
            "target_novelty_mean": _safe_mean(group, "target_novelty"),
            "target_novelty_std": _safe_std(group, "target_novelty"),
            "target_similarity_mean": _safe_mean(group, "target_similarity"),
            "target_similarity_std": _safe_std(group, "target_similarity"),
            "top_salience_mean": _safe_mean(group, "top_salience"),
            "top_salience_std": _safe_std(group, "top_salience"),
            "top_source_counts": group["top_source"].value_counts().to_dict()
            if "top_source" in group
            else {},
        }
    return out


def save_summary(summary: dict[str, Any], path: str | Path) -> None:
    """Write a summary dict to JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2, default=_json_default))


# ---------- internals ----------

def _safe_mean(group: pd.DataFrame, column: str) -> float | None:
    if column not in group:
        return None
    series = group[column].dropna()
    if series.empty:
        return None
    return float(series.mean())


def _safe_std(group: pd.DataFrame, column: str) -> float | None:
    if column not in group:
        return None
    series = group[column].dropna()
    if len(series) < 2:
        return None
    return float(series.std())


def _json_default(obj: Any) -> Any:
    # numpy scalars, etc.
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")
