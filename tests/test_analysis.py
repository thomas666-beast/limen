import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from limen.analysis import load_trials, summarize, save_summary


def _write_trials(path: Path) -> None:
    data = {
        "meta": {"n_pairs": 2, "n_trials": 4, "pairs": [["a", "b"], ["c", "d"]]},
        "trials": [
            {
                "pair_id": 0, "condition": "related",
                "prime": "a", "target": "b",
                "target_novelty": 0.2, "target_similarity": 0.8,
                "target_best_match": "a",
                "top_source": "memory", "top_salience": 0.7,
            },
            {
                "pair_id": 0, "condition": "unrelated",
                "prime": "c", "target": "b",
                "target_novelty": 1.0, "target_similarity": 0.0,
                "target_best_match": None,
                "top_source": "memory", "top_salience": 0.9,
            },
            {
                "pair_id": 1, "condition": "related",
                "prime": "c", "target": "d",
                "target_novelty": 0.3, "target_similarity": 0.7,
                "target_best_match": "c",
                "top_source": "sentiment", "top_salience": 0.5,
            },
            {
                "pair_id": 1, "condition": "unrelated",
                "prime": "a", "target": "d",
                "target_novelty": 1.0, "target_similarity": 0.0,
                "target_best_match": None,
                "top_source": "sentiment", "top_salience": 0.6,
            },
        ],
    }
    path.write_text(json.dumps(data))


def test_load_trials_returns_dataframe(tmp_path: Path):
    p = tmp_path / "priming.json"
    _write_trials(p)

    df = load_trials(p)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4
    assert set(df.columns) >= {
        "pair_id", "condition", "prime", "target",
        "target_novelty", "target_similarity", "top_source",
    }


def test_load_trials_preserves_condition_values(tmp_path: Path):
    p = tmp_path / "priming.json"
    _write_trials(p)
    df = load_trials(p)
    assert set(df["condition"]) == {"related", "unrelated"}


def test_load_trials_raises_on_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_trials(tmp_path / "nope.json")


def test_summarize_groups_by_condition(tmp_path: Path):
    p = tmp_path / "priming.json"
    _write_trials(p)
    df = load_trials(p)

    summary = summarize(df)

    assert set(summary.keys()) == {"related", "unrelated"}
    assert summary["related"]["target_novelty_mean"] == pytest.approx(0.25)
    assert summary["unrelated"]["target_novelty_mean"] == pytest.approx(1.0)
    assert summary["related"]["n"] == 2


def test_summarize_includes_top_source_counts(tmp_path: Path):
    p = tmp_path / "priming.json"
    _write_trials(p)
    df = load_trials(p)
    summary = summarize(df)

    # related trials: 1 memory + 1 sentiment
    assert summary["related"]["top_source_counts"] == {"memory": 1, "sentiment": 1}


def test_save_summary_writes_json(tmp_path: Path):
    p = tmp_path / "priming.json"
    _write_trials(p)
    df = load_trials(p)
    summary = summarize(df)

    out = tmp_path / "summary.json"
    save_summary(summary, out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert "related" in data
    assert "unrelated" in data
