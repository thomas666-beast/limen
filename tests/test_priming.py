import json
from pathlib import Path

import pytest

from experiments.priming import (
    PAIRS,
    Condition,
    Trial,
    build_trials,
    run_experiment,
    run_trial,
)


# ---------- trial construction ----------

def test_build_trials_produces_two_conditions_per_pair():
    trials = build_trials(PAIRS)
    # 5 pairs × 2 conditions
    assert len(trials) == 10
    conditions = {t.condition for t in trials}
    assert conditions == {"related", "unrelated"}


def test_related_trial_uses_pair_itself():
    trials = build_trials([("doctor", "nurse"), ("bread", "butter")])
    related = [
        t for t in trials
        if t.condition == "related" and t.prime == "doctor"
    ][0]
    assert related.prime == "doctor"
    assert related.target == "nurse"


def test_unrelated_trial_uses_different_prime():
    trials = build_trials([("doctor", "nurse"), ("bread", "butter")])
    unrelated = [
        t for t in trials if t.condition == "unrelated" and t.target == "nurse"
    ][0]
    assert unrelated.prime != "doctor"
    assert unrelated.target == "nurse"


def test_trial_is_hashable_dataclass():
    t = Trial(condition="related", prime="a", target="b", pair_id=0)
    assert t.condition == "related"
    # frozen=True → hashable
    assert hash(t)


# ---------- single trial ----------

def test_run_trial_returns_expected_keys():
    t = Trial(condition="related", prime="doctor", target="nurse", pair_id=0)
    result = run_trial(t)

    assert set(result.keys()) >= {
        "pair_id", "condition", "prime", "target",
        "target_novelty", "target_similarity", "target_best_match",
        "top_source", "top_salience",
    }
    assert result["condition"] == "related"
    assert result["prime"] == "doctor"
    assert result["target"] == "nurse"


def test_run_trial_related_has_nonzero_similarity():
    """After priming with 'doctor', the target 'nurse' should not be
    perfectly novel to memory *if* it shares any token — but they share none.
    So similarity stays 0. We test the mechanism, not the outcome."""
    t = Trial(condition="related", prime="doctor", target="doctor nurse", pair_id=0)
    # prime 'doctor', target 'doctor nurse' — shares token 'doctor'
    result = run_trial(t)
    assert result["target_similarity"] > 0.0


def test_run_trial_isolates_between_trials():
    """Each trial must start with a fresh bot — priming does not leak."""
    t1 = Trial(condition="related", prime="alpha", target="alpha", pair_id=0)
    t2 = Trial(condition="related", prime="beta", target="alpha", pair_id=1)

    r1 = run_trial(t1)
    r2 = run_trial(t2)

    # both trials prime their target first; memory should be identical
    # (novelty on the target cycle after priming with itself is 0 in both)
    assert r1["target_novelty"] == pytest.approx(0.0)
    assert r2["target_novelty"] == pytest.approx(1.0)  # 'beta' did not prime 'alpha'


# ---------- whole experiment ----------

def test_run_experiment_writes_json(tmp_path: Path):
    out = tmp_path / "priming.json"
    results = run_experiment(PAIRS, out_path=out)

    assert out.exists()
    data = json.loads(out.read_text())
    assert "trials" in data
    assert "meta" in data
    assert len(data["trials"]) == 10
    assert len(results) == 10


def test_run_experiment_meta_records_pairs(tmp_path: Path):
    out = tmp_path / "priming.json"
    run_experiment(PAIRS, out_path=out)
    data = json.loads(out.read_text())
    assert data["meta"]["n_pairs"] == len(PAIRS)
    assert data["meta"]["n_trials"] == 10

def test_build_trials_requires_at_least_two_pairs():
    with pytest.raises(ValueError, match="at least two"):
        build_trials([("doctor", "nurse")])
