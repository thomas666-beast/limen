import pytest

from limen.config import SalienceWeights
from limen.salience import ConstantScorer, SalienceContext, WeightedScorer
from limen.workspace import Content


def make(payload=None, source="x"):
    return Content(source=source, payload=payload or {})


# ---------- ConstantScorer ----------

def test_constant_scorer_returns_given_value():
    s = ConstantScorer(0.42)
    assert s(make()) == pytest.approx(0.42)


def test_constant_scorer_rejects_out_of_range():
    with pytest.raises(ValueError, match="constant"):
        ConstantScorer(1.5)
    with pytest.raises(ValueError, match="constant"):
        ConstantScorer(-0.1)


# ---------- SalienceContext ----------

def test_context_starts_empty():
    ctx = SalienceContext()
    assert ctx.recent_payloads == ()


def test_context_records_payloads_in_order():
    ctx = SalienceContext(capacity=3)
    ctx.remember({"a": 1})
    ctx.remember({"b": 2})
    assert ctx.recent_payloads == ({"a": 1}, {"b": 2})


def test_context_caps_at_capacity_oldest_dropped():
    ctx = SalienceContext(capacity=2)
    ctx.remember({"a": 1})
    ctx.remember({"b": 2})
    ctx.remember({"c": 3})
    assert ctx.recent_payloads == ({"b": 2}, {"c": 3})


def test_context_capacity_must_be_positive():
    with pytest.raises(ValueError, match="capacity"):
        SalienceContext(capacity=0)


# ---------- WeightedScorer ----------

def weights(n=0.4, e=0.4, r=0.2):
    return SalienceWeights(novelty=n, emotion=e, relevance=r)


def test_weighted_scorer_defaults_when_payload_empty():
    """No novelty hint, no sentiment, no relevance → neutral mid."""
    s = WeightedScorer(weights=weights(), context=SalienceContext())
    score = s(make({"novelty": 0.0, "relevance": 0.5}))
    # 0.4*0 + 0.4*0 + 0.2*0.5 = 0.1
    assert score == pytest.approx(0.1, abs=1e-6)


def test_weighted_scorer_uses_all_three_signals():
    s = WeightedScorer(weights=weights(0.5, 0.3, 0.2), context=SalienceContext())
    c = make({"novelty": 1.0, "sentiment": 0.8, "relevance": 0.5})
    # 0.5*1 + 0.3*0.8 + 0.2*0.5 = 0.5 + 0.24 + 0.1 = 0.84
    assert s(c) == pytest.approx(0.84, abs=1e-6)


def test_weighted_scorer_clamps_components_to_unit_interval():
    s = WeightedScorer(weights=weights(1.0, 0.0, 0.0), context=SalienceContext())
    # novelty=5.0 should be clamped to 1.0
    assert s(make({"novelty": 5.0})) == pytest.approx(1.0, abs=1e-6)


def test_weighted_scorer_uses_absolute_sentiment():
    s = WeightedScorer(weights=weights(0.0, 1.0, 0.0), context=SalienceContext())
    # -0.9 emotion should count as 0.9 salience
    assert s(make({"sentiment": -0.9})) == pytest.approx(0.9, abs=1e-6)


def test_weighted_scorer_novelty_uses_context():
    ctx = SalienceContext(capacity=5)
    ctx.remember({"k": "seen"})

    s = WeightedScorer(weights=weights(1.0, 0.0, 0.0), context=ctx)

    seen = make({"k": "seen"})
    new = make({"k": "different"})

    assert s(seen) == pytest.approx(0.0, abs=1e-6)
    assert s(new) == pytest.approx(1.0, abs=1e-6)


def test_weighted_scorer_does_not_remember_by_itself():
    """Scoring must be side-effect-free with respect to context."""
    ctx = SalienceContext()
    s = WeightedScorer(weights=weights(), context=ctx)
    s(make({"novelty": 1.0}))
    assert ctx.recent_payloads == ()


def test_weighted_scorer_returns_float_in_unit_interval():
    s = WeightedScorer(weights=weights(), context=SalienceContext())
    score = s(make({"novelty": 1.0, "sentiment": 1.0, "relevance": 1.0}))
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
