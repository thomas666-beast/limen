import pytest

from limen.workspace import Content, Cycle, GlobalWorkspace


def make(source: str, salience_hint: float = 0.0) -> Content:
    return Content(source=source, payload={"hint": salience_hint})


def scorer_by_hint(c: Content) -> float:
    """Trivial scorer: reads the 'hint' from payload."""
    return float(c.payload.get("hint", 0.0))


# ---------- construction ----------

def test_capacity_must_be_positive():
    with pytest.raises(ValueError, match="capacity"):
        GlobalWorkspace(capacity=0)


def test_history_limit_must_be_positive():
    with pytest.raises(ValueError, match="history_limit"):
        GlobalWorkspace(capacity=3, history_limit=0)


# ---------- submit ----------

def test_submit_adds_to_pool():
    ws = GlobalWorkspace(capacity=3)
    c = make("a", 0.5)
    ws.submit(c)
    assert len(ws.pool) == 1
    assert ws.pool[0] is c


def test_submit_rejects_non_content():
    ws = GlobalWorkspace(capacity=3)
    with pytest.raises(TypeError, match="Content"):
        ws.submit("not a content")  # type: ignore[arg-type]


# ---------- compete ----------

def test_compete_broadcasts_top_k_by_salience():
    ws = GlobalWorkspace(capacity=2)
    ws.submit(make("low", 0.1))
    ws.submit(make("high", 0.9))
    ws.submit(make("mid", 0.5))

    cycle = ws.compete(scorer_by_hint)

    broadcast_sources = [c.source for c in cycle.broadcast]
    assert broadcast_sources == ["high", "mid"]

    remainder_sources = [c.source for c in cycle.unconscious]
    assert remainder_sources == ["low"]


def test_compete_sets_salience_on_each_content():
    ws = GlobalWorkspace(capacity=1)
    ws.submit(make("a", 0.42))
    cycle = ws.compete(scorer_by_hint)
    assert cycle.broadcast[0].salience == pytest.approx(0.42)


def test_compete_is_stable_on_ties():
    ws = GlobalWorkspace(capacity=2)
    ws.submit(make("first", 0.5))
    ws.submit(make("second", 0.5))
    ws.submit(make("third", 0.5))

    cycle = ws.compete(scorer_by_hint)

    assert [c.source for c in cycle.broadcast] == ["first", "second"]
    assert [c.source for c in cycle.unconscious] == ["third"]


def test_compete_capacity_larger_than_pool():
    ws = GlobalWorkspace(capacity=10)
    ws.submit(make("a", 0.3))
    ws.submit(make("b", 0.6))

    cycle = ws.compete(scorer_by_hint)

    assert len(cycle.broadcast) == 2
    assert cycle.unconscious == ()


def test_compete_empty_pool_is_allowed():
    ws = GlobalWorkspace(capacity=3)
    cycle = ws.compete(scorer_by_hint)
    assert cycle.broadcast == ()
    assert cycle.unconscious == ()


def test_compete_rejects_scorer_returning_out_of_range():
    ws = GlobalWorkspace(capacity=1)
    ws.submit(make("x", 2.0))  # hint=2.0 → scorer returns 2.0, invalid
    with pytest.raises(ValueError, match="salience"):
        ws.compete(scorer_by_hint)


# ---------- state after cycle ----------

def test_pool_is_emptied_after_compete():
    ws = GlobalWorkspace(capacity=1)
    ws.submit(make("a", 0.5))
    ws.compete(scorer_by_hint)
    assert ws.pool == ()


def test_history_grows_by_one_per_cycle():
    ws = GlobalWorkspace(capacity=1)
    ws.submit(make("a", 0.5))
    ws.compete(scorer_by_hint)
    ws.submit(make("b", 0.5))
    ws.compete(scorer_by_hint)
    assert len(ws.history) == 2


def test_history_records_broadcast_and_unconscious():
    ws = GlobalWorkspace(capacity=1)
    ws.submit(make("hi", 0.9))
    ws.submit(make("lo", 0.1))
    ws.compete(scorer_by_hint)

    cycle = ws.history[0]
    assert [c.source for c in cycle.broadcast] == ["hi"]
    assert [c.source for c in cycle.unconscious] == ["lo"]


def test_history_is_capped_at_limit():
    ws = GlobalWorkspace(capacity=1, history_limit=3)
    for i in range(5):
        ws.submit(make(f"item-{i}", 0.5))
        ws.compete(scorer_by_hint)

    assert len(ws.history) == 3
    # The three *most recent* cycles survive
    assert ws.history[-1].broadcast[0].source == "item-4"
    assert ws.history[0].broadcast[0].source == "item-2"


def test_cycle_has_a_started_at_timestamp():
    ws = GlobalWorkspace(capacity=1)
    ws.submit(make("a", 0.5))
    cycle = ws.compete(scorer_by_hint)
    assert cycle.started_at > 0
    assert cycle.cycle_id  # non-empty string
