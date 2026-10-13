import pytest

from limen.modules.memory import MemoryModule
from limen.workspace import Content


# ---------- contract ----------

def test_module_has_name():
    m = MemoryModule()
    assert m.name == "memory"


def test_produce_returns_content():
    m = MemoryModule()
    c = m.produce("the sky is blue")
    assert isinstance(c, Content)
    assert c.source == "memory"


# ---------- first call ----------

def test_first_call_is_maximally_novel():
    m = MemoryModule()
    c = m.produce("hello world")
    assert c.payload["novelty"] == pytest.approx(1.0)
    assert c.payload["similarity"] == pytest.approx(0.0)
    assert c.payload["best_match"] is None


# ---------- repeat call ----------

def test_repeat_exact_text_is_not_novel():
    m = MemoryModule()
    m.produce("the sky is blue")
    c = m.produce("the sky is blue")
    assert c.payload["novelty"] == pytest.approx(0.0)
    assert c.payload["similarity"] == pytest.approx(1.0)
    assert c.payload["best_match"] == "the sky is blue"


def test_partial_overlap_is_partially_novel():
    m = MemoryModule()
    m.produce("the sky is blue")
    c = m.produce("the sky is gray")
    # overlap: the, sky, is  → 3 of 4
    assert 0.0 < c.payload["novelty"] < 1.0
    assert c.payload["best_match"] == "the sky is blue"


# ---------- ring buffer ----------

def test_memory_is_bounded():
    m = MemoryModule(capacity=2)
    m.produce("a a")
    m.produce("b b")
    m.produce("c c")
    # 'a a' has been dropped
    c = m.produce("a a")
    assert c.payload["best_match"] != "a a"


def test_reset_clears_memory():
    m = MemoryModule()
    m.produce("something")
    m.reset()
    c = m.produce("something")
    assert c.payload["novelty"] == pytest.approx(1.0)
    assert c.payload["best_match"] is None


# ---------- payload ----------

def test_novelty_and_similarity_in_unit_interval():
    m = MemoryModule()
    for text in ["a", "a b", "a b c", "unrelated tokens here"]:
        c = m.produce(text)
        assert 0.0 <= c.payload["novelty"] <= 1.0
        assert 0.0 <= c.payload["similarity"] <= 1.0


def test_relevance_mirrors_novelty():
    m = MemoryModule()
    m.produce("the same thing")
    c = m.produce("the same thing")
    assert c.payload["relevance"] == pytest.approx(c.payload["novelty"])


def test_empty_text_handled():
    m = MemoryModule()
    c = m.produce("")
    assert c.payload["novelty"] == pytest.approx(1.0)
    assert c.payload["n_tokens"] == 0


# ---------- constructor validation ----------

def test_capacity_must_be_positive():
    with pytest.raises(ValueError, match="capacity"):
        MemoryModule(capacity=0)


# ---------- determinism ----------

def test_deterministic_given_same_history():
    m1 = MemoryModule()
    m2 = MemoryModule()
    for t in ["a b", "c d", "a b e"]:
        m1.produce(t)
        m2.produce(t)
    c1 = m1.produce("a b f")
    c2 = m2.produce("a b f")
    assert c1.payload == c2.payload
