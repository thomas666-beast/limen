import pytest

from limen.bot import Bot, describe_cycle
from limen.config import SalienceWeights
from limen.modules.memory import MemoryModule
from limen.modules.sentiment import LexiconSentiment
from limen.modules.topic import TopicClassifier
from limen.salience import ConstantScorer, SalienceContext, WeightedScorer
from limen.workspace import Content, Cycle


# ---------- a fake module for controlled tests ----------

class FixedModule:
    name = "fixed"

    def __init__(self, payload):
        self._payload = payload

    def produce(self, _text):
        return Content(source=self.name, payload=dict(self._payload))


class ExplodingModule:
    name = "boom"

    def produce(self, _text):
        raise RuntimeError("intentional failure")


# ---------- construction ----------

def test_bot_requires_modules():
    with pytest.raises(ValueError, match="modules"):
        Bot(modules=[], scorer=ConstantScorer(0.5), capacity=3)


def test_bot_requires_positive_capacity():
    with pytest.raises(ValueError, match="capacity"):
        Bot(modules=[FixedModule({})], scorer=ConstantScorer(0.5), capacity=0)


# ---------- one-shot ----------

def test_respond_returns_cycle():
    bot = Bot(
        modules=[FixedModule({"novelty": 0.9})],
        scorer=ConstantScorer(0.5),
        capacity=1,
    )
    cycle = bot.respond("hello")
    assert isinstance(cycle, Cycle)
    assert len(cycle.broadcast) == 1
    assert cycle.broadcast[0].source == "fixed"


def test_respond_submits_from_every_module():
    bot = Bot(
        modules=[
            FixedModule({"novelty": 0.3}),
            FixedModule({"novelty": 0.6}),
        ],
        scorer=ConstantScorer(0.5),
        capacity=5,
    )
    cycle = bot.respond("anything")
    assert len(cycle.broadcast) == 2


# ---------- competition is real ----------

def test_scorer_decides_what_wins():
    class Tagged:
        def __init__(self, name, novelty):
            self.name = name
            self._n = novelty

        def produce(self, _text):
            return Content(source=self.name, payload={"novelty": self._n})

    bot = Bot(
        modules=[Tagged("low", 0.1), Tagged("high", 0.9)],
        scorer=WeightedScorer(
            weights=SalienceWeights(novelty=1.0, emotion=0.0, relevance=0.0),
            context=SalienceContext(),
        ),
        capacity=1,
    )
    cycle = bot.respond("x")
    assert cycle.broadcast[0].source == "high"
    assert cycle.unconscious[0].source == "low"


# ---------- error tolerance ----------

def test_failing_module_does_not_kill_cycle():
    bot = Bot(
        modules=[ExplodingModule(), FixedModule({"novelty": 0.9})],
        scorer=ConstantScorer(0.5),
        capacity=3,
    )
    cycle = bot.respond("x")
    sources = [c.source for c in cycle.broadcast]
    assert "fixed" in sources
    assert "boom" not in sources


def test_describe_cycle_reports_errors():
    bot = Bot(
        modules=[ExplodingModule()],
        scorer=ConstantScorer(0.5),
        capacity=3,
    )
    cycle = bot.respond("x")
    described = bot.describe(cycle)
    assert described["module_errors"] == [
        {"module": "boom", "error": "intentional failure"}
    ]


# ---------- memory of conscious content ----------

def test_broadcast_items_are_remembered_in_context():
    ctx = SalienceContext(capacity=10)
    bot = Bot(
        modules=[FixedModule({"novelty": 0.9})],
        scorer=ConstantScorer(0.5),
        capacity=1,
        context=ctx,
    )
    bot.respond("x")
    assert ctx.recent_payloads == ({"novelty": 0.9},)


def test_unconscious_items_are_not_remembered():
    ctx = SalienceContext(capacity=10)
    bot = Bot(
        modules=[FixedModule({"novelty": 0.9}), FixedModule({"novelty": 0.1})],
        scorer=ConstantScorer(0.5),
        capacity=1,
        context=ctx,
    )
    bot.respond("x")
    # only the broadcast payload ends up in context
    assert len(ctx.recent_payloads) == 1


# ---------- logging is optional ----------

def test_bot_without_logger_does_not_write(tmp_path):
    bot = Bot(
        modules=[FixedModule({"novelty": 0.9})],
        scorer=ConstantScorer(0.5),
        capacity=1,
    )
    bot.respond("x")
    assert not (tmp_path / "logs.db").exists()


def test_bot_with_logger_writes_a_cycle(tmp_path):
    from limen.logger import Logger

    db = tmp_path / "logs.db"
    with Logger(db) as log:
        bot = Bot(
            modules=[FixedModule({"novelty": 0.9})],
            scorer=ConstantScorer(0.5),
            capacity=1,
            logger=log,
        )
        bot.respond("x")

    with Logger(db) as log:
        assert log.count_cycles() == 1


# ---------- describe_cycle ----------

def test_describe_cycle_shape():
    bot = Bot(
        modules=[FixedModule({"novelty": 0.9})],
        scorer=ConstantScorer(0.5),
        capacity=1,
    )
    cycle = bot.respond("x")
    d = bot.describe(cycle)

    assert set(d.keys()) >= {
        "cycle_id", "started_at", "capacity",
        "broadcast", "unconscious", "module_errors",
    }
    assert d["capacity"] == 1
    assert isinstance(d["broadcast"], list)
    assert d["broadcast"][0]["source"] == "fixed"
    assert isinstance(d["broadcast"][0]["payload"], dict)


# ---------- end-to-end with real modules ----------

def test_end_to_end_with_real_modules():
    bot = Bot(
        modules=[LexiconSentiment(), TopicClassifier(), MemoryModule()],
        scorer=WeightedScorer(
            weights=SalienceWeights(),
            context=SalienceContext(),
        ),
        capacity=2,
    )
    cycle = bot.respond("I love this, what is it?")
    sources = {c.source for c in cycle.broadcast} | {c.source for c in cycle.unconscious}
    assert sources == {"sentiment", "topic", "memory"}
    assert len(cycle.broadcast) == 2
    assert len(cycle.unconscious) == 1
