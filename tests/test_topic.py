import pytest

from limen.modules.topic import (
    DEFAULT_GREETINGS,
    DEFAULT_GREETING_PHRASES,
    DEFAULT_QUESTION_WORDS,
    DEFAULT_COMMANDS,
    TopicClassifier,
)
from limen.workspace import Content


# ---------- basic contract ----------

def test_module_has_name():
    m = TopicClassifier()
    assert m.name == "topic"


def test_produce_returns_content():
    m = TopicClassifier()
    c = m.produce("hello")
    assert isinstance(c, Content)
    assert c.source == "topic"


# ---------- classification ----------

def test_greeting_detected():
    m = TopicClassifier()
    for text in ["hi", "Hello!", "hey there", "good morning", "Hi."]:
        c = m.produce(text)
        assert c.payload["topic"] == "greeting", text


def test_question_detected():
    m = TopicClassifier()
    for text in ["what is this?", "why?", "how are you", "where do we go?", "who is that?"]:
        c = m.produce(text)
        assert c.payload["topic"] == "question", text


def test_command_detected():
    m = TopicClassifier()
    for text in ["open the door", "stop talking", "run now", "look at this", "listen carefully"]:
        c = m.produce(text)
        assert c.payload["topic"] == "command", text


def test_statement_is_fallback():
    m = TopicClassifier()
    for text in ["the sky is blue", "water boils at 100 degrees", "a table has four legs"]:
        c = m.produce(text)
        assert c.payload["topic"] == "statement", text


def test_greeting_takes_priority_over_question():
    m = TopicClassifier()
    # "hi, what's up?" — greeting should win
    c = m.produce("hi, what's up?")
    assert c.payload["topic"] == "greeting"


# ---------- relevance ----------

def test_relevance_per_topic():
    m = TopicClassifier()
    assert m.produce("what is this?").payload["relevance"] == pytest.approx(0.9)
    assert m.produce("open it").payload["relevance"] == pytest.approx(0.8)
    assert m.produce("hi").payload["relevance"] == pytest.approx(0.6)
    assert m.produce("the sky is blue").payload["relevance"] == pytest.approx(0.3)


# ---------- confidence ----------

def test_confidence_in_unit_interval():
    m = TopicClassifier()
    for text in ["hi", "what?", "stop", "the sky is blue", "", "???"]:
        c = m.produce(text)
        assert 0.0 <= c.payload["confidence"] <= 1.0


def test_empty_text_is_statement_with_low_confidence():
    m = TopicClassifier()
    c = m.produce("")
    assert c.payload["topic"] == "statement"
    assert c.payload["confidence"] <= 0.4


# ---------- custom vocabularies ----------

def test_custom_greetings_are_used():
    m = TopicClassifier(greetings={"yo"})
    assert m.produce("yo").payload["topic"] == "greeting"


def test_custom_commands_are_used():
    m = TopicClassifier(commands={"banish"})
    assert m.produce("banish the dark").payload["topic"] == "command"


# ---------- determinism / normalization ----------

def test_case_and_punctuation_normalized():
    m = TopicClassifier()
    a = m.produce("HELLO!!!")
    b = m.produce("hello")
    assert a.payload["topic"] == b.payload["topic"]
    assert a.payload["confidence"] == b.payload["confidence"]


def test_defaults_contain_expected_words():
    for w in ("hi", "hello", "hey"):
        assert w in DEFAULT_GREETINGS
    for w in ("what", "why", "how", "where", "who", "when"):
        assert w in DEFAULT_QUESTION_WORDS
    for w in ("open", "stop", "run", "look", "listen"):
        assert w in DEFAULT_COMMANDS

def test_greeting_phrases_detected():
    m = TopicClassifier()
    for text in ["good morning", "Good Evening.", "good afternoon friend", "good night"]:
        c = m.produce(text)
        assert c.payload["topic"] == "greeting", text


def test_morning_alone_is_not_greeting():
    m = TopicClassifier()
    # 'morning' by itself is not a greeting; this sentence has no
    # question words, no command verbs, and no greetings.
    c = m.produce("morning light fills the room")
    assert c.payload["topic"] == "statement"


def test_default_greeting_phrases_contain_expected():
    for phrase in ("good morning", "good evening"):
        assert phrase in DEFAULT_GREETING_PHRASES
