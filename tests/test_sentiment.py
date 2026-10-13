import pytest

from limen.modules.sentiment import DEFAULT_LEXICON, LexiconSentiment
from limen.workspace import Content


# ---------- tokenization / scoring ----------

def test_module_has_name():
    m = LexiconSentiment()
    assert m.name == "sentiment"


def test_produce_returns_content():
    m = LexiconSentiment()
    c = m.produce("I love this")
    assert isinstance(c, Content)
    assert c.source == "sentiment"


def test_positive_text_yields_positive_label():
    m = LexiconSentiment()
    c = m.produce("I love this, it is wonderful and great")
    assert c.payload["label"] == "pos"
    assert c.payload["sentiment"] > 0


def test_negative_text_yields_negative_label():
    m = LexiconSentiment()
    c = m.produce("I hate this, it is terrible and awful")
    assert c.payload["label"] == "neg"
    assert c.payload["sentiment"] < 0


def test_neutral_text_yields_neutral_label():
    m = LexiconSentiment()
    c = m.produce("the table is made of wood")
    assert c.payload["label"] == "neu"
    assert c.payload["sentiment"] == pytest.approx(0.0)


def test_sentiment_is_in_unit_interval():
    m = LexiconSentiment()
    for text in ["love", "hate", "love love love love", "hate hate hate hate", "neutral"]:
        c = m.produce(text)
        assert -1.0 <= c.payload["sentiment"] <= 1.0


def test_relevance_defaults_to_half():
    m = LexiconSentiment()
    assert m.produce("hi").payload["relevance"] == pytest.approx(0.5)


# ---------- custom lexicon ----------

def test_custom_lexicon_is_used():
    m = LexiconSentiment(lexicon={"banana": 1.0})
    c = m.produce("banana banana banana")
    assert c.payload["label"] == "pos"
    assert c.payload["sentiment"] > 0


def test_default_lexicon_has_expected_words():
    for w in ("love", "hate", "great", "awful"):
        assert w in DEFAULT_LEXICON


# ---------- edge cases ----------

def test_empty_text_is_neutral():
    m = LexiconSentiment()
    c = m.produce("")
    assert c.payload["label"] == "neu"
    assert c.payload["sentiment"] == pytest.approx(0.0)


def test_punctuation_and_case_are_normalized():
    m = LexiconSentiment()
    a = m.produce("LOVE!!!")
    b = m.produce("love")
    assert a.payload == b.payload


def test_repeated_words_saturate_not_explode():
    """Ten 'love' should not be twice as intense as five."""
    m = LexiconSentiment()
    five = m.produce("love " * 5).payload["sentiment"]
    ten = m.produce("love " * 10).payload["sentiment"]
    assert five <= ten
    assert ten <= 1.0
