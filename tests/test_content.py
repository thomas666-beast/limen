from dataclasses import FrozenInstanceError

import pytest

from limen.workspace import Content


def test_create_content_with_defaults():
    c = Content(source="sentiment", payload={"label": "pos", "score": 0.9})

    assert c.source == "sentiment"
    assert c.payload == {"label": "pos", "score": 0.9}
    assert c.salience == 0.0
    assert isinstance(c.timestamp, float)
    assert c.timestamp > 0
    assert isinstance(c.id, str)
    assert len(c.id) == 12


def test_ids_are_unique():
    a = Content(source="x", payload={})
    b = Content(source="x", payload={})
    assert a.id != b.id


def test_explicit_timestamp_is_kept():
    c = Content(source="x", payload={}, timestamp=1234.5)
    assert c.timestamp == 1234.5


def test_salience_can_be_set_at_creation():
    c = Content(source="x", payload={}, salience=0.7)
    assert c.salience == 0.7


def test_content_is_frozen():
    c = Content(source="x", payload={})
    with pytest.raises(FrozenInstanceError):
        c.salience = 0.5  # type: ignore[misc]


def test_payload_must_be_a_dict():
    with pytest.raises(TypeError, match="payload"):
        Content(source="x", payload=[1, 2, 3])  # type: ignore[arg-type]


def test_source_must_be_non_empty_string():
    with pytest.raises(ValueError, match="source"):
        Content(source="", payload={})


def test_salience_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="salience"):
        Content(source="x", payload={}, salience=1.5)
    with pytest.raises(ValueError, match="salience"):
        Content(source="x", payload={}, salience=-0.1)


def test_equality_is_identity_based():
    a = Content(source="x", payload={"k": 1})
    b = Content(source="x", payload={"k": 1})
    assert a != b
    assert a == a
