import json
import sqlite3
from pathlib import Path

import pytest

from limen.logger import Logger
from limen.workspace import Content, Cycle


def make_cycle(cycle_id="c1", broadcast_sources=("hi",), unconscious_sources=("lo",)):
    b = tuple(Content(source=s, payload={"n": i}, salience=0.9 - i * 0.1)
              for i, s in enumerate(broadcast_sources))
    u = tuple(Content(source=s, payload={"n": i}, salience=0.1 - i * 0.01)
              for i, s in enumerate(unconscious_sources))
    return Cycle(
        cycle_id=cycle_id,
        started_at=1234.5,
        broadcast=b,
        unconscious=u,
    )


# ---------- construction / schema ----------

def test_logger_creates_db_file(tmp_path: Path):
    db = tmp_path / "logs.db"
    with Logger(db):
        pass
    assert db.exists()


def test_logger_creates_tables(tmp_path: Path):
    db = tmp_path / "logs.db"
    with Logger(db):
        pass
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    con.close()
    assert {"cycles", "items"} <= tables


def test_logger_is_reentrant_on_same_path(tmp_path: Path):
    db = tmp_path / "logs.db"
    with Logger(db):
        pass
    # opening again must not wipe data
    with Logger(db):
        pass


# ---------- recording ----------

def test_record_cycle_persists_cycle_row(tmp_path: Path):
    db = tmp_path / "logs.db"
    with Logger(db) as log:
        log.record(make_cycle("c-abc"))

    con = sqlite3.connect(db)
    rows = list(con.execute("SELECT cycle_id, started_at FROM cycles"))
    con.close()
    assert rows == [("c-abc", 1234.5)]


def test_record_cycle_persists_items(tmp_path: Path):
    db = tmp_path / "logs.db"
    cycle = make_cycle(
        broadcast_sources=("a", "b"),
        unconscious_sources=("c",),
    )
    with Logger(db) as log:
        log.record(cycle)

    con = sqlite3.connect(db)
    rows = list(con.execute(
        "SELECT source, broadcast, salience FROM items ORDER BY source"
    ))
    con.close()

    assert [(r[0], r[1]) for r in rows] == [("a", 1), ("b", 1), ("c", 0)]
    assert [r[2] for r in rows] == pytest.approx([0.9, 0.8, 0.1])


def test_record_serializes_payload_as_json(tmp_path: Path):
    db = tmp_path / "logs.db"
    with Logger(db) as log:
        log.record(make_cycle(broadcast_sources=("x",)))

    con = sqlite3.connect(db)
    payload_text, = con.execute(
        "SELECT payload FROM items WHERE source='x'"
    ).fetchone()
    con.close()
    assert json.loads(payload_text) == {"n": 0}


def test_record_rejects_non_cycle(tmp_path: Path):
    with Logger(tmp_path / "logs.db") as log:
        with pytest.raises(TypeError, match="Cycle"):
            log.record("not a cycle")  # type: ignore[arg-type]


def test_record_requires_open(tmp_path: Path):
    log = Logger(tmp_path / "logs.db")
    with pytest.raises(RuntimeError, match="not open"):
        log.record(make_cycle())


# ---------- queries ----------

def test_count_cycles(tmp_path: Path):
    with Logger(tmp_path / "logs.db") as log:
        log.record(make_cycle("c1"))
        log.record(make_cycle("c2"))
        assert log.count_cycles() == 2


def test_broadcast_rows_returns_only_broadcast(tmp_path: Path):
    with Logger(tmp_path / "logs.db") as log:
        log.record(make_cycle(
            broadcast_sources=("hi",),
            unconscious_sources=("lo",),
        ))
        rows = log.broadcast_rows()
    sources = [r["source"] for r in rows]
    assert sources == ["hi"]


def test_unconscious_rows_returns_only_unconscious(tmp_path: Path):
    with Logger(tmp_path / "logs.db") as log:
        log.record(make_cycle(
            broadcast_sources=("hi",),
            unconscious_sources=("lo",),
        ))
        rows = log.unconscious_rows()
    sources = [r["source"] for r in rows]
    assert sources == ["lo"]


def test_iter_items_returns_all_with_cycle_id(tmp_path: Path):
    with Logger(tmp_path / "logs.db") as log:
        log.record(make_cycle(
            cycle_id="c-X",
            broadcast_sources=("hi",),
            unconscious_sources=("lo",),
        ))
        rows = list(log.iter_items())

    assert {r["source"] for r in rows} == {"hi", "lo"}
    assert {r["cycle_id"] for r in rows} == {"c-X"}


def test_close_is_idempotent(tmp_path: Path):
    log = Logger(tmp_path / "logs.db")
    log.open()
    log.close()
    log.close()  # must not raise
