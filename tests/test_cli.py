import json
import subprocess
import sys
from pathlib import Path

import pytest

from limen.bot import Bot
from limen.cli import build_default_bot, format_cycle, main


# ---------- build_default_bot ----------

def test_build_default_bot_returns_bot():
    bot = build_default_bot()
    assert isinstance(bot, Bot)
    assert {m.name for m in bot.modules} == {"sentiment", "topic", "memory"}


def test_build_default_bot_respects_capacity_override():
    bot = build_default_bot(capacity=1)
    assert bot.workspace.capacity == 1


def test_build_default_bot_uses_config(tmp_path: Path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("workspace:\n  capacity: 2\n")
    bot = build_default_bot(config_path=cfg)
    assert bot.workspace.capacity == 2


# ---------- format_cycle ----------

def _sample_cycle_dict():
    return {
        "cycle_id": "abc",
        "started_at": 1234.5,
        "capacity": 2,
        "broadcast": [
            {"id": "b1", "source": "topic", "salience": 0.9,
             "timestamp": 1234.5, "payload": {"topic": "question"}},
        ],
        "unconscious": [
            {"id": "u1", "source": "sentiment", "salience": 0.2,
             "timestamp": 1234.5, "payload": {"label": "neu"}},
        ],
        "module_errors": [],
    }


def test_format_cycle_mentions_broadcast_source():
    text = format_cycle(_sample_cycle_dict())
    assert "topic" in text
    assert "question" in text


def test_format_cycle_mentions_unconscious_source():
    text = format_cycle(_sample_cycle_dict())
    assert "sentiment" in text


def test_format_cycle_mentions_errors_when_present():
    d = _sample_cycle_dict()
    d["module_errors"] = [{"module": "boom", "error": "kaboom"}]
    text = format_cycle(d)
    assert "boom" in text
    assert "kaboom" in text


# ---------- main() ----------

def test_main_prints_something(capsys):
    rc = main(["hello there"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() != ""


def test_main_json_mode_is_valid_json(capsys):
    rc = main(["--json", "hello there"])
    out = capsys.readouterr().out
    assert rc == 0
    data = json.loads(out)
    assert set(data.keys()) >= {"cycle_id", "broadcast", "unconscious"}


def test_main_writes_log_when_requested(tmp_path, capsys):
    db = tmp_path / "logs.db"
    rc = main(["--log", str(db), "hi"])
    assert rc == 0
    assert db.exists()


def test_main_reads_stdin_when_dash(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", type("S", (), {"read": staticmethod(lambda: "from stdin")})())
    rc = main(["-"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.strip() != ""


def test_main_errors_on_empty_input(capsys):
    rc = main([""])
    err = capsys.readouterr().err
    assert rc == 2
    assert "empty" in err.lower()


# ---------- console script (install-time) ----------

def test_console_script_available():
    """Only runs if `limen` is installed as a console script."""
    result = subprocess.run(
        ["limen", "--help"],
        capture_output=True, text=True,
    )
    # If the console script isn't on PATH, skip rather than fail.
    if result.returncode == 127 or "not found" in result.stderr.lower():
        pytest.skip("limen console script not installed")
    assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()
