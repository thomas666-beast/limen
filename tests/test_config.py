from pathlib import Path

import pytest

from limen.config import Config, load_config


def test_load_config_from_file(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        """
workspace:
  capacity: 5
salience:
  weights:
    novelty: 0.5
    emotion: 0.3
    relevance: 0.2
logging:
  db_path: data/test.db
""".strip()
    )

    cfg = load_config(cfg_file)

    assert isinstance(cfg, Config)
    assert cfg.workspace.capacity == 5
    assert cfg.salience.weights.novelty == 0.5
    assert cfg.salience.weights.emotion == 0.3
    assert cfg.salience.weights.relevance == 0.2
    assert cfg.logging.db_path == "data/test.db"


def test_load_config_missing_file_uses_defaults(tmp_path: Path):
    cfg = load_config(tmp_path / "does-not-exist.yaml")

    assert cfg.workspace.capacity == 3
    assert cfg.salience.weights.novelty == 0.4
    assert cfg.salience.weights.emotion == 0.4
    assert cfg.salience.weights.relevance == 0.2
    assert cfg.logging.db_path == "data/logs.db"


def test_load_config_partial_file_fills_defaults(tmp_path: Path):
    cfg_file = tmp_path / "partial.yaml"
    cfg_file.write_text(
        """
workspace:
  capacity: 7
""".strip()
    )

    cfg = load_config(cfg_file)

    assert cfg.workspace.capacity == 7
    # untouched sections come from defaults
    assert cfg.salience.weights.novelty == 0.4
    assert cfg.logging.db_path == "data/logs.db"


def test_capacity_must_be_positive(tmp_path: Path):
    cfg_file = tmp_path / "bad.yaml"
    cfg_file.write_text("workspace:\n  capacity: 0\n")

    with pytest.raises(ValueError, match="capacity"):
        load_config(cfg_file)


def test_salience_weights_must_sum_to_one(tmp_path: Path):
    cfg_file = tmp_path / "bad_weights.yaml"
    cfg_file.write_text(
        """
salience:
  weights:
    novelty: 0.9
    emotion: 0.9
    relevance: 0.9
""".strip()
    )

    with pytest.raises(ValueError, match="sum"):
        load_config(cfg_file)
