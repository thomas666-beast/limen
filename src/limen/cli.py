"""Command-line entry point for limen.

Usage examples
--------------
    limen "I love this, what is it?"
    limen --json "hi"
    echo "text here" | limen -
    limen --capacity 1 --log data/run.db "test"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from limen.bot import Bot
from limen.config import Config, load_config
from limen.logger import Logger
from limen.modules.memory import MemoryModule
from limen.modules.sentiment import LexiconSentiment
from limen.modules.topic import TopicClassifier
from limen.salience import SalienceContext, WeightedScorer


# ---------- factory ----------

def build_default_bot(
    *,
    config_path: str | Path | None = None,
    capacity: int | None = None,
    logger: Logger | None = None,
    memory_feedback: bool = True,
) -> Bot:
    """Build the standard bot: sentiment + topic + memory + weighted scorer.

    Parameters
    ----------
    config_path : optional path to a YAML config.
    capacity : overrides config.workspace.capacity when given.
    logger : attach a Logger for persistence.
    memory_feedback : whether broadcast items feed back into SalienceContext.
    """
    cfg: Config = load_config(config_path)

    cap = capacity if capacity is not None else cfg.workspace.capacity

    context = SalienceContext(capacity=100) if memory_feedback else None

    scorer = WeightedScorer(weights=cfg.salience.weights, context=context or SalienceContext())

    return Bot(
        modules=[LexiconSentiment(), TopicClassifier(), MemoryModule()],
        scorer=scorer,
        capacity=cap,
        context=context,
        logger=logger,
    )


# ---------- human-readable formatting ----------

def _fmt_item(item: dict[str, Any]) -> str:
    payload = item.get("payload", {})
    sal = item.get("salience", 0.0)
    src = item.get("source", "?")
    payload_str = ", ".join(f"{k}={v!r}" for k, v in payload.items())
    return f"  [{sal:.3f}] {src}: {payload_str}"


def format_cycle(described: dict[str, Any]) -> str:
    """Turn `bot.describe(cycle)` output into a plain-text report."""
    lines: list[str] = []
    lines.append(f"cycle {described.get('cycle_id', '?')} "
                 f"(capacity={described.get('capacity')})")

    lines.append("conscious (broadcast):")
    bcast = described.get("broadcast", [])
    if bcast:
        for item in bcast:
            lines.append(_fmt_item(item))
    else:
        lines.append("  (none)")

    lines.append("unconscious:")
    uncon = described.get("unconscious", [])
    if uncon:
        for item in uncon:
            lines.append(_fmt_item(item))
    else:
        lines.append("  (none)")

    errors = described.get("module_errors", [])
    if errors:
        lines.append("module errors:")
        for e in errors:
            lines.append(f"  {e.get('module', '?')}: {e.get('error', '?')}")

    return "\n".join(lines)


# ---------- argparse ----------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="limen",
        description="Run the limen Global Workspace bot on a piece of text.",
    )
    p.add_argument(
        "text",
        nargs="?",
        help="input text, or '-' to read from stdin",
    )
    p.add_argument(
        "--config", type=Path, default=None,
        help="path to a YAML config (defaults to config.yaml if present)",
    )
    p.add_argument(
        "--capacity", type=int, default=None,
        help="override workspace capacity",
    )
    p.add_argument(
        "--log", type=Path, default=None,
        help="path to a SQLite log file; disables in-memory-only mode",
    )
    p.add_argument(
        "--json", action="store_true",
        help="print the cycle as JSON instead of human-readable text",
    )
    p.add_argument(
        "--no-memory-feedback", action="store_true",
        help="do not feed broadcast payloads back into the salience context",
    )
    return p


# ---------- main ----------

def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # resolve input text
    if args.text is None or args.text == "-":
        text = sys.stdin.read()
    else:
        text = args.text

    if not text.strip():
        print("error: input text is empty", file=sys.stderr)
        return 2

    # optional logger
    logger_cm: Any
    if args.log is not None:
        logger_cm = Logger(args.log)
    else:
        logger_cm = _NullContext()

    try:
        with logger_cm as log:
            bot = build_default_bot(
                config_path=args.config,
                capacity=args.capacity,
                logger=log if isinstance(log, Logger) else None,
                memory_feedback=not args.no_memory_feedback,
            )
            cycle = bot.respond(text)
            described = bot.describe(cycle)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(described, indent=2))
    else:
        print(format_cycle(described))

    return 0


# ---------- tiny null context manager ----------

class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
