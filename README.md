# limen

*A Global Workspace model for studying conscious and unconscious processing.*

**limen** (Latin: *threshold*) — the boundary between what is processed
unconsciously and what enters conscious access.

## What this is

A small, transparent AI system that implements a toy version of
**Global Workspace Theory** (Baars; Dehaene). Input is processed by
several parallel modules; only a small subset of their outputs
"wins" a salience competition and is broadcast as the system's
conscious content. Everything else stays in the unconscious pool
and is logged for analysis.

## What this is *not*

- Not a claim that the bot is conscious
- Not a theory of consciousness itself
- Not a replacement for human-subject research

It is a **model** and a **research tool**.

## Status

v0.0.1 — scaffolding only.

## Install (dev)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Test

```bash
pytest
```

## Layout

```
src/limen/        core package
tests/            pytest suite
experiments/      runnable paradigms (priming, capacity, ...)
notebooks/        analysis
ui/               Streamlit dashboard
data/             SQLite logs (git-ignored)
docs/             notes, references, write-up
```

## License

MIT
