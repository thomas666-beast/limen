"""Memory module: recall recent inputs, emit a novelty signal.

This is the first module with internal state. Each call to ``produce``:

1. tokenizes the input
2. compares it against the ring buffer of past entries (Jaccard)
3. emits Content with ``novelty`` (1 - best_similarity) and ``best_match``
4. appends the current input to memory

Design notes
------------
- Similarity is Jaccard on token sets: |A∩B| / |A∪B|. Fast, deterministic,
  good enough for a research prototype. Swap for embeddings later without
  changing the interface.
- Memory is a bounded deque; oldest entries drop first.
- ``relevance`` is set to ``novelty`` — the module's job is novelty.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from limen.workspace import Content


_TOKEN_RE = re.compile(r"[a-z']+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0  # two empties are trivially identical
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass(slots=True)
class _Entry:
    text: str
    tokens: frozenset[str]


@dataclass(slots=True)
class MemoryModule:
    """Recall recent inputs; emit novelty Content."""

    name: str = field(default="memory", init=False)
    capacity: int = 50
    _memory: deque[_Entry] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        self._memory = deque(self._memory, maxlen=self.capacity)

    # ---------- public API ----------

    def produce(self, text: str) -> Content:
        tokens_list = _tokens(text)
        tokens = frozenset(tokens_list)

        best_similarity = 0.0
        best_match: str | None = None

        for entry in self._memory:
            sim = _jaccard(tokens, entry.tokens)
            if sim > best_similarity:
                best_similarity = sim
                best_match = entry.text

        novelty = 1.0 - best_similarity

        # remember this input for future calls
        self._memory.append(_Entry(text=text, tokens=tokens))

        return Content(
            source=self.name,
            payload={
                "novelty": novelty,
                "similarity": best_similarity,
                "best_match": best_match,
                "relevance": novelty,
                "n_tokens": len(tokens_list),
            },
        )

    def reset(self) -> None:
        """Forget all remembered inputs. Useful for reproducible runs."""
        self._memory.clear()
