"""Lexicon-based sentiment module.

Produces one Content per call with payload:

    {
      "label": "pos" | "neg" | "neu",
      "sentiment": float in [-1, 1],
      "relevance": float in [0, 1],
    }

The lexicon is a plain dict injected at construction. No ML, no model
loading, no randomness — reproducible by design.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from limen.workspace import Content


# A small, intentionally modest default lexicon.
# Positive = +1..+3, Negative = -1..-3 (magnitudes matter for weighting).
DEFAULT_LEXICON: dict[str, float] = {
    # positive
    "love": 2.5,
    "like": 1.0,
    "great": 2.0,
    "wonderful": 2.5,
    "good": 1.0,
    "excellent": 2.5,
    "amazing": 2.5,
    "happy": 1.5,
    "joy": 2.0,
    "beautiful": 2.0,
    "kind": 1.0,
    # negative
    "hate": -2.5,
    "dislike": -1.0,
    "bad": -1.0,
    "awful": -2.5,
    "terrible": -2.5,
    "horrible": -2.5,
    "sad": -1.5,
    "angry": -2.0,
    "ugly": -2.0,
    "cruel": -2.0,
    "pain": -1.5,
}


_TOKEN_RE = re.compile(r"[a-z']+")
_POS_THRESHOLD = 0.15
_NEG_THRESHOLD = -0.15


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class LexiconSentiment:
    """Emits a sentiment Content for the given text.

    Parameters
    ----------
    lexicon : mapping of word -> weight (positive or negative).
    relevance : fixed relevance value to attach (default 0.5).
    max_magnitude : denominator used to squash the raw sum into [-1, 1].
    """

    name: str = field(default="sentiment", init=False)
    lexicon: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_LEXICON))
    relevance: float = 0.5
    max_magnitude: float = 6.0

    def __post_init__(self) -> None:
        if self.max_magnitude <= 0:
            raise ValueError("max_magnitude must be positive")
        if not (0.0 <= self.relevance <= 1.0):
            raise ValueError(f"relevance must be in [0, 1], got {self.relevance}")

    def produce(self, text: str) -> Content:
        tokens = _tokens(text)

        raw = 0.0
        for t in tokens:
            raw += self.lexicon.get(t, 0.0)

        # Squash to [-1, 1] via tanh-like scaling. Deterministic, bounded.
        scaled = math.tanh(raw / self.max_magnitude) if raw else 0.0

        if scaled > _POS_THRESHOLD:
            label = "pos"
        elif scaled < _NEG_THRESHOLD:
            label = "neg"
        else:
            label = "neu"

        return Content(
            source=self.name,
            payload={
                "label": label,
                "sentiment": scaled,
                "relevance": self.relevance,
                "n_tokens": len(tokens),
            },
        )
