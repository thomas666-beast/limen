"""Salience scoring: which submitted items deserve to reach consciousness.

A scorer is any callable ``Content -> float in [0, 1]``. The workspace
(Task 3) calls it for every item in the pool. Higher = more likely to
be broadcast.

Two scorers are provided:

- ``ConstantScorer`` — every item gets the same score. Useful as a
  baseline in experiments.
- ``WeightedScorer`` — the default. Combines novelty, emotional weight,
  and relevance using weights from config.

Nothing here has hidden state. If a scorer needs memory of recent items,
it reads from an injected ``SalienceContext`` and the *caller* decides
when to remember.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from limen.config import SalienceWeights
from limen.workspace import Content


# ---------- context ----------

@dataclass(slots=True)
class SalienceContext:
    """A tiny ring buffer of recently seen payloads.

    Used only for computing novelty. The caller decides what to remember
    and when; scoring itself never writes here.
    """

    capacity: int = 100
    _recent: deque[dict[str, Any]] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError(f"capacity must be positive, got {self.capacity}")
        # rebind deque with maxlen now that capacity is validated
        self._recent = deque(self._recent, maxlen=self.capacity)

    def remember(self, payload: dict[str, Any]) -> None:
        self._recent.append(dict(payload))

    @property
    def recent_payloads(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._recent)

    def is_novel(self, payload: dict[str, Any]) -> bool:
        """True if no remembered payload equals this one."""
        return all(p != payload for p in self._recent)


# ---------- scorers ----------

@dataclass(frozen=True, slots=True)
class ConstantScorer:
    """Every item gets the same salience. Baseline for experiments."""

    value: float = 0.5

    def __post_init__(self) -> None:
        if not (0.0 <= self.value <= 1.0):
            raise ValueError(
                f"constant must be in [0, 1], got {self.value}"
            )

    def __call__(self, _content: Content) -> float:
        return float(self.value)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


@dataclass(frozen=True, slots=True)
class WeightedScorer:
    """Combine novelty, emotional weight, and relevance.

    Reads from ``Content.payload``:

    - ``novelty``   : optional float in [0, 1]. If absent, computed from
                      context: 1.0 if payload unseen, else 0.0.
    - ``sentiment`` : optional float in [-1, 1]. Absolute value is used.
    - ``relevance`` : optional float in [0, 1]. Defaults to 0.5.

    Components are clamped to [0, 1] before weighting so a misbehaving
    module cannot dominate competition.
    """

    weights: SalienceWeights
    context: SalienceContext

    def __call__(self, content: Content) -> float:
        p = content.payload

        # novelty
        if "novelty" in p:
            novelty = _clamp01(float(p["novelty"]))
        else:
            novelty = 1.0 if self.context.is_novel(p) else 0.0

        # emotional weight (absolute value of sentiment)
        emotion = _clamp01(abs(float(p.get("sentiment", 0.0))))

        # relevance
        relevance = _clamp01(float(p.get("relevance", 0.5)))

        w = self.weights
        score = w.novelty * novelty + w.emotion * emotion + w.relevance * relevance
        return _clamp01(score)
