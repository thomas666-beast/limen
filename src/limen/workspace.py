"""The Global Workspace: contents and (later) the competition/broadcast core.

`Content` is the atom. Every module produces `Content`; the workspace
competes over many `Content` items and broadcasts only the winners.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


def _new_id() -> str:
    """Short, collision-resistant id for tracing items through logs."""
    return uuid.uuid4().hex[:12]


@dataclass(frozen=True, slots=True, eq=False)
class Content:
    """A single item produced by a module.

    Immutable by design: once a module emits it, nobody mutates it.
    Salience is assigned by the competition step (Task 3) via `dataclasses.replace`.
    """

    source: str
    payload: dict[str, Any]
    salience: float = 0.0
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=_new_id)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("Content.source must be a non-empty string")
        if not isinstance(self.payload, dict):
            raise TypeError(
                f"Content.payload must be a dict, got {type(self.payload).__name__}"
            )
        if not (0.0 <= self.salience <= 1.0):
            raise ValueError(
                f"Content.salience must be in [0, 1], got {self.salience}"
            )
