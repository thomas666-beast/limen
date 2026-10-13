"""The Bot: wires modules to a GlobalWorkspace.

Responsibilities
----------------
- Hold a list of modules and a scorer.
- On ``respond(text)``:
    1. ask each module to ``produce(text)``
    2. submit every emitted Content to the workspace
    3. call ``compete(scorer)``
    4. remember broadcast payloads in the context (optional)
    5. persist the cycle via Logger (optional)
- Expose ``describe_cycle`` for CLI/UI/JSON output.

Design notes
------------
- A failing module does NOT abort the cycle. Errors are recorded on
  ``bot.last_module_errors`` and surfaced by ``describe_cycle``.
- Memory feedback is limited to broadcast items only. This is a modeling
  choice: only conscious content feeds back into recent context.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Protocol

from limen.logger import Logger
from limen.salience import SalienceContext
from limen.workspace import Content, Cycle, GlobalWorkspace


class Module(Protocol):
    """The module contract: a name and a produce() method."""

    name: str

    def produce(self, text: str) -> Content: ...


class Bot:
    def __init__(
        self,
        modules: Sequence[Module],
        scorer: Callable[[Content], float],
        capacity: int,
        *,
        context: SalienceContext | None = None,
        logger: Logger | None = None,
    ) -> None:
        if not modules:
            raise ValueError("modules must be a non-empty sequence")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.modules = list(modules)
        self.scorer = scorer
        self.workspace = GlobalWorkspace(capacity=capacity)
        self.context = context
        self.logger = logger
        self.last_module_errors: list[dict[str, str]] = []

    # ---------- main entry point ----------

    def respond(self, text: str) -> Cycle:
        self.last_module_errors = []

        for module in self.modules:
            try:
                content = module.produce(text)
            except Exception as exc:  # noqa: BLE001 - we want to catch all module errors
                self.last_module_errors.append(
                    {"module": getattr(module, "name", type(module).__name__),
                     "error": str(exc)}
                )
                continue
            self.workspace.submit(content)

        cycle = self.workspace.compete(self.scorer)

        # feedback: only broadcast content enters recent context
        if self.context is not None:
            for item in cycle.broadcast:
                self.context.remember(item.payload)

        # optional persistence
        if self.logger is not None:
            self.logger.record(cycle)

        return cycle

    def describe(self, cycle: Cycle) -> dict[str, Any]:
        """Describe a cycle this bot produced, including capacity."""
        return describe_cycle(
            cycle,
            module_errors=self.last_module_errors,
            capacity=self.workspace.capacity,
        )


# ---------- description helper ----------

def _content_to_dict(c: Content) -> dict[str, Any]:
    return {
        "id": c.id,
        "source": c.source,
        "salience": c.salience,
        "timestamp": c.timestamp,
        "payload": c.payload,
    }


def describe_cycle(
    cycle: Cycle,
    module_errors: list[dict[str, str]] | None = None,
    capacity: int | None = None,
) -> dict[str, Any]:
    """A JSON-safe summary of a cycle, suitable for CLI/UI/logs."""
    return {
        "cycle_id": cycle.cycle_id,
        "started_at": cycle.started_at,
        "capacity": capacity,
        "broadcast": [_content_to_dict(c) for c in cycle.broadcast],
        "unconscious": [_content_to_dict(c) for c in cycle.unconscious],
        "module_errors": list(module_errors or []),
    }
