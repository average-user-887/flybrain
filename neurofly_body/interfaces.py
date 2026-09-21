"""Dependency-injection protocols for embodied co-simulation."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class NeuralBackend(Protocol):
    """Minimal interface implemented by ``ConnectomeServer`` and test fakes."""

    def reset(self) -> None: ...

    def get_status(self) -> dict[str, Any]: ...

    def step(self, sensory: dict[str, Any], duration_ms: float = 2.0) -> dict[str, Any]: ...


@runtime_checkable
class BodyBackend(Protocol):
    """Physical body seam; values returned by ``observe`` must be JSON-compatible."""

    @property
    def physics_dt_s(self) -> float: ...

    def reset(self, seed: int) -> dict[str, Any]: ...

    def step(self, cpg_drive: tuple[float, float], substeps: int) -> dict[str, Any]: ...

    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None: ...
