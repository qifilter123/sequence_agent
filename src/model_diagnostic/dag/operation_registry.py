# from __future__ import annotations

from collections.abc import Callable
from typing import Any


DagOperation = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], Any]


class OperationRegistry:
    """Registry mapping declarative DAG operation names to Python callables."""

    def __init__(self) -> None:
        self._operations: dict[str, DagOperation] = {}

    def register(self, name: str, operation: DagOperation | None = None):
        """Register an operation directly or use as a decorator."""
        if not name or not isinstance(name, str):
            raise ValueError("Operation name must be a non-empty string")

        def decorator(fn: DagOperation) -> DagOperation:
            if name in self._operations:
                raise ValueError(f"DAG operation already registered: {name}")
            self._operations[name] = fn
            return fn

        if operation is not None:
            return decorator(operation)
        return decorator

    def get(self, name: str) -> DagOperation:
        try:
            return self._operations[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._operations)) or "<none>"
            raise KeyError(
                f"Unknown DAG operation '{name}'. Registered operations: {available}"
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._operations))
