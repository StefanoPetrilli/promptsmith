"""Backend registry.

Backends (generation, labeling, dataset adapters) are Python classes registered under
string names; config files reference them by name. This is the single seam that keeps the
pipeline model-agnostic and reusable.

Usage in a backend module:
    from numeri.registry import generation_backends
    @generation_backends.register("diffusion")
    class DiffusionBackend: ...
"""
from __future__ import annotations

from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self.kind} backend '{name}' already registered")
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"Unknown {self.kind} backend '{name}'. Available: {available}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)


# Public registries — import these in backend modules to register, and in cli/runner to
# instantiate from config.
dataset_adapters: Registry = Registry("dataset")
generation_backends: Registry = Registry("generation")
labeling_backends: Registry = Registry("labeling")