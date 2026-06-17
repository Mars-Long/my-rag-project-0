"""Service registry — decorator-based plugin registration.

Replaces the ``if/elif/else`` factory functions and ``pass`` stubs
in the old codebase with a clean, discoverable registration system.

Usage::

    from rag0.connectors.registry import loader_registry

    @loader_registry.register(".pdf")
    class PDFLoader:
        ...

    # Later:
    loader_cls = loader_registry.get(".pdf")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T", bound=type)


class Registry:
    """A typed registry mapping string keys to classes or callables.

    Each registry instance is independent — loaders, splitters, and
    vector stores each have their own :class:`Registry`.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._items: dict[str, Any] = {}

    def register(self, key: str) -> Callable[[T], T]:
        """Decorator that registers a class under *key*."""

        def decorator(cls: T) -> T:
            self._items[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> Any | None:
        """Return the registered item for *key*, or ``None``."""
        return self._items.get(key)

    def get_required(self, key: str) -> Any:
        """Return the registered item for *key*, or raise ``KeyError``."""
        if key not in self._items:
            available = ", ".join(sorted(self._items.keys())) or "(none)"
            raise KeyError(
                f"No {self.name} registered for '{key}'. "
                f"Available: {available}"
            )
        return self._items[key]

    def list_keys(self) -> list[str]:
        """Return all registered keys."""
        return sorted(self._items.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._items

    def __repr__(self) -> str:
        return f"Registry({self.name!r}, keys={len(self._items)})"


# Global registries (module-level singletons are acceptable for registries
# because they carry no state — they are configuration, not resources).
loader_registry = Registry("loader")
"""Registry mapping file extensions (``".pdf"``, ``".docx"``) to loader classes."""

splitter_registry = Registry("splitter")
"""Registry mapping splitter names to splitter classes."""

vector_store_registry = Registry("vector_store")
"""Registry mapping vector store type names to ``VectorStoreInterface`` subclasses."""
