"""FastAPI dependency injection helpers.

Replaces the old ``KBServiceFactory`` global singleton pattern.
"""

from __future__ import annotations

from contextvars import ContextVar

from rag0.container import Container

# Thread-safe context variable for the container
_container_ctx: ContextVar[Container | None] = ContextVar("container", default=None)


def set_container(container: Container) -> None:
    """Store the container for the current async context."""
    _container_ctx.set(container)


def get_container() -> Container:
    """Retrieve the DI container (FastAPI dependency)."""
    container = _container_ctx.get()
    if container is None:
        raise RuntimeError(
            "Container not initialized. Call set_container() or "
            "pass a container to create_app()."
        )
    return container
