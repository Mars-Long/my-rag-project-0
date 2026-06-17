"""Optional Langfuse telemetry for RAG pipeline observability.

Key fix over the old ``server/trace.py``:
- Langfuse is initialized lazily (not at module import time).
- Controlled by ``TelemetryConfig.enabled`` — off by default.
- Does not set global env vars as a side effect.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterator

from rag0.config import TelemetryConfig
from rag0.logging import get_logger

logger = get_logger(__name__)


class Telemetry:
    """Optional Langfuse observability wrapper.

    Args:
        config: Telemetry configuration.
    """

    def __init__(self, config: TelemetryConfig) -> None:
        self._config = config
        self._langfuse = None
        self._initialized = False

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def _ensure_initialized(self) -> None:
        if self._initialized or not self._config.enabled:
            return

        try:
            import langfuse  # type: ignore[import-untyped]

            self._langfuse = langfuse.Langfuse(
                public_key=self._config.public_key,
                secret_key=self._config.secret_key,
                host=self._config.host,
            )
            self._initialized = True
            logger.info("Langfuse telemetry initialized")
        except ImportError:
            logger.warning("langfuse not installed; telemetry disabled")
            self._config.enabled = False
        except Exception as exc:
            logger.warning("Langfuse init failed; telemetry disabled", error=str(exc))
            self._config.enabled = False

    @contextmanager
    def trace(self, name: str, **metadata: Any) -> Iterator[Any]:
        """Context manager that creates a Langfuse trace span.

        Usage::

            with telemetry.trace("retrieval", query=query, kb=kb_name) as span:
                docs = await chain.retrieve(...)
                span.update(output={"doc_count": len(docs)})
        """
        self._ensure_initialized()

        if not self._langfuse:
            yield _NoopSpan()
            return

        try:
            trace = self._langfuse.trace(name=name, metadata=metadata)
            yield trace
        except Exception as exc:
            logger.debug("Telemetry trace error", error=str(exc))
            yield _NoopSpan()


class _NoopSpan:
    """No-op trace span when telemetry is disabled."""

    def update(self, **kwargs: Any) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
