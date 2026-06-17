"""Unified exception handling middleware.

Maps domain exceptions to consistent HTTP responses.
Fix: No more bare ``except:`` — every exception gets a proper HTTP status.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag0.exceptions import (
    ConfigurationError,
    ConnectionError,
    DocumentError,
    GenerationError,
    Rag0Error,
    RetrievalError,
    ValidationError,
)


@dataclass
class RAG0Exception(Exception):
    """An exception that knows its HTTP status code."""

    status_code: int
    message: str
    detail: dict | None = None


# Map domain exceptions to HTTP status codes
_EXCEPTION_MAP = {
    ValidationError: 400,
    DocumentError: 400,
    RetrievalError: 500,
    GenerationError: 500,
    ConnectionError: 503,
    ConfigurationError: 500,
    Rag0Error: 500,
    ValueError: 400,
}


def map_exception_to_response(exc: Exception) -> RAG0Exception:
    """Convert a domain exception to an HTTP-aware exception.

    Args:
        exc: The exception raised during request processing.

    Returns:
        A :class:`RAG0Exception` with appropriate HTTP status code.
    """
    status_code = 500
    detail: dict | None = None

    for exc_type, code in _EXCEPTION_MAP.items():
        if isinstance(exc, exc_type):
            status_code = code
            break

    # Include structured context if available
    if isinstance(exc, Rag0Error) and exc.context:
        detail = exc.context

    return RAG0Exception(
        status_code=status_code,
        message=str(exc),
        detail=detail,
    )
