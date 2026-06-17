"""Domain exception hierarchy for the RAG0 application.

All exceptions inherit from :class:`Rag0Error` for consistent error handling.
Each exception carries a human-readable ``message``, an optional ``cause``
(the underlying exception), and an optional ``context`` dict with structured data.
"""

from __future__ import annotations


class Rag0Error(Exception):
    """Base exception for all RAG0 errors."""

    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
        context: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause
        self.context = context or {}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class ConfigurationError(Rag0Error):
    """Raised when the configuration is invalid or missing required values."""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
class ConnectionError(Rag0Error):
    """Base for external-service connection failures."""


class LLMConnectionError(ConnectionError):
    """Raised when the LLM service cannot be reached or returns an error."""


class VectorStoreConnectionError(ConnectionError):
    """Raised when the vector store cannot be reached."""


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
class DocumentError(Rag0Error):
    """Base for document processing failures."""


class DocumentLoadError(DocumentError):
    """Raised when a document cannot be loaded or parsed."""


class DocumentSplitError(DocumentError):
    """Raised when text splitting fails."""


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
class RetrievalError(Rag0Error):
    """Raised when retrieval fails (query expansion, routing, search)."""


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
class GenerationError(Rag0Error):
    """Raised when LLM generation fails."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class ValidationError(Rag0Error):
    """Raised when input validation fails (API request, file type, etc.)."""
