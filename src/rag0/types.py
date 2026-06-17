"""Shared type definitions for RAG0.

These types are used across the entire codebase to avoid coupling
to any specific library's document representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoredDocument:
    """A document chunk with its retrieval score.

    Replaces the deprecated ``DocumentWithVSId`` from the old codebase.
    """

    content: str
    """The text content of the document chunk."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Metadata: filename, page number, chunk index, parent_id, etc."""

    score: float = 0.0
    """Relevance score from vector similarity, RRF fusion, or reranking."""

    doc_id: str = ""
    """Unique identifier for this chunk (typically a UUID)."""


@dataclass
class Message:
    """A chat message with role and content."""

    role: str
    """One of 'system', 'user', 'assistant'."""

    content: str
    """The message body."""


@dataclass
class IndexingResult:
    """Result of indexing a single file."""

    filename: str
    success: bool
    chunks_count: int = 0
    error: str | None = None
