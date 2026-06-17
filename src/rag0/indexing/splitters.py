"""Text splitters with Chinese-language awareness.

Key improvements over the old splitters:
- **Critical fix**: ``ChineseRecursiveTextSplitter.split_documents`` now processes
  **all** documents, not just ``documents[:1]``.
- Docstrings on every method.
- Registered via ``@splitter_registry.register()`` decorator.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag0.connectors.registry import splitter_registry
from rag0.exceptions import DocumentSplitError
from rag0.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Chinese sentence boundary regex
# ---------------------------------------------------------------------------
_SENTENCE_END = re.compile(r"(?<=[。！？?!])\s*")


# =============================================================================
# Chinese Recursive Text Splitter
# =============================================================================
@splitter_registry.register("ChineseRecursiveTextSplitter")
class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    """Recursive text splitter with Chinese-aware separators.

    Separator hierarchy (tried in order)::

        double-newline → single-newline → Chinese sentence end →
        English sentence end → Chinese semicolon → Chinese comma → space

    **Fix**: The old code's ``split_documents`` only processed ``documents[:1]``.
    This implementation correctly iterates over **all** documents.
    """

    CHINESE_SEPARATORS = [
        "\n\n",
        "\n",
        "。|！|？",
        r"\.\s|\!\s|\?\s",
        r"；|;\s",
        r"，|,\s",
        " ",
        "",   # character-level fallback — handles long strings with no punctuation
    ]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, **kwargs: Any) -> None:
        super().__init__(
            separators=self.CHINESE_SEPARATORS,
            keep_separator=True,
            is_separator_regex=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs,
        )

    def split_documents(self, documents: list[Any]) -> list[Any]:
        """Split all documents — **fixed** from the old ``documents[:1]`` bug.

        Args:
            documents: List of LangChain ``Document`` objects.

        Returns:
            List of split ``Document`` objects with updated metadata.
        """
        if not documents:
            return []

        all_splits: list[Any] = []
        for doc in documents:
            try:
                chunks = self.split_text(doc.page_content)
            except Exception as exc:
                raise DocumentSplitError(
                    f"Failed to split document: {doc.metadata.get('source', 'unknown')}",
                    cause=exc,
                ) from exc

            for i, chunk in enumerate(chunks):
                meta = dict(doc.metadata)
                meta["chunk_index"] = i
                meta["chunk_count"] = len(chunks)
                all_splits.append(
                    type(doc)(page_content=chunk, metadata=meta)
                )

        logger.debug(
            "Documents split",
            input_count=len(documents),
            output_count=len(all_splits),
        )
        return all_splits


# =============================================================================
# Chinese Text Splitter (simple, sentence-boundary based)
# =============================================================================
@splitter_registry.register("ChineseTextSplitter")
class ChineseTextSplitter(ChineseRecursiveTextSplitter):
    """A simpler Chinese splitter that splits at sentence boundaries.

    Uses lookbehind patterns to find Chinese/English sentence endings
    and splits accordingly. Falls back to recursive character splitting
    for very long sentences.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, **kwargs: Any) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    def split_text(self, text: str) -> list[str]:
        """Split *text* at sentence boundaries, then enforce chunk_size."""
        if not text.strip():
            return []

        # Split by sentence boundaries
        sentences = _SENTENCE_END.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Merge sentences into chunks respecting chunk_size
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sent in sentences:
            if current_len + len(sent) > self._chunk_size and current:
                chunks.append("".join(current))
                # Keep overlap: retain last sentence
                overlap_start = max(0, len(current) - 1)
                current = current[overlap_start:]
                current_len = sum(len(s) for s in current)
            current.append(sent)
            current_len += len(sent)

        if current:
            chunks.append("".join(current))

        # For any chunk still too large, fall back to recursive splitting
        final: list[str] = []
        for chunk in chunks:
            if len(chunk) <= self._chunk_size:
                final.append(chunk)
            else:
                final.extend(super().split_text(chunk))

        return final
