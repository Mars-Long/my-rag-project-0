"""Multi-vector indexing strategies.

Three strategies for enhanced retrieval:
1. **Small-to-big**: Split chunks into smaller sub-chunks for embedding,
   but return parent chunks at retrieval time.
2. **Text summaries**: Generate LLM summaries for each chunk.
3. **Table summaries**: Generate LLM summaries for table content.

All strategies are **async** (fix: old code used sync LLM calls).
"""

from __future__ import annotations

import uuid

from langchain_core.documents import Document

from rag0.connectors.llm import LLMConnector
from rag0.logging import get_logger
from rag0.types import ScoredDocument

logger = get_logger(__name__)

# Multi-vector type labels (stored in metadata)
TYPE_SMALL_TO_BIG = "text small-to-big"
TYPE_TEXT_SUMMARY = "text summary"
TYPE_TABLE_SUMMARY = "table summary"


# =============================================================================
# 1. Small-to-Big
# =============================================================================
def split_smaller_chunks(
    documents: list[Document],
    smaller_chunk_size: int = 200,
) -> list[ScoredDocument]:
    """Create smaller sub-chunks linked to parent documents.

    At retrieval time, the smaller chunk's embedding matches the query,
    but the **parent** chunk's full text is returned to the LLM.

    Args:
        documents: The parent (full-size) chunks.
        smaller_chunk_size: Target size for sub-chunks (0 overlap).

    Returns:
        List of :class:`ScoredDocument` — one per sub-chunk, each with
        ``parent_id`` metadata pointing to the parent document.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if smaller_chunk_size <= 0:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=smaller_chunk_size,
        chunk_overlap=0,
    )

    sub_docs: list[ScoredDocument] = []
    for parent in documents:
        parent_id = parent.metadata.get("doc_id", str(uuid.uuid4()))
        sub_texts = splitter.split_text(parent.page_content)
        for sub_text in sub_texts:
            sub_docs.append(
                ScoredDocument(
                    content=sub_text,
                    metadata={
                        **parent.metadata,
                        "parent_id": parent_id,
                        "multi_vector_type": TYPE_SMALL_TO_BIG,
                    },
                    doc_id=str(uuid.uuid4()),
                )
            )

    logger.debug(
        "Small-to-big chunks created",
        parents=len(documents),
        sub_chunks=len(sub_docs),
        sub_size=smaller_chunk_size,
    )
    return sub_docs


# =============================================================================
# 2. Text Summaries
# =============================================================================
async def generate_text_summaries(
    documents: list[Document],
    llm: LLMConnector,
) -> list[ScoredDocument]:
    """Generate LLM summaries for each document chunk.

    The summary is embedded separately; at retrieval time the parent
    document is returned when the summary matches.

    Args:
        documents: The full-text chunks.
        llm: LLM connector for summary generation.

    Returns:
        List of :class:`ScoredDocument` with summary content and
        ``parent_id`` linking to the original chunk.
    """
    from rag0.types import Message

    results: list[ScoredDocument] = []
    for doc in documents:
        parent_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
        try:
            summary = await llm.generate(
                [
                    Message(
                        role="user",
                        content=(
                            f"请为以下文档内容生成一个简洁的中文摘要（不超过100字）：\n\n"
                            f"{doc.page_content[:2000]}"
                        ),
                    )
                ]
            )
        except Exception:
            logger.warning("Summary generation failed, skipping chunk")
            continue

        if not summary:
            continue

        results.append(
            ScoredDocument(
                content=summary,
                metadata={
                    **doc.metadata,
                    "parent_id": parent_id,
                    "multi_vector_type": TYPE_TEXT_SUMMARY,
                },
                doc_id=str(uuid.uuid4()),
            )
        )

    logger.debug("Text summaries generated", count=len(results))
    return results


# =============================================================================
# 3. Table Summaries (NEW — was defined but never called in old codebase)
# =============================================================================
async def generate_table_summaries(
    documents: list[Document],
    llm: LLMConnector,
) -> list[ScoredDocument]:
    """Generate LLM summaries for table-like content.

    Detects markdown tables or pipe-separated data in chunks, then
    asks the LLM to summarize the table in natural language.

    Args:
        documents: Document chunks that may contain tables.
        llm: LLM connector.

    Returns:
        List of :class:`ScoredDocument` with table summaries.
    """
    from rag0.types import Message

    results: list[ScoredDocument] = []
    for doc in documents:
        if not _contains_table(doc.page_content):
            continue

        parent_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
        try:
            summary = await llm.generate(
                [
                    Message(
                        role="user",
                        content=(
                            f"请用自然语言简洁地总结以下表格的内容（不超过100字）：\n\n"
                            f"{doc.page_content[:2000]}"
                        ),
                    )
                ]
            )
        except Exception:
            logger.warning("Table summary generation failed, skipping chunk")
            continue

        if not summary:
            continue

        results.append(
            ScoredDocument(
                content=summary,
                metadata={
                    **doc.metadata,
                    "parent_id": parent_id,
                    "multi_vector_type": TYPE_TABLE_SUMMARY,
                },
                doc_id=str(uuid.uuid4()),
            )
        )

    logger.debug("Table summaries generated", count=len(results))
    return results


def _contains_table(text: str) -> bool:
    """Heuristic: does *text* contain a table (markdown or pipe-separated)?"""
    # Markdown table: contains |---|---| or at least 2 pipe-separated lines
    lines = text.strip().split("\n")
    pipe_lines = [ln for ln in lines if ln.strip().startswith("|")]
    return bool(len(pipe_lines) >= 2 and any("---" in ln for ln in pipe_lines))
