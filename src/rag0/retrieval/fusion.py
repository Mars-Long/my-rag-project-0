"""Result fusion — Reciprocal Rank Fusion (RRF) + hybrid search orchestration.

The RRF implementation from the old codebase is mathematically correct.
This version adds:
- Deduplication by ``doc_id`` (more robust than ``page_content``).
- Hybrid search orchestration (dense + sparse → RRF).
- Cleaner separation of concerns.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import List

from rag0.connectors.vector_store import BM25Retriever, VectorStoreInterface
from rag0.logging import get_logger
from rag0.types import ScoredDocument

logger = get_logger(__name__)


# =============================================================================
# Reciprocal Rank Fusion (RRF)
# =============================================================================
def reciprocal_rank_fusion(
    doc_lists: list[list[ScoredDocument]],
    weights: list[float] | None = None,
    k: int = 60,
) -> list[ScoredDocument]:
    """Fuse multiple ranked document lists using weighted RRF.

    Args:
        doc_lists: One or more ranked lists of documents.
        weights: Optional per-list weight. Defaults to 1.0 for each.
        k: RRF smoothing constant (default 60, standard value).

    Returns:
        Single merged list sorted by descending RRF score.
    """
    if weights is None:
        weights = [1.0] * len(doc_lists)

    scores: dict[str, tuple[float, ScoredDocument]] = {}

    for weight, doc_list in zip(weights, doc_lists):
        for rank, doc in enumerate(doc_list):
            key = doc.doc_id or doc.content[:80]  # fallback to content fingerprint
            rrf_score = weight / (k + rank + 1)

            if key in scores:
                current_score, _ = scores[key]
                scores[key] = (current_score + rrf_score, doc)
            else:
                scores[key] = (rrf_score, doc)

    # Sort by descending RRF score
    fused = sorted(scores.values(), key=lambda x: x[0], reverse=True)
    result: list[ScoredDocument] = []
    for score, doc in fused:
        doc.score = score
        result.append(doc)

    return result


# =============================================================================
# Hybrid Search (Dense + Sparse)
# =============================================================================
async def hybrid_search(
    query: str,
    vector_store: VectorStoreInterface,
    collection: str,
    query_embedding: list[float],
    bm25: BM25Retriever | None = None,
    top_k: int = 20,
    dense_weight: float = 0.7,
    sparse_weight: float = 0.3,
) -> list[ScoredDocument]:
    """Perform hybrid search: dense (vector) + sparse (BM25) → RRF fusion.

    Args:
        query: Raw query text (for BM25).
        vector_store: Dense vector store.
        collection: Collection name.
        query_embedding: Pre-computed query embedding.
        bm25: Optional BM25 retriever. If ``None``, only dense search is done.
        top_k: Final number of results after fusion.
        dense_weight: RRF weight for dense results.
        sparse_weight: RRF weight for sparse (BM25) results.

    Returns:
        Fused and re-ranked document list.
    """
    doc_lists: list[list[ScoredDocument]] = []
    weights: list[float] = []

    # 1. Dense search
    dense_results = vector_store.search(
        collection=collection,
        query_embedding=query_embedding,
        top_k=top_k,
    )
    doc_lists.append(dense_results)
    weights.append(dense_weight)
    logger.debug("Dense search", results=len(dense_results))

    # 2. Sparse search (BM25)
    if bm25 is not None:
        bm25_hits = bm25.search(query, top_k=top_k)
        # Convert BM25 hits (index, score) to ScoredDocument list
        # BM25Retriever doesn't store docs — the caller must provide the corpus
        # as ScoredDocuments. For now, BM25 returns indices; the caller is
        # responsible for mapping back. If no mapping is available, skip.
        if bm25_hits and bm25._corpus:
            bm25_docs: list[ScoredDocument] = []
            for idx, score in bm25_hits:
                if idx < len(bm25._corpus):
                    bm25_docs.append(
                        ScoredDocument(
                            content=bm25._corpus[idx],
                            score=float(score),
                            doc_id=f"bm25_{idx}",
                        )
                    )
            doc_lists.append(bm25_docs)
            weights.append(sparse_weight)
            logger.debug("BM25 search", results=len(bm25_docs))

    # 3. Fuse
    if len(doc_lists) == 1:
        return doc_lists[0][:top_k]

    fused = reciprocal_rank_fusion(doc_lists, weights=weights)
    return fused[:top_k]
