"""RetrievalChain — orchestrates the full retrieval pipeline.

Key improvements over the old ``rag/chains/retrieval.py``:
- Hybrid search (dense + sparse) via RRF — **new feature**.
- Metadata filtering support — **new feature**.
- Proper ``RetrievalError`` propagation (no silent ``return []``).
- No ``print(docs)`` debugging statements.
- Optional HyDE expansion.
"""

from __future__ import annotations

from typing import Any

from rag0.chains.indexing import ScoredDocument
from rag0.connectors.vector_store import BM25Retriever
from rag0.container import Container
from rag0.logging import get_logger
from rag0.retrieval.fusion import hybrid_search, reciprocal_rank_fusion
from rag0.retrieval.query_expansion import generate_hyde_document, generate_multi_queries
from rag0.retrieval.reranker import CrossEncoderReranker, RerankerInterface
from rag0.retrieval.routing import route_query_to_file

logger = get_logger(__name__)


class RetrievalChain:
    """Orchestrate retrieval: expand → route → search → fuse → rerank.

    Args:
        container: The DI container.
    """

    def __init__(self, container: Container) -> None:
        self._container = container
        self._config = container.config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def retrieve(
        self,
        query: str,
        knowledge_base_name: str,
        *,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        enable_multi_query: bool = True,
        enable_hyde: bool = False,
        enable_route: bool = False,
        bm25: BM25Retriever | None = None,
    ) -> list[ScoredDocument]:
        """Retrieve relevant documents for *query*.

        Args:
            query: The user's question.
            knowledge_base_name: Target knowledge base (Milvus collection).
            top_k: Final number of documents to return.
            filters: Optional metadata filters (e.g., ``{"file_name": "report.pdf"}``).
            enable_multi_query: Generate variant queries for broader recall.
            enable_hyde: Use HyDE (hypothetical document embedding).
            enable_route: Route the query to a specific file.
            bm25: Optional BM25 retriever for hybrid search.

        Returns:
            Ranked list of :class:`ScoredDocument`.
        """
        emb = self._container.embedding
        vs = self._container.vector_store

        # ── 1. Pre-retrieval: query expansion ──
        queries = [query]

        if enable_multi_query:
            expanded = await generate_multi_queries(
                query, self._container.llm, num_queries=3
            )
            queries = expanded

        if enable_hyde:
            hyde_doc = await generate_hyde_document(query, self._container.llm)
            if hyde_doc:
                # Use the HyDE document as an additional query
                queries.append(hyde_doc)

        # ── 2. Pre-retrieval: route to file ──
        route_filter: dict[str, Any] | None = dict(filters) if filters else None
        if enable_route:
            from rag0.connectors.database import KnowledgeFileRepository

            session = self._container.new_session()
            try:
                file_repo = KnowledgeFileRepository(session)
                files = file_repo.list_files(knowledge_base_name)
                file_names = [f.file_name for f in files]
                matched = await route_query_to_file(query, file_names, self._container.llm)
                if matched:
                    route_filter = route_filter or {}
                    route_filter["file_name"] = matched
                    logger.debug("Query routed", file=matched)
            finally:
                session.close()

        # ── 3. Retrieval: search per query ──
        all_results: list[list[ScoredDocument]] = []

        for q in queries:
            q_embedding = emb.embed_query(q)

            if bm25 is not None:
                # Hybrid search
                results = await hybrid_search(
                    query=q,
                    vector_store=vs,
                    collection=knowledge_base_name,
                    query_embedding=q_embedding,
                    bm25=bm25,
                    top_k=top_k * 2,  # Oversample before rerank
                )
            else:
                # Dense-only search
                results = vs.search(
                    collection=knowledge_base_name,
                    query_embedding=q_embedding,
                    top_k=top_k * 2,
                    filters=route_filter,
                )
            all_results.append(results)

        # ── 4. Post-retrieval: fuse multi-query results ──
        docs = all_results[0] if len(all_results) == 1 else reciprocal_rank_fusion(all_results)

        # Deduplicate by doc_id
        seen: set[str] = set()
        deduped: list[ScoredDocument] = []
        for doc in docs:
            key = doc.doc_id or doc.content[:80]
            if key not in seen:
                seen.add(key)
                deduped.append(doc)

        # ── 5. Post-retrieval: rerank ──
        reranker = _create_reranker(self._config.reranker, self._container.llm)
        ranked = await _rerank(reranker, query, deduped, top_k)

        logger.debug(
            "Retrieval complete",
            query=query[:50],
            results=len(ranked),
            multi_query=enable_multi_query,
            hyde=enable_hyde,
        )

        return ranked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _create_reranker(config, llm) -> RerankerInterface | None:
    """Create a reranker instance from config."""
    if config.type == "cross-encoder":
        try:
            return CrossEncoderReranker(config)
        except Exception:
            logger.warning("Cross-encoder reranker unavailable, skipping rerank")
            return None
    elif config.type == "llm":
        from rag0.retrieval.reranker import LLMReranker

        return LLMReranker(config, llm)
    return None


async def _rerank(
    reranker: RerankerInterface | None,
    query: str,
    documents: list[ScoredDocument],
    top_k: int,
) -> list[ScoredDocument]:
    """Apply reranker if available, otherwise return documents as-is."""
    if reranker is None or not documents:
        return documents[:top_k]

    try:
        if hasattr(reranker, "rank"):
            result = reranker.rank(query, documents, top_k)
            # Check if it's a coroutine
            import asyncio

            if asyncio.iscoroutine(result):
                result = await result
            return result
    except Exception as exc:
        logger.warning("Reranking failed, returning unranked results", error=str(exc))

    return documents[:top_k]
