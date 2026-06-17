"""Vector store connector — Milvus integration + BM25 sparse retrieval.

Key improvements over the old ``MilvusVectorStore``:
- Direct pymilvus usage (no LangChain wrapper).
- **Critical fix**: No ``@lru_cache`` on the factory — each KB gets its own instance.
- Proper collection lifecycle (create on demand, existence checks).
- Metadata filtering via parameterized expressions.
- BM25 sparse retriever for hybrid search (new feature).
- All metadata values preserved with correct types (not force-cast to ``str()``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusClient,
    connections,
    utility,
)

from rag0.config import VectorStoreConfig
from rag0.exceptions import VectorStoreConnectionError
from rag0.logging import get_logger
from rag0.types import ScoredDocument

logger = get_logger(__name__)


# =============================================================================
# Abstract interface
# =============================================================================
class VectorStoreInterface(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection (schema + index)."""

    @abstractmethod
    def add_documents(
        self,
        collection: str,
        docs: list[ScoredDocument],
        embeddings: list[list[float]],
    ) -> list[str]:
        """Insert documents with their embeddings. Returns insert IDs."""

    @abstractmethod
    def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        """Search for similar documents."""

    @abstractmethod
    def delete_by_filter(self, collection: str, filter_expr: str) -> int:
        """Delete documents matching *filter_expr*. Returns count deleted."""

    @abstractmethod
    def drop_collection(self, name: str) -> None:
        """Delete a collection and all its data."""

    @abstractmethod
    def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""


# =============================================================================
# Milvus Implementation
# =============================================================================
class MilvusVectorStore(VectorStoreInterface):
    """Milvus-backed vector store using pymilvus directly.

    Args:
        config: Vector store configuration.
    """

    # Milvus field names
    _ID_FIELD = "id"
    _VECTOR_FIELD = "vector"
    _CONTENT_FIELD = "content"
    _METADATA_FIELD = "metadata"

    def __init__(self, config: VectorStoreConfig) -> None:
        self._config = config
        self._client: MilvusClient | None = None
        self._alias = "rag0"
        self._connect()

    # ------------------------------------------------------------------
    # VectorStoreInterface
    # ------------------------------------------------------------------
    def create_collection(self, name: str, dimension: int) -> None:
        if self.collection_exists(name):
            logger.info("Collection already exists, skipping create", collection=name)
            return

        fields = [
            FieldSchema(name=self._ID_FIELD, dtype=DataType.VARCHAR, max_length=64, is_primary=True),
            FieldSchema(name=self._VECTOR_FIELD, dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name=self._CONTENT_FIELD, dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name=self._METADATA_FIELD, dtype=DataType.JSON),
        ]
        schema = CollectionSchema(fields, description=f"RAG0 collection: {name}")

        logger.info("Creating Milvus collection", collection=name, dimension=dimension)
        Collection(name, schema, using=self._alias)

        # Build index
        index_params = {
            "metric_type": self._config.metric_type,
            "index_type": self._config.index_type,
            "params": {"M": 8, "efConstruction": 64},
        }
        col = Collection(name, using=self._alias)
        col.create_index(self._VECTOR_FIELD, index_params)
        col.load()

    def add_documents(
        self,
        collection: str,
        docs: list[ScoredDocument],
        embeddings: list[list[float]],
    ) -> list[str]:
        if not docs:
            return []

        self._ensure_collection(collection)

        rows: list[dict[str, Any]] = []
        for doc, vec in zip(docs, embeddings, strict=False):
            rows.append({
                self._ID_FIELD: doc.doc_id,
                self._VECTOR_FIELD: vec,
                self._CONTENT_FIELD: doc.content[:65535],
                self._METADATA_FIELD: doc.metadata,
            })

        logger.debug("Inserting documents", collection=collection, count=len(rows))
        result = self.client.insert(collection_name=collection, data=rows)
        return result["primary_keys"] if isinstance(result, dict) else result.primary_keys  # type: ignore[no-any-return]

    def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredDocument]:
        self._ensure_collection(collection)

        filter_expr = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=collection,
            data=[query_embedding],
            anns_field=self._VECTOR_FIELD,
            limit=top_k,
            output_fields=[self._CONTENT_FIELD, self._METADATA_FIELD],
            filter=filter_expr,
        )

        if not results or not results[0]:
            return []

        docs: list[ScoredDocument] = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            docs.append(
                ScoredDocument(
                    doc_id=hit.get("id", ""),
                    content=entity.get(self._CONTENT_FIELD, ""),
                    metadata=entity.get(self._METADATA_FIELD, {}),
                    score=hit.get("distance", 0.0),
                )
            )
        return docs

    def delete_by_filter(self, collection: str, filter_expr: str) -> int:
        if not self.collection_exists(collection):
            return 0
        result = self.client.delete(collection_name=collection, filter=filter_expr)
        count = result.get("delete_count", 0) if isinstance(result, dict) else 0
        logger.debug("Deleted documents", collection=collection, count=count)
        return count  # type: ignore[no-any-return]

    def drop_collection(self, name: str) -> None:
        if self.collection_exists(name):
            utility.drop_collection(name, using=self._alias)
            logger.info("Dropped collection", collection=name)

    def collection_exists(self, name: str) -> bool:
        return bool(utility.has_collection(name, using=self._alias))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            self._connect()
        assert self._client is not None
        return self._client

    def _connect(self) -> None:
        try:
            connections.connect(
                alias=self._alias,
                host=self._config.host,
                port=self._config.port,
                user=self._config.user,
                password=self._config.password,
                secure=self._config.secure,
            )
            self._client = MilvusClient(
                uri=f"http{'s' if self._config.secure else ''}://{self._config.host}:{self._config.port}",
                user=self._config.user,
                password=self._config.password,
            )
            logger.info(
                "Connected to Milvus",
                host=self._config.host,
                port=self._config.port,
            )
        except Exception as exc:
            raise VectorStoreConnectionError(
                f"Failed to connect to Milvus at {self._config.host}:{self._config.port}",
                cause=exc,
            ) from exc

    def _ensure_collection(self, name: str) -> None:
        if not self.collection_exists(name):
            raise VectorStoreConnectionError(
                f"Collection '{name}' does not exist. Create it first with create_collection()."
            )
        col = Collection(name, using=self._alias)
        col.load()

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> str:
        """Build a Milvus scalar filter expression from a dict.

        Simple equality-only for now. Extend as needed.
        """
        parts = []
        for key, value in filters.items():
            if isinstance(value, str):
                parts.append(f'{key} == "{value}"')
            elif isinstance(value, bool):
                parts.append(f"{key} == {str(value).lower()}")
            else:
                parts.append(f"{key} == {value}")
        return " && ".join(parts)


# =============================================================================
# BM25 Sparse Retriever (for hybrid search)
# =============================================================================
class BM25Retriever:
    """BM25-based sparse retrieval for hybrid (dense + sparse) search.

    Indexes a document corpus and returns scored results keyed by
    integer indices (matching the order of documents passed to ``index()``).
    """

    def __init__(self) -> None:
        self._bm25: Any = None
        self._corpus: list[str] = []

    def index(self, corpus: list[str]) -> None:
        """Build the BM25 index from *corpus*."""
        from rank_bm25 import BM25Okapi

        self._corpus = corpus
        tokenized = [self._tokenize(doc) for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.debug("BM25 indexed", doc_count=len(corpus))

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """Search the indexed corpus.

        Returns:
            List of ``(corpus_index, score)`` sorted by descending score.
        """
        if self._bm25 is None:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Sort by score descending, return top_k
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer for Chinese/English text."""
        import re

        # Split on whitespace and common punctuation, keep CJK chars
        tokens = re.findall(r"[一-鿿]+|[a-zA-Z0-9]+", text.lower())
        return tokens
