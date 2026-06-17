"""Caching layer — reduce redundant LLM and Embedding calls.

Uses ``diskcache`` as the default backend (zero extra dependencies).
Optional Redis backend for distributed deployments.

Cache strategies:
- **Embedding cache**: Same text → cached vector (LRU, in-memory).
- **LLM semantic cache**: Same query + same doc IDs → cached answer.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any

from rag0.logging import get_logger

logger = get_logger(__name__)


class CacheBackend:
    """Simple in-memory + optional disk-backed cache.

    Args:
        max_size: Maximum number of entries in memory.
        disk_path: Optional path for disk persistence via diskcache.
        ttl: Default TTL in seconds (0 = no expiration).
    """

    def __init__(
        self,
        max_size: int = 1024,
        disk_path: str | None = None,
        ttl: int = 3600,
    ) -> None:
        self._lock = threading.Lock()
        self._ttl = ttl
        self._disk = None

        # In-memory LRU (simple dict + access-order tracking, Python 3.7+)
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expiry)
        self._max_size = max_size
        self._access_order: list[str] = []

        # Optional disk persistence
        if disk_path:
            try:
                from diskcache import Cache

                self._disk = Cache(disk_path)
                logger.info("Disk cache enabled", path=disk_path)
            except ImportError:
                logger.warning("diskcache not installed; using in-memory only")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, key: str) -> Any | None:
        """Retrieve a cached value, or ``None`` if missing/expired."""
        import time

        # Check in-memory first
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                value, expiry = entry
                if expiry == 0 or time.time() < expiry:
                    # Move to end (most recently used)
                    if key in self._access_order:
                        self._access_order.remove(key)
                    self._access_order.append(key)
                    return value
                # Expired — remove
                del self._store[key]
                if key in self._access_order:
                    self._access_order.remove(key)

        # Fall back to disk
        if self._disk is not None:
            disk_val = self._disk.get(key)
            if disk_val is not None:
                # Promote to memory
                expiry = time.time() + self._ttl if self._ttl > 0 else 0
                with self._lock:
                    self._set_in_memory(key, disk_val, expiry)
                return disk_val

        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in the cache."""
        import time

        effective_ttl = ttl if ttl is not None else self._ttl
        # ttl <= 0 means "never expire"; ttl < 0 means "immediately expire"
        if effective_ttl > 0:
            expiry = time.time() + effective_ttl
        elif effective_ttl < 0:
            expiry = time.time() - 1  # Already expired
        else:
            expiry = 0  # Never expire

        with self._lock:
            self._set_in_memory(key, value, expiry)

        if self._disk is not None:
            self._disk.set(key, value, expire=effective_ttl if effective_ttl > 0 else None)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._store.clear()
            self._access_order.clear()
        if self._disk is not None:
            self._disk.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _set_in_memory(self, key: str, value: Any, expiry: float) -> None:
        """Set a value in the in-memory store, evicting if needed."""
        # Evict least recently used if at capacity
        while len(self._store) >= self._max_size and self._access_order:
            oldest = self._access_order.pop(0)
            if oldest in self._store:
                del self._store[oldest]

        self._store[key] = (value, expiry)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)


# =============================================================================
# Application-level caches
# =============================================================================
class LLMResponseCache:
    """Semantic cache for LLM responses.

    Caches by (query_hash, sorted_doc_ids_hash). When the same query is
    asked with the same retrieved documents, returns the cached answer.
    """

    def __init__(self, backend: CacheBackend) -> None:
        self._backend = backend

    @staticmethod
    def _make_key(query: str, doc_ids: list[str]) -> str:
        """Build a deterministic cache key from query + doc IDs."""
        sorted_ids = sorted(doc_ids)
        fingerprint = f"{query}|{','.join(sorted_ids)}"
        return f"llm:{hashlib.sha256(fingerprint.encode()).hexdigest()[:32]}"

    def get(self, query: str, doc_ids: list[str]) -> str | None:
        return self._backend.get(self._make_key(query, doc_ids))

    def set(self, query: str, doc_ids: list[str], answer: str) -> None:
        self._backend.set(self._make_key(query, doc_ids), answer)


class EmbeddingCache:
    """Cache for text embeddings — same text always produces the same vector."""

    def __init__(self, backend: CacheBackend) -> None:
        self._backend = backend

    @staticmethod
    def _make_key(text: str) -> str:
        fingerprint = hashlib.sha256(text.encode()).hexdigest()[:32]
        return f"emb:{fingerprint}"

    def get(self, text: str) -> list[float] | None:
        return self._backend.get(self._make_key(text))

    def set(self, text: str, embedding: list[float]) -> None:
        self._backend.set(self._make_key(text), embedding)


# =============================================================================
# Module-level singleton (lightweight — no heavy deps)
# =============================================================================
_cache_backend: CacheBackend | None = None
_llm_cache: LLMResponseCache | None = None
_embedding_cache: EmbeddingCache | None = None


def get_cache_backend(disk_path: str | None = "data/cache") -> CacheBackend:
    """Return (or create) the global cache backend singleton."""
    global _cache_backend
    if _cache_backend is None:
        _cache_backend = CacheBackend(max_size=2048, disk_path=disk_path, ttl=3600)
    return _cache_backend


def get_llm_cache() -> LLMResponseCache:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMResponseCache(get_cache_backend())
    return _llm_cache


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(get_cache_backend())
    return _embedding_cache
