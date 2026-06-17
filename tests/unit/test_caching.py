"""Tests for the caching layer (src/rag0/caching.py)."""

from __future__ import annotations

from rag0.caching import CacheBackend, EmbeddingCache, LLMResponseCache


class TestCacheBackend:
    """Tests for the LRU in-memory cache backend."""

    def test_set_and_get(self) -> None:
        cache = CacheBackend(max_size=100)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self) -> None:
        cache = CacheBackend(max_size=100)
        assert cache.get("nonexistent") is None

    def test_lru_eviction(self) -> None:
        cache = CacheBackend(max_size=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict "a" (least recently used)
        assert cache.get("a") is None
        assert cache.get("b") == 2

    def test_get_reorders_access(self) -> None:
        cache = CacheBackend(max_size=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # Make "a" recently used
        cache.set("c", 3)  # Should evict "b" not "a"
        assert cache.get("a") == 1
        assert cache.get("b") is None

    def test_clear(self) -> None:
        cache = CacheBackend(max_size=100)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.clear()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_ttl_expiry(self) -> None:
        cache = CacheBackend(max_size=100, ttl=-1)  # -1 = immediately expire
        cache.set("a", 1, ttl=-1)
        assert cache.get("a") is None


class TestLLMResponseCache:
    """Tests for the LLM semantic cache."""

    def test_same_query_same_docs_hits(self) -> None:
        backend = CacheBackend(max_size=100)
        cache = LLMResponseCache(backend)
        cache.set("什么是Python？", ["doc1", "doc2"], "Python是编程语言")
        result = cache.get("什么是Python？", ["doc1", "doc2"])
        assert result == "Python是编程语言"

    def test_different_query_misses(self) -> None:
        backend = CacheBackend(max_size=100)
        cache = LLMResponseCache(backend)
        cache.set("什么是Python？", ["doc1"], "答案1")
        assert cache.get("什么是Java？", ["doc1"]) is None

    def test_different_docs_misses(self) -> None:
        backend = CacheBackend(max_size=100)
        cache = LLMResponseCache(backend)
        cache.set("什么是Python？", ["doc1"], "答案1")
        assert cache.get("什么是Python？", ["doc2"]) is None

    def test_doc_order_does_not_matter(self) -> None:
        backend = CacheBackend(max_size=100)
        cache = LLMResponseCache(backend)
        cache.set("什么是Python？", ["b", "a"], "答案")
        result = cache.get("什么是Python？", ["a", "b"])  # Different order
        assert result == "答案"


class TestEmbeddingCache:
    """Tests for the embedding cache."""

    def test_same_text_hits(self) -> None:
        backend = CacheBackend(max_size=100)
        cache = EmbeddingCache(backend)
        vec = [0.1, 0.2, 0.3]
        cache.set("你好世界", vec)
        assert cache.get("你好世界") == vec

    def test_different_text_misses(self) -> None:
        backend = CacheBackend(max_size=100)
        cache = EmbeddingCache(backend)
        cache.set("你好", [0.1, 0.2])
        assert cache.get("世界") is None
