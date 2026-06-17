"""Shared test fixtures and configuration for RAG0."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_config_dict() -> dict:
    """Return a minimal valid configuration dict for testing."""
    return {
        "llm": {"model_name": "test-model", "temperature": 0.0},
        "embedding": {"model_name": "test-embeddings", "dimensions": 768},
        "vector_store": {"host": "127.0.0.1", "port": 19530},
        "database": {"url": "sqlite:///test.db"},
        "splitter": {"chunk_size": 500, "chunk_overlap": 50},
        "reranker": {"model_name": "test-reranker"},
        "server": {"host": "127.0.0.1", "port": 7861},
        "telemetry": {"enabled": False},
    }
