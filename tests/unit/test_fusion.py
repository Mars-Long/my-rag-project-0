"""Tests for RRF fusion algorithm (src/rag0/retrieval/fusion.py)."""

from __future__ import annotations

import pytest

from rag0.retrieval.fusion import reciprocal_rank_fusion
from rag0.types import ScoredDocument


def _make_docs(ids_and_scores: list[tuple[str, float]]) -> list[ScoredDocument]:
    """Helper to create scored docs from (doc_id, score) tuples."""
    return [
        ScoredDocument(content=f"doc_{doc_id}", score=score, doc_id=doc_id)
        for doc_id, score in ids_and_scores
    ]


class TestReciprocalRankFusion:
    """Comprehensive tests for reciprocal_rank_fusion()."""

    def test_single_list_returns_same_order(self) -> None:
        docs = _make_docs([("a", 0.9), ("b", 0.8), ("c", 0.7)])
        result = reciprocal_rank_fusion([docs])
        assert [d.doc_id for d in result] == ["a", "b", "c"]

    def test_two_lists_with_overlap(self) -> None:
        list1 = _make_docs([("a", 0.9), ("b", 0.8)])
        list2 = _make_docs([("b", 0.9), ("c", 0.7)])
        result = reciprocal_rank_fusion([list1, list2])
        ids = [d.doc_id for d in result]
        # "b" appears in both lists at high ranks → boosted
        assert len(ids) == 3
        assert ids[0] == "b"  # Appears in both, should rank highest

    def test_identical_lists(self) -> None:
        docs = _make_docs([("a", 0.9), ("b", 0.8)])
        result = reciprocal_rank_fusion([docs, docs])
        ids = [d.doc_id for d in result]
        assert ids == ["a", "b"]
        assert len(ids) == 2

    def test_empty_input(self) -> None:
        result = reciprocal_rank_fusion([])
        assert result == []

    def test_one_empty_one_nonempty(self) -> None:
        list1 = _make_docs([("a", 0.9), ("b", 0.8)])
        result = reciprocal_rank_fusion([list1, []])
        assert [d.doc_id for d in result] == ["a", "b"]

    def test_weighted_fusion(self) -> None:
        list1 = _make_docs([("a", 0.9), ("b", 0.8)])
        list2 = _make_docs([("b", 0.9), ("a", 0.7)])
        # Give list1 10x weight — "a" should win
        result = reciprocal_rank_fusion([list1, list2], weights=[10.0, 1.0])
        assert result[0].doc_id == "a"

    def test_three_lists_different_sizes(self) -> None:
        list1 = _make_docs([("a", 0.9)])
        list2 = _make_docs([("a", 0.9), ("b", 0.8), ("c", 0.7)])
        list3 = _make_docs([("d", 0.9), ("a", 0.6)])
        result = reciprocal_rank_fusion([list1, list2, list3])
        ids = [d.doc_id for d in result]
        assert "a" in ids
        assert "a" == ids[0]  # Appears in all 3 lists

    def test_deduplication_by_doc_id(self) -> None:
        """Documents with the same doc_id are deduplicated."""
        docs_a = [ScoredDocument(content="content_v1", score=0.9, doc_id="same_id")]
        docs_b = [ScoredDocument(content="content_v2", score=0.8, doc_id="same_id")]
        result = reciprocal_rank_fusion([docs_a, docs_b])
        assert len(result) == 1
        assert result[0].doc_id == "same_id"

    def test_default_weights(self) -> None:
        list1 = _make_docs([("a", 0.9)])
        list2 = _make_docs([("b", 0.9)])
        # Default weights should be [1.0, 1.0]
        result = reciprocal_rank_fusion([list1, list2])
        ids = [d.doc_id for d in result]
        assert len(ids) == 2
        assert result[0].score == result[1].score  # Equal weight at rank 0
