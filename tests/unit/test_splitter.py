"""Tests for Chinese text splitters (src/rag0/indexing/splitters.py)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag0.indexing.splitters import ChineseRecursiveTextSplitter, ChineseTextSplitter


class TestChineseRecursiveTextSplitter:
    """Tests for ChineseRecursiveTextSplitter — the primary chunker."""

    def test_empty_input(self) -> None:
        splitter = ChineseRecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
        result = splitter.split_documents([])
        assert result == []

    def test_single_document(self) -> None:
        splitter = ChineseRecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
        doc = Document(
            page_content="这是一个测试文档。它包含多个句子。用于验证分割功能。",
            metadata={"source": "test.txt"},
        )
        result = splitter.split_documents([doc])
        assert len(result) >= 1
        for chunk in result:
            assert chunk.page_content.strip()
            assert chunk.metadata["source"] == "test.txt"
            assert "chunk_index" in chunk.metadata

    def test_multiple_documents(self) -> None:
        """Critical fix verification: ALL documents are split, not just doc[0]."""
        splitter = ChineseRecursiveTextSplitter(chunk_size=300, chunk_overlap=30)
        docs = [
            Document(
                page_content="文档一的内容。" * 50,
                metadata={"source": "doc1.txt"},
            ),
            Document(
                page_content="文档二的内容。" * 50,
                metadata={"source": "doc2.txt"},
            ),
            Document(
                page_content="文档三的内容。" * 50,
                metadata={"source": "doc3.txt"},
            ),
        ]
        result = splitter.split_documents(docs)

        # Every document should produce chunks
        sources = {chunk.metadata.get("source") for chunk in result}
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources
        assert "doc3.txt" in sources

        # Each source should have >1 chunk (since content is long)
        for source in sources:
            source_chunks = [
                c for c in result if c.metadata.get("source") == source
            ]
            assert len(source_chunks) > 1, (
                f"Source '{source}' has only {len(source_chunks)} chunk(s) "
                f"— the documents[:1] bug may still be present!"
            )

    def test_short_document_not_split(self) -> None:
        splitter = ChineseRecursiveTextSplitter(chunk_size=1000, chunk_overlap=50)
        doc = Document(
            page_content="短文本。",
            metadata={"source": "short.txt"},
        )
        result = splitter.split_documents([doc])
        assert len(result) == 1
        assert result[0].page_content == "短文本。"

    def test_chunk_size_respected(self) -> None:
        splitter = ChineseRecursiveTextSplitter(chunk_size=100, chunk_overlap=0)
        doc = Document(
            page_content="这是一个很长的句子。" * 50,
            metadata={"source": "long.txt"},
        )
        result = splitter.split_documents([doc])
        for chunk in result:
            assert len(chunk.page_content) <= 150, (
                f"Chunk size {len(chunk.page_content)} exceeds limit"
            )

    def test_metadata_preserved(self) -> None:
        splitter = ChineseRecursiveTextSplitter(chunk_size=200, chunk_overlap=20)
        doc = Document(
            page_content="元数据测试。" * 30,
            metadata={"source": "meta.txt", "page": 3, "author": "test"},
        )
        result = splitter.split_documents([doc])
        for chunk in result:
            assert chunk.metadata["source"] == "meta.txt"
            assert chunk.metadata["page"] == 3
            assert chunk.metadata["author"] == "test"
            assert "chunk_index" in chunk.metadata
            assert "chunk_count" in chunk.metadata

    def test_mixed_chinese_english(self) -> None:
        splitter = ChineseRecursiveTextSplitter(chunk_size=300, chunk_overlap=30)
        doc = Document(
            page_content="这是中文内容。This is English content. "
            "混合在一起进行测试。More English here for good measure.",
            metadata={"source": "mixed.txt"},
        )
        result = splitter.split_documents([doc])
        assert len(result) >= 1
        combined = "".join(c.page_content for c in result)
        # Original content should be roughly preserved
        assert "中文" in combined
        assert "English" in combined


class TestChineseTextSplitter:
    """Tests for ChineseTextSplitter — the simpler sentence-boundary splitter."""

    def test_splits_on_sentence_boundary(self) -> None:
        splitter = ChineseTextSplitter(chunk_size=500, chunk_overlap=0)
        doc = Document(
            page_content="第一句话。第二句话！第三句话？第四句话。",
            metadata={"source": "sentences.txt"},
        )
        result = splitter.split_documents([doc])
        assert len(result) >= 1

    def test_handles_very_long_sentence(self) -> None:
        splitter = ChineseTextSplitter(chunk_size=100, chunk_overlap=0)
        # A very long single sentence (no punctuation)
        doc = Document(
            page_content="A" * 500,
            metadata={"source": "long.txt"},
        )
        result = splitter.split_documents([doc])
        assert len(result) > 1  # Must be split
        for chunk in result:
            assert len(chunk.page_content) <= 150
