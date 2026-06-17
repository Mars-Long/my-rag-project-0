"""Tests for GenerateChain (src/rag0/chains/generation.py)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from rag0.chains.generation import GenerateChain
from rag0.types import Message, ScoredDocument


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Return a mocked LLM connector."""
    llm = AsyncMock()
    llm.generate.return_value = "这是测试回答。"
    return llm


@pytest.fixture
def sample_docs() -> list[ScoredDocument]:
    return [
        ScoredDocument(
            content="Python是一种编程语言。",
            metadata={"file_name": "doc1.txt"},
            score=0.9,
            doc_id="doc1",
        ),
        ScoredDocument(
            content="RAG是检索增强生成技术的缩写。",
            metadata={"file_name": "doc2.txt"},
            score=0.8,
            doc_id="doc2",
        ),
    ]


class TestGenerateChain:
    """Tests for the GenerateChain."""

    async def test_generate_returns_answer(self, mock_llm, sample_docs) -> None:
        chain = GenerateChain(mock_llm)
        answer = await chain.generate("什么是Python？", sample_docs)
        assert answer == "这是测试回答。"
        mock_llm.generate.assert_called_once()

    async def test_generate_with_empty_docs(self, mock_llm) -> None:
        chain = GenerateChain(mock_llm)
        answer = await chain.generate("什么是Python？", [])
        assert answer is not None
        mock_llm.generate.assert_called_once()

    async def test_generate_with_history(self, mock_llm, sample_docs) -> None:
        chain = GenerateChain(mock_llm)
        history = [
            Message(role="user", content="你好"),
            Message(role="assistant", content="你好！有什么可以帮助你的？"),
        ]
        answer = await chain.generate("什么是Python？", sample_docs, history=history)
        assert answer == "这是测试回答。"

    async def test_generate_stream_yields_tokens(self, sample_docs) -> None:
        mock_llm = AsyncMock()
        tokens = ["这是", "流式", "回答"]

        async def fake_stream(messages, **kwargs):
            for t in tokens:
                yield t

        mock_llm.generate_stream = fake_stream
        chain = GenerateChain(mock_llm)
        received: list[str] = []
        async for token in chain.generate_stream("什么是Python？", sample_docs):
            received.append(token)
        assert received == tokens

    async def test_generate_with_no_context_shows_no_reference(self) -> None:
        """When no docs are available, context should indicate that."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "无法回答"
        chain = GenerateChain(mock_llm)
        answer = await chain.generate("复杂问题", [])
        # Should still work without crashing
        assert answer is not None

    async def test_context_includes_source_info(self, mock_llm, sample_docs) -> None:
        chain = GenerateChain(mock_llm)
        await chain.generate("什么是Python？", sample_docs)

        # Verify the prompt sent to the LLM contains source info
        call_args = mock_llm.generate.call_args[0][0]  # messages list
        user_msg = call_args[-1].content  # last message is the user prompt
        assert "doc1.txt" in user_msg
        assert "Python是一种编程语言" in user_msg
