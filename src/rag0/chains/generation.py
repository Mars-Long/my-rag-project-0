"""GenerateChain — augments queries with retrieved context and calls the LLM.

Key improvements over the old ``rag/chains/generate.py``:
- Async LLM calls throughout (was sync, blocking the event loop).
- Streaming via ``AsyncIterator[str]`` — truly async, not sync-in-async-wrapper.
- Configurable system prompt.
- Graceful degradation when no documents are retrieved.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from rag0.connectors.llm import LLMConnector
from rag0.logging import get_logger
from rag0.types import Message, ScoredDocument

logger = get_logger(__name__)

# Default templates (override via config)
_DEFAULT_SYSTEM_PROMPT = (
    "你是一个乐于助人的AI助手。请根据提供的参考信息回答用户的问题。"
    "如果参考信息不足以回答，请如实说明。回答时请保持简洁、准确。"
)

_DEFAULT_RAG_TEMPLATE = """<指令> 根据已知信息，简洁、准确地回答用户的问题。如果无法从已知信息中找到答案，请说"根据已知信息无法回答该问题"，不允许在答案中添加编造成分。 </指令>

<已知信息>
{context}
</已知信息>

<问题>
{query}
</问题>"""

_DEFAULT_CHAT_TEMPLATE = """{context}

基于以上对话历史和参考信息，回答用户的问题。

用户：{query}
助手："""


class GenerateChain:
    """Generate answers using retrieved documents and an LLM.

    Args:
        llm: The LLM connector (from the DI container).
        system_prompt: Optional system-level instruction override.
    """

    def __init__(
        self,
        llm: LLMConnector,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def generate(
        self,
        query: str,
        documents: list[ScoredDocument],
        history: list[Message] | None = None,
    ) -> str:
        """Generate a complete answer.

        Args:
            query: The user's question.
            documents: Retrieved context documents.
            history: Optional conversation history (user/assistant pairs).

        Returns:
            The generated answer text.
        """
        context = self._build_context(documents)
        messages = self._build_messages(query, context, history)

        logger.debug("Generating answer", query=query[:50], doc_count=len(documents))
        start = time.perf_counter()

        try:
            answer = await self._llm.generate(messages)
        except Exception:
            logger.warning("LLM generation failed, returning fallback")
            return "抱歉，生成回答时遇到了问题，请稍后重试。"

        elapsed = time.perf_counter() - start
        logger.debug("Answer generated", length=len(answer), elapsed_ms=int(elapsed * 1000))
        return answer

    async def generate_stream(
        self,
        query: str,
        documents: list[ScoredDocument],
        history: list[Message] | None = None,
    ) -> AsyncIterator[str]:
        """Generate an answer token-by-token (SSE streaming).

        Args:
            query: The user's question.
            documents: Retrieved context documents.
            history: Optional conversation history.

        Yields:
            Text tokens as they arrive from the LLM.
        """
        context = self._build_context(documents)
        messages = self._build_messages(query, context, history)

        logger.debug("Streaming answer", query=query[:50], doc_count=len(documents))

        try:
            async for token in self._llm.generate_stream(messages):
                yield token
        except Exception:
            logger.warning("LLM stream failed mid-response")
            yield "抱歉，生成回答时遇到了问题。"

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------
    def _build_context(self, documents: list[ScoredDocument]) -> str:
        """Concatenate retrieved documents into a context string."""
        if not documents:
            return "（无相关参考信息）"

        parts: list[str] = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("file_name", doc.metadata.get("source", "未知"))
            parts.append(f"[参考{i}] 来源: {source}\n{doc.content}")
        return "\n\n".join(parts)

    def _build_messages(
        self,
        query: str,
        context: str,
        history: list[Message] | None,
    ) -> list[Message]:
        """Build the full message list for the LLM call."""
        messages: list[Message] = [
            Message(role="system", content=self._system_prompt),
        ]

        # Include conversation history
        if history:
            messages.extend(history)

        # Build the RAG prompt
        if context and "无相关参考信息" not in context:
            prompt = _DEFAULT_RAG_TEMPLATE.format(context=context, query=query)
        else:
            prompt = query

        messages.append(Message(role="user", content=prompt))
        return messages
