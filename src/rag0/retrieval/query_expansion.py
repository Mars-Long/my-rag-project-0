"""Query expansion strategies — generate variant queries to improve recall.

Two strategies:
1. **Multi-query**: Ask the LLM to rephrase the question from different angles.
2. **HyDE** (Hypothetical Document Embeddings): Ask the LLM to generate a
   hypothetical answer, then use its embedding for retrieval.

Key improvements over the old ``multi_query.py``:
- Async LLM calls.
- Configurable query count.
- Deduplication and basic quality checks.
"""

from __future__ import annotations

from rag0.connectors.llm import LLMConnector
from rag0.logging import get_logger
from rag0.types import Message

logger = get_logger(__name__)

# Prompt for multi-query generation
_MULTI_QUERY_PROMPT = """你是一个AI助手。你的任务是为给定的用户问题生成 {num_queries} 个不同角度的查询表述，
以帮助从向量数据库中检索相关文档。用中文生成查询。

请从不同角度、用不同措辞来表述问题，以克服基于距离的相似性搜索的局限性。
每行一个查询，不要编号，不要多余的解释。

原始问题：{question}

生成的查询："""

# Prompt for HyDE
_HYDE_PROMPT = """你是一个AI助手。请根据以下问题，生成一段假想的回答段落。
不要回答问题本身，而是生成一段看起来像是在回答这个问题的文本。
这段文本将用于在文档库中进行语义搜索。

问题：{question}

假想回答："""


async def generate_multi_queries(
    question: str,
    llm: LLMConnector,
    num_queries: int = 3,
) -> list[str]:
    """Generate variant queries for expanded retrieval.

    Args:
        question: The original user question.
        llm: LLM connector.
        num_queries: Number of variant queries to generate.

    Returns:
        List of queries (including the original question as first element).
    """
    if num_queries < 1:
        return [question]

    try:
        response = await llm.generate(
            [
                Message(
                    role="user",
                    content=_MULTI_QUERY_PROMPT.format(
                        num_queries=num_queries,
                        question=question,
                    ),
                )
            ]
        )
    except Exception as exc:
        logger.warning("Multi-query generation failed", error=str(exc))
        return [question]

    # Parse: one query per line, skip empty/commented lines
    lines = [line.strip() for line in response.split("\n") if line.strip()]
    lines = [l for l in lines if not l.startswith("#") and not l.startswith("//")]

    # Remove numbering prefixes like "1." or "1、"
    import re

    cleaned: list[str] = []
    for line in lines:
        cleaned.append(re.sub(r"^\d+[.、)]\s*", "", line))

    # Deduplicate, preserving order
    seen = {question}
    queries = [question]
    for q in cleaned:
        if q not in seen:
            seen.add(q)
            queries.append(q)

    logger.debug("Multi-queries generated", original=question, count=len(queries))
    return queries[: num_queries + 1]  # +1 for the original


async def generate_hyde_document(
    question: str,
    llm: LLMConnector,
) -> str | None:
    """Generate a hypothetical answer document for HyDE retrieval.

    Args:
        question: The user's question.
        llm: LLM connector.

    Returns:
        A synthetic document text, or ``None`` if generation fails.
    """
    try:
        response = await llm.generate(
            [
                Message(
                    role="user",
                    content=_HYDE_PROMPT.format(question=question),
                )
            ]
        )
    except Exception as exc:
        logger.warning("HyDE generation failed", error=str(exc))
        return None

    if not response.strip():
        return None

    logger.debug("HyDE document generated", length=len(response))
    return response
