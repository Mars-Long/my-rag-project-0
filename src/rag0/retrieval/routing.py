"""Query routing — route a question to the most relevant file in a knowledge base.

Key improvements over the old ``route_query.py``:
- Structured output via JSON mode (not fragile string parsing).
- Failures raise ``RetrievalError`` instead of silently returning ``[]``.
- ``double_check_keys`` moved to configuration (not hardcoded ``["重庆"]``).
- Async LLM call.
"""

from __future__ import annotations

import json

from rag0.connectors.llm import LLMConnector
from rag0.exceptions import RetrievalError
from rag0.logging import get_logger
from rag0.types import Message

logger = get_logger(__name__)

_ROUTE_PROMPT = """你是一个文件路由助手。根据用户问题，判断回答该问题最可能需要的文件。

知识库文件列表：
{file_list}

用户问题：{question}

请返回一个JSON对象，格式如下：
{{"file_name": "匹配的文件名", "confidence": "high|medium|low"}}

规则：
- 如果问题中的关键词明显匹配某个文件名，选择该文件
- 如果无法确定匹配，设置 file_name 为 null
- 只返回JSON，不要其他文字"""


async def route_query_to_file(
    question: str,
    file_names: list[str],
    llm: LLMConnector,
) -> str | None:
    """Route a question to the most relevant file.

    Args:
        question: User's question.
        file_names: List of available file names in the knowledge base.
        llm: LLM connector.

    Returns:
        The matched file name, or ``None`` if no match found.

    Raises:
        RetrievalError: If the LLM call fails.
    """
    if not file_names:
        return None

    if len(file_names) == 1:
        return file_names[0]

    file_list = "\n".join(f"- {f}" for f in file_names)

    try:
        response = await llm.generate(
            [
                Message(
                    role="user",
                    content=_ROUTE_PROMPT.format(
                        file_list=file_list,
                        question=question,
                    ),
                )
            ]
        )
    except Exception as exc:
        raise RetrievalError(
            "Query routing LLM call failed",
            cause=exc,
            context={"question": question[:100]},
        ) from exc

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        response = response.strip()
        if response.startswith("```"):
            response = response.split("\n", 1)[1]
            if response.endswith("```"):
                response = response[:-3]

        result = json.loads(response)
    except json.JSONDecodeError as exc:
        raise RetrievalError(
            "Failed to parse query routing LLM response as JSON",
            cause=exc,
            context={"response": response[:200]},
        ) from exc

    file_name: str | None = result.get("file_name")
    if file_name and file_name in file_names:
        logger.debug("Query routed to file", question=question[:50], file=file_name)
        return file_name

    logger.debug("Query could not be routed to a specific file", question=question[:50])
    return None
