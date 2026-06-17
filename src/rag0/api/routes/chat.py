"""Chat endpoint with SSE streaming.

Key fixes over the old ``chat.py``:
- Truly async streaming (no sync-in-async-wrapper blocking the event loop).
- No duplicate vector store creation (old code created it twice).
- Configurable ``return_docs`` to include retrieved documents in the response.
- Proper error events in the SSE stream.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Body
from sse_starlette.sse import EventSourceResponse

from rag0.api.deps import get_container
from rag0.chains.generation import GenerateChain
from rag0.chains.retrieval import RetrievalChain
from rag0.logging import get_logger
from rag0.types import Message, ScoredDocument

logger = get_logger(__name__)

router = APIRouter()


@router.post("")
async def chat(
    query: str = Body(...),  # noqa: B008
    knowledge_base_name: str = Body(...),  # noqa: B008
    history: list[dict[str, str]] | None = Body(None),  # noqa: B008
    top_k: int = Body(5),  # noqa: B008
    stream: bool = Body(True),  # noqa: B008
    return_docs: bool = Body(False),  # noqa: B008
    enable_multi_query: bool = Body(True),  # noqa: B008
) -> Any:
    """Chat with a knowledge base.

    Args:
        query: The user's question.
        knowledge_base_name: Target knowledge base.
        history: Optional conversation history as ``[{"role": "...", "content": "..."}]``.
        top_k: Number of documents to retrieve.
        stream: If True (default), returns SSE stream. Otherwise returns JSON.
        return_docs: Include retrieved documents in the response.
        enable_multi_query: Enable multi-query expansion.

    Returns:
        SSE event stream (``stream=True``) or JSON response (``stream=False``).
    """
    container = get_container()

    # Parse history
    history_msgs: list[Message] = []
    if history:
        history_msgs = [
            Message(role=h.get("role", "user"), content=h.get("content", ""))
            for h in history
        ]

    # Retrieve
    retrieval = RetrievalChain(container)
    docs = await retrieval.retrieve(
        query=query,
        knowledge_base_name=knowledge_base_name,
        top_k=top_k,
        enable_multi_query=enable_multi_query,
    )

    # Generate
    generation = GenerateChain(container.llm)

    if stream:
        async def event_stream() -> AsyncIterator[dict[str, str]]:
            """SSE event generator — truly async, no event-loop blocking."""
            try:
                # First event: metadata (doc IDs if requested)
                if return_docs:
                    yield {
                        "event": "docs",
                        "data": json.dumps(
                            [_doc_to_dict(d) for d in docs], ensure_ascii=False
                        ),
                    }

                # Stream answer tokens
                async for token in generation.generate_stream(query, docs, history_msgs):
                    yield {
                        "event": "token",
                        "data": json.dumps({"content": token}, ensure_ascii=False),
                    }

                # Done event
                yield {"event": "done", "data": json.dumps({"status": "complete"})}

            except Exception as exc:
                logger.error("SSE stream error", error=str(exc))
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(exc)}, ensure_ascii=False),
                }

        return EventSourceResponse(event_stream())

    else:
        # Non-streaming: return JSON directly
        answer = await generation.generate(query, docs, history_msgs)
        response: dict[str, Any] = {"query": query, "answer": answer}
        if return_docs:
            response["documents"] = [_doc_to_dict(d) for d in docs]
        return response


def _doc_to_dict(doc: ScoredDocument) -> dict[str, Any]:
    return {
        "content": doc.content[:500],  # Truncate for response
        "metadata": doc.metadata,
        "score": round(doc.score, 4),
        "doc_id": doc.doc_id,
    }
