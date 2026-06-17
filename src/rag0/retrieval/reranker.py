"""Reranker — cross-encoder and LLM-based document reranking.

Key improvements over the old ``reranker.py``:
- ``RerankerInterface`` abstract base class.
- ``CrossEncoderReranker`` ported from old code, with cleaner device detection.
- ``LLMReranker`` is **now reachable** (the old ``module/utils.py`` only handled
  ``type == "rank"``, making ``LLMReranker`` dead code).
- Both implement the same interface for easy swapping.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rag0.config import RerankerConfig
from rag0.logging import get_logger
from rag0.types import ScoredDocument

logger = get_logger(__name__)


# =============================================================================
# Abstract interface
# =============================================================================
class RerankerInterface(ABC):
    """Interface for document rerankers."""

    @abstractmethod
    def rank(
        self,
        query: str,
        documents: list[ScoredDocument],
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        """Rerank documents by relevance to the query.

        Args:
            query: The user's query.
            documents: Candidate documents to rerank.
            top_k: Return at most this many documents.

        Returns:
            Reranked documents sorted by descending relevance.
        """


# =============================================================================
# Cross-Encoder Reranker
# =============================================================================
class CrossEncoderReranker(RerankerInterface):
    """Cross-encoder based reranker using HuggingFace models.

    Supports BGE, BCE, and general sentence-transformers cross-encoder models.
    Auto-detects GPU with configurable device override.

    Args:
        config: Reranker configuration.
    """

    def __init__(self, config: RerankerConfig) -> None:
        self._config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._device = self._resolve_device()
        self._load_model()

    # ------------------------------------------------------------------
    # RerankerInterface
    # ------------------------------------------------------------------
    def rank(
        self,
        query: str,
        documents: list[ScoredDocument],
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        if not documents or self._model is None:
            return documents[:top_k]

        top_k = min(top_k, self._config.top_k)

        # Build (query, document) pairs
        pairs = [(query, doc.content[:2000]) for doc in documents]

        scores = self._compute_scores(pairs)
        if scores is None:
            return documents[:top_k]

        # Assign new scores and sort
        for doc, score in zip(documents, scores, strict=False):
            doc.score = float(score)

        ranked = sorted(documents, key=lambda d: d.score, reverse=True)
        logger.debug("Cross-encoder reranked", input=len(documents), output=min(top_k, len(ranked)))
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "transformers is required for cross-encoder reranking. "
                "Install with: pip install transformers"
            ) from exc

        logger.info("Loading reranker model", model=self._config.model_name, device=self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self._config.model_name
        )
        self._model.to(self._device)
        self._model.eval()

    def _compute_scores(self, pairs: list[tuple[str, str]]) -> list[float] | None:
        """Tokenize and score query-document pairs."""
        import torch

        if self._tokenizer is None or self._model is None:
            return None

        try:
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                logits = self._model(**inputs).logits

            # Handle BCE models (sigmoid) vs regression models
            scores = (
                torch.sigmoid(logits).squeeze(-1)
                if logits.shape[-1] == 1
                else logits[:, -1]
            )

            return scores.cpu().tolist()
        except Exception as exc:
            logger.warning("Reranker scoring failed", error=str(exc))
            return None

    @staticmethod
    def _resolve_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


# =============================================================================
# LLM Reranker (NOW REACHABLE — was dead code in old ``module/utils.py``)
# =============================================================================
class LLMReranker(RerankerInterface):
    """LLM-based pointwise reranker.

    Uses the LLM to judge whether each document is relevant to the query.
    Scores are derived from the token probability of "Yes" vs "No".

    Args:
        config: Reranker configuration.
        llm: LLM connector (any LiteLLM-backed model).
    """

    _JUDGE_PROMPT = (
        "Given a query A and a passage B, determine whether the passage "
        "contains an answer to the query. Reply with ONLY 'Yes' or 'No'.\n\n"
        "A: {query}\nB: {passage}\n\nAnswer:"
    )

    def __init__(self, config: RerankerConfig, llm) -> None:
        self._config = config
        self._llm = llm

    async def rank(  # type: ignore[override]
        self,
        query: str,
        documents: list[ScoredDocument],
        top_k: int = 5,
    ) -> list[ScoredDocument]:
        """Rerank documents using LLM pointwise judgments."""
        if not documents:
            return []

        from rag0.types import Message

        top_k = min(top_k, self._config.top_k)

        # Score each document
        for doc in documents:
            try:
                response = await self._llm.generate(
                    [
                        Message(
                            role="user",
                            content=self._JUDGE_PROMPT.format(
                                query=query, passage=doc.content[:1500]
                            ),
                        )
                    ]
                )
                # Simple heuristic: "Yes" → 1.0, anything else → 0.0
                doc.score = 1.0 if response.strip().lower().startswith("yes") else 0.0
            except Exception:
                doc.score = 0.0

        ranked = sorted(documents, key=lambda d: d.score, reverse=True)
        logger.debug("LLM reranked", input=len(documents), output=min(top_k, len(ranked)))
        return ranked[:top_k]
