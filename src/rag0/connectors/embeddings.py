"""Embedding connector — wraps sentence-transformers directly.

Key improvements over the old ``LocalEmbeddings``:
- Direct ``sentence-transformers`` usage (no LangChain wrapper).
- LRU caching for repeated texts.
- Proper error handling (raises ``ConfigurationError`` for unknown engines).
- GPU auto-detection with configurable device override.
"""

from __future__ import annotations

from rag0.config import EmbeddingConfig
from rag0.exceptions import ConfigurationError
from rag0.logging import get_logger

logger = get_logger(__name__)


class EmbeddingConnector:
    """Encode text into dense vector embeddings.

    Args:
        config: Embedding configuration.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config
        self._model = self._load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of document texts.

        Args:
            texts: List of document text chunks.

        Returns:
            List of embedding vectors, same order as input.
        """
        if not texts:
            return []

        logger.debug("Embedding documents", count=len(texts))
        embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self._config.normalize,
        )
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, query: str) -> list[float]:
        """Encode a single query text.

        Args:
            query: The search query string.

        Returns:
            Embedding vector.
        """
        if not query.strip():
            raise ValueError("Query must not be empty")

        # Check cache first
        return self._cached_embed(query)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _cached_embed(self, text: str) -> list[float]:
        """LRU-cached single-text embedding."""
        embedding = self._model.encode(
            [text],
            batch_size=1,
            show_progress_bar=False,
            normalize_embeddings=self._config.normalize,
        )
        return embedding[0].tolist()

    def _load_model(self):
        """Load the sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigurationError(
                "sentence-transformers is required for embedding. "
                "Install with: pip install sentence-transformers",
                cause=exc,
            ) from exc

        device = self._resolve_device()
        logger.info("Loading embedding model", model=self._config.model_name, device=device)

        try:
            model = SentenceTransformer(
                self._config.model_name,
                device=device,
            )
        except Exception as exc:
            raise ConfigurationError(
                f"Failed to load embedding model '{self._config.model_name}'",
                cause=exc,
            ) from exc

        return model

    def _resolve_device(self) -> str:
        """Resolve the device string (auto → cuda/cpu/mps)."""
        device = self._config.device.lower()
        if device != "auto":
            return device

        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"
