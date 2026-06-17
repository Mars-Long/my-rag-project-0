"""Dependency injection container — wires all components together.

Replaces the old module-level globals (``rag/connector/base.py``,
``rag/module/base.py``) with an explicit, testable container.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag0.config import RagConfig, get_config
from rag0.connectors.database import (
    KnowledgeBaseRepository,
    KnowledgeFileRepository,
    create_engine_and_sessionmaker,
)
from rag0.connectors.embeddings import EmbeddingConnector
from rag0.connectors.llm import LLMConnector
from rag0.connectors.vector_store import MilvusVectorStore, VectorStoreInterface
from rag0.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Container:
    """Holds all application-level services.

    Create once at startup, inject into API handlers and pipeline chains.

    Usage::

        container = Container.create()
        # Inject into chains:
        chain = IndexingChain(container)
    """

    config: RagConfig
    llm: LLMConnector
    embedding: EmbeddingConnector
    vector_store: VectorStoreInterface
    session_factory: object  # sessionmaker[Session]

    # Repositories are created per-session, not stored here.
    # The session_factory is what callers use.

    @classmethod
    def create(cls, config: RagConfig | None = None) -> "Container":
        """Create a fully-wired container from configuration.

        Args:
            config: Optional config. If ``None``, loads from the global singleton.
        """
        if config is None:
            config = get_config()

        logger.info("Initializing LLM connector", model=config.llm.model_name)
        llm = LLMConnector(config.llm)

        logger.info("Initializing embedding connector", model=config.embedding.model_name)
        embedding = EmbeddingConnector(config.embedding)

        logger.info("Initializing vector store", host=config.vector_store.host)
        vector_store = MilvusVectorStore(config.vector_store)

        _, session_factory = create_engine_and_sessionmaker(config.database)

        logger.info("Container initialized")
        return cls(
            config=config,
            llm=llm,
            embedding=embedding,
            vector_store=vector_store,
            session_factory=session_factory,
        )

    def new_session(self):
        """Create a new database session. Caller is responsible for closing it."""
        return self.session_factory()

    def kb_repo(self, session) -> KnowledgeBaseRepository:
        return KnowledgeBaseRepository(session)

    def file_repo(self, session) -> KnowledgeFileRepository:
        return KnowledgeFileRepository(session)
