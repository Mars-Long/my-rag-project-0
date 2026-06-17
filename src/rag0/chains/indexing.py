"""IndexingChain — orchestrates document indexing: load → split → multi-vector → store.

Replaces the old ``rag/chains/indexing.py``.

Key improvements:
- ``asyncio.gather`` for true parallel file processing (was ``ThreadPoolExecutor``
  with no max_workers limit).
- Structured result: ``IndexingResult`` per file instead of bare tuple.
- Graceful partial failure (one bad file does not abort the batch).
- Async LLM calls for summary generation.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from pathlib import Path

from langchain_core.documents import Document

from rag0.connectors.registry import loader_registry, splitter_registry
from rag0.container import Container
from rag0.exceptions import DocumentLoadError
from rag0.indexing.multi_vector import (
    generate_table_summaries,
    generate_text_summaries,
    split_smaller_chunks,
)
from rag0.logging import get_logger
from rag0.types import IndexingResult, ScoredDocument

logger = get_logger(__name__)


class IndexingChain:
    """Indexes files into a knowledge base.

    Args:
        container: The DI container providing vector_store, embedding, llm, etc.
    """

    def __init__(self, container: Container) -> None:
        self._container = container
        self._config = container.config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def index_files(
        self,
        file_paths: Sequence[str | Path],
        knowledge_base_name: str,
        *,
        splitter_name: str | None = None,
        enable_multi_vector: bool = False,
    ) -> list[IndexingResult]:
        """Index a batch of files into a knowledge base.

        Files are processed in parallel. A single file failure does not
        affect other files.

        Args:
            file_paths: List of paths to files on disk.
            knowledge_base_name: Target knowledge base.
            splitter_name: Override the default splitter from config.
            enable_multi_vector: If True, generate sub-chunks and summaries.

        Returns:
            One :class:`IndexingResult` per file.
        """
        if not file_paths:
            return []

        # Ensure the collection exists
        dimensions = self._config.embedding.dimensions
        if not self._container.vector_store.collection_exists(knowledge_base_name):
            self._container.vector_store.create_collection(
                knowledge_base_name, dimensions
            )

        tasks = [
            self._index_one_file(
                Path(fp),
                knowledge_base_name,
                splitter_name,
                enable_multi_vector,
            )
            for fp in file_paths
        ]
        results = await asyncio.gather(*tasks)
        return list(results)

    async def _index_one_file(
        self,
        file_path: Path,
        kb_name: str,
        splitter_name_override: str | None,
        enable_multi_vector: bool,
    ) -> IndexingResult:
        """Index a single file and return its result."""
        try:
            # 1. Load
            docs = self._load(file_path)

            # 2. Split
            splitter_name = splitter_name_override or self._config.splitter.name
            chunks = self._split(docs, splitter_name)

            # 3. Assign IDs
            for chunk in chunks:
                chunk.metadata["doc_id"] = str(uuid.uuid4())

            # 4. Multi-vector (optional)
            multi_vector_docs = await self._multi_vector(chunks, enable_multi_vector)

            # 5. Store — convert LangChain Documents to ScoredDocuments
            scored_chunks = [
                ScoredDocument(
                    content=c.page_content,
                    metadata=c.metadata,
                    doc_id=c.metadata.get("doc_id", str(uuid.uuid4())),
                )
                for c in chunks
            ]
            all_docs = scored_chunks + multi_vector_docs
            await self._store(file_path, kb_name, all_docs)

            logger.info(
                "File indexed",
                file=file_path.name,
                kb=kb_name,
                chunks=len(chunks),
                multi_vector=len(multi_vector_docs),
            )
            return IndexingResult(
                filename=file_path.name,
                success=True,
                chunks_count=len(chunks),
            )

        except Exception as exc:
            logger.error(
                "File indexing failed",
                file=file_path.name,
                error=str(exc),
            )
            return IndexingResult(
                filename=file_path.name,
                success=False,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------
    def _load(self, file_path: Path) -> list[Document]:
        """Load a file using the registered loader for its extension."""
        ext = file_path.suffix.lower()
        loader_cls = loader_registry.get(ext)

        if loader_cls is None:
            available = loader_registry.list_keys()
            raise DocumentLoadError(
                f"No loader registered for '{ext}'. Available: {available}",
                context={"extension": ext},
            )

        loader = loader_cls(str(file_path))
        docs: list[Document] = loader.load()

        if not docs:
            raise DocumentLoadError(
                f"No content extracted from '{file_path.name}'",
                context={"file": str(file_path)},
            )

        return docs

    @staticmethod
    def _split(documents: list[Document], splitter_name: str) -> list[Document]:
        """Split documents using the registered splitter."""
        splitter_cls = splitter_registry.get_required(splitter_name)
        splitter = splitter_cls()
        return list(splitter.split_documents(documents))  # type: ignore[no-any-return]

    async def _multi_vector(
        self,
        chunks: list[Document],
        enabled: bool,
    ) -> list[ScoredDocument]:
        """Generate multi-vector documents (small-to-big, summaries)."""
        if not enabled:
            return []

        result: list[ScoredDocument] = []

        # 1. Small-to-big
        smaller_size = self._config.splitter.smaller_chunk_size
        if smaller_size > 0:
            result.extend(split_smaller_chunks(chunks, smaller_size))

        # 2. Text summaries
        if self._config.splitter.enable_summary:
            summaries = await generate_text_summaries(chunks, self._container.llm)
            result.extend(summaries)

        # 3. Table summaries
        if self._config.splitter.enable_table_summary:
            table_summaries = await generate_table_summaries(chunks, self._container.llm)
            result.extend(table_summaries)

        return result

    async def _store(
        self,
        file_path: Path,
        kb_name: str,
        docs: list[ScoredDocument],
    ) -> None:
        """Store documents into the vector store and metadata database."""
        vs = self._container.vector_store
        emb = self._container.embedding

        # Batch embed
        texts = [d.content for d in docs]
        embeddings = emb.embed_documents(texts)

        # Delete old chunks for this file (by source metadata)
        vs.delete_by_filter(kb_name, f'metadata["source"] == "{str(file_path)}"')

        # Insert new
        vs.add_documents(kb_name, docs, embeddings)

        # Save metadata to database
        session = self._container.new_session()
        try:
            file_repo = self._container.file_repo(session)
            kb_repo = self._container.kb_repo(session)

            # File record
            file_stat = file_path.stat()
            file_repo.add_file(
                kb_name=kb_name,
                file_name=file_path.name,
                file_ext=file_path.suffix.lower(),
                file_size=file_stat.st_size,
                loader_name=type(
                    loader_registry.get(file_path.suffix.lower())
                ).__name__,
                splitter_name=self._config.splitter.name,
                docs_count=len(docs),
            )

            # Doc records (chunk IDs)
            doc_infos = [
                {"doc_id": d.doc_id, "metadata": d.metadata} for d in docs
            ]
            file_repo.add_docs(kb_name, file_path.name, doc_infos)

            # Update file count
            kb_repo.increment_file_count(kb_name)

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
