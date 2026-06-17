"""Database connector — SQLAlchemy 2.0 models and repositories.

Key improvements over the old database layer:
- SQLAlchemy 2.0 declarative style with ``Mapped`` annotations.
- Alembic for schema migrations (no more ad-hoc ``create_tables``).
- Repository methods receive a ``Session`` parameter (caller manages lifecycle).
- No ``@with_session`` double-commit bug.
- Proper error handling — ``None`` doc_infos raises instead of printing.
- Foreign key constraints and indexes on frequently queried columns.
- ``pathlib.Path`` for all filesystem paths.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from rag0.config import DatabaseConfig
from rag0.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# ORM Base
# =============================================================================
class Base(DeclarativeBase):
    pass


# =============================================================================
# Models
# =============================================================================
class KnowledgeBase(Base):
    """A knowledge base — a logical container for indexed documents."""

    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(String(500), default=None)
    vs_type: Mapped[str] = mapped_column(String(50), default="milvus")
    embed_model: Mapped[str] = mapped_column(String(200), default="")
    file_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    files: Mapped[list[KnowledgeFile]] = relationship(
        back_populates="knowledge_base", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<KnowledgeBase name={self.name!r} files={self.file_count}>"


class KnowledgeFile(Base):
    """Metadata for a file indexed into a knowledge base."""

    __tablename__ = "knowledge_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kb_name: Mapped[str] = mapped_column(
        String(100), ForeignKey("knowledge_bases.name", ondelete="CASCADE"), nullable=False, index=True
    )
    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    file_ext: Mapped[str] = mapped_column(String(20), default="")
    file_size: Mapped[int] = mapped_column(Integer, default=0)
    loader_name: Mapped[str] = mapped_column(String(100), default="")
    splitter_name: Mapped[str] = mapped_column(String(100), default="")
    docs_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="files")
    docs: Mapped[list[FileDocument]] = relationship(
        back_populates="file", cascade="all, delete-orphan"
    )

    __table_args__ = (UniqueConstraint("kb_name", "file_name", name="uq_kb_file"),)

    def __repr__(self) -> str:
        return f"<KnowledgeFile kb={self.kb_name!r} name={self.file_name!r}>"


class FileDocument(Base):
    """Links a file to its vector store chunk IDs."""

    __tablename__ = "file_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kb_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("knowledge_files.id", ondelete="CASCADE"), nullable=False
    )
    doc_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, default=None)

    # Relationships
    file: Mapped[KnowledgeFile] = relationship(back_populates="docs")

    def __repr__(self) -> str:
        return f"<FileDocument doc_id={self.doc_id!r}>"


# =============================================================================
# Engine & Session factory
# =============================================================================
def create_engine_and_sessionmaker(
    config: DatabaseConfig,
) -> tuple[Any, sessionmaker[Session]]:
    """Create SQLAlchemy engine and session factory.

    Returns a tuple of ``(engine, SessionLocal)``.
    """
    # Resolve the URL — if relative SQLite path, make it relative to cwd
    url = config.url
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        db_path = url.removeprefix("sqlite:///")
        # Ensure parent directory exists
        path = Path(db_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{path.as_posix()}"

    engine = create_engine(
        url,
        echo=False,
        connect_args={"check_same_thread": False} if "sqlite" in url else {},
        pool_pre_ping=True,
    )

    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    logger.info("Database engine created", url=url.split("///")[-1])
    return engine, SessionLocal


def create_tables(engine: Any) -> None:
    """Create all tables that don't exist yet."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")


# =============================================================================
# Repositories
# =============================================================================
class KnowledgeBaseRepository:
    """CRUD operations for :class:`KnowledgeBase`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def list_all(self) -> list[KnowledgeBase]:
        return self._session.query(KnowledgeBase).order_by(KnowledgeBase.created_at.desc()).all()

    def get_by_name(self, name: str) -> KnowledgeBase | None:
        return self._session.query(KnowledgeBase).filter(KnowledgeBase.name == name).first()

    def create(
        self,
        name: str,
        description: str = "",
        vs_type: str = "milvus",
        embed_model: str = "",
    ) -> KnowledgeBase:
        if self.get_by_name(name):
            raise ValueError(f"Knowledge base '{name}' already exists")
        kb = KnowledgeBase(
            name=name,
            description=description,
            vs_type=vs_type,
            embed_model=embed_model,
        )
        self._session.add(kb)
        self._session.flush()
        logger.info("Knowledge base created", name=name)
        return kb

    def delete(self, name: str) -> bool:
        kb = self.get_by_name(name)
        if kb is None:
            return False
        self._session.delete(kb)
        self._session.flush()
        logger.info("Knowledge base deleted", name=name)
        return True

    def increment_file_count(self, name: str, delta: int = 1) -> None:
        kb = self.get_by_name(name)
        if kb:
            kb.file_count += delta

    def reset_file_count(self, name: str) -> None:
        kb = self.get_by_name(name)
        if kb:
            kb.file_count = 0


class KnowledgeFileRepository:
    """CRUD operations for :class:`KnowledgeFile` and :class:`FileDocument`."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # -- File operations --
    def list_files(self, kb_name: str) -> list[KnowledgeFile]:
        return (
            self._session.query(KnowledgeFile)
            .filter(KnowledgeFile.kb_name == kb_name)
            .order_by(KnowledgeFile.created_at.desc())
            .all()
        )

    def get_file(self, kb_name: str, file_name: str) -> KnowledgeFile | None:
        return (
            self._session.query(KnowledgeFile)
            .filter(
                KnowledgeFile.kb_name == kb_name,
                KnowledgeFile.file_name == file_name,
            )
            .first()
        )

    def add_file(
        self,
        kb_name: str,
        file_name: str,
        file_ext: str = "",
        file_size: int = 0,
        loader_name: str = "",
        splitter_name: str = "",
        docs_count: int = 0,
    ) -> KnowledgeFile:
        existing = self.get_file(kb_name, file_name)
        if existing:
            # Update existing
            existing.file_size = file_size
            existing.docs_count = docs_count
            existing.loader_name = loader_name
            existing.splitter_name = splitter_name
            self._session.flush()
            return existing

        kf = KnowledgeFile(
            kb_name=kb_name,
            file_name=file_name,
            file_ext=file_ext,
            file_size=file_size,
            loader_name=loader_name,
            splitter_name=splitter_name,
            docs_count=docs_count,
        )
        self._session.add(kf)
        self._session.flush()
        return kf

    def delete_file(self, kb_name: str, file_name: str) -> bool:
        kf = self.get_file(kb_name, file_name)
        if kf is None:
            return False
        # Delete associated doc records first
        self._session.query(FileDocument).filter(
            FileDocument.file_id == kf.id
        ).delete()
        self._session.delete(kf)
        self._session.flush()
        return True

    def delete_all_files(self, kb_name: str) -> int:
        count = (
            self._session.query(KnowledgeFile)
            .filter(KnowledgeFile.kb_name == kb_name)
            .delete()
        )
        # Also delete orphaned doc records
        self._session.query(FileDocument).filter(
            FileDocument.kb_name == kb_name
        ).delete()
        self._session.flush()
        return count

    # -- Document (chunk) operations --
    def add_docs(
        self,
        kb_name: str,
        file_name: str,
        doc_infos: list[dict[str, Any]],
    ) -> list[FileDocument]:
        """Add chunk records for a file.

        Args:
            kb_name: Knowledge base name.
            file_name: Source file name.
            doc_infos: List of ``{"doc_id": str, "metadata": dict}``.

        Returns:
            List of created :class:`FileDocument` instances.

        Raises:
            ValueError: If *doc_infos* is ``None`` or empty.
        """
        if not doc_infos:
            raise ValueError("doc_infos must not be None or empty")

        kf = self.get_file(kb_name, file_name)
        if kf is None:
            raise ValueError(f"File '{file_name}' not found in KB '{kb_name}'")

        docs = []
        for info in doc_infos:
            doc = FileDocument(
                kb_name=kb_name,
                file_name=file_name,
                file_id=kf.id,
                doc_id=info.get("doc_id", str(uuid.uuid4())),
                metadata_=info.get("metadata"),
            )
            self._session.add(doc)
            docs.append(doc)

        self._session.flush()
        return docs

    def list_docs(self, kb_name: str, file_name: str | None = None) -> list[FileDocument]:
        q = self._session.query(FileDocument).filter(FileDocument.kb_name == kb_name)
        if file_name:
            q = q.filter(FileDocument.file_name == file_name)
        return q.all()

    def delete_docs(self, kb_name: str, file_name: str) -> list[dict[str, Any]]:
        """Delete all doc records for a file and return their metadata."""
        docs = self.list_docs(kb_name, file_name)
        result = [{"doc_id": d.doc_id, "metadata": d.metadata_} for d in docs]
        self._session.query(FileDocument).filter(
            FileDocument.kb_name == kb_name,
            FileDocument.file_name == file_name,
        ).delete()
        self._session.flush()
        return result
