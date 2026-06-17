"""Initial migration — creates knowledge_bases, knowledge_files, file_documents tables.

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "knowledge_bases",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("description", sa.String(length=500), nullable=True),
        sa.Column("vs_type", sa.String(length=50), nullable=False, server_default="milvus"),
        sa.Column("embed_model", sa.String(length=200), nullable=False, server_default=""),
        sa.Column("file_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index("ix_knowledge_bases_name", "knowledge_bases", ["name"])

    op.create_table(
        "knowledge_files",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("kb_name", sa.String(length=100), nullable=False),
        sa.Column("file_name", sa.String(length=500), nullable=False),
        sa.Column("file_ext", sa.String(length=20), nullable=False, server_default=""),
        sa.Column("file_size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("loader_name", sa.String(length=100), nullable=False, server_default=""),
        sa.Column("splitter_name", sa.String(length=100), nullable=False, server_default=""),
        sa.Column("docs_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["kb_name"], ["knowledge_bases.name"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("kb_name", "file_name", name="uq_kb_file"),
    )
    op.create_index("ix_knowledge_files_kb_name", "knowledge_files", ["kb_name"])

    op.create_table(
        "file_documents",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("kb_name", sa.String(length=100), nullable=False),
        sa.Column("file_name", sa.String(length=500), nullable=False),
        sa.Column("file_id", sa.Integer(), nullable=False),
        sa.Column("doc_id", sa.String(length=64), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(["file_id"], ["knowledge_files.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_file_documents_kb_name", "file_documents", ["kb_name"])
    op.create_index("ix_file_documents_doc_id", "file_documents", ["doc_id"])


def downgrade() -> None:
    op.drop_table("file_documents")
    op.drop_table("knowledge_files")
    op.drop_table("knowledge_bases")
