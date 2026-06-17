"""Knowledge base management routes.

Key fixes over the old API:
- Correct HTTP methods: GET for list, DELETE for delete (was all POST).
- Path-based parameters instead of body-only params.
- Proper path traversal protection (``Path.resolve()`` not ``"../" in name``).
- Structured error responses via the unified middleware.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from rag0.api.deps import get_container
from rag0.exceptions import ValidationError
from rag0.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Where uploaded files are stored before indexing
_UPLOAD_DIR = Path("data/uploads")


# ---------------------------------------------------------------------------
# GET /knowledge-bases — List all (was POST!)
# ---------------------------------------------------------------------------
@router.get("")
async def list_knowledge_bases() -> list[dict]:
    """List all knowledge bases."""
    container = get_container()
    session = container.new_session()
    try:
        kb_repo = container.kb_repo(session)
        kbs = kb_repo.list_all()
        return [
            {
                "name": kb.name,
                "description": kb.description,
                "vs_type": kb.vs_type,
                "file_count": kb.file_count,
                "created_at": kb.created_at.isoformat() if kb.created_at else None,
            }
            for kb in kbs
        ]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# POST /knowledge-bases — Create
# ---------------------------------------------------------------------------
@router.post("", status_code=201)
async def create_knowledge_base(
    name: str = Form(...),
    description: str = Form(""),
    vs_type: str = Form("milvus"),
    embed_model: str = Form(""),
) -> dict:
    """Create a new knowledge base."""
    _validate_name(name)

    container = get_container()
    session = container.new_session()
    try:
        kb_repo = container.kb_repo(session)
        try:
            kb = kb_repo.create(name, description, vs_type, embed_model)
            session.commit()
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        # Also create the vector store collection
        dims = container.config.embedding.dimensions
        container.vector_store.create_collection(name, dims)

        return {"name": kb.name, "message": "Knowledge base created"}
    finally:
        session.close()


# ---------------------------------------------------------------------------
# DELETE /knowledge-bases/{name} — Delete
# ---------------------------------------------------------------------------
@router.delete("/{name}")
async def delete_knowledge_base(name: str) -> dict:
    """Delete a knowledge base and all its data."""
    _validate_name(name)

    container = get_container()
    session = container.new_session()
    try:
        kb_repo = container.kb_repo(session)
        deleted = kb_repo.delete(name)
        session.commit()

        if not deleted:
            raise HTTPException(status_code=404, detail=f"Knowledge base '{name}' not found")

        # Drop the vector collection
        container.vector_store.drop_collection(name)

        # Clean up uploaded files
        kb_upload_dir = _UPLOAD_DIR / name
        if kb_upload_dir.exists():
            shutil.rmtree(kb_upload_dir)

        return {"name": name, "message": "Knowledge base deleted"}
    finally:
        session.close()


# ---------------------------------------------------------------------------
# DELETE /knowledge-bases/{name}/documents — Clear
# ---------------------------------------------------------------------------
@router.delete("/{name}/documents")
async def clear_knowledge_base(name: str) -> dict:
    """Remove all documents from a knowledge base (keeps the KB itself)."""
    _validate_name(name)

    container = get_container()
    session = container.new_session()
    try:
        file_repo = container.file_repo(session)
        kb_repo = container.kb_repo(session)

        # Clear vector store collection
        container.vector_store.drop_collection(name)
        dims = container.config.embedding.dimensions
        container.vector_store.create_collection(name, dims)

        # Clear DB records
        count = file_repo.delete_all_files(name)
        kb_repo.reset_file_count(name)
        session.commit()

        return {"name": name, "message": f"Cleared {count} file(s)"}
    finally:
        session.close()


# ---------------------------------------------------------------------------
# POST /knowledge-bases/{name}/documents — Upload & Index
# ---------------------------------------------------------------------------
@router.post("/{name}/documents")
async def upload_documents(
    name: str,
    files: List[UploadFile] = File(...),
    splitter_name: str | None = Form(None),
    enable_multi_vector: bool = Form(False),
) -> dict:
    """Upload and index files into a knowledge base."""
    _validate_name(name)

    container = get_container()
    kb_upload_dir = _UPLOAD_DIR / name
    kb_upload_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save uploaded files to disk
    saved_paths: list[Path] = []
    for f in files:
        if f.filename is None:
            continue
        dest = kb_upload_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        saved_paths.append(dest)

    if not saved_paths:
        raise ValidationError("No valid files uploaded")

    # 2. Index files
    from rag0.chains.indexing import IndexingChain

    chain = IndexingChain(container)
    results = await chain.index_files(
        saved_paths,
        knowledge_base_name=name,
        splitter_name=splitter_name,
        enable_multi_vector=enable_multi_vector,
    )

    # 3. Build response
    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    return {
        "name": name,
        "total": len(results),
        "succeeded": [{"filename": r.filename, "chunks": r.chunks_count} for r in succeeded],
        "failed": [{"filename": r.filename, "error": r.error} for r in failed],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_name(name: str) -> None:
    """Validate knowledge base name — no path traversal, no empty name."""
    if not name or not name.strip():
        raise ValidationError("Knowledge base name must not be empty")

    # Resolve against the upload dir to catch path traversal
    resolved = (_UPLOAD_DIR / name).resolve()
    if not str(resolved).startswith(str(_UPLOAD_DIR.resolve())):
        raise ValidationError("Invalid knowledge base name")
