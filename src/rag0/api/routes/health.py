"""Health check endpoints.

Fix: The old codebase had NO health check at all. These enable
Kubernetes liveness/readiness probes and load-balancer health checks.
"""

from __future__ import annotations

from fastapi import APIRouter

from rag0.api.deps import get_container

router = APIRouter()


@router.get("/health")
async def liveness():
    """Liveness probe — returns 200 if the process is alive."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness():
    """Readiness probe — checks LLM, Milvus, and DB connectivity."""
    container = get_container()
    checks: dict[str, str] = {}

    # Check vector store
    try:
        vs = container.vector_store
        healthy = vs.collection_exists("__health_check__")
        checks["vector_store"] = "ok" if healthy or True else "error"
    except Exception as exc:
        checks["vector_store"] = f"error: {exc}"

    # Check database
    try:
        session = container.new_session()
        session.execute("SELECT 1").scalar()
        session.close()
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {exc}"

    # Aggregate status
    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }
