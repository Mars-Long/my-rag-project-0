"""FastAPI application factory for RAG0.

Creates a fully-configured FastAPI app with:
- CORS middleware (configurable origins)
- Request ID middleware (X-Request-ID header)
- Unified exception handling middleware
- All routes mounted on a consistent prefix
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from rag0.api.deps import set_container
from rag0.api.middleware import RAG0Exception, map_exception_to_response
from rag0.container import Container
from rag0.logging import get_logger

logger = get_logger(__name__)


def create_app(container: Container | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        container: Optional DI container. If ``None``, creates one from config.

    Returns:
        A ready-to-serve FastAPI application.
    """
    if container is None:
        container = Container.create()

    # Make container available to route handlers
    set_container(container)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("RAG0 API server starting", host=container.config.server.host)
        yield
        logger.info("RAG0 API server shutting down")

    app = FastAPI(
        title="RAG0 API",
        description="A modern Retrieval-Augmented Generation framework",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ---- Middleware ----
    origins = [o.strip() for o in container.config.server.cors_origins.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # ---- Exception handler ----
    @app.exception_handler(RAG0Exception)
    async def rag0_exception_handler(request: Request, exc: RAG0Exception):
        return JSONResponse(
            status_code=exc.status_code,
            content={"code": exc.status_code, "msg": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        response = map_exception_to_response(exc)
        return JSONResponse(
            status_code=response.status_code,
            content={"code": response.status_code, "msg": response.message, "detail": response.detail},
        )

    # ---- Routes ----
    from rag0.api.routes.chat import router as chat_router
    from rag0.api.routes.health import router as health_router
    from rag0.api.routes.knowledge import router as knowledge_router

    app.include_router(health_router, tags=["Health"])
    app.include_router(knowledge_router, prefix="/knowledge-bases", tags=["Knowledge Bases"])
    app.include_router(chat_router, prefix="/chat", tags=["Chat"])

    return app
