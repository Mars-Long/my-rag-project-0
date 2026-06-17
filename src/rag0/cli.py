"""CLI entry point for RAG0.

Usage::

    python -m rag0 serve           # Start the API server
    python -m rag0 create-tables   # Initialize/reset the database
    python -m rag0 config validate # Validate config file
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Main CLI dispatcher."""
    parser = argparse.ArgumentParser(
        prog="rag0",
        description="RAG0 — A modern Retrieval-Augmented Generation framework",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---- serve ----
    serve_parser = sub.add_parser("serve", help="Start the API server")
    serve_parser.add_argument("--host", default=None, help="API server host")
    serve_parser.add_argument("--port", type=int, default=None, help="API server port")

    # ---- create-tables ----
    sub.add_parser("create-tables", help="Create database tables")

    # ---- config validate ----
    sub.add_parser("config", parents=[
        argparse.ArgumentParser(add_help=False)
    ])
    # subparser for config validate
    sub.add_parser("config-validate", help="Validate configuration")
    # Workaround: let's just use simple command names

    args = parser.parse_args(argv)

    if args.command == "serve":
        _cmd_serve(args.host, args.port)
    elif args.command == "create-tables":
        _cmd_create_tables()
    elif args.command == "config-validate":
        _cmd_config_validate()
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_serve(host: str | None, port: int | None) -> None:
    """Start the FastAPI server."""
    import uvicorn

    from rag0.api.app import create_app
    from rag0.config import get_config

    config = get_config()
    host = host or config.server.host
    port = port or config.server.port

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


def _cmd_create_tables() -> None:
    """Create all database tables."""
    from rag0.config import get_config
    from rag0.connectors.database import create_engine_and_sessionmaker, create_tables

    config = get_config()
    engine, _ = create_engine_and_sessionmaker(config.database)
    create_tables(engine)
    print("Database tables created.")


def _cmd_config_validate() -> None:
    """Validate the configuration file."""
    from rag0.config import get_config

    try:
        config = get_config()
        print("Configuration loaded successfully.")
        print(f"  LLM model:      {config.llm.model_name}")
        print(f"  Embedding:      {config.embedding.model_name}")
        print(f"  Vector store:   {config.vector_store.host}:{config.vector_store.port}")
        print(f"  Database:       {config.database.url}")
        print(f"  Server:         {config.server.host}:{config.server.port}")
        print(f"  Telemetry:      {'enabled' if config.telemetry.enabled else 'disabled'}")
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
