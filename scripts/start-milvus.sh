#!/usr/bin/env bash
# =============================================================================
# Start Milvus Standalone without Docker (for local development)
# =============================================================================
# Prerequisites:
#   1. Install milvus:  pip install pymilvus>=2.4
#   2. Download milvus binary: https://milvus.io/docs/install_standalone-docker.md
#      (or use the Docker Compose setup: docker compose up -d milvus)
#
# This script uses the embedded Milvus mode if available (pymilvus >= 2.4).
# =============================================================================

set -euo pipefail

MILVUS_HOST="${MILVUS_HOST:-127.0.0.1}"
MILVUS_PORT="${MILVUS_PORT:-19530}"

echo "==> Checking Milvus connection at ${MILVUS_HOST}:${MILVUS_PORT}..."

python -c "
from pymilvus import connections
try:
    connections.connect(host='${MILVUS_HOST}', port='${MILVUS_PORT}')
    print('Milvus is ready.')
except Exception as e:
    print(f'Cannot connect to Milvus: {e}')
    print('Start Milvus with: docker compose up -d milvus')
    exit(1)
"

echo "==> Milvus is running. You can now start RAG0:"
echo "    python -m rag0 serve"
