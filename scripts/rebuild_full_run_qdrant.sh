#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
SECTIONS_PATH="${SECTIONS_PATH:-/home/paper/Benchmark/artifacts/full_run/markdown_sections.jsonl}"
MANIFEST_PATH="${MANIFEST_PATH:-/home/paper/Benchmark/artifacts/qdrant_index_manifest.full_run.json}"
PYTHON_BIN="${PYTHON_BIN:-/home/huang/miniconda3/envs/paper_benchmark/bin/python}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "env file not found: $ENV_FILE" >&2
  exit 1
fi

if [[ ! -f "$SECTIONS_PATH" ]]; then
  echo "sections file not found: $SECTIONS_PATH" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "python interpreter not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
QDRANT_USE_LOCAL="${QDRANT_USE_LOCAL:-false}"
QDRANT_COLLECTION="${FULL_RUN_QDRANT_COLLECTION:-benchmark_sections_full_run}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-embedding-3}"
EMBEDDING_DIM="${EMBEDDING_DIM:-1024}"

export QDRANT_URL
export QDRANT_USE_LOCAL
export QDRANT_COLLECTION
export EMBEDDING_MODEL
export EMBEDDING_DIM

echo "Rebuilding Qdrant index for full_run"
echo "  sections_path: $SECTIONS_PATH"
echo "  manifest_path: $MANIFEST_PATH"
echo "  qdrant_url: $QDRANT_URL"
echo "  qdrant_use_local: $QDRANT_USE_LOCAL"
echo "  qdrant_collection: $QDRANT_COLLECTION"
echo "  embedding_model: $EMBEDDING_MODEL"
echo "  embedding_dim: $EMBEDDING_DIM"
echo "  python_bin: $PYTHON_BIN"

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/build_qdrant_index.py" \
  --sections "$SECTIONS_PATH" \
  --manifest-path "$MANIFEST_PATH" \
  --recreate
