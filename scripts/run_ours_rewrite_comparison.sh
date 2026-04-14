#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/outputs/full_run/benchmark_dataset.jsonl}"
SECTIONS_PATH="${SECTIONS_PATH:-$PROJECT_ROOT/artifacts/full_run/markdown_sections.jsonl}"
KNOWLEDGE_PATH="${KNOWLEDGE_PATH:-$PROJECT_ROOT/artifacts/full_run/knowledge_extraction.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/experiments/ours_rewrite_comparison}"

TOP_K="${TOP_K:-5}"
LIMIT="${LIMIT:-100}"
MAX_WORKERS="${MAX_WORKERS:-4}"
SKELETON_MODE="${SKELETON_MODE:-oracle}"
REWRITE_MODES="${REWRITE_MODES:-original,splicing,sparse_llm,dense_llm,hybrid_llm}"
RETRIEVAL_MODES="${RETRIEVAL_MODES:-lexical,dense,hybrid}"
SCORER_MODES="${SCORER_MODES:-plain,relation_driven}"
LLM_REWRITE_MAX_TOKENS="${LLM_REWRITE_MAX_TOKENS:-128}"
LLM_REWRITE_TEMPERATURE="${LLM_REWRITE_TEMPERATURE:-0.0}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT/src"
export GENERATOR_BACKEND="llm"
export GENERATOR_PROVIDER="${GENERATOR_PROVIDER:-dashscope}"
export GENERATOR_API_KEY="${GENERATOR_API_KEY:-${DASHSCOPE_API_KEY:-}}"
export GENERATOR_BASE_URL="${GENERATOR_BASE_URL:-${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}}"
export GENERATOR_MODEL="${GENERATOR_MODEL:-qwen2.5-7b-instruct-1m}"
export GENERATOR_JSON_MODE="true"
export GENERATOR_TEMPERATURE="${GENERATOR_TEMPERATURE:-0.0}"

python "$PROJECT_ROOT/scripts/run_ours_retrieval_lab.py" \
  --dataset "$DATASET_PATH" \
  --sections "$SECTIONS_PATH" \
  --knowledge "$KNOWLEDGE_PATH" \
  --top-k "$TOP_K" \
  --limit "$LIMIT" \
  --max-workers "$MAX_WORKERS" \
  --skeleton-mode "$SKELETON_MODE" \
  --rewrite-modes "$REWRITE_MODES" \
  --retrieval-modes "$RETRIEVAL_MODES" \
  --scorer-modes "$SCORER_MODES" \
  --output-dir "$OUTPUT_DIR" \
  --overwrite \
  --llm-rewrite-max-tokens "$LLM_REWRITE_MAX_TOKENS" \
  --llm-rewrite-temperature "$LLM_REWRITE_TEMPERATURE"
