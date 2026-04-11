#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATASET_PATH="${DATASET_PATH:-$PROJECT_ROOT/outputs/full_run/benchmark_dataset.jsonl}"
SECTIONS_PATH="${SECTIONS_PATH:-$PROJECT_ROOT/artifacts/full_run/markdown_sections.jsonl}"
KNOWLEDGE_PATH="${KNOWLEDGE_PATH:-$PROJECT_ROOT/artifacts/full_run/knowledge_extraction.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/experiments/ours_retrieval_lab_full}"

TOP_K="${TOP_K:-5}"
MAX_WORKERS="${MAX_WORKERS:-10}"
SKELETON_MODE="${SKELETON_MODE:-oracle}"
REWRITE_MODES="${REWRITE_MODES:-original,template,rule_based,llm}"
RETRIEVAL_MODES="${RETRIEVAL_MODES:-lexical,dense,hybrid}"
SCORER_MODES="${SCORER_MODES:-plain,relation_driven}"
LLM_REWRITE_MAX_TOKENS="${LLM_REWRITE_MAX_TOKENS:-640}"
LLM_REWRITE_TEMPERATURE="${LLM_REWRITE_TEMPERATURE:-0.0}"

if [[ ! -f "/home/huang/miniconda3/etc/profile.d/conda.sh" ]]; then
  echo "conda init script not found: /home/huang/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

source /home/huang/miniconda3/etc/profile.d/conda.sh
conda activate paper_benchmark

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is not set. Export it before running this script." >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT/src"
export GENERATOR_BACKEND="llm"
export GENERATOR_PROVIDER="dashscope"
export GENERATOR_API_KEY="$DASHSCOPE_API_KEY"
export GENERATOR_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export GENERATOR_MODEL="qwen2.5-7b-instruct-1m"
export DASHSCOPE_MODEL="qwen2.5-7b-instruct-1m"
export DASHSCOPE_BASE_URL="${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export GENERATOR_JSON_MODE="true"
export GENERATOR_TEMPERATURE="${GENERATOR_TEMPERATURE:-0.0}"

echo "Running retrieval lab on full dataset"
echo "  dataset: $DATASET_PATH"
echo "  sections: $SECTIONS_PATH"
echo "  knowledge: $KNOWLEDGE_PATH"
echo "  output_dir: $OUTPUT_DIR"
echo "  rewrite_modes: $REWRITE_MODES"
echo "  retrieval_modes: $RETRIEVAL_MODES"
echo "  scorer_modes: $SCORER_MODES"
echo "  llm_model: $GENERATOR_MODEL"
echo "  llm_base_url: $GENERATOR_BASE_URL"

python "$PROJECT_ROOT/scripts/run_ours_retrieval_lab.py" \
  --dataset "$DATASET_PATH" \
  --sections "$SECTIONS_PATH" \
  --knowledge "$KNOWLEDGE_PATH" \
  --top-k "$TOP_K" \
  --max-workers "$MAX_WORKERS" \
  --skeleton-mode "$SKELETON_MODE" \
  --rewrite-modes "$REWRITE_MODES" \
  --retrieval-modes "$RETRIEVAL_MODES" \
  --scorer-modes "$SCORER_MODES" \
  --output-dir "$OUTPUT_DIR" \
  --overwrite \
  --llm-rewrite-max-tokens "$LLM_REWRITE_MAX_TOKENS" \
  --llm-rewrite-temperature "$LLM_REWRITE_TEMPERATURE"
