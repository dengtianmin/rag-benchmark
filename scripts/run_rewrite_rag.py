from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataio.loaders import load_benchmark_samples
from evaluation.answer_metrics import aggregate_answer_metrics
from evaluation.retrieval_metrics import aggregate_retrieval_metrics
from pipelines.base import PublicIndex
from pipelines.rewrite_rag import RewriteRAGConfig, RewriteRAGPipeline, summarize_rewrite_improvements
from retrievers import build_reranker, build_text_retriever
from runtime_config import build_trace_metadata, load_runtime_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Rewrite-RAG baseline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/two_file_demo/benchmark_dataset.jsonl"),
        help="Path to benchmark_dataset.jsonl",
    )
    parser.add_argument(
        "--sections",
        type=Path,
        default=Path("artifacts/two_file_demo/markdown_sections.jsonl"),
        help="Path to markdown_sections.jsonl",
    )
    parser.add_argument("--mode", choices=["naive", "entity_relation"], default="naive", help="Rewrite mode.")
    parser.add_argument("--top-k", type=int, default=5, help="Retriever top-k.")
    parser.add_argument("--limit", type=int, default=None, help="Only run first n samples.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/rewrite_rag"),
        help="Directory for metrics.json and predictions.jsonl",
    )
    parser.add_argument("--disable-rerank", action="store_true", help="Disable mock reranker.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_runtime_settings()
    samples = load_benchmark_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    index = PublicIndex.from_markdown_sections(args.sections)
    retriever = build_text_retriever(index=index, settings=settings)
    reranker = None if args.disable_rerank else build_reranker(settings)
    pipeline = RewriteRAGPipeline(
        index=index,
        retriever=retriever,
        reranker=reranker,
        config=RewriteRAGConfig(
            top_k=args.top_k,
            rerank=(not args.disable_rerank) and settings.rerank.enabled,
            rerank_top_n=settings.rerank.top_n,
            mode=args.mode,
            retrieval_mode=settings.retrieval.mode,
            trace_metadata=build_trace_metadata(settings),
        ),
    )
    records = [pipeline.run(sample) for sample in samples]
    metrics = {
        "method_name": pipeline.method_name,
        "rewrite_mode": args.mode,
        "sample_count": len(records),
        "top_k": args.top_k,
        "answer": aggregate_answer_metrics(records),
        "retrieval": aggregate_retrieval_metrics(records, k=args.top_k),
        "rewrite_effect": summarize_rewrite_improvements(records),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"

    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_output_dict(), ensure_ascii=False) + "\n")

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(f"predictions: {prediction_path}")
    print(f"metrics: {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
