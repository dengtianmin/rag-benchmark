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
from generators import build_generator
from parallel_runner import run_samples
from pipelines.base import PublicIndex
from pipelines.graph_enhanced_rag import (
    GraphEnhancedRAGConfig,
    GraphEnhancedRAGPipeline,
    build_graph_retriever_config,
    summarize_graph_retrieval,
)
from retrievers import build_reranker, build_text_retriever
from retrievers.graph_retriever import GraphRetriever
from runtime_config import build_trace_metadata, load_runtime_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Graph-enhanced RAG baseline.")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/two_file_demo/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=5, help="Final kept top-k.")
    parser.add_argument("--seed-top-k", type=int, default=5)
    parser.add_argument("--expand-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--hint-mode", choices=["gold", "none"], default="gold")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/graph_enhanced_rag"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_runtime_settings()
    samples = load_benchmark_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    seed_index = PublicIndex.from_markdown_sections(args.sections)
    shared_seed_retriever = build_text_retriever(index=seed_index, settings=settings)

    def build_pipeline() -> GraphEnhancedRAGPipeline:
        pipeline_config = GraphEnhancedRAGConfig(
            seed_top_k=args.seed_top_k,
            expand_k=args.expand_k,
            final_top_k=args.top_k,
            rerank=(not args.disable_rerank) and settings.rerank.enabled,
            rerank_top_n=settings.rerank.top_n,
            use_gold_hints=args.hint_mode == "gold",
            retrieval_mode=settings.retrieval.mode,
            trace_metadata=build_trace_metadata(settings),
        )
        retriever = GraphRetriever.from_paths(
            sections_path=args.sections,
            knowledge_path=args.knowledge,
            seed_retriever=shared_seed_retriever,
            config=build_graph_retriever_config(pipeline_config),
        )
        reranker = None if args.disable_rerank else build_reranker(settings)
        generator = build_generator(settings)
        return GraphEnhancedRAGPipeline(retriever, config=pipeline_config, reranker=reranker, generator=generator)

    records = run_samples(
        samples,
        build_pipeline=build_pipeline,
        run_sample=lambda pipeline, sample: pipeline.run(sample),
        description="Graph-enhanced RAG",
        max_workers=args.max_workers,
    )

    metrics = {
        "method_name": GraphEnhancedRAGPipeline.method_name,
        "sample_count": len(records),
        "top_k": args.top_k,
        "seed_top_k": args.seed_top_k,
        "expand_k": args.expand_k,
        "hint_mode": args.hint_mode,
        "max_workers": args.max_workers,
        "answer": aggregate_answer_metrics(records),
        "retrieval": aggregate_retrieval_metrics(records, k=args.top_k),
        "graph": summarize_graph_retrieval(records),
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
