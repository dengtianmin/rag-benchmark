from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataio.loaders import load_benchmark_samples, load_graph_section_records
from evaluation.answer_metrics import aggregate_answer_metrics
from evaluation.retrieval_metrics import aggregate_retrieval_metrics
from modules.graph_expander import GraphIndex
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractor
from modules.text_compensator import TextCompensator
from pipelines.base import PublicIndex
from pipelines.ours_ch4 import OursCh4Config, OursCh4Pipeline, summarize_ablation
from retrievers import build_reranker, build_text_retriever
from runtime_config import build_trace_metadata, load_runtime_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ours-Ch4 pipeline.")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/two_file_demo/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skeleton-mode", choices=["oracle", "stub_predicted"], default="oracle")
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/ours_ch4"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_runtime_settings()
    samples = load_benchmark_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    text_index = PublicIndex.from_markdown_sections(args.sections)
    text_retriever = build_text_retriever(index=text_index, settings=settings)
    graph_index = GraphIndex.build(load_graph_section_records(args.knowledge))
    config = OursCh4Config(
        top_k=args.top_k,
        skeleton_mode=args.skeleton_mode,
        rerank=(not args.disable_rerank) and settings.rerank.enabled,
        rerank_top_n=settings.rerank.top_n,
        retrieval_mode=settings.retrieval.mode,
        trace_metadata=build_trace_metadata(settings),
    )
    reranker = None if args.disable_rerank else build_reranker(settings)
    pipeline = OursCh4Pipeline(
        text_index,
        SkeletonExtractor(graph_index),
        RelationDrivenRetriever(text_index, graph_index, text_retriever=text_retriever),
        TextCompensator(text_index, graph_index, text_retriever=text_retriever),
        config=config,
        reranker=reranker,
    )
    records = [pipeline.run(sample) for sample in samples]
    metrics = {
        "method_name": pipeline.method_name,
        "sample_count": len(records),
        "top_k": args.top_k,
        "skeleton_mode": args.skeleton_mode,
        "answer": aggregate_answer_metrics(records),
        "retrieval": aggregate_retrieval_metrics(records, k=args.top_k),
        "ours": summarize_ablation(records),
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
