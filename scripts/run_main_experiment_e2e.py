from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataio.loaders import load_benchmark_samples, load_graph_section_records
from evaluation.answer_metrics import aggregate_answer_metrics
from evaluation.retrieval_metrics import aggregate_retrieval_metrics
from generators import build_generator
from modules.entity_linker import EntityLinker
from modules.graph_expander import GraphIndex
from modules.kb_executor import KBExecutor
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.relation_matcher import RelationMatcher
from modules.skeleton_extractor import SkeletonExtractor
from modules.text_compensator import TextCompensator
from parallel_runner import run_samples
from pipelines.base import PublicIndex
from pipelines.graph_enhanced_rag import (
    GraphEnhancedRAGConfig,
    GraphEnhancedRAGPipeline,
    build_graph_retriever_config,
    summarize_graph_retrieval,
)
from pipelines.kbqa_baseline import KBQABaselineConfig, KBQABaselinePipeline
from pipelines.ours_ch4 import OursCh4Config, OursCh4Pipeline, summarize_ablation
from pipelines.traditional_rag import TraditionalRAGConfig, TraditionalRAGPipeline
from retrievers import build_reranker, build_text_retriever
from retrievers.graph_retriever import GraphRetriever
from runtime_config import build_trace_metadata, load_runtime_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end main experiment with no gold/oracle modes."
    )
    parser.add_argument("--dataset", type=Path, default=Path("outputs/full_run/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/full_run/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/full_run/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-top-k", type=int, default=5)
    parser.add_argument("--expand-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/main_experiment_e2e"),
    )
    return parser.parse_args()


def _write_outputs(records: list[Any], metrics: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_output_dict(), ensure_ascii=False) + "\n")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return {"predictions": str(prediction_path), "metrics": str(metrics_path)}


def _build_summary_entry(metrics: dict[str, Any], extras: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = {
        "method_name": metrics["method_name"],
        "sample_count": metrics["sample_count"],
        "answer": metrics["answer"],
        "retrieval": metrics["retrieval"],
    }
    if extras:
        entry.update(extras)
    return entry


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_benchmark_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    settings = load_runtime_settings()
    text_index = PublicIndex.from_markdown_sections(args.sections)
    graph_records = load_graph_section_records(args.knowledge)
    graph_index = GraphIndex.build(graph_records)
    rerank_enabled = (not args.disable_rerank) and settings.rerank.enabled
    trace_metadata = build_trace_metadata(settings)
    shared_text_retriever = build_text_retriever(index=text_index, settings=settings)

    protocol = {
        "traditional_rag": {"rewrite": None, "gold": False},
        "graph_enhanced_rag": {"hint_mode": "none", "gold": False},
        "kbqa_baseline": {"entity_mode": "heuristic", "relation_mode": "heuristic", "gold": False},
        "ours_ch4": {"skeleton_mode": "stub_predicted", "gold": False},
    }
    summary: dict[str, Any] = {
        "dataset": str(args.dataset),
        "sections": str(args.sections),
        "knowledge": str(args.knowledge),
        "sample_count": len(samples),
        "top_k": args.top_k,
        "max_workers": args.max_workers,
        "experiment_type": "main_e2e_no_gold",
        "protocol": protocol,
        "pipelines": {},
    }

    def build_traditional_pipeline() -> TraditionalRAGPipeline:
        return TraditionalRAGPipeline(
            index=text_index,
            retriever=shared_text_retriever,
            reranker=None if args.disable_rerank else build_reranker(settings),
            generator=build_generator(settings),
            config=TraditionalRAGConfig(
                top_k=args.top_k,
                rerank=rerank_enabled,
                rerank_top_n=settings.rerank.top_n,
                retrieval_mode=settings.retrieval.mode,
                trace_metadata=trace_metadata,
            ),
        )

    traditional_records = run_samples(
        samples,
        build_pipeline=build_traditional_pipeline,
        run_sample=lambda pipeline, sample: pipeline.run(sample),
        description="Traditional RAG",
        max_workers=args.max_workers,
    )
    traditional_metrics = {
        "method_name": TraditionalRAGPipeline.method_name,
        "sample_count": len(traditional_records),
        "top_k": args.top_k,
        "max_workers": args.max_workers,
        "answer": aggregate_answer_metrics(traditional_records),
        "retrieval": aggregate_retrieval_metrics(traditional_records, k=args.top_k),
    }
    traditional_outputs = _write_outputs(traditional_records, traditional_metrics, args.output_dir / "traditional_rag")
    summary["pipelines"]["traditional_rag"] = _build_summary_entry(
        traditional_metrics,
        {"protocol": protocol["traditional_rag"], "output_dir": str(args.output_dir / "traditional_rag"), **traditional_outputs},
    )

    def build_graph_pipeline() -> GraphEnhancedRAGPipeline:
        graph_config = GraphEnhancedRAGConfig(
            seed_top_k=args.seed_top_k,
            expand_k=args.expand_k,
            final_top_k=args.top_k,
            rerank=rerank_enabled,
            rerank_top_n=settings.rerank.top_n,
            use_gold_hints=False,
            retrieval_mode=settings.retrieval.mode,
            trace_metadata=trace_metadata,
        )
        graph_retriever = GraphRetriever.from_paths(
            sections_path=args.sections,
            knowledge_path=args.knowledge,
            seed_retriever=shared_text_retriever,
            config=build_graph_retriever_config(graph_config),
        )
        return GraphEnhancedRAGPipeline(
            graph_retriever,
            config=graph_config,
            reranker=None if args.disable_rerank else build_reranker(settings),
            generator=build_generator(settings),
        )

    graph_records_out = run_samples(
        samples,
        build_pipeline=build_graph_pipeline,
        run_sample=lambda pipeline, sample: pipeline.run(sample),
        description="Graph-enhanced RAG",
        max_workers=args.max_workers,
    )
    graph_metrics = {
        "method_name": GraphEnhancedRAGPipeline.method_name,
        "sample_count": len(graph_records_out),
        "top_k": args.top_k,
        "seed_top_k": args.seed_top_k,
        "expand_k": args.expand_k,
        "hint_mode": "none",
        "max_workers": args.max_workers,
        "answer": aggregate_answer_metrics(graph_records_out),
        "retrieval": aggregate_retrieval_metrics(graph_records_out, k=args.top_k),
        "graph": summarize_graph_retrieval(graph_records_out),
    }
    graph_outputs = _write_outputs(graph_records_out, graph_metrics, args.output_dir / "graph_enhanced_rag")
    summary["pipelines"]["graph_enhanced_rag"] = _build_summary_entry(
        graph_metrics,
        {
            "protocol": protocol["graph_enhanced_rag"],
            "hint_mode": "none",
            "graph": graph_metrics["graph"],
            "output_dir": str(args.output_dir / "graph_enhanced_rag"),
            **graph_outputs,
        },
    )

    def build_kbqa_pipeline() -> KBQABaselinePipeline:
        return KBQABaselinePipeline(
            EntityLinker(graph_index),
            RelationMatcher(graph_index),
            KBExecutor(graph_index, text_index),
            config=KBQABaselineConfig(
                top_k=args.top_k,
                entity_mode="heuristic",
                relation_mode="heuristic",
            ),
        )

    kbqa_records = run_samples(
        samples,
        build_pipeline=build_kbqa_pipeline,
        run_sample=lambda pipeline, sample: pipeline.run(sample),
        description="KBQA baseline",
        max_workers=args.max_workers,
    )
    kbqa_metrics = {
        "method_name": KBQABaselinePipeline.method_name,
        "sample_count": len(kbqa_records),
        "top_k": args.top_k,
        "entity_mode": "heuristic",
        "relation_mode": "heuristic",
        "max_workers": args.max_workers,
        "answer": aggregate_answer_metrics(kbqa_records),
        "retrieval": aggregate_retrieval_metrics(kbqa_records, k=args.top_k),
    }
    kbqa_outputs = _write_outputs(kbqa_records, kbqa_metrics, args.output_dir / "kbqa_baseline")
    summary["pipelines"]["kbqa_baseline"] = _build_summary_entry(
        kbqa_metrics,
        {
            "protocol": protocol["kbqa_baseline"],
            "entity_mode": "heuristic",
            "relation_mode": "heuristic",
            "output_dir": str(args.output_dir / "kbqa_baseline"),
            **kbqa_outputs,
        },
    )

    def build_ours_pipeline() -> OursCh4Pipeline:
        return OursCh4Pipeline(
            text_index,
            SkeletonExtractor(graph_index),
            RelationDrivenRetriever(
                text_index,
                graph_index,
                text_retriever=shared_text_retriever,
            ),
            TextCompensator(
                text_index,
                graph_index,
                text_retriever=shared_text_retriever,
            ),
            reranker=None if args.disable_rerank else build_reranker(settings),
            generator=build_generator(settings),
            config=OursCh4Config(
                top_k=args.top_k,
                skeleton_mode="stub_predicted",
                rerank=rerank_enabled,
                rerank_top_n=settings.rerank.top_n,
                retrieval_mode=settings.retrieval.mode,
                trace_metadata=trace_metadata,
            ),
        )

    ours_records = run_samples(
        samples,
        build_pipeline=build_ours_pipeline,
        run_sample=lambda pipeline, sample: pipeline.run(sample),
        description="Ours-Ch4",
        max_workers=args.max_workers,
    )
    ours_metrics = {
        "method_name": OursCh4Pipeline.method_name,
        "sample_count": len(ours_records),
        "top_k": args.top_k,
        "skeleton_mode": "stub_predicted",
        "max_workers": args.max_workers,
        "answer": aggregate_answer_metrics(ours_records),
        "retrieval": aggregate_retrieval_metrics(ours_records, k=args.top_k),
        "ours": summarize_ablation(ours_records),
    }
    ours_outputs = _write_outputs(ours_records, ours_metrics, args.output_dir / "ours_ch4")
    summary["pipelines"]["ours_ch4"] = _build_summary_entry(
        ours_metrics,
        {
            "protocol": protocol["ours_ch4"],
            "skeleton_mode": "stub_predicted",
            "ours": ours_metrics["ours"],
            "output_dir": str(args.output_dir / "ours_ch4"),
            **ours_outputs,
        },
    )

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
