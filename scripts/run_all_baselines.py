from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

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
from pipelines.base import PublicIndex
from pipelines.graph_enhanced_rag import (
    GraphEnhancedRAGConfig,
    GraphEnhancedRAGPipeline,
    build_graph_retriever_config,
    summarize_graph_retrieval,
)
from pipelines.kbqa_baseline import KBQABaselineConfig, KBQABaselinePipeline
from pipelines.ours_ch4 import OursCh4Config, OursCh4Pipeline, summarize_ablation
from pipelines.rewrite_rag import RewriteRAGConfig, RewriteRAGPipeline, summarize_rewrite_improvements
from pipelines.traditional_rag import TraditionalRAGConfig, TraditionalRAGPipeline
from retrievers import build_reranker, build_text_retriever
from retrievers.graph_retriever import GraphRetriever
from runtime_config import build_trace_metadata, load_runtime_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all baseline pipelines and collect outputs in one directory.")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/two_file_demo/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed-top-k", type=int, default=5)
    parser.add_argument("--expand-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rewrite-mode", choices=["naive", "entity_relation"], default="entity_relation")
    parser.add_argument("--graph-hint-mode", choices=["gold", "none"], default="gold")
    parser.add_argument("--kbqa-entity-mode", choices=["gold", "heuristic"], default="gold")
    parser.add_argument("--kbqa-relation-mode", choices=["gold", "heuristic"], default="gold")
    parser.add_argument("--skeleton-mode", choices=["oracle", "stub_predicted"], default="oracle")
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--include-ablation", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/all_baselines"))
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
    answer = metrics["answer"]
    retrieval = metrics["retrieval"]
    entry = {
        "method_name": metrics["method_name"],
        "sample_count": metrics["sample_count"],
        "answer": answer,
        "retrieval": retrieval,
    }
    if extras:
        entry.update(extras)
    return entry


def _existing_jsonl_examples(base_dir: Path, *, pattern: str = "*.jsonl") -> list[str]:
    if not base_dir.exists():
        return []
    return sorted(str(path) for path in base_dir.rglob(pattern))[:10]


def _validate_input_path(path: Path, label: str, *, search_root: Path) -> None:
    if path.exists():
        return
    message = [f"{label} file not found: {path}"]
    examples = _existing_jsonl_examples(search_root, pattern=path.name)
    if not examples:
        examples = _existing_jsonl_examples(search_root)
    if examples:
        message.append(f"Available {search_root} examples:")
        message.extend(f"  - {item}" for item in examples)
    raise FileNotFoundError("\n".join(message))


def _run_pipeline_with_progress(samples: list[Any], pipeline: Any, description: str) -> list[Any]:
    records = []
    for sample in tqdm(samples, desc=description):
        records.append(pipeline.run(sample))
    return records


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _validate_input_path(args.dataset, "Dataset", search_root=PROJECT_ROOT / "outputs")
    _validate_input_path(args.sections, "Sections", search_root=PROJECT_ROOT / "artifacts")
    _validate_input_path(args.knowledge, "Knowledge", search_root=PROJECT_ROOT / "artifacts")

    samples = load_benchmark_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    settings = load_runtime_settings()
    text_index = PublicIndex.from_markdown_sections(args.sections)
    graph_records = load_graph_section_records(args.knowledge)
    graph_index = GraphIndex.build(graph_records)
    rerank_enabled = (not args.disable_rerank) and settings.rerank.enabled
    text_retriever = build_text_retriever(index=text_index, settings=settings)
    reranker = None if args.disable_rerank else build_reranker(settings)
    generator = build_generator(settings)
    trace_metadata = build_trace_metadata(settings)

    summary: dict[str, Any] = {
        "dataset": str(args.dataset),
        "sections": str(args.sections),
        "knowledge": str(args.knowledge),
        "sample_count": len(samples),
        "top_k": args.top_k,
        "pipelines": {},
    }

    traditional = TraditionalRAGPipeline(
        index=text_index,
        retriever=text_retriever,
        reranker=reranker,
        generator=generator,
        config=TraditionalRAGConfig(
            top_k=args.top_k,
            rerank=rerank_enabled,
            rerank_top_n=settings.rerank.top_n,
            retrieval_mode=settings.retrieval.mode,
            trace_metadata=trace_metadata,
        ),
    )
    traditional_records = _run_pipeline_with_progress(samples, traditional, "Traditional RAG")
    traditional_metrics = {
        "method_name": traditional.method_name,
        "sample_count": len(traditional_records),
        "top_k": args.top_k,
        "answer": aggregate_answer_metrics(traditional_records),
        "retrieval": aggregate_retrieval_metrics(traditional_records, k=args.top_k),
    }
    traditional_outputs = _write_outputs(traditional_records, traditional_metrics, args.output_dir / "traditional_rag")
    summary["pipelines"]["traditional_rag"] = _build_summary_entry(
        traditional_metrics,
        {"output_dir": str(args.output_dir / "traditional_rag"), **traditional_outputs},
    )

    rewrite = RewriteRAGPipeline(
        index=text_index,
        retriever=text_retriever,
        reranker=reranker,
        generator=generator,
        config=RewriteRAGConfig(
            top_k=args.top_k,
            rerank=rerank_enabled,
            rerank_top_n=settings.rerank.top_n,
            mode=args.rewrite_mode,
            retrieval_mode=settings.retrieval.mode,
            trace_metadata=trace_metadata,
        ),
    )
    rewrite_records = _run_pipeline_with_progress(samples, rewrite, "Rewrite-RAG")
    rewrite_metrics = {
        "method_name": rewrite.method_name,
        "sample_count": len(rewrite_records),
        "top_k": args.top_k,
        "rewrite_mode": args.rewrite_mode,
        "answer": aggregate_answer_metrics(rewrite_records),
        "retrieval": aggregate_retrieval_metrics(rewrite_records, k=args.top_k),
        "rewrite_effect": summarize_rewrite_improvements(rewrite_records),
    }
    rewrite_outputs = _write_outputs(rewrite_records, rewrite_metrics, args.output_dir / "rewrite_rag")
    summary["pipelines"]["rewrite_rag"] = _build_summary_entry(
        rewrite_metrics,
        {
            "rewrite_mode": args.rewrite_mode,
            "rewrite_effect": rewrite_metrics["rewrite_effect"],
            "output_dir": str(args.output_dir / "rewrite_rag"),
            **rewrite_outputs,
        },
    )

    graph_config = GraphEnhancedRAGConfig(
        seed_top_k=args.seed_top_k,
        expand_k=args.expand_k,
        final_top_k=args.top_k,
        rerank=rerank_enabled,
        rerank_top_n=settings.rerank.top_n,
        use_gold_hints=args.graph_hint_mode == "gold",
        retrieval_mode=settings.retrieval.mode,
        trace_metadata=trace_metadata,
    )
    graph_retriever = GraphRetriever.from_paths(
        sections_path=args.sections,
        knowledge_path=args.knowledge,
        seed_retriever=text_retriever,
        config=build_graph_retriever_config(graph_config),
    )
    graph_pipeline = GraphEnhancedRAGPipeline(
        graph_retriever,
        config=graph_config,
        reranker=reranker,
        generator=generator,
    )
    graph_records_out = _run_pipeline_with_progress(samples, graph_pipeline, "Graph-enhanced RAG")
    graph_metrics = {
        "method_name": graph_pipeline.method_name,
        "sample_count": len(graph_records_out),
        "top_k": args.top_k,
        "seed_top_k": args.seed_top_k,
        "expand_k": args.expand_k,
        "hint_mode": args.graph_hint_mode,
        "answer": aggregate_answer_metrics(graph_records_out),
        "retrieval": aggregate_retrieval_metrics(graph_records_out, k=args.top_k),
        "graph": summarize_graph_retrieval(graph_records_out),
    }
    graph_outputs = _write_outputs(graph_records_out, graph_metrics, args.output_dir / "graph_enhanced_rag")
    summary["pipelines"]["graph_enhanced_rag"] = _build_summary_entry(
        graph_metrics,
        {
            "hint_mode": args.graph_hint_mode,
            "graph": graph_metrics["graph"],
            "output_dir": str(args.output_dir / "graph_enhanced_rag"),
            **graph_outputs,
        },
    )

    kbqa = KBQABaselinePipeline(
        EntityLinker(graph_index),
        RelationMatcher(graph_index),
        KBExecutor(graph_index, text_index),
        config=KBQABaselineConfig(
            top_k=args.top_k,
            entity_mode=args.kbqa_entity_mode,
            relation_mode=args.kbqa_relation_mode,
        ),
    )
    kbqa_records = _run_pipeline_with_progress(samples, kbqa, "KBQA baseline")
    kbqa_metrics = {
        "method_name": kbqa.method_name,
        "sample_count": len(kbqa_records),
        "top_k": args.top_k,
        "entity_mode": args.kbqa_entity_mode,
        "relation_mode": args.kbqa_relation_mode,
        "answer": aggregate_answer_metrics(kbqa_records),
        "retrieval": aggregate_retrieval_metrics(kbqa_records, k=args.top_k),
    }
    kbqa_outputs = _write_outputs(kbqa_records, kbqa_metrics, args.output_dir / "kbqa_baseline")
    summary["pipelines"]["kbqa_baseline"] = _build_summary_entry(
        kbqa_metrics,
        {
            "entity_mode": args.kbqa_entity_mode,
            "relation_mode": args.kbqa_relation_mode,
            "output_dir": str(args.output_dir / "kbqa_baseline"),
            **kbqa_outputs,
        },
    )

    ours = OursCh4Pipeline(
        text_index,
        SkeletonExtractor(graph_index),
        RelationDrivenRetriever(text_index, graph_index, text_retriever=text_retriever),
        TextCompensator(text_index, graph_index, text_retriever=text_retriever),
        reranker=reranker,
        generator=generator,
        config=OursCh4Config(
            top_k=args.top_k,
            skeleton_mode=args.skeleton_mode,
            rerank=rerank_enabled,
            rerank_top_n=settings.rerank.top_n,
            retrieval_mode=settings.retrieval.mode,
            trace_metadata=trace_metadata,
        ),
    )
    ours_records = _run_pipeline_with_progress(samples, ours, "Ours-Ch4")
    ours_metrics = {
        "method_name": ours.method_name,
        "sample_count": len(ours_records),
        "top_k": args.top_k,
        "skeleton_mode": args.skeleton_mode,
        "answer": aggregate_answer_metrics(ours_records),
        "retrieval": aggregate_retrieval_metrics(ours_records, k=args.top_k),
        "ours": summarize_ablation(ours_records),
    }
    ours_outputs = _write_outputs(ours_records, ours_metrics, args.output_dir / "ours_ch4")
    summary["pipelines"]["ours_ch4"] = _build_summary_entry(
        ours_metrics,
        {
            "skeleton_mode": args.skeleton_mode,
            "ours": ours_metrics["ours"],
            "output_dir": str(args.output_dir / "ours_ch4"),
            **ours_outputs,
        },
    )

    if args.include_ablation:
        from pipelines.ours_ch4 import OursCh4Config as AblationConfig

        ablations = ["full", "w/o_relation_driven", "w/o_skeleton_rewrite", "w/o_text_compensation"]
        comparison = []
        ablation_dir = args.output_dir / "ablation"
        ablation_dir.mkdir(parents=True, exist_ok=True)
        for mode in ablations:
            config = AblationConfig.from_ablation(mode, top_k=args.top_k, skeleton_mode=args.skeleton_mode)
            config.retrieval_mode = settings.retrieval.mode
            config.rerank = rerank_enabled
            config.rerank_top_n = settings.rerank.top_n
            config.trace_metadata = trace_metadata
            pipeline = OursCh4Pipeline(
                text_index,
                SkeletonExtractor(graph_index),
                RelationDrivenRetriever(text_index, graph_index, text_retriever=text_retriever),
                TextCompensator(text_index, graph_index, text_retriever=text_retriever),
                config=config,
                reranker=reranker,
                generator=generator,
            )
            records = _run_pipeline_with_progress(samples, pipeline, f"Ablation {mode}")
            metrics = {
                "mode": mode,
                "retrieval_mode": settings.retrieval.mode,
                "use_rerank": rerank_enabled,
                "answer": aggregate_answer_metrics(records),
                "retrieval": aggregate_retrieval_metrics(records, k=args.top_k),
                "ours": summarize_ablation(records),
            }
            comparison.append(metrics)
            variant_dir = ablation_dir / mode.replace("/", "_")
            _write_outputs(records, metrics, variant_dir)
        comparison_path = ablation_dir / "comparison.json"
        with comparison_path.open("w", encoding="utf-8") as handle:
            json.dump(comparison, handle, ensure_ascii=False, indent=2)
        summary["pipelines"]["ablation"] = {
            "output_dir": str(ablation_dir),
            "comparison": str(comparison_path),
            "variants": len(comparison),
        }

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
