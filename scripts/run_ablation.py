from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataio.loaders import load_benchmark_samples, load_graph_section_records
from evaluation.answer_metrics import aggregate_answer_metrics
from evaluation.retrieval_metrics import aggregate_retrieval_metrics
from generators import build_generator
from modules.graph_expander import GraphIndex
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractor
from modules.text_compensator import TextCompensator
from parallel_runner import run_samples
from pipelines.base import PublicIndex
from pipelines.ours_ch4 import OursCh4Config, OursCh4Pipeline, summarize_ablation
from retrievers import build_reranker, build_text_retriever
from runtime_config import build_trace_metadata, load_runtime_settings


ABLATIONS = ["full", "w/o_relation_driven", "w/o_skeleton_rewrite", "w/o_text_compensation"]


def _build_llm_client(settings):
    from clients.chat_llm_client import ChatLLMClient

    return ChatLLMClient(
        api_key=settings.generator.api_key,
        base_url=settings.generator.base_url,
        model=settings.generator.model,
        timeout=settings.generator.timeout,
        max_tokens=settings.generator.max_tokens,
        temperature=settings.generator.temperature,
        json_mode=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Ours-Ch4 ablations.")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/two_file_demo/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skeleton-mode", choices=["oracle", "stub_predicted", "llm_predicted"], default="oracle")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/ablation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_runtime_settings()
    samples = load_benchmark_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    text_index = PublicIndex.from_markdown_sections(args.sections)
    graph_index = GraphIndex.build(load_graph_section_records(args.knowledge))
    shared_text_retriever = build_text_retriever(index=text_index, settings=settings)
    llm_client = _build_llm_client(settings) if args.skeleton_mode == "llm_predicted" else None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    comparison = []
    for mode in tqdm(ABLATIONS, desc="Ablation modes"):
        config = OursCh4Config.from_ablation(mode, top_k=args.top_k, skeleton_mode=args.skeleton_mode)
        config.retrieval_mode = settings.retrieval.mode
        config.rerank = settings.rerank.enabled
        config.rerank_top_n = settings.rerank.top_n
        config.trace_metadata = build_trace_metadata(settings)

        def build_pipeline() -> OursCh4Pipeline:
            reranker = build_reranker(settings) if settings.rerank.enabled else None
            generator = build_generator(settings)
            return OursCh4Pipeline(
                text_index,
                SkeletonExtractor(graph_index, llm_client=llm_client),
                RelationDrivenRetriever(text_index, graph_index, text_retriever=shared_text_retriever),
                TextCompensator(text_index, graph_index, text_retriever=shared_text_retriever),
                config=config,
                reranker=reranker,
                generator=generator,
            )

        records = run_samples(
            samples,
            build_pipeline=build_pipeline,
            run_sample=lambda pipeline, sample: pipeline.run(sample),
            description=f"Ablation {mode}",
            max_workers=args.max_workers,
        )
        metrics = {
            "mode": mode,
            "retrieval_mode": settings.retrieval.mode,
            "use_rerank": settings.rerank.enabled,
            "max_workers": args.max_workers,
            "answer": aggregate_answer_metrics(records),
            "retrieval": aggregate_retrieval_metrics(records, k=args.top_k),
            "ours": summarize_ablation(records),
        }
        comparison.append(
            {
                "mode": mode,
                "retrieval_mode": settings.retrieval.mode,
                "use_rerank": settings.rerank.enabled,
                "em": metrics["answer"]["em"],
                "token_f1": metrics["answer"]["token_f1"],
                f"hit@{args.top_k}": metrics["retrieval"][f"hit@{args.top_k}"],
                "mrr": metrics["retrieval"]["mrr"],
                "text_compensation_activation_rate": metrics["ours"]["text_compensation_activation_rate"],
            }
        )
        variant_dir = args.output_dir / mode.replace("/", "_")
        variant_dir.mkdir(parents=True, exist_ok=True)
        with (variant_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
        with (variant_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record.to_output_dict(), ensure_ascii=False) + "\n")

    comparison_path = args.output_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as handle:
        json.dump(comparison, handle, ensure_ascii=False, indent=2)
    print(f"comparison: {comparison_path}")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
