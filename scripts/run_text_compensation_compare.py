from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):  # type: ignore[no-redef]
        del args, kwargs
        return False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare no/legacy/new text compensation under Ours-Ch4."
    )
    parser.add_argument("--dataset", type=Path, default=Path("outputs/full_run/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/full_run/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/full_run/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--skeleton-mode", choices=["oracle", "stub_predicted", "llm_predicted"], default="oracle")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/text_compensation_compare"))
    parser.add_argument("--generator-provider", default="dashscope")
    parser.add_argument("--generator-model", default="qwen2.5-7b-instruct-1m")
    parser.add_argument("--disable-rerank", action="store_true")
    return parser.parse_args()


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


def _apply_generator_env(args: argparse.Namespace) -> None:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    provider = args.generator_provider.strip().lower()
    model = args.generator_model.strip()

    os.environ["GENERATOR_BACKEND"] = "llm"
    os.environ["GENERATOR_PROVIDER"] = provider
    os.environ["GENERATOR_MODEL"] = model
    os.environ["RETRIEVAL_MODE"] = "hybrid"

    if provider == "dashscope":
        os.environ["DASHSCOPE_MODEL"] = model
        dashscope_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        dashscope_base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip()
        if dashscope_key:
            os.environ["GENERATOR_API_KEY"] = dashscope_key
        if dashscope_base_url:
            os.environ["GENERATOR_BASE_URL"] = dashscope_base_url


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    _validate_input_path(args.dataset, "Dataset", search_root=PROJECT_ROOT / "outputs")
    _validate_input_path(args.sections, "Sections", search_root=PROJECT_ROOT / "artifacts")
    _validate_input_path(args.knowledge, "Knowledge", search_root=PROJECT_ROOT / "artifacts")

    _apply_generator_env(args)
    settings = load_runtime_settings()

    samples = load_benchmark_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    text_index = PublicIndex.from_markdown_sections(args.sections)
    graph_index = GraphIndex.build(load_graph_section_records(args.knowledge))
    shared_text_retriever = build_text_retriever(index=text_index, settings=settings)
    rerank_enabled = (not args.disable_rerank) and settings.rerank.enabled
    trace_metadata = build_trace_metadata(settings)

    variant_specs = [
        {
            "name": "no_text_compensation",
            "use_text_compensation": False,
            "text_compensation_strategy": "legacy_compensation",
        },
        {
            "name": "legacy_compensation",
            "use_text_compensation": True,
            "text_compensation_strategy": "legacy_compensation",
        },
        {
            "name": "new_compensation",
            "use_text_compensation": True,
            "text_compensation_strategy": "new_compensation",
        },
    ]

    summary: dict[str, Any] = {
        "experiment_type": "text_compensation_compare",
        "dataset": str(args.dataset.resolve()),
        "sections": str(args.sections.resolve()),
        "knowledge": str(args.knowledge.resolve()),
        "sample_count": len(samples),
        "top_k": args.top_k,
        "max_workers": args.max_workers,
        "retrieval_mode": "hybrid",
        "scorer_mode": "relation_driven",
        "generator": {
            "backend": settings.generator.backend,
            "provider": settings.generator.provider,
            "model": settings.generator.model,
            "base_url": settings.generator.base_url,
        },
        "variants": {},
    }

    for spec in variant_specs:
        variant_name = spec["name"]

        def build_pipeline() -> OursCh4Pipeline:
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
                    skeleton_mode=args.skeleton_mode,
                    use_relation_driven=True,
                    use_skeleton_rewrite=True,
                    use_text_compensation=spec["use_text_compensation"],
                    text_compensation_strategy=spec["text_compensation_strategy"],
                    rerank=rerank_enabled,
                    rerank_top_n=settings.rerank.top_n,
                    retrieval_mode="hybrid",
                    trace_metadata=trace_metadata,
                ),
            )

        records = run_samples(
            samples,
            build_pipeline=build_pipeline,
            run_sample=lambda pipeline, sample: pipeline.run(sample),
            description=f"Text compensation [{variant_name}]",
            max_workers=args.max_workers,
        )
        metrics = {
            "method_name": OursCh4Pipeline.method_name,
            "variant": variant_name,
            "sample_count": len(records),
            "top_k": args.top_k,
            "skeleton_mode": args.skeleton_mode,
            "max_workers": args.max_workers,
            "retrieval_mode": "hybrid",
            "scorer_mode": "relation_driven",
            "use_text_compensation": spec["use_text_compensation"],
            "text_compensation_strategy": spec["text_compensation_strategy"],
            "generator": {
                "backend": settings.generator.backend,
                "provider": settings.generator.provider,
                "model": settings.generator.model,
                "base_url": settings.generator.base_url,
            },
            "answer": aggregate_answer_metrics(records),
            "retrieval": aggregate_retrieval_metrics(records, k=args.top_k),
            "ours": summarize_ablation(records),
        }
        outputs = _write_outputs(records, metrics, args.output_dir / variant_name)
        summary["variants"][variant_name] = {
            "protocol": {
                "pipeline": "ours_ch4",
                "retrieval_mode": "hybrid",
                "scorer_mode": "relation_driven",
                "skeleton_mode": args.skeleton_mode,
                "use_text_compensation": spec["use_text_compensation"],
                "text_compensation_strategy": spec["text_compensation_strategy"],
            },
            "output_dir": str((args.output_dir / variant_name).resolve()),
            **outputs,
            **metrics,
        }

    summary_path = args.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
