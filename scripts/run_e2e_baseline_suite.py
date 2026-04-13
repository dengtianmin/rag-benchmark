from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from core.schema import PipelineRunRecord
from dataio.loaders import load_benchmark_samples, load_graph_section_records
from evaluation.answer_metrics import aggregate_answer_metrics
from evaluation.retrieval_metrics import aggregate_retrieval_metrics
from generators import build_generator
from clients.chat_llm_client import ChatLLMClient
from modules.entity_linker import EntityLinker
from modules.graph_expander import GraphIndex
from modules.kb_executor import KBExecutor
from modules.query_rewriters import RetrievalLabQueryRewriter
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.relation_matcher import RelationMatcher
from modules.skeleton_extractor import SkeletonExtractor
from modules.text_compensator import TextCompensator
from parallel_runner import run_samples
from pipelines.base import BasePipeline, PublicIndex
from pipelines.graph_enhanced_rag import (
    GraphEnhancedRAGConfig,
    GraphEnhancedRAGPipeline,
    build_graph_retriever_config,
    summarize_graph_retrieval,
)
from pipelines.kbqa_baseline import KBQABaselineConfig, KBQABaselinePipeline
from pipelines.traditional_rag import TraditionalRAGConfig, TraditionalRAGPipeline
from retrievers import build_reranker, build_text_retriever
from retrievers.graph_retriever import GraphRetriever
from runtime_config import build_trace_metadata, load_runtime_settings


@dataclass(slots=True)
class OursVariantSpec:
    name: str
    rewrite_mode: str
    retrieval_mode: str
    scorer_mode: str
    skeleton_mode: str = "stub_predicted"


class OursVariantRunner(BasePipeline):
    method_name = "ours_variant"

    def __init__(
        self,
        *,
        name: str,
        skeleton_extractor: SkeletonExtractor,
        query_rewriter: RetrievalLabQueryRewriter,
        text_retriever: Any,
        relation_driven: RelationDrivenRetriever,
        text_compensator: TextCompensator,
        reranker: Any | None,
        generator: Any,
        top_k: int,
        rerank: bool,
        rerank_top_n: int | None,
        use_text_compensation: bool,
        rewrite_mode: str,
        retrieval_mode: str,
        scorer_mode: str,
        skeleton_mode: str,
        trace_metadata: dict[str, Any],
    ) -> None:
        self.name = name
        self.skeleton_extractor = skeleton_extractor
        self.query_rewriter = query_rewriter
        self.text_retriever = text_retriever
        self.relation_driven = relation_driven
        self.text_compensator = text_compensator
        self.reranker = reranker
        self.generator = generator
        self.top_k = top_k
        self.rerank = rerank
        self.rerank_top_n = rerank_top_n
        self.use_text_compensation = use_text_compensation
        self.rewrite_mode = rewrite_mode
        self.retrieval_mode = retrieval_mode
        self.scorer_mode = scorer_mode
        self.skeleton_mode = skeleton_mode
        self.trace_metadata = trace_metadata

    def run(self, sample) -> PipelineRunRecord:
        skeleton = self.skeleton_extractor.extract(sample, mode=self.skeleton_mode)
        rewrite_result = self.query_rewriter.rewrite(sample, skeleton, mode=self.rewrite_mode)
        rewritten_query = rewrite_result.rewritten_query

        if self.scorer_mode == "plain":
            documents = self.text_retriever.retrieve(rewritten_query, top_k=self.top_k)
            retrieval_trace = {
                "mode": "plain",
                "stage1_query": rewritten_query,
                "query_source": "rewrite_mode",
            }
        elif self.scorer_mode == "relation_driven":
            retrieve_result = self.relation_driven.retrieve(
                sample,
                skeleton,
                top_k=self.top_k,
                use_relation_driven=True,
                use_skeleton_rewrite=False,
                query_override=rewritten_query,
            )
            documents = retrieve_result.documents
            retrieval_trace = retrieve_result.details
        else:
            raise ValueError(f"Unsupported scorer mode: {self.scorer_mode}")

        compensation = self.text_compensator.compensate(
            sample,
            skeleton,
            documents,
            top_k=self.top_k,
            enabled=self.use_text_compensation,
        )
        documents = compensation.documents
        pre_rerank_documents = [document.model_copy() for document in documents]
        if self.rerank and documents:
            # Keep retrieval on the rewritten query, but rerank against the original question
            # to stay aligned with the Ours-Ch4 experiment pipeline.
            documents = self.reranker.rerank(sample.question, documents, top_n=self.rerank_top_n)
        rerank_scores = {
            document.section_id: float(document.metadata.get("rerank_score"))
            for document in documents
            if "rerank_score" in document.metadata
        }
        rerank_trace = self.build_rerank_trace(
            reranker=self.reranker if self.rerank else None,
            before_documents=pre_rerank_documents,
            after_documents=documents,
            top_n=self.rerank_top_n,
        )
        retrieval_result = self.build_retrieval_result(sample.question_id, rewritten_query, documents)
        answer_result = self.generator.generate(sample, documents)
        return PipelineRunRecord(
            question_id=sample.question_id,
            system_name=self.name,
            sample=sample,
            retrieval=retrieval_result,
            answer=answer_result,
            rewritten_query=rewritten_query if rewritten_query != sample.question else None,
            trace={
                "pipeline": "question -> skeleton -> rewrite -> retrieve -> text compensation -> rerank -> generate",
                "variant": self.name,
                "rewrite_mode": self.rewrite_mode,
                "retrieval_mode": self.retrieval_mode,
                "scorer_mode": self.scorer_mode,
                "skeleton_mode": skeleton.mode,
                "skeleton": skeleton.to_trace_dict(),
                "skeleton_details": skeleton.details,
                "rewrite_details": rewrite_result.details,
                **self.trace_metadata,
                "use_rerank": self.rerank,
                "retrieval_trace": retrieval_trace,
                "text_compensation_enabled": self.use_text_compensation,
                "text_compensation_activated": compensation.activated,
                "text_compensation_reason": compensation.reason,
                "text_compensation_details": compensation.details,
                **rerank_trace,
                "rerank_scores": rerank_scores,
                "final_reranked_candidates": [document.section_id for document in documents],
                "final_kept_sections": [document.section_id for document in documents],
                "generator_metadata": answer_result.metadata,
            },
        )


def _build_llm_client(max_tokens: int, temperature: float) -> ChatLLMClient:
    settings = load_runtime_settings()
    return ChatLLMClient(
        api_key=settings.generator.api_key,
        base_url=settings.generator.base_url,
        model=settings.generator.model,
        timeout=settings.generator.timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run e2e baselines plus two configurable Ours variants in one command."
    )
    parser.add_argument("--dataset", type=Path, default=Path("outputs/full_run/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/full_run/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/full_run/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed-top-k", type=int, default=10)
    parser.add_argument("--expand-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--disable-rerank", action="store_true")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--generator-backend", choices=["mock", "llm", "dashscope", "openai"], default="llm")
    parser.add_argument("--generator-provider", default="dashscope")
    parser.add_argument("--generator-model", default="qwen2.5-7b-instruct-1m")
    parser.add_argument("--graph-hint-mode", choices=["none", "gold"], default="none")
    parser.add_argument("--kbqa-entity-mode", choices=["heuristic", "gold"], default="heuristic")
    parser.add_argument("--kbqa-relation-mode", choices=["heuristic", "gold"], default="heuristic")
    parser.add_argument("--ours-main-recall-rewrite-mode", default="template")
    parser.add_argument("--ours-main-recall-retrieval-mode", default="dense")
    parser.add_argument("--ours-main-recall-scorer-mode", choices=["plain", "relation_driven"], default="plain")
    parser.add_argument("--ours-main-rerank-rewrite-mode", default="llm")
    parser.add_argument("--ours-main-rerank-retrieval-mode", default="dense")
    parser.add_argument("--ours-main-rerank-scorer-mode", choices=["plain", "relation_driven"], default="relation_driven")
    parser.add_argument("--ours-use-text-compensation", action="store_true")
    parser.add_argument("--llm-rewrite-max-tokens", type=int, default=640)
    parser.add_argument("--llm-rewrite-temperature", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/experiments/e2e_baseline_suite"),
    )
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
    provider = args.generator_provider.strip().lower()
    os.environ["GENERATOR_BACKEND"] = args.generator_backend
    os.environ["GENERATOR_PROVIDER"] = provider
    os.environ["GENERATOR_MODEL"] = args.generator_model

    if provider == "dashscope":
        os.environ["DASHSCOPE_MODEL"] = args.generator_model
        dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        dashscope_base_url = os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ).strip()
        if dashscope_api_key:
            os.environ["GENERATOR_API_KEY"] = dashscope_api_key
        os.environ["GENERATOR_BASE_URL"] = dashscope_base_url
        return

    if provider == "deepseek":
        deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        deepseek_base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()
        if deepseek_api_key:
            os.environ["GENERATOR_API_KEY"] = deepseek_api_key
        os.environ["GENERATOR_BASE_URL"] = deepseek_base_url


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


def _ours_metrics(
    name: str,
    records: list[PipelineRunRecord],
    *,
    top_k: int,
    max_workers: int,
    spec: OursVariantSpec,
    use_text_compensation: bool,
) -> dict[str, Any]:
    return {
        "method_name": name,
        "sample_count": len(records),
        "top_k": top_k,
        "max_workers": max_workers,
        "answer": aggregate_answer_metrics(records),
        "retrieval": aggregate_retrieval_metrics(records, k=top_k),
        "protocol": {
            "pipeline": "ours_variant",
            "skeleton_mode": spec.skeleton_mode,
            "rewrite_mode": spec.rewrite_mode,
            "retrieval_mode": spec.retrieval_mode,
            "scorer_mode": spec.scorer_mode,
            "use_text_compensation": use_text_compensation,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _validate_input_path(args.dataset, "Dataset", search_root=PROJECT_ROOT / "outputs")
    _validate_input_path(args.sections, "Sections", search_root=PROJECT_ROOT / "artifacts")
    _validate_input_path(args.knowledge, "Knowledge", search_root=PROJECT_ROOT / "artifacts")

    _apply_generator_env(args)
    settings = load_runtime_settings()

    samples = load_benchmark_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]

    text_index = PublicIndex.from_markdown_sections(args.sections)
    graph_records = load_graph_section_records(args.knowledge)
    graph_index = GraphIndex.build(graph_records)
    rerank_enabled = (not args.disable_rerank) and settings.rerank.enabled
    base_trace_metadata = build_trace_metadata(settings)
    shared_text_retriever = build_text_retriever(index=text_index, settings=settings)

    summary: dict[str, Any] = {
        "experiment_type": "e2e_baseline_suite",
        "dataset": str(args.dataset),
        "sections": str(args.sections),
        "knowledge": str(args.knowledge),
        "sample_count": len(samples),
        "top_k": args.top_k,
        "max_workers": args.max_workers,
        "generator": {
            "backend": settings.generator.backend,
            "provider": settings.generator.provider,
            "model": settings.generator.model,
        },
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
                trace_metadata=base_trace_metadata,
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
        {"protocol": {"rewrite": None, "gold": False}, "output_dir": str(args.output_dir / "traditional_rag"), **traditional_outputs},
    )

    def build_graph_pipeline() -> GraphEnhancedRAGPipeline:
        graph_config = GraphEnhancedRAGConfig(
            seed_top_k=args.seed_top_k,
            expand_k=args.expand_k,
            final_top_k=args.top_k,
            rerank=rerank_enabled,
            rerank_top_n=settings.rerank.top_n,
            use_gold_hints=args.graph_hint_mode == "gold",
            retrieval_mode=settings.retrieval.mode,
            trace_metadata=base_trace_metadata,
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
        "hint_mode": args.graph_hint_mode,
        "max_workers": args.max_workers,
        "answer": aggregate_answer_metrics(graph_records_out),
        "retrieval": aggregate_retrieval_metrics(graph_records_out, k=args.top_k),
        "graph": summarize_graph_retrieval(graph_records_out),
    }
    graph_outputs = _write_outputs(graph_records_out, graph_metrics, args.output_dir / "graph_enhanced_rag")
    summary["pipelines"]["graph_enhanced_rag"] = _build_summary_entry(
        graph_metrics,
        {
            "protocol": {"hint_mode": args.graph_hint_mode, "gold": args.graph_hint_mode == "gold"},
            "hint_mode": args.graph_hint_mode,
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
                entity_mode=args.kbqa_entity_mode,
                relation_mode=args.kbqa_relation_mode,
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
        "entity_mode": args.kbqa_entity_mode,
        "relation_mode": args.kbqa_relation_mode,
        "max_workers": args.max_workers,
        "answer": aggregate_answer_metrics(kbqa_records),
        "retrieval": aggregate_retrieval_metrics(kbqa_records, k=args.top_k),
    }
    kbqa_outputs = _write_outputs(kbqa_records, kbqa_metrics, args.output_dir / "kbqa_baseline")
    summary["pipelines"]["kbqa_baseline"] = _build_summary_entry(
        kbqa_metrics,
        {
            "protocol": {"entity_mode": args.kbqa_entity_mode, "relation_mode": args.kbqa_relation_mode},
            "entity_mode": args.kbqa_entity_mode,
            "relation_mode": args.kbqa_relation_mode,
            "output_dir": str(args.output_dir / "kbqa_baseline"),
            **kbqa_outputs,
        },
    )

    llm_client = None
    if "llm" in {args.ours_main_recall_rewrite_mode, args.ours_main_rerank_rewrite_mode}:
        llm_client = _build_llm_client(args.llm_rewrite_max_tokens, args.llm_rewrite_temperature)
    query_rewriter = RetrievalLabQueryRewriter(
        llm_client=llm_client,
        llm_max_tokens=args.llm_rewrite_max_tokens,
        llm_temperature=args.llm_rewrite_temperature,
    )

    ours_specs = [
        OursVariantSpec(
            name="ours_template__dense__plain",
            rewrite_mode=args.ours_main_recall_rewrite_mode,
            retrieval_mode=args.ours_main_recall_retrieval_mode,
            scorer_mode=args.ours_main_recall_scorer_mode,
        ),
        OursVariantSpec(
            name="ours_llm__dense__relation_driven",
            rewrite_mode=args.ours_main_rerank_rewrite_mode,
            retrieval_mode=args.ours_main_rerank_retrieval_mode,
            scorer_mode=args.ours_main_rerank_scorer_mode,
        ),
    ]

    for spec in ours_specs:
        variant_settings = load_runtime_settings()
        variant_settings.retrieval.mode = spec.retrieval_mode
        variant_trace_metadata = build_trace_metadata(variant_settings)
        variant_text_retriever = build_text_retriever(index=text_index, settings=variant_settings)
        variant_relation_driven = RelationDrivenRetriever(
            text_index,
            graph_index,
            text_retriever=variant_text_retriever,
        )
        variant_text_compensator = TextCompensator(
            text_index,
            graph_index,
            text_retriever=variant_text_retriever,
        )

        def build_ours_variant_pipeline(spec: OursVariantSpec = spec) -> OursVariantRunner:
            return OursVariantRunner(
                name=spec.name,
                skeleton_extractor=SkeletonExtractor(graph_index),
                query_rewriter=query_rewriter,
                text_retriever=variant_text_retriever,
                relation_driven=variant_relation_driven,
                text_compensator=variant_text_compensator,
                reranker=None if args.disable_rerank else build_reranker(variant_settings),
                generator=build_generator(settings),
                top_k=args.top_k,
                rerank=rerank_enabled,
                rerank_top_n=settings.rerank.top_n,
                use_text_compensation=args.ours_use_text_compensation,
                rewrite_mode=spec.rewrite_mode,
                retrieval_mode=spec.retrieval_mode,
                scorer_mode=spec.scorer_mode,
                skeleton_mode=spec.skeleton_mode,
                trace_metadata=variant_trace_metadata,
            )

        ours_records = run_samples(
            samples,
            build_pipeline=build_ours_variant_pipeline,
            run_sample=lambda pipeline, sample: pipeline.run(sample),
            description=spec.name,
            max_workers=args.max_workers,
        )
        ours_metrics = _ours_metrics(
            spec.name,
            ours_records,
            top_k=args.top_k,
            max_workers=args.max_workers,
            spec=spec,
            use_text_compensation=args.ours_use_text_compensation,
        )
        ours_outputs = _write_outputs(ours_records, ours_metrics, args.output_dir / spec.name)
        summary["pipelines"][spec.name] = _build_summary_entry(
            ours_metrics,
            {
                "protocol": ours_metrics["protocol"],
                "output_dir": str(args.output_dir / spec.name),
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
