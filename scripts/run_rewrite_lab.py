from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipelines.base import PublicIndex
from retrievers.text_retriever import LexicalTextRetriever, build_reranker
from rewrite_lab.dataset import load_lab_dataset
from rewrite_lab.evaluators.buckets import assign_bucket_tags, summarize_buckets
from rewrite_lab.evaluators.intrinsic import aggregate_intrinsic_metrics, evaluate_intrinsic_metrics
from rewrite_lab.evaluators.retrieval import aggregate_retrieval_metrics, evaluate_retrieval_pair
from rewrite_lab.reporting.writer import write_report_bundle
from rewrite_lab.rewriters import NaiveCurrentAdapter, RuleBasedV1Rewriter, SkipBaselineRewriter
from rewrite_lab.schema import RewriteLabRecord, RewriteLabSummary
from rewrite_lab.utils import question_type_label
from runtime_config import load_runtime_settings


LOGGER = logging.getLogger("rewrite_lab")


def _bool_flag(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _build_rewriters(names: list[str]) -> dict[str, object]:
    registry = {
        "naive_current_adapter": NaiveCurrentAdapter,
        "skip_baseline": SkipBaselineRewriter,
        "rule_based_v1": RuleBasedV1Rewriter,
    }
    resolved = {}
    for name in names:
        key = name.strip()
        if key not in registry:
            raise ValueError(f"Unsupported rewriter: {key}")
        resolved[key] = registry[key]()
    return resolved


def _aggregate_question_types(records: list[RewriteLabRecord], top_k: int) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RewriteLabRecord]] = defaultdict(list)
    for record in records:
        grouped[record.question_type or "unknown"].append(record)
    payload: dict[str, dict[str, float]] = {}
    for question_type, rows in grouped.items():
        payload[question_type] = {
            "sample_count": float(len(rows)),
            "rewritten_hit_rate": sum(float(row.retrieval_metrics.get(f"rewritten_hit@{top_k}", 0.0)) for row in rows) / len(rows),
            "win_rate": sum(float(row.retrieval_metrics.get("win_rate", 0.0)) for row in rows) / len(rows),
            "loss_rate": sum(float(row.retrieval_metrics.get("loss_rate", 0.0)) for row in rows) / len(rows),
        }
    return payload


def _make_retriever(sections_path: str):
    index = PublicIndex.from_markdown_sections(sections_path)
    lexical = LexicalTextRetriever(index)
    return lexical.retrieve


def run_rewrite_lab(
    *,
    dataset: str,
    sections: str,
    knowledge: str | None = None,
    rewriters: list[str],
    top_k: int,
    use_rerank: bool,
    output_dir: str,
    max_samples: int | None = None,
    question_types: list[str] | None = None,
    write_html_report: bool = False,
) -> Path:
    del knowledge
    settings = load_runtime_settings(
        overrides={
            "rerank": {"enabled": use_rerank, "use_rerank": use_rerank, "backend": "mock" if not use_rerank else "mock"}
        }
    )
    reranker = build_reranker(settings, logger=LOGGER) if use_rerank else None
    retrieve = _make_retriever(sections)
    samples = load_lab_dataset(
        dataset,
        max_samples=max_samples,
        question_types=set(question_types or []),
    )
    if not samples:
        raise ValueError("No samples found for Rewrite Lab. Check dataset path and question-type filters.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rewriter_map = _build_rewriters(rewriters)

    all_candidates = []
    all_records = []
    summaries = []

    for strategy_name, rewriter in rewriter_map.items():
        LOGGER.info("Evaluating rewriter=%s on %s sample(s).", strategy_name, len(samples))
        records: list[RewriteLabRecord] = []
        candidates = rewriter.batch_rewrite(samples)
        for sample, candidate in zip(samples, candidates, strict=True):
            intrinsic_metrics = evaluate_intrinsic_metrics(sample, candidate)
            original_snapshot, rewritten_snapshot, retrieval_metrics = evaluate_retrieval_pair(
                sample,
                original_query=sample.question,
                rewritten_query=candidate.rewritten_query,
                retrieve=retrieve,
                top_k=top_k,
                rerank=(lambda query, docs, top_n: reranker.rerank(query, docs, top_n=top_n)) if reranker else None,
                rerank_top_n=top_k,
            )
            bucket_tags = assign_bucket_tags(candidate, intrinsic_metrics, retrieval_metrics)
            record = RewriteLabRecord(
                question_id=sample.question_id,
                strategy_name=strategy_name,
                question_type=question_type_label(sample.question_type),
                original_question=sample.question,
                rewritten_query=candidate.rewritten_query,
                flags=candidate.flags,
                bucket_tags=bucket_tags,
                intrinsic_metrics=intrinsic_metrics,
                retrieval_metrics=retrieval_metrics,
                original_retrieval=original_snapshot,
                rewritten_retrieval=rewritten_snapshot,
                notes=candidate.notes,
            )
            records.append(record)
        strategy_dir = output_path / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        write_report_bundle(
            output_dir=strategy_dir,
            candidates=candidates,
            records=records,
            summaries=[],
            write_html=write_html_report,
        )
        combined_metrics = {}
        combined_metrics.update(aggregate_intrinsic_metrics([record.intrinsic_metrics for record in records]))
        combined_metrics.update(aggregate_retrieval_metrics([record.retrieval_metrics for record in records]))
        summary = RewriteLabSummary(
            strategy_name=strategy_name,
            sample_count=len(records),
            metrics=combined_metrics,
            bucket_summary=summarize_buckets([record.bucket_tags for record in records]),
            by_question_type=_aggregate_question_types(records, top_k),
        )
        write_report_bundle(
            output_dir=strategy_dir,
            candidates=candidates,
            records=records,
            summaries=[summary],
            write_html=write_html_report,
        )
        all_candidates.extend(candidates)
        all_records.extend(records)
        summaries.append(summary)

    write_report_bundle(
        output_dir=output_path,
        candidates=all_candidates,
        records=all_records,
        summaries=summaries,
        write_html=write_html_report,
    )
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Rewrite Lab for offline query rewrite evaluation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--sections", required=True)
    parser.add_argument("--knowledge", default=None)
    parser.add_argument("--rewriters", default="naive_current_adapter,skip_baseline,rule_based_v1")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-rerank", default="false")
    parser.add_argument("--output-dir", default="outputs/rewrite_lab/default")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--question-types", default="")
    parser.add_argument("--write-html-report", default="false")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    output_path = run_rewrite_lab(
        dataset=args.dataset,
        sections=args.sections,
        knowledge=args.knowledge,
        rewriters=[item.strip() for item in args.rewriters.split(",") if item.strip()],
        top_k=args.top_k,
        use_rerank=_bool_flag(args.use_rerank),
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        question_types=[item.strip() for item in args.question_types.split(",") if item.strip()],
        write_html_report=_bool_flag(args.write_html_report),
    )
    LOGGER.info("Rewrite Lab outputs written to %s", output_path)


if __name__ == "__main__":
    main()
