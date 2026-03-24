from __future__ import annotations

from collections import Counter
import logging

import pandas as pd

from benchmark_builder.config import Settings
from benchmark_builder.models import FinalBenchmarkRecord, QACandidate, QAValidationResult
from benchmark_builder.utils.io_utils import dump_json, load_jsonl_as_models, write_jsonl

logger = logging.getLogger(__name__)


def run_build_dataset(settings: Settings) -> list[FinalBenchmarkRecord]:
    candidates = {item.qid: item for item in load_jsonl_as_models(settings.artifacts_dir / "qa_candidates.jsonl", QACandidate)}
    validations = {
        item.qid: item
        for item in load_jsonl_as_models(settings.artifacts_dir / "qa_validation.jsonl", QAValidationResult)
        if item.is_valid
    }

    records: list[FinalBenchmarkRecord] = []
    for qid, validation in validations.items():
        candidate = candidates.get(qid)
        if not candidate:
            continue
        records.append(
            FinalBenchmarkRecord(
                **candidate.model_dump(),
                support_score=validation.support_score,
                completeness_score=validation.completeness_score,
            )
        )

    jsonl_path = settings.outputs_dir / "benchmark_dataset.jsonl"
    json_path = settings.outputs_dir / "benchmark_dataset.json"
    summary_path = settings.outputs_dir / "benchmark_summary.csv"

    write_jsonl(jsonl_path, records)
    dump_json(json_path, [item.model_dump() for item in records])

    type_counter = Counter(item.question_type for item in records)
    doc_counter = Counter(item.doc_id for item in records)
    scope_counter = Counter(item.source_scope for item in records)
    compensation_counter = Counter(str(item.requires_text_compensation) for item in records)
    avg_evidence = round(sum(len(item.evidence) for item in records) / max(1, len(records)), 4)

    summary_rows = [
        {"metric": "total_samples", "value": len(records)},
        {"metric": "avg_evidence_count", "value": avg_evidence},
    ]
    summary_rows.extend({"metric": f"question_type::{key}", "value": value} for key, value in sorted(type_counter.items()))
    summary_rows.extend({"metric": f"doc_id::{key}", "value": value} for key, value in sorted(doc_counter.items()))
    summary_rows.extend({"metric": f"source_scope::{key}", "value": value} for key, value in sorted(scope_counter.items()))
    summary_rows.extend(
        {"metric": f"requires_text_compensation::{key}", "value": value}
        for key, value in sorted(compensation_counter.items())
    )

    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    logger.info("Built dataset with %s records", len(records))
    return records
