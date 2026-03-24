from __future__ import annotations

from core.schema import PipelineRunRecord
from evaluation.common import mean


def _gold_section_ids(record: PipelineRunRecord) -> set[str]:
    return {item.section_id for item in record.sample.evidence}


def _retrieved_section_ids(record: PipelineRunRecord) -> list[str]:
    return [item.section_id for item in record.retrieval.retrieved_documents]


def recall_at_k(record: PipelineRunRecord, k: int) -> float:
    gold = _gold_section_ids(record)
    if not gold:
        return 0.0
    retrieved = set(_retrieved_section_ids(record)[:k])
    return len(gold & retrieved) / len(gold)


def precision_at_k(record: PipelineRunRecord, k: int) -> float:
    retrieved = _retrieved_section_ids(record)[:k]
    if not retrieved:
        return 0.0
    gold = _gold_section_ids(record)
    return len(gold & set(retrieved)) / len(retrieved)


def hit_at_k(record: PipelineRunRecord, k: int) -> float:
    gold = _gold_section_ids(record)
    retrieved = set(_retrieved_section_ids(record)[:k])
    return 1.0 if gold & retrieved else 0.0


def reciprocal_rank(record: PipelineRunRecord) -> float:
    gold = _gold_section_ids(record)
    for rank, section_id in enumerate(_retrieved_section_ids(record), start=1):
        if section_id in gold:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval_record(record: PipelineRunRecord, k: int) -> dict[str, float]:
    return {
        f"recall@{k}": recall_at_k(record, k),
        f"precision@{k}": precision_at_k(record, k),
        f"hit@{k}": hit_at_k(record, k),
        "mrr": reciprocal_rank(record),
    }


def aggregate_retrieval_metrics(records: list[PipelineRunRecord], k: int) -> dict[str, float]:
    per_record = [evaluate_retrieval_record(record, k) for record in records]
    return {
        f"recall@{k}": mean([item[f"recall@{k}"] for item in per_record]),
        f"precision@{k}": mean([item[f"precision@{k}"] for item in per_record]),
        f"hit@{k}": mean([item[f"hit@{k}"] for item in per_record]),
        "mrr": mean([item["mrr"] for item in per_record]),
    }
