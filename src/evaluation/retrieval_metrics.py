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


def recall_at_k_from_ids(gold_section_ids: set[str], retrieved_section_ids: list[str], k: int) -> float:
    if not gold_section_ids:
        return 0.0
    retrieved = set(retrieved_section_ids[:k])
    return len(gold_section_ids & retrieved) / len(gold_section_ids)


def precision_at_k_from_ids(gold_section_ids: set[str], retrieved_section_ids: list[str], k: int) -> float:
    retrieved = retrieved_section_ids[:k]
    if not retrieved:
        return 0.0
    return len(gold_section_ids & set(retrieved)) / len(retrieved)


def hit_at_k_from_ids(gold_section_ids: set[str], retrieved_section_ids: list[str], k: int) -> float:
    return 1.0 if gold_section_ids & set(retrieved_section_ids[:k]) else 0.0


def reciprocal_rank_from_ids(gold_section_ids: set[str], retrieved_section_ids: list[str]) -> float:
    for rank, section_id in enumerate(retrieved_section_ids, start=1):
        if section_id in gold_section_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval_ids(gold_section_ids: set[str], retrieved_section_ids: list[str], k: int) -> dict[str, float]:
    return {
        f"recall@{k}": recall_at_k_from_ids(gold_section_ids, retrieved_section_ids, k),
        f"precision@{k}": precision_at_k_from_ids(gold_section_ids, retrieved_section_ids, k),
        f"hit@{k}": hit_at_k_from_ids(gold_section_ids, retrieved_section_ids, k),
        "mrr": reciprocal_rank_from_ids(gold_section_ids, retrieved_section_ids),
    }


def aggregate_retrieval_metric_dicts(metrics_list: list[dict[str, float]], k: int) -> dict[str, float]:
    if not metrics_list:
        return {
            f"recall@{k}": 0.0,
            f"precision@{k}": 0.0,
            f"hit@{k}": 0.0,
            "mrr": 0.0,
        }
    return {
        f"recall@{k}": mean([item.get(f"recall@{k}", 0.0) for item in metrics_list]),
        f"precision@{k}": mean([item.get(f"precision@{k}", 0.0) for item in metrics_list]),
        f"hit@{k}": mean([item.get(f"hit@{k}", 0.0) for item in metrics_list]),
        "mrr": mean([item.get("mrr", 0.0) for item in metrics_list]),
    }


def evaluate_retrieval_record(record: PipelineRunRecord, k: int) -> dict[str, float]:
    return evaluate_retrieval_ids(_gold_section_ids(record), _retrieved_section_ids(record), k)


def aggregate_retrieval_metrics(records: list[PipelineRunRecord], k: int) -> dict[str, float]:
    per_record = [evaluate_retrieval_record(record, k) for record in records]
    return aggregate_retrieval_metric_dicts(per_record, k)
