from __future__ import annotations

from collections.abc import Callable

from core.schema import BenchmarkSample, RetrievedDocument

from rewrite_lab.schema import RetrievalSnapshot


RetrieverFn = Callable[[str, int], list[RetrievedDocument]]
RerankerFn = Callable[[str, list[RetrievedDocument], int | None], list[RetrievedDocument]]


def rank_of_first_gold(sample: BenchmarkSample, documents: list[RetrievedDocument]) -> int | None:
    gold = {item.section_id for item in sample.evidence}
    for rank, document in enumerate(documents, start=1):
        if document.section_id in gold:
            return rank
    return None


def _snapshot(sample: BenchmarkSample, query: str, documents: list[RetrievedDocument], top_k: int) -> RetrievalSnapshot:
    gold = {item.section_id for item in sample.evidence}
    section_ids = [document.section_id for document in documents[:top_k]]
    doc_ids = [document.source_id for document in documents[:top_k]]
    hits = gold & set(section_ids)
    first_rank = rank_of_first_gold(sample, documents[:top_k])
    mrr = 0.0 if first_rank is None else 1.0 / first_rank
    recall = len(hits) / len(gold) if gold else 0.0
    return RetrievalSnapshot(
        query=query,
        section_ids=section_ids,
        doc_ids=doc_ids,
        scores=[float(document.score) for document in documents[:top_k]],
        hit_at_k=1.0 if hits else 0.0,
        recall_at_k=recall,
        mrr=mrr,
        rank_of_first_gold=first_rank,
    )


def evaluate_retrieval_pair(
    sample: BenchmarkSample,
    *,
    original_query: str,
    rewritten_query: str,
    retrieve: RetrieverFn,
    top_k: int,
    rerank: RerankerFn | None = None,
    rerank_top_n: int | None = None,
) -> tuple[RetrievalSnapshot, RetrievalSnapshot, dict[str, float | int | bool | None]]:
    original_documents = retrieve(original_query, top_k)
    rewritten_documents = retrieve(rewritten_query, top_k)
    if rerank is not None:
        if original_documents:
            original_documents = rerank(original_query, original_documents, rerank_top_n)
        if rewritten_documents:
            rewritten_documents = rerank(rewritten_query, rewritten_documents, rerank_top_n)

    original = _snapshot(sample, original_query, original_documents, top_k)
    rewritten = _snapshot(sample, rewritten_query, rewritten_documents, top_k)

    original_top = set(original.section_ids)
    rewritten_top = set(rewritten.section_ids)
    intersection = len(original_top & rewritten_top)
    union = len(original_top | rewritten_top)
    candidate_jaccard = intersection / union if union else 1.0

    original_rank = original.rank_of_first_gold or 10**9
    rewritten_rank = rewritten.rank_of_first_gold or 10**9

    metrics: dict[str, float | int | bool | None] = {
        f"original_hit@{top_k}": original.hit_at_k,
        f"rewritten_hit@{top_k}": rewritten.hit_at_k,
        f"original_recall@{top_k}": original.recall_at_k,
        f"rewritten_recall@{top_k}": rewritten.recall_at_k,
        "original_mrr": original.mrr,
        "rewritten_mrr": rewritten.mrr,
        "original_rank_of_first_gold": original.rank_of_first_gold,
        "rewritten_rank_of_first_gold": rewritten.rank_of_first_gold,
        "win_rate": 1.0 if rewritten.hit_at_k and not original.hit_at_k else 0.0,
        "loss_rate": 1.0 if original.hit_at_k and not rewritten.hit_at_k else 0.0,
        "non_inferiority_rate": 1.0 if rewritten_rank <= original_rank else 0.0,
        "top1_preservation_rate": 1.0
        if original.rank_of_first_gold == 1 and rewritten.rank_of_first_gold == 1
        else (0.0 if original.rank_of_first_gold == 1 else 1.0),
        "candidate_jaccard": candidate_jaccard,
    }
    return original, rewritten, metrics


def aggregate_retrieval_metrics(metric_rows: list[dict[str, float | int | bool | None]]) -> dict[str, float]:
    if not metric_rows:
        return {}
    numeric_keys = [key for key, value in metric_rows[0].items() if isinstance(value, (int, float, bool)) or value is None]
    summary: dict[str, float] = {}
    for key in numeric_keys:
        values = [0.0 if row[key] is None else float(row[key]) for row in metric_rows]
        summary[key] = sum(values) / len(values)
    return summary
