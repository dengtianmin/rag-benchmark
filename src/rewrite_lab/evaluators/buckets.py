from __future__ import annotations

from rewrite_lab.schema import RewriteCandidate


def assign_bucket_tags(
    candidate: RewriteCandidate,
    intrinsic_metrics: dict[str, float | bool],
    retrieval_metrics: dict[str, float | int | bool | None],
) -> list[str]:
    buckets: list[str] = []
    if retrieval_metrics.get("win_rate") == 1.0:
        buckets.append("win")
    if retrieval_metrics.get("loss_rate") == 1.0:
        buckets.append("loss")

    original_rank = retrieval_metrics.get("original_rank_of_first_gold")
    rewritten_rank = retrieval_metrics.get("rewritten_rank_of_first_gold")
    original_hit = bool(retrieval_metrics.get(next((key for key in retrieval_metrics if key.startswith("original_hit@")), ""), 0.0))
    rewritten_hit = bool(retrieval_metrics.get(next((key for key in retrieval_metrics if key.startswith("rewritten_hit@")), ""), 0.0))

    if original_hit and rewritten_hit and isinstance(original_rank, int) and isinstance(rewritten_rank, int):
        if rewritten_rank < original_rank:
            buckets.append("same_hit_rank_up")
        elif rewritten_rank > original_rank:
            buckets.append("same_hit_rank_down")
    if not original_hit and not rewritten_hit:
        buckets.append("same_miss")

    if "skip_worthy" in candidate.flags:
        buckets.append("skip_worthy")
    if "entity_dropped" in candidate.flags or float(intrinsic_metrics.get("entity_preservation_rate", 1.0)) < 1.0:
        buckets.append("entity_dropped")
    if float(intrinsic_metrics.get("noise_rate", 0.0)) >= 0.5:
        buckets.append("noisy_rewrite")
    if float(intrinsic_metrics.get("compression_ratio", 1.0)) > 1.2 or float(intrinsic_metrics.get("rewrite_length", 0.0)) >= 16.0:
        buckets.append("overlong_rewrite")
    return buckets


def summarize_buckets(bucket_rows: list[list[str]]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for row in bucket_rows:
        for tag in row:
            summary[tag] = summary.get(tag, 0) + 1
    return dict(sorted(summary.items()))
