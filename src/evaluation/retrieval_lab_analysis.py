from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean


QUESTION_TYPE_BUCKETS: tuple[str, ...] = (
    "fact_attribute",
    "cause_explanation",
    "procedure_method",
    "relation_compare",
    "fallback_balanced",
)


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def build_question_type_breakdown(prediction_rows: list[dict], *, top_k: int) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in prediction_rows:
        bucket = str(row.get("question_type", "fallback_balanced") or "fallback_balanced")
        buckets[bucket].append(row)

    payload: dict[str, dict[str, float | int]] = {}
    for bucket in QUESTION_TYPE_BUCKETS:
        rows = buckets.get(bucket, [])
        payload[bucket] = {
            "sample_count": len(rows),
            f"precision@{top_k}": _safe_mean([float(item.get("retrieval_metrics", {}).get(f"precision@{top_k}", 0.0)) for item in rows]),
            f"recall@{top_k}": _safe_mean([float(item.get("retrieval_metrics", {}).get(f"recall@{top_k}", 0.0)) for item in rows]),
            f"hit@{top_k}": _safe_mean([float(item.get("retrieval_metrics", {}).get(f"hit@{top_k}", 0.0)) for item in rows]),
            "mrr": _safe_mean([float(item.get("retrieval_metrics", {}).get("mrr", 0.0)) for item in rows]),
            "stage1_pool_recall_rate": _safe_mean([1.0 if bool(item.get("gold_in_stage1_pool")) else 0.0 for item in rows]),
            "in_stage1_but_filtered_rate": _safe_mean([1.0 if item.get("miss_type") == "in_stage1_but_filtered" else 0.0 for item in rows]),
            "in_final_candidates_but_rank_too_low_rate": _safe_mean(
                [1.0 if item.get("miss_type") == "in_final_candidates_but_rank_too_low" else 0.0 for item in rows]
            ),
        }
    return payload


def build_combo_analysis(prediction_rows: list[dict], *, top_k: int) -> dict[str, object]:
    miss_counter = Counter(str(item.get("miss_type", "unknown")) for item in prediction_rows)
    return {
        "question_type_breakdown": build_question_type_breakdown(prediction_rows, top_k=top_k),
        "miss_type_distribution": dict(miss_counter),
        "stage1_pool_recall_rate": _safe_mean([1.0 if bool(item.get("gold_in_stage1_pool")) else 0.0 for item in prediction_rows]),
        "in_stage1_but_filtered_rate": _safe_mean(
            [1.0 if item.get("miss_type") == "in_stage1_but_filtered" else 0.0 for item in prediction_rows]
        ),
        "in_final_candidates_but_rank_too_low_rate": _safe_mean(
            [1.0 if item.get("miss_type") == "in_final_candidates_but_rank_too_low" else 0.0 for item in prediction_rows]
        ),
        "avg_gold_best_rank_final": _safe_mean(
            [float(item["gold_best_rank_final"]) for item in prediction_rows if item.get("gold_best_rank_final") is not None]
        ),
    }


def build_dynamic_compare_report(
    *,
    relation_summary: dict,
    dynamic_summary: dict,
    relation_predictions: list[dict],
    dynamic_predictions: list[dict],
    top_k: int,
) -> dict[str, object]:
    relation_analysis = build_combo_analysis(relation_predictions, top_k=top_k)
    dynamic_analysis = build_combo_analysis(dynamic_predictions, top_k=top_k)
    retrieval_keys = (f"recall@{top_k}", f"hit@{top_k}", "mrr")

    total_metric_delta = {
        metric: float(dynamic_summary["retrieval"].get(metric, 0.0)) - float(relation_summary["retrieval"].get(metric, 0.0))
        for metric in retrieval_keys
    }
    total_metric_delta["stage1_pool_recall_rate"] = float(dynamic_analysis["stage1_pool_recall_rate"]) - float(
        relation_analysis["stage1_pool_recall_rate"]
    )
    question_type_delta: dict[str, dict[str, float]] = {}
    for question_type in QUESTION_TYPE_BUCKETS:
        relation_bucket = relation_analysis["question_type_breakdown"][question_type]
        dynamic_bucket = dynamic_analysis["question_type_breakdown"][question_type]
        question_type_delta[question_type] = {
            "sample_count": int(dynamic_bucket["sample_count"]),
            f"recall@{top_k}_delta": float(dynamic_bucket[f"recall@{top_k}"]) - float(relation_bucket[f"recall@{top_k}"]),
            f"hit@{top_k}_delta": float(dynamic_bucket[f"hit@{top_k}"]) - float(relation_bucket[f"hit@{top_k}"]),
            "mrr_delta": float(dynamic_bucket["mrr"]) - float(relation_bucket["mrr"]),
            "stage1_pool_recall_rate_delta": float(dynamic_bucket["stage1_pool_recall_rate"]) - float(relation_bucket["stage1_pool_recall_rate"]),
            "in_stage1_but_filtered_rate_delta": float(dynamic_bucket["in_stage1_but_filtered_rate"])
            - float(relation_bucket["in_stage1_but_filtered_rate"]),
            "in_final_candidates_but_rank_too_low_rate_delta": float(dynamic_bucket["in_final_candidates_but_rank_too_low_rate"])
            - float(relation_bucket["in_final_candidates_but_rank_too_low_rate"]),
        }

    miss_delta_keys = set(relation_analysis["miss_type_distribution"]) | set(dynamic_analysis["miss_type_distribution"])
    miss_type_distribution_delta = {
        key: int(dynamic_analysis["miss_type_distribution"].get(key, 0)) - int(relation_analysis["miss_type_distribution"].get(key, 0))
        for key in sorted(miss_delta_keys)
    }

    return {
        "relation_driven": {
            "summary": relation_summary,
            "analysis": relation_analysis,
        },
        "dynamic_relation_drive": {
            "summary": dynamic_summary,
            "analysis": dynamic_analysis,
        },
        "delta": {
            "total_metrics": total_metric_delta,
            "question_type_breakdown": question_type_delta,
            "miss_type_distribution": miss_type_distribution_delta,
            "in_stage1_but_filtered_rate": float(dynamic_analysis["in_stage1_but_filtered_rate"])
            - float(relation_analysis["in_stage1_but_filtered_rate"]),
            "in_final_candidates_but_rank_too_low_rate": float(dynamic_analysis["in_final_candidates_but_rank_too_low_rate"])
            - float(relation_analysis["in_final_candidates_but_rank_too_low_rate"]),
            "avg_gold_best_rank_final": float(dynamic_analysis["avg_gold_best_rank_final"])
            - float(relation_analysis["avg_gold_best_rank_final"]),
        },
    }


def build_multi_scorer_compare_report(
    *,
    summary_payload: dict[str, dict],
    combo_prediction_rows: dict[str, list[dict]],
    grouped_compare_keys: dict[tuple[str, str], dict[str, str]],
    top_k: int,
) -> dict[str, object]:
    report: dict[str, object] = {}
    for (rewrite_mode, retrieval_mode), scorer_map in grouped_compare_keys.items():
        combo_key = f"{rewrite_mode}__{retrieval_mode}"
        scorer_entries: dict[str, object] = {}
        for scorer_mode, combo in scorer_map.items():
            scorer_entries[scorer_mode] = {
                "summary": summary_payload[combo],
                "analysis": build_combo_analysis(combo_prediction_rows[combo], top_k=top_k),
            }
        pairwise: dict[str, object] = {}
        for left in scorer_map:
            for right in scorer_map:
                if left >= right:
                    continue
                left_combo = scorer_map[left]
                right_combo = scorer_map[right]
                left_analysis = build_combo_analysis(combo_prediction_rows[left_combo], top_k=top_k)
                right_analysis = build_combo_analysis(combo_prediction_rows[right_combo], top_k=top_k)
                pairwise[f"{left}__vs__{right}"] = {
                    "total_metrics": {
                        f"precision@{top_k}_delta": float(summary_payload[right_combo]["retrieval"].get(f"precision@{top_k}", 0.0))
                        - float(summary_payload[left_combo]["retrieval"].get(f"precision@{top_k}", 0.0)),
                        f"recall@{top_k}_delta": float(summary_payload[right_combo]["retrieval"].get(f"recall@{top_k}", 0.0))
                        - float(summary_payload[left_combo]["retrieval"].get(f"recall@{top_k}", 0.0)),
                        f"hit@{top_k}_delta": float(summary_payload[right_combo]["retrieval"].get(f"hit@{top_k}", 0.0))
                        - float(summary_payload[left_combo]["retrieval"].get(f"hit@{top_k}", 0.0)),
                        "mrr_delta": float(summary_payload[right_combo]["retrieval"].get("mrr", 0.0))
                        - float(summary_payload[left_combo]["retrieval"].get("mrr", 0.0)),
                    },
                    "in_stage1_but_filtered_rate_delta": float(right_analysis["in_stage1_but_filtered_rate"])
                    - float(left_analysis["in_stage1_but_filtered_rate"]),
                    "in_final_candidates_but_rank_too_low_rate_delta": float(right_analysis["in_final_candidates_but_rank_too_low_rate"])
                    - float(left_analysis["in_final_candidates_but_rank_too_low_rate"]),
                }
        report[combo_key] = {
            "scorers": scorer_entries,
            "pairwise": pairwise,
        }
    return report


def build_regression_cases(
    *,
    relation_rows: list[dict],
    dynamic_rows: list[dict],
    dynamic_new_rows: list[dict],
    top_k: int,
) -> dict[str, object]:
    relation_map = {row["sample_id"]: row for row in relation_rows}
    dynamic_map = {row["sample_id"]: row for row in dynamic_rows}
    dynamic_new_map = {row["sample_id"]: row for row in dynamic_new_rows}

    def _snapshot(row: dict) -> dict[str, object]:
        return {
            "sample_id": row["sample_id"],
            "question": row.get("question", ""),
            "question_type": row.get("question_type", "fallback_balanced"),
            f"hit@{top_k}": float(row.get("retrieval_metrics", {}).get(f"hit@{top_k}", 0.0)),
            "mrr": float(row.get("retrieval_metrics", {}).get("mrr", 0.0)),
            "miss_type": row.get("miss_type"),
            "gold_best_rank_final": row.get("gold_best_rank_final"),
            "gold_filter_reasons": row.get("gold_filter_reasons", []),
        }

    relation_hits_dynamic_new_miss = []
    dynamic_new_hits_relation_miss = []
    dynamic_new_repairs_vs_dynamic = []
    dynamic_new_still_failed_vs_dynamic = []
    for sample_id, relation_row in relation_map.items():
        dynamic_row = dynamic_map.get(sample_id)
        dynamic_new_row = dynamic_new_map.get(sample_id)
        if dynamic_row is None or dynamic_new_row is None:
            continue
        relation_hit = bool(relation_row.get("retrieval_metrics", {}).get(f"hit@{top_k}", 0.0))
        dynamic_hit = bool(dynamic_row.get("retrieval_metrics", {}).get(f"hit@{top_k}", 0.0))
        dynamic_new_hit = bool(dynamic_new_row.get("retrieval_metrics", {}).get(f"hit@{top_k}", 0.0))
        if relation_hit and not dynamic_new_hit:
            relation_hits_dynamic_new_miss.append(
                {"relation_driven": _snapshot(relation_row), "dynamic_relation_drive_new": _snapshot(dynamic_new_row)}
            )
        if dynamic_new_hit and not relation_hit:
            dynamic_new_hits_relation_miss.append(
                {"relation_driven": _snapshot(relation_row), "dynamic_relation_drive_new": _snapshot(dynamic_new_row)}
            )
        if dynamic_new_hit and not dynamic_hit:
            dynamic_new_repairs_vs_dynamic.append(
                {"dynamic_relation_drive": _snapshot(dynamic_row), "dynamic_relation_drive_new": _snapshot(dynamic_new_row)}
            )
        if not dynamic_new_hit and not dynamic_hit:
            dynamic_new_still_failed_vs_dynamic.append(
                {
                    "dynamic_relation_drive": _snapshot(dynamic_row),
                    "dynamic_relation_drive_new": _snapshot(dynamic_new_row),
                }
            )

    return {
        "relation_driven_hit_but_dynamic_new_miss": relation_hits_dynamic_new_miss,
        "dynamic_new_hit_but_relation_driven_miss": dynamic_new_hits_relation_miss,
        "dynamic_new_repairs_vs_dynamic": dynamic_new_repairs_vs_dynamic,
        "dynamic_new_still_failed_vs_dynamic": dynamic_new_still_failed_vs_dynamic,
    }
