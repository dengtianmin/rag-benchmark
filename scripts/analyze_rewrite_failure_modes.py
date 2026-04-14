from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataio.loaders import iter_jsonl


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_rank_delta(stage1_rank: int | None, final_rank: int | None) -> float:
    if stage1_rank is None or final_rank is None:
        return 0.0
    return float(stage1_rank - final_rank)


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    return [row for row in iter_jsonl(path)]


def _collect_combo_rows(prediction_paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in prediction_paths:
        combo = path.parent.name
        grouped[combo].extend(_read_predictions(path))
    return grouped


def _aggregate_combo(rows: list[dict[str, Any]]) -> dict[str, Any]:
    miss_counter = Counter(str(row.get("miss_type", "unknown")) for row in rows)
    filter_reason_counter = Counter()
    gold_filter_reason_counter = Counter()
    top1_relation_scores: list[float] = []
    top5_relation_scores: list[float] = []
    rank_improvements: list[float] = []
    final_source_counter = Counter()

    for row in rows:
        for reasons in (row.get("filtered_out_reason_map", {}) or {}).values():
            for reason in reasons:
                filter_reason_counter[str(reason)] += 1
        for reason in row.get("gold_filter_reasons", []) or []:
            gold_filter_reason_counter[str(reason)] += 1
        relation_scores = row.get("relation_scores", {}) or {}
        final_ids = row.get("final_topk_ids", []) or []
        if final_ids:
            top1_relation_scores.append(float(relation_scores.get(final_ids[0], 0.0)))
        for section_id in final_ids[:5]:
            top5_relation_scores.append(float(relation_scores.get(section_id, 0.0)))
        rank_improvements.append(
            _safe_rank_delta(row.get("gold_best_rank_stage1"), row.get("gold_best_rank_final"))
        )
        for source_name, count in (row.get("final_topk_source_breakdown", {}) or {}).items():
            final_source_counter[str(source_name)] += int(count)

    total = len(rows)
    return {
        "sample_count": total,
        "stage1_pool_recall_rate": _mean([1.0 if row.get("gold_in_stage1_pool") else 0.0 for row in rows]),
        "avg_gold_best_rank_stage1": _mean(
            [float(row["gold_best_rank_stage1"]) for row in rows if row.get("gold_best_rank_stage1") is not None]
        ),
        "stage1_miss_rate": _mean([1.0 if not row.get("gold_in_stage1_pool") else 0.0 for row in rows]),
        "final_hit_rate": _mean([1.0 if row.get("gold_in_final_topk") else 0.0 for row in rows]),
        "avg_gold_best_rank_final": _mean(
            [float(row["gold_best_rank_final"]) for row in rows if row.get("gold_best_rank_final") is not None]
        ),
        "rank_improvement_from_stage1_to_final": _mean(rank_improvements),
        "miss_type_counts": dict(miss_counter),
        "miss_type_rates": {key: count / total for key, count in miss_counter.items()} if total else {},
        "avg_gold_semantic_score": _mean(
            [float(row["gold_semantic_score"]) for row in rows if row.get("gold_semantic_score") is not None]
        ),
        "avg_gold_relation_score": _mean(
            [float(row["gold_relation_score"]) for row in rows if row.get("gold_relation_score") is not None]
        ),
        "avg_gold_constraint_score": _mean(
            [float(row["gold_constraint_score"]) for row in rows if row.get("gold_constraint_score") is not None]
        ),
        "avg_gold_fusion_score": _mean(
            [float(row["gold_fusion_score"]) for row in rows if row.get("gold_fusion_score") is not None]
        ),
        "top1_avg_relation_score": _mean(top1_relation_scores),
        "top5_avg_relation_score": _mean(top5_relation_scores),
        "filter_reason_counts": dict(filter_reason_counter),
        "gold_filter_reason_counts": dict(gold_filter_reason_counter),
        "gold_only_in_lexical_rate": _mean([1.0 if row.get("gold_only_in_lexical") else 0.0 for row in rows]),
        "gold_only_in_dense_rate": _mean([1.0 if row.get("gold_only_in_dense") else 0.0 for row in rows]),
        "gold_in_both_pools_rate": _mean([1.0 if row.get("gold_in_both_pools") else 0.0 for row in rows]),
        "final_topk_source_breakdown": dict(final_source_counter),
    }


def _find_combo_by_rewrite(per_combo: dict[str, dict[str, Any]], rewrite_mode: str) -> tuple[str, dict[str, Any]] | None:
    for combo, payload in per_combo.items():
        if payload.get("rewrite_mode") == rewrite_mode:
            return combo, payload
    return None


def _diff_metrics(left: dict[str, Any], right: dict[str, Any], keys: list[str]) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key in keys:
        delta[key] = float(left.get(key, 0.0)) - float(right.get(key, 0.0))
    return delta


def _build_focus_comparison(per_combo: dict[str, dict[str, Any]], left_mode: str, right_mode: str) -> dict[str, Any]:
    left = _find_combo_by_rewrite(per_combo, left_mode)
    right = _find_combo_by_rewrite(per_combo, right_mode)
    if left is None or right is None:
        return {"available": False}
    left_combo, left_payload = left
    right_combo, right_payload = right
    return {
        "available": True,
        "left_combo": left_combo,
        "right_combo": right_combo,
        "stage1_pool_diff": _diff_metrics(
            left_payload,
            right_payload,
            ["stage1_pool_recall_rate", "avg_gold_best_rank_stage1", "stage1_miss_rate"],
        ),
        "final_topk_diff": _diff_metrics(
            left_payload,
            right_payload,
            ["final_hit_rate", "avg_gold_best_rank_final", "rank_improvement_from_stage1_to_final"],
        ),
        "gold_score_diff": _diff_metrics(
            left_payload,
            right_payload,
            ["avg_gold_semantic_score", "avg_gold_relation_score", "avg_gold_constraint_score", "avg_gold_fusion_score"],
        ),
        "filter_reason_diff": {
            reason: int(left_payload.get("filter_reason_counts", {}).get(reason, 0))
            - int(right_payload.get("filter_reason_counts", {}).get(reason, 0))
            for reason in sorted(
                set(left_payload.get("filter_reason_counts", {})) | set(right_payload.get("filter_reason_counts", {}))
            )
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_report(path: Path, summary: dict[str, Any], comparisons: dict[str, Any]) -> None:
    lines = ["# Rewrite Failure Analysis", ""]
    lines.append("## Global Summary")
    lines.append(f"- combo_count: {summary['combo_count']}")
    lines.append(f"- sample_count: {summary['sample_count']}")
    lines.append("")
    lines.append("## Focus Comparisons")
    for title, payload in comparisons.items():
        lines.append(f"### {title}")
        if not payload.get("available"):
            lines.append("- unavailable")
            lines.append("")
            continue
        lines.append(f"- left_combo: {payload['left_combo']}")
        lines.append(f"- right_combo: {payload['right_combo']}")
        lines.append(f"- stage1_pool_diff: {json.dumps(payload['stage1_pool_diff'], ensure_ascii=False)}")
        lines.append(f"- final_topk_diff: {json.dumps(payload['final_topk_diff'], ensure_ascii=False)}")
        lines.append(f"- gold_score_diff: {json.dumps(payload['gold_score_diff'], ensure_ascii=False)}")
        lines.append(f"- filter_reason_diff: {json.dumps(payload['filter_reason_diff'], ensure_ascii=False)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_predictions(prediction_paths: list[Path]) -> dict[str, Any]:
    combo_rows = _collect_combo_rows(prediction_paths)
    per_combo_analysis: dict[str, dict[str, Any]] = {}
    total_samples = 0
    for combo, rows in combo_rows.items():
        payload = _aggregate_combo(rows)
        if rows:
            payload["combo"] = combo
            payload["rewrite_mode"] = rows[0].get("rewrite_mode")
            payload["retrieval_mode"] = rows[0].get("retrieval_mode")
            payload["scorer_mode"] = rows[0].get("scorer_mode")
        per_combo_analysis[combo] = payload
        total_samples += len(rows)

    focus_comparisons = {
        "sparse_llm_vs_dense_llm": _build_focus_comparison(per_combo_analysis, "sparse_llm", "dense_llm"),
        "dense_llm_vs_hybrid_llm": _build_focus_comparison(per_combo_analysis, "dense_llm", "hybrid_llm"),
        "original_vs_splicing": _build_focus_comparison(per_combo_analysis, "original", "splicing"),
    }
    summary = {
        "combo_count": len(per_combo_analysis),
        "sample_count": total_samples,
        "combos": sorted(per_combo_analysis),
    }
    failure_breakdown = {
        combo: {
            "miss_type_counts": payload.get("miss_type_counts", {}),
            "filter_reason_counts": payload.get("filter_reason_counts", {}),
            "gold_filter_reason_counts": payload.get("gold_filter_reason_counts", {}),
        }
        for combo, payload in per_combo_analysis.items()
    }
    return {
        "summary": summary,
        "failure_breakdown": failure_breakdown,
        "per_combo_analysis": per_combo_analysis,
        "focus_comparisons": focus_comparisons,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze retrieval-lab failure modes from predictions.jsonl outputs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_paths = sorted(args.input_dir.glob("combos/*/predictions.jsonl"))
    if not prediction_paths:
        raise ValueError(f"No predictions.jsonl files found under {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    analysis = analyze_predictions(prediction_paths)
    _write_json(args.output_dir / "summary.json", analysis["summary"])
    _write_json(args.output_dir / "failure_breakdown.json", analysis["failure_breakdown"])
    _write_json(args.output_dir / "per_combo_analysis.json", analysis["per_combo_analysis"])
    _write_json(args.output_dir / "focus_comparisons.json", analysis["focus_comparisons"])
    _write_report(args.output_dir / "report.md", analysis["summary"], analysis["focus_comparisons"])
    print(f"analysis_outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
