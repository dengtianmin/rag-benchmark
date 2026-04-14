from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_rewrite_failure_modes import analyze_predictions


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_analyze_rewrite_failure_modes_reads_predictions_and_aggregates(tmp_path: Path) -> None:
    combo_a = tmp_path / "combos" / "sparse_llm__hybrid__plain" / "predictions.jsonl"
    combo_b = tmp_path / "combos" / "dense_llm__hybrid__plain" / "predictions.jsonl"
    _write_jsonl(
        combo_a,
        [
            {
                "rewrite_mode": "sparse_llm",
                "retrieval_mode": "hybrid",
                "scorer_mode": "plain",
                "gold_in_stage1_pool": True,
                "gold_best_rank_stage1": 2,
                "gold_in_final_topk": False,
                "gold_best_rank_final": None,
                "miss_type": "in_final_candidates_but_rank_too_low",
                "gold_semantic_score": None,
                "gold_relation_score": None,
                "gold_constraint_score": None,
                "gold_fusion_score": None,
                "filtered_out_reason_map": {"sec_x": ["target_relation_weak"]},
                "gold_filter_reasons": [],
                "relation_scores": {},
                "final_topk_ids": ["sec_a"],
                "gold_only_in_lexical": True,
                "gold_only_in_dense": False,
                "gold_in_both_pools": False,
                "final_topk_source_breakdown": {"lexical": 1, "dense": 0, "both": 0, "unknown": 0},
            }
        ],
    )
    _write_jsonl(
        combo_b,
        [
            {
                "rewrite_mode": "dense_llm",
                "retrieval_mode": "hybrid",
                "scorer_mode": "plain",
                "gold_in_stage1_pool": True,
                "gold_best_rank_stage1": 1,
                "gold_in_final_topk": True,
                "gold_best_rank_final": 1,
                "miss_type": "in_final_topk",
                "gold_semantic_score": 0.8,
                "gold_relation_score": 0.6,
                "gold_constraint_score": 0.5,
                "gold_fusion_score": 0.7,
                "filtered_out_reason_map": {},
                "gold_filter_reasons": [],
                "relation_scores": {"sec_gold": 0.6},
                "final_topk_ids": ["sec_gold"],
                "gold_only_in_lexical": False,
                "gold_only_in_dense": True,
                "gold_in_both_pools": False,
                "final_topk_source_breakdown": {"lexical": 0, "dense": 1, "both": 0, "unknown": 0},
            }
        ],
    )

    payload = analyze_predictions([combo_a, combo_b])

    assert payload["summary"]["combo_count"] == 2
    assert "sparse_llm__hybrid__plain" in payload["per_combo_analysis"]
    assert "dense_llm__hybrid__plain" in payload["per_combo_analysis"]
    assert payload["per_combo_analysis"]["dense_llm__hybrid__plain"]["final_hit_rate"] == 1.0
    assert payload["failure_breakdown"]["sparse_llm__hybrid__plain"]["miss_type_counts"]["in_final_candidates_but_rank_too_low"] == 1
