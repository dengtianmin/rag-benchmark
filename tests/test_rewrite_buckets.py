from __future__ import annotations

from rewrite_lab.evaluators.buckets import assign_bucket_tags
from rewrite_lab.schema import RewriteCandidate


def test_bucket_assignment_win_and_entity_dropped() -> None:
    candidate = RewriteCandidate(
        question_id="q1",
        original_question="原问题",
        rewritten_query="改写",
        strategy_name="rule_based_v1",
        flags=["entity_dropped"],
    )
    intrinsic = {
        "entity_preservation_rate": 0.0,
        "noise_rate": 0.2,
        "compression_ratio": 0.8,
        "rewrite_length": 3.0,
    }
    retrieval = {
        "original_hit@5": 0.0,
        "rewritten_hit@5": 1.0,
        "original_rank_of_first_gold": None,
        "rewritten_rank_of_first_gold": 2,
        "win_rate": 1.0,
        "loss_rate": 0.0,
    }
    buckets = assign_bucket_tags(candidate, intrinsic, retrieval)
    assert "win" in buckets
    assert "entity_dropped" in buckets


def test_bucket_assignment_rank_down_and_same_miss() -> None:
    candidate = RewriteCandidate(
        question_id="q2",
        original_question="原问题",
        rewritten_query="改写",
        strategy_name="rule_based_v1",
    )
    intrinsic = {
        "entity_preservation_rate": 1.0,
        "noise_rate": 0.8,
        "compression_ratio": 1.4,
        "rewrite_length": 18.0,
    }
    retrieval_rank_down = {
        "original_hit@5": 1.0,
        "rewritten_hit@5": 1.0,
        "original_rank_of_first_gold": 1,
        "rewritten_rank_of_first_gold": 3,
        "win_rate": 0.0,
        "loss_rate": 0.0,
    }
    assert "same_hit_rank_down" in assign_bucket_tags(candidate, intrinsic, retrieval_rank_down)
    assert "noisy_rewrite" in assign_bucket_tags(candidate, intrinsic, retrieval_rank_down)
    assert "overlong_rewrite" in assign_bucket_tags(candidate, intrinsic, retrieval_rank_down)

    retrieval_same_miss = {
        "original_hit@5": 0.0,
        "rewritten_hit@5": 0.0,
        "original_rank_of_first_gold": None,
        "rewritten_rank_of_first_gold": None,
        "win_rate": 0.0,
        "loss_rate": 0.0,
    }
    assert "same_miss" in assign_bucket_tags(candidate, intrinsic, retrieval_same_miss)
