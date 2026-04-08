from __future__ import annotations

from core.schema import BenchmarkSample

from rewrite_lab.evaluators.intrinsic import evaluate_intrinsic_metrics
from rewrite_lab.schema import RewriteCandidate


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        question_id="q1",
        question="S6850交换机的交换容量是多少？",
        answer_short="1Tbps",
        question_type="fact",
        entities=["S6850交换机"],
        relations=["交换容量"],
        constraints=[],
        evidence=[{"source_id": "doc1", "section_id": "sec1", "quote": "1Tbps"}],
        source_scope="single_section",
    )


def test_intrinsic_metrics_preserve_entity_and_relation() -> None:
    candidate = RewriteCandidate(
        question_id="q1",
        original_question="S6850交换机的交换容量是多少？",
        rewritten_query="S6850交换机 交换容量",
        strategy_name="rule_based_v1",
    )
    metrics = evaluate_intrinsic_metrics(_sample(), candidate)
    assert metrics["entity_preservation_rate"] == 1.0
    assert metrics["attribute_or_relation_preservation_rate"] == 1.0
    assert metrics["noise_rate"] == 0.0
    assert metrics["parse_success"] is True


def test_intrinsic_metrics_detect_noise_and_entity_drop() -> None:
    candidate = RewriteCandidate(
        question_id="q1",
        original_question="S6850交换机的交换容量是多少？",
        rewritten_query="随机新增词 端口统计",
        strategy_name="rule_based_v1",
        flags=["entity_dropped"],
    )
    metrics = evaluate_intrinsic_metrics(_sample(), candidate)
    assert metrics["entity_preservation_rate"] == 0.0
    assert metrics["noise_rate"] > 0.0
