from __future__ import annotations

from core.schema import BenchmarkSample
from core.types import QuestionType, SourceScope
from dataio.loaders import GraphRelationRecord, GraphSectionRecord
from evaluation.rewrite_diagnostics import evaluate_rewrite_diagnostics
from modules.graph_expander import GraphIndex
from modules.skeleton_extractor import SkeletonExtractor


def _build_graph() -> GraphIndex:
    return GraphIndex.build(
        [
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_a",
                content="2019年 Alpha 与 Beta 合作原因是为了降低成本。",
                entities=["Alpha", "Beta"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="合作原因", object="降低成本")],
                constraints=["2019年"],
            )
        ]
    )


def _make_sample(question: str) -> BenchmarkSample:
    return BenchmarkSample(
        question_id=question,
        question=question,
        answer_short="为了降低成本",
        question_type=QuestionType.EXPLANATION,
        source_scope=SourceScope.SINGLE_DOC,
        entities=["Alpha", "Beta"],
        relations=["合作原因"],
        constraints=["2019年"],
    )


def test_rewrite_diagnostics_compute_retention_metrics() -> None:
    extractor = SkeletonExtractor(_build_graph())
    original = extractor.extract(_make_sample("2019年 Alpha 与 Beta 合作的原因是什么？"), mode="stub_predicted")
    rewritten = extractor.extract(_make_sample("2019年 Alpha 合作 原因"), mode="stub_predicted")

    metrics = evaluate_rewrite_diagnostics(original, rewritten)

    assert metrics["entity_retention_precision"] == 1.0
    assert metrics["entity_retention_recall"] < 1.0
    assert metrics["relation_retention_recall"] == 1.0
    assert metrics["semantic_drift"] is False


def test_rewrite_diagnostics_flags_semantic_drift_when_structure_lost() -> None:
    extractor = SkeletonExtractor(_build_graph())
    original = extractor.extract(_make_sample("2019年 Alpha 与 Beta 合作的原因是什么？"), mode="stub_predicted")
    rewritten = extractor.extract(_make_sample("无关问题"), mode="stub_predicted")

    metrics = evaluate_rewrite_diagnostics(original, rewritten)

    assert metrics["entity_retention_recall"] == 0.0
    assert metrics["relation_retention_recall"] == 0.0
    assert metrics["semantic_drift"] is True
