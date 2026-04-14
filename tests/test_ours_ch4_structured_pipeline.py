from __future__ import annotations

from core.schema import BenchmarkSample, RetrievedDocument
from core.types import QuestionType, SourceScope
from dataio.loaders import GraphRelationRecord, GraphSectionRecord
from modules.graph_expander import GraphIndex
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractor
from modules.text_compensator import QueryAnalysisPayload, TextCompensator
from pipelines.base import PublicIndex, SectionDocument
from pipelines.ours_ch4 import OursCh4Config, OursCh4Pipeline


def _build_indexes() -> tuple[PublicIndex, GraphIndex]:
    text_index = PublicIndex(
        [
            SectionDocument(
                source_id="doc_a",
                section_id="sec_summary",
                content="2019年 Alpha 与 Beta 建立合作关系，双方开始联合研发。",
                doc_title="Alpha Beta 概览",
            ),
            SectionDocument(
                source_id="doc_a",
                section_id="sec_reason",
                content="合作原因是为了降低成本并扩大市场覆盖范围，这解释了双方为何在2019年推进合作。",
                doc_title="Alpha Beta 原因",
            ),
            SectionDocument(
                source_id="doc_a",
                section_id="sec_noise",
                content="Gamma 在 2021 年调整了内部流程，与 Alpha Beta 合作无关。",
                doc_title="无关段落",
            ),
            SectionDocument(
                source_id="doc_a",
                section_id="sec_wrong_relation",
                content="2019年 Alpha 与 Beta 建立合作关系，但这里只介绍双方背景，没有说明原因。",
                doc_title="Alpha Beta 背景",
            ),
            SectionDocument(
                source_id="doc_b",
                section_id="sec_condition",
                content="交换机出现风扇告警时，若复位无效，则更换风扇模组硬件。",
                doc_title="风扇告警处理",
            ),
            SectionDocument(
                source_id="doc_b",
                section_id="sec_fact",
                content="Alpha 设备的管理地址是 10.0.0.1，默认版本为 8.0。",
                doc_title="Alpha 属性",
            ),
        ]
    )
    graph_index = GraphIndex.build(
        [
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_summary",
                content="2019年 Alpha 与 Beta 建立合作关系，双方开始联合研发。",
                entities=["Alpha", "Beta"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="合作", object="Beta")],
                constraints=["2019年", "联合研发"],
            ),
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_reason",
                content="合作原因是为了降低成本并扩大市场覆盖范围。",
                entities=["Alpha", "Beta"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="合作原因", object="降低成本")],
                constraints=["2019年", "为了降低成本"],
            ),
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_noise",
                content="Gamma 在 2021 年调整了内部流程。",
                entities=["Gamma"],
                relations=[GraphRelationRecord(subject="Gamma", predicate="调整", object="流程")],
                constraints=["2021年"],
            ),
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_wrong_relation",
                content="2019年 Alpha 与 Beta 建立合作关系，但这里只介绍双方背景，没有说明原因。",
                entities=["Alpha", "Beta"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="合作", object="Beta")],
                constraints=["2019年"],
            ),
            GraphSectionRecord(
                source_id="doc_b",
                section_id="sec_condition",
                content="交换机出现风扇告警时，若复位无效，则更换风扇模组硬件。",
                entities=["交换机", "风扇模组"],
                relations=[GraphRelationRecord(subject="交换机", predicate="更换硬件条件", object="风扇模组")],
                constraints=["风扇告警", "若复位无效则更换硬件"],
            ),
            GraphSectionRecord(
                source_id="doc_b",
                section_id="sec_fact",
                content="Alpha 设备的管理地址是 10.0.0.1，默认版本为 8.0。",
                entities=["Alpha 设备"],
                relations=[GraphRelationRecord(subject="Alpha 设备", predicate="管理地址", object="10.0.0.1")],
                constraints=["版本 8.0"],
            ),
        ]
    )
    return text_index, graph_index


def _build_sample() -> BenchmarkSample:
    return BenchmarkSample(
        question_id="q1",
        question="2019年 Alpha 与 Beta 合作的原因是什么？",
        answer_short="为了降低成本并扩大市场覆盖范围",
        question_type=QuestionType.EXPLANATION,
        source_scope=SourceScope.SINGLE_DOC,
        entities=["LeakEntity"],
        relations=["LeakRelation"],
        constraints=["LeakConstraint"],
        requires_text_compensation=True,
    )


def _build_condition_sample(*, requires_text_compensation: bool = False) -> BenchmarkSample:
    return BenchmarkSample(
        question_id="q_condition",
        question="交换机风扇告警时，什么条件下需要更换硬件？",
        answer_short="若复位无效，则需要更换风扇模组硬件。",
        question_type=QuestionType.EXPLANATION,
        source_scope=SourceScope.SINGLE_DOC,
        entities=["交换机"],
        relations=["更换硬件条件"],
        constraints=["风扇告警", "什么条件下需要更换硬件"],
        requires_text_compensation=requires_text_compensation,
    )


def _build_fact_sample() -> BenchmarkSample:
    return BenchmarkSample(
        question_id="q_fact",
        question="Alpha 设备的管理地址是什么？",
        answer_short="10.0.0.1",
        question_type=QuestionType.FACT,
        source_scope=SourceScope.SINGLE_DOC,
        entities=["Alpha 设备"],
        relations=["管理地址"],
        constraints=["版本 8.0"],
    )


def test_predicted_skeleton_and_compensation_do_not_use_label_shortcuts() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample()
    extractor = SkeletonExtractor(graph_index)
    retriever = RelationDrivenRetriever(text_index, graph_index)
    compensator = TextCompensator(text_index, graph_index)

    skeleton = extractor.extract(sample, mode="stub_predicted")

    assert "LeakEntity" not in skeleton.entities
    assert "LeakRelation" not in skeleton.relations
    assert "LeakConstraint" not in skeleton.constraints
    assert any(entity.normalized_name == "alpha" for entity in skeleton.structured_entities)
    assert any(entity.role == "anchor" for entity in skeleton.structured_entities)
    assert any(relation.role == "target" for relation in skeleton.structured_relations)
    assert any(relation.relation_type == "cause" for relation in skeleton.structured_relations)
    assert any(constraint.kind == "time" for constraint in skeleton.structured_constraints)

    retrieved = retriever.retrieve(sample, skeleton, top_k=2, use_relation_driven=True, use_skeleton_rewrite=True)
    compensation = compensator.compensate(sample, skeleton, retrieved.documents, top_k=2, enabled=True)

    assert compensation.activated is False
    assert compensation.reason == "not_needed"


def test_text_compensation_is_gap_triggered_and_skeleton_guided() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample()
    extractor = SkeletonExtractor(graph_index)
    retriever = RelationDrivenRetriever(text_index, graph_index)
    compensator = TextCompensator(text_index, graph_index)

    skeleton = extractor.extract(sample, mode="predicted")
    base_documents = retriever.retrieve(sample, skeleton, top_k=1, use_relation_driven=True, use_skeleton_rewrite=True).documents
    compensation = compensator.compensate(sample, skeleton, base_documents, top_k=2, enabled=True)

    assert compensation.activated is True
    assert compensation.reason == "evidence_too_thin"
    assert compensation.details["compensation_query"]
    assert compensation.details["max_additions"] == 1
    assert compensation.details["guaranteed_primary"] == ["sec_reason"]
    assert len(compensation.details["added_sections"]) <= 1


def test_relation_driven_retrieval_filters_wrong_relation_candidate() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample()
    extractor = SkeletonExtractor(graph_index)
    retriever = RelationDrivenRetriever(text_index, graph_index)

    skeleton = extractor.extract(sample, mode="predicted")
    result = retriever.retrieve(sample, skeleton, top_k=2, use_relation_driven=True, use_skeleton_rewrite=True)

    kept_sections = [document.section_id for document in result.documents]
    assert "sec_reason" in kept_sections
    assert "sec_wrong_relation" not in kept_sections
    wrong_relation = result.details["filtered_out"].get("sec_wrong_relation")
    assert wrong_relation is not None
    assert "target_relation_weak" in wrong_relation["gate"]["reasons"] or "target_relation_missing" in wrong_relation["gate"]["reasons"]


def test_ours_ch4_ablation_modes_run_with_structured_trace() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample()

    for mode in ("full", "w/o_relation_driven", "w/o_skeleton_rewrite", "w/o_text_compensation"):
        pipeline = OursCh4Pipeline(
            text_index,
            SkeletonExtractor(graph_index),
            RelationDrivenRetriever(text_index, graph_index),
            TextCompensator(text_index, graph_index),
            config=OursCh4Config.from_ablation(mode, top_k=2, skeleton_mode="predicted"),
        )
        record = pipeline.run(sample)
        assert record.trace["skeleton"]["entities"]
        assert "rewrite_payload" in record.trace["skeleton"]
        assert "relation_driven_details" in record.trace
        assert record.trace["ablation"]["use_relation_driven"] == (mode != "w/o_relation_driven")
        assert record.trace["ablation"]["use_skeleton_rewrite"] == (mode != "w/o_skeleton_rewrite")
        assert record.trace["ablation"]["use_text_compensation"] == (mode != "w/o_text_compensation")


def test_new_compensation_triggers_when_doc_count_is_enough_but_condition_uncovered() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_condition_sample()
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    base_documents = [
        RetrievedDocument(
            source_id="doc_a",
            section_id="sec_summary",
            content="2019年 Alpha 与 Beta 建立合作关系，双方开始联合研发。",
            score=1.0,
            rank=index,
            metadata={},
        )
        for index in range(1, 6)
    ]
    compensation = TextCompensator(text_index, graph_index).compensate(
        sample,
        skeleton,
        base_documents,
        top_k=5,
        enabled=True,
        strategy="new_compensation",
    )
    assert compensation.activated is True
    assert compensation.reason in {"primary_relation_uncovered", "primary_constraint_uncovered", "coverage_below_threshold"}
    assert compensation.details["coverage_before"]["unmatched_constraints"]


def test_new_compensation_skips_simple_fact_question_when_coverage_is_complete() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_fact_sample()
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    base_documents = [
        RetrievedDocument(
            source_id="doc_b",
            section_id="sec_fact",
            content="Alpha 设备的管理地址是 10.0.0.1，默认版本为 8.0。",
            score=1.0,
            rank=1,
            metadata={},
        )
    ]
    compensation = TextCompensator(text_index, graph_index).compensate(
        sample,
        skeleton,
        base_documents,
        top_k=3,
        enabled=True,
        strategy="new_compensation",
    )
    assert compensation.activated is False
    assert compensation.reason == "not_needed"


def test_new_compensation_hard_triggers_when_primary_relation_missing() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_condition_sample()
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    base_documents = [
        RetrievedDocument(
            source_id="doc_b",
            section_id="sec_fact",
            content="Alpha 设备的管理地址是 10.0.0.1，默认版本为 8.0。",
            score=0.9,
            rank=1,
            metadata={},
        )
    ]
    compensation = TextCompensator(text_index, graph_index).compensate(
        sample,
        skeleton,
        base_documents,
        top_k=3,
        enabled=True,
        strategy="new_compensation",
    )
    assert compensation.activated is True
    assert compensation.reason == "primary_relation_uncovered"


def test_condition_template_matching_is_semantic() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_condition_sample()
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    document = RetrievedDocument(
        source_id="doc_b",
        section_id="sec_condition",
        content="交换机出现风扇告警时，若复位无效，则更换风扇模组硬件。",
        score=1.0,
        rank=1,
        metadata={},
    )
    result = TextCompensator(text_index, graph_index).match_constraint(
        "什么条件下需要更换硬件",
        skeleton,
        [document],
        query_analysis=QueryAnalysisPayload(entities=["交换机"], relations=["更换硬件条件"], constraints=["什么条件下需要更换硬件"]),
    )
    assert result["covered"] is True
    assert result["status"] == "semantic"


def test_new_compensation_fallback_keeps_old_rule_available() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_condition_sample(requires_text_compensation=True)
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    base_documents = [
        RetrievedDocument(
            source_id="doc_b",
            section_id="sec_fact",
            content="Alpha 设备的管理地址是 10.0.0.1，默认版本为 8.0。",
            score=0.9,
            rank=1,
            metadata={},
        )
    ]
    compensation = TextCompensator(text_index, graph_index).compensate(
        sample,
        skeleton,
        base_documents,
        top_k=3,
        enabled=True,
        query_analysis={"question_type": "fallback_balanced", "question_type_confidence": "low"},
        strategy="new_compensation",
    )
    assert compensation.activated is True
    assert compensation.details["fallback_used"] is True
    assert compensation.details["old_rule_triggered"] is True
