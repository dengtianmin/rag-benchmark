from __future__ import annotations

from core.schema import BenchmarkSample, RetrievedDocument
from core.types import QuestionType, SourceScope
from dataio.loaders import GraphRelationRecord, GraphSectionRecord
from modules.dynamic_relation_drive_new_retriever import (
    DynamicRelationDriveNewQuestionTypeAdjuster,
    DynamicRelationDriveNewRetriever,
)
from modules.dynamic_relation_drive_retriever import DynamicRelationDriveRetriever
from modules.graph_expander import GraphIndex
from modules.question_type_classifier import QuestionTypeClassification
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractor
from pipelines.base import PublicIndex, SectionDocument


class _RecordingRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents

    def retrieve(self, query, top_k: int) -> list[RetrievedDocument]:
        del query
        return self.documents[:top_k]


def _build_indexes() -> tuple[PublicIndex, GraphIndex]:
    text_index = PublicIndex(
        [
            SectionDocument(source_id="doc_a", section_id="sec_attr", content="Alpha 支持应用加速与病毒过滤。", doc_title="attr"),
            SectionDocument(source_id="doc_a", section_id="sec_proc", content="配置步骤包括登录、校验、下发策略。", doc_title="proc"),
            SectionDocument(source_id="doc_a", section_id="sec_rel", content="Alpha 与 Beta 的区别在于部署模式。", doc_title="rel"),
        ]
    )
    graph_index = GraphIndex.build(
        [
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_attr",
                content="Alpha 支持应用加速与病毒过滤。",
                entities=["Alpha"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="支持功能", object="应用加速")],
                constraints=[],
            ),
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_proc",
                content="配置步骤包括登录、校验、下发策略。",
                entities=["Alpha"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="配置步骤", object="登录")],
                constraints=[],
            ),
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_rel",
                content="Alpha 与 Beta 的区别在于部署模式。",
                entities=["Alpha", "Beta"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="区别", object="Beta")],
                constraints=[],
            ),
        ]
    )
    return text_index, graph_index


def _build_sample(question: str, relations: list[str]) -> BenchmarkSample:
    return BenchmarkSample(
        question_id="q1",
        question=question,
        answer_short="a",
        question_type=QuestionType.FACT,
        source_scope=SourceScope.SINGLE_DOC,
        entities=["Alpha"],
        relations=relations,
        constraints=[],
    )


def test_dynamic_new_adjuster_prefers_fact_or_fallback_for_attribute_like_questions() -> None:
    _, graph_index = _build_indexes()
    sample = _build_sample("该设备支持哪些应用/功能？", ["支持功能"])
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="stub_predicted")
    adjuster = DynamicRelationDriveNewQuestionTypeAdjuster()

    adjusted = adjuster.adjust(
        sample.question,
        QuestionTypeClassification(
            question_type="procedure_method",
            question_type_confidence="medium",
            question_type_evidence={},
        ),
        skeleton,
    )

    assert adjusted.adjusted_question_type == "fact_attribute"


def test_dynamic_new_adjuster_keeps_explicit_procedure_questions() -> None:
    _, graph_index = _build_indexes()
    sample = _build_sample("如何配置设备的步骤？", ["配置步骤"])
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="stub_predicted")
    adjuster = DynamicRelationDriveNewQuestionTypeAdjuster()

    adjusted = adjuster.adjust(
        sample.question,
        QuestionTypeClassification(
            question_type="procedure_method",
            question_type_confidence="high",
            question_type_evidence={},
        ),
        skeleton,
    )

    assert adjusted.adjusted_question_type == "procedure_method"


def test_dynamic_new_does_not_change_old_retrievers() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample("Alpha 支持哪些应用/功能？", ["支持功能"])
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    docs = [RetrievedDocument(source_id="doc_a", section_id="sec_attr", content="Alpha 支持应用加速与病毒过滤。", score=0.9, rank=1, metadata={"dense_score": 0.9})]

    relation = RelationDrivenRetriever(text_index, graph_index, text_retriever=_RecordingRetriever(docs))
    dynamic = DynamicRelationDriveRetriever(text_index, graph_index, text_retriever=_RecordingRetriever(docs))

    relation_result = relation.retrieve(sample, skeleton, top_k=1, use_relation_driven=True, use_skeleton_rewrite=False, query_override="q")
    dynamic_result = dynamic.retrieve(
        sample,
        skeleton,
        top_k=1,
        use_skeleton_rewrite=False,
        query_override="q",
        question_type_info=QuestionTypeClassification(question_type="fact_attribute", question_type_confidence="high", question_type_evidence={}),
    )

    assert relation_result.details["mode"] == "relation_driven"
    assert dynamic_result.details["mode"] == "dynamic_relation_drive"


def test_dynamic_new_uses_new_weight_profile_and_softens_fact_attribute_relation_miss() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample("Alpha 支持哪些应用/功能？", ["不存在关系"])
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    skeleton.structured_relations[0].name = "不存在关系"
    skeleton.structured_relations[0].normalized_name = "不存在关系"
    docs = [RetrievedDocument(source_id="doc_a", section_id="sec_attr", content="Alpha 支持应用加速与病毒过滤。", score=0.9, rank=1, metadata={"dense_score": 0.9})]
    retriever = DynamicRelationDriveNewRetriever(text_index, graph_index, text_retriever=_RecordingRetriever(docs))

    result = retriever.retrieve(
        sample,
        skeleton,
        top_k=1,
        use_skeleton_rewrite=False,
        query_override="q",
        question_type_info=QuestionTypeClassification(question_type="procedure_method", question_type_confidence="medium", question_type_evidence={}),
    )

    metadata = result.documents[0].metadata
    assert metadata["dynamic_new_weight_profile"] == {"w_sem": 0.64, "w_ent": 0.22, "w_rel": 0.04, "w_con": 0.1}
    assert metadata["adjusted_question_type_for_dynamic_new"] == "fact_attribute"


def test_dynamic_new_high_semantic_protection_and_compare_drop_thresholds() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample("Alpha 和 Beta 的区别是什么？", ["区别"])
    sample.entities = ["Alpha", "Beta"]
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    weak_docs = [
        RetrievedDocument(source_id="doc_a", section_id="sec_attr", content="完全无关。", score=0.2, rank=1, metadata={"dense_score": 0.2}),
        RetrievedDocument(source_id="doc_a", section_id="sec_unknown", content="完全无关。", score=0.2, rank=1, metadata={"dense_score": 0.2}),
        RetrievedDocument(
            source_id="doc_a",
            section_id="sec_rel",
            content="Alpha 与 Beta 的区别在于部署模式。",
            score=0.9,
            rank=2,
            metadata={"dense_score": 0.9},
        ),
    ]
    strong_docs = [RetrievedDocument(source_id="doc_a", section_id="sec_rel", content="Alpha 与 Beta 的区别在于部署模式。", score=0.8, rank=1, metadata={"dense_score": 0.8})]

    weak_result = DynamicRelationDriveNewRetriever(text_index, graph_index, text_retriever=_RecordingRetriever(weak_docs)).retrieve(
        sample,
        skeleton,
        top_k=1,
        use_skeleton_rewrite=False,
        query_override="q",
        question_type_info=QuestionTypeClassification(question_type="relation_compare", question_type_confidence="high", question_type_evidence={}),
    )
    strong_result = DynamicRelationDriveNewRetriever(text_index, graph_index, text_retriever=_RecordingRetriever(strong_docs)).retrieve(
        sample,
        skeleton,
        top_k=1,
        use_skeleton_rewrite=False,
        query_override="q",
        question_type_info=QuestionTypeClassification(question_type="relation_compare", question_type_confidence="high", question_type_evidence={}),
    )

    weak_breakdown = weak_result.details["filtered_out"]["sec_unknown"]
    assert weak_breakdown["hard_drop_triggered"] is True
    assert strong_result.documents[0].metadata["high_semantic_protection_triggered"] is True
