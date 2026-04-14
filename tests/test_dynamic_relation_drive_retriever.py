from __future__ import annotations

from core.schema import BenchmarkSample, RetrievedDocument
from core.types import QuestionType, SourceScope
from dataio.loaders import GraphRelationRecord, GraphSectionRecord
from modules.dynamic_relation_drive_retriever import DynamicRelationDriveRetriever
from modules.graph_expander import GraphIndex
from modules.question_type_classifier import QuestionTypeClassification
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractionResult, SkeletonExtractor
from pipelines.base import PublicIndex, SectionDocument


class _RecordingRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents

    def retrieve(self, query, top_k: int) -> list[RetrievedDocument]:
        del query
        return self.documents[:top_k]


def _build_sample(question: str = "Alpha 的版本是什么？") -> BenchmarkSample:
    return BenchmarkSample(
        question_id="q1",
        question=question,
        answer_short="v1",
        question_type=QuestionType.FACT,
        source_scope=SourceScope.SINGLE_DOC,
        entities=["Alpha"],
        relations=["版本"],
        constraints=[],
    )


def _build_indexes() -> tuple[PublicIndex, GraphIndex]:
    text_index = PublicIndex(
        [
            SectionDocument(
                source_id="doc_a",
                section_id="sec_gold",
                content="Alpha 当前版本是 v1。",
                doc_title="gold",
            ),
            SectionDocument(
                source_id="doc_a",
                section_id="sec_compare",
                content="Alpha 与 Beta 的区别在于部署方式和兼容性。",
                doc_title="compare",
            ),
        ]
    )
    graph_index = GraphIndex.build(
        [
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_gold",
                content="Alpha 当前版本是 v1。",
                entities=["Alpha"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="版本", object="v1")],
                constraints=[],
            ),
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_compare",
                content="Alpha 与 Beta 的区别在于部署方式和兼容性。",
                entities=["Alpha", "Beta"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="区别", object="Beta")],
                constraints=[],
            ),
        ]
    )
    return text_index, graph_index


def test_dynamic_relation_drive_keeps_relation_driven_unchanged() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample("Alpha 的版本是什么？")
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    docs = [
        RetrievedDocument(
            source_id="doc_a",
            section_id="sec_gold",
            content="Alpha 当前版本是 v1。",
            score=0.9,
            rank=1,
            metadata={"dense_score": 0.9},
        )
    ]
    retriever = _RecordingRetriever(docs)
    relation = RelationDrivenRetriever(text_index, graph_index, text_retriever=retriever)

    result = relation.retrieve(sample, skeleton, top_k=1, use_relation_driven=True, use_skeleton_rewrite=False, query_override="q")

    assert result.details["mode"] == "relation_driven"
    assert result.documents[0].metadata["retriever"] == "relation_driven"


def test_dynamic_relation_drive_uses_question_type_weights_and_high_semantic_protection() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample("Alpha 的版本是什么？")
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    docs = [
        RetrievedDocument(
            source_id="doc_a",
            section_id="sec_gold",
            content="Alpha 当前版本是 v1。",
            score=0.92,
            rank=1,
            metadata={"dense_score": 0.92},
        )
    ]
    retriever = DynamicRelationDriveRetriever(text_index, graph_index, text_retriever=_RecordingRetriever(docs))

    result = retriever.retrieve(
        sample,
        skeleton,
        top_k=1,
        use_skeleton_rewrite=False,
        query_override="q",
        question_type_info=QuestionTypeClassification(
            question_type="fact_attribute",
            question_type_confidence="high",
            question_type_evidence={"trigger_terms": ["是什么"], "entity_count": 1, "relation_count": 1, "constraint_count": 0},
        ),
    )

    metadata = result.documents[0].metadata
    assert metadata["dynamic_weight_profile"] == {"w_sem": 0.58, "w_ent": 0.22, "w_rel": 0.08, "w_con": 0.12}
    assert metadata["whether_high_semantic_protection_triggered"] is True


def test_dynamic_relation_drive_fact_attribute_does_not_hard_drop_on_missing_relation() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample("Alpha 的版本是什么？")
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    skeleton.structured_relations[0].name = "不存在的关系"
    skeleton.structured_relations[0].normalized_name = "不存在的关系"
    docs = [
        RetrievedDocument(
            source_id="doc_a",
            section_id="sec_gold",
            content="Alpha 当前版本是 v1。",
            score=0.9,
            rank=1,
            metadata={"dense_score": 0.9},
        )
    ]
    retriever = DynamicRelationDriveRetriever(text_index, graph_index, text_retriever=_RecordingRetriever(docs))

    result = retriever.retrieve(
        sample,
        skeleton,
        top_k=1,
        use_skeleton_rewrite=False,
        query_override="q",
        question_type_info=QuestionTypeClassification(
            question_type="fact_attribute",
            question_type_confidence="high",
            question_type_evidence={"trigger_terms": ["是什么"], "entity_count": 1, "relation_count": 1, "constraint_count": 0},
        ),
    )

    assert [item.section_id for item in result.documents] == ["sec_gold"]


def test_dynamic_relation_drive_relation_compare_hard_drop_requires_weak_entity_and_relation() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample("Alpha 和 Beta 的区别是什么？")
    sample.entities = ["Alpha", "Beta"]
    sample.relations = ["区别"]
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    docs = [
        RetrievedDocument(
            source_id="doc_a",
            section_id="sec_unknown",
            content="完全无关文本。",
            score=0.1,
            rank=1,
            metadata={"dense_score": 0.1},
        ),
        RetrievedDocument(
            source_id="doc_a",
            section_id="sec_compare",
            content="Alpha 与 Beta 的区别在于部署方式和兼容性。",
            score=0.8,
            rank=2,
            metadata={"dense_score": 0.8},
        ),
    ]
    retriever = DynamicRelationDriveRetriever(text_index, graph_index, text_retriever=_RecordingRetriever(docs))

    result = retriever.retrieve(
        sample,
        skeleton,
        top_k=1,
        use_skeleton_rewrite=False,
        query_override="q",
        question_type_info=QuestionTypeClassification(
            question_type="relation_compare",
            question_type_confidence="high",
            question_type_evidence={"trigger_terms": ["区别"], "entity_count": 2, "relation_count": 1, "constraint_count": 0},
        ),
    )

    filtered_breakdown = result.details["filtered_out"]["sec_unknown"]
    assert "structure_filter_drop" in filtered_breakdown["gate"]["reasons"]
