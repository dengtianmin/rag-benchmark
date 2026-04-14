from __future__ import annotations

from core.schema import BenchmarkSample, RetrievedDocument
from core.types import QuestionType, SourceScope
from dataio.loaders import GraphRelationRecord, GraphSectionRecord
from modules.graph_expander import GraphIndex
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractor
from pipelines.base import PublicIndex, SectionDocument
from retrievers.text_retriever import TextRetrieverQuery


class _RecordingRetriever:
    def __init__(self, documents: list[RetrievedDocument]) -> None:
        self.documents = documents
        self.calls: list[tuple[str | TextRetrieverQuery, int]] = []

    def retrieve(self, query: str | TextRetrieverQuery, top_k: int) -> list[RetrievedDocument]:
        self.calls.append((query, top_k))
        return self.documents[:top_k]


def _build_sample() -> BenchmarkSample:
    return BenchmarkSample(
        question_id="q1",
        question="2019年 Alpha 与 Beta 合作的原因是什么？",
        answer_short="为了降低成本",
        question_type=QuestionType.EXPLANATION,
        source_scope=SourceScope.SINGLE_DOC,
        entities=["Alpha", "Beta"],
        relations=["合作原因"],
        constraints=["2019年"],
    )


def _build_indexes() -> tuple[PublicIndex, GraphIndex]:
    text_index = PublicIndex(
        [
            SectionDocument(
                source_id="doc_a",
                section_id="sec_reason",
                content="2019年 Alpha 与 Beta 合作原因是为了降低成本。",
                doc_title="原因",
            )
        ]
    )
    graph_index = GraphIndex.build(
        [
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_reason",
                content="2019年 Alpha 与 Beta 合作原因是为了降低成本。",
                entities=["Alpha", "Beta"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="合作原因", object="降低成本")],
                constraints=["2019年"],
            )
        ]
    )
    return text_index, graph_index


def test_relation_driven_retriever_uses_query_override() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample()
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="stub_predicted")
    base_documents = [
        RetrievedDocument(
            source_id="doc_a",
            section_id="sec_reason",
            content="2019年 Alpha 与 Beta 合作原因是为了降低成本。",
            score=0.9,
            rank=1,
            metadata={"dense_score": 0.9},
        )
    ]
    text_retriever = _RecordingRetriever(base_documents)
    retriever = RelationDrivenRetriever(text_index, graph_index, text_retriever=text_retriever)

    result = retriever.retrieve(
        sample,
        skeleton,
        top_k=1,
        use_relation_driven=True,
        use_skeleton_rewrite=True,
        query_override="override query",
    )

    assert text_retriever.calls == [("override query", 4)]
    assert result.details["query_source"] == "override"
    assert result.details["stage1_query"] == "override query"


def test_relation_driven_retriever_preserves_hybrid_query_override() -> None:
    text_index, graph_index = _build_indexes()
    sample = _build_sample()
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="stub_predicted")
    base_documents = [
        RetrievedDocument(
            source_id="doc_a",
            section_id="sec_reason",
            content="2019年 Alpha 与 Beta 合作原因是为了降低成本。",
            score=0.9,
            rank=1,
            metadata={"dense_score": 0.9},
        )
    ]
    text_retriever = _RecordingRetriever(base_documents)
    retriever = RelationDrivenRetriever(text_index, graph_index, text_retriever=text_retriever)
    query_override = TextRetrieverQuery(
        lexical_query="Alpha 合作原因 2019年",
        dense_query="2019年 Alpha 与 Beta 合作原因",
    )

    result = retriever.retrieve(
        sample,
        skeleton,
        top_k=1,
        use_relation_driven=True,
        use_skeleton_rewrite=True,
        query_override=query_override,
    )

    assert text_retriever.calls == [(query_override, 4)]
    assert result.details["query_source"] == "override"
    assert result.details["stage1_query"] == {
        "lexical_query": "Alpha 合作原因 2019年",
        "dense_query": "2019年 Alpha 与 Beta 合作原因",
    }
    assert result.details["candidate_pool"][0]["section_id"] == "sec_reason"
