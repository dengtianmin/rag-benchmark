from __future__ import annotations

from pathlib import Path

from core.schema import RetrievedDocument
from dataio.loaders import load_benchmark_samples
from modules.query_rewriter import QueryRewriter
from modules.skeleton_utils import build_entity_relation_query, build_question_skeleton, flatten_skeleton_to_query_parts
from pipelines.base import PublicIndex
from pipelines.rewrite_rag import (
    RewriteRAGConfig,
    RewriteRAGPipeline,
    compare_retrievals,
    summarize_rewrite_improvements,
)


class _FakeRetriever:
    def __init__(self, mapping: dict[str, list[RetrievedDocument]]) -> None:
        self.mapping = mapping
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        self.calls.append((query, top_k))
        return self.mapping.get(query, [])[:top_k]


class _FakeReranker:
    def rerank(self, query: str, documents: list[RetrievedDocument], top_n: int | None = None) -> list[RetrievedDocument]:
        ordered = sorted(documents, key=lambda item: float(item.metadata.get("rerank_score", 0.0)), reverse=True)
        if top_n is not None:
            ordered = ordered[:top_n]
        return [document.model_copy(update={"rank": rank}) for rank, document in enumerate(ordered, start=1)]


def test_naive_rewrite_returns_keywords() -> None:
    rewriter = QueryRewriter()
    output = rewriter.rewrite("北京总部的具体地址是什么？", mode="naive")
    assert output.rewritten_query
    assert output.details["strategy"] == "naive"
    assert "keywords" in output.details


def test_entity_relation_rewrite_injects_schema_fields() -> None:
    sample = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))[0]
    rewriter = QueryRewriter()
    output = rewriter.rewrite(sample.question, sample=sample, mode="entity_relation")
    assert sample.entities[0] in output.rewritten_query
    assert output.details["relations"] == sample.relations
    assert output.details["constraints"] == sample.constraints
    assert output.details["strategy"] == "entity_relation"


def test_skeleton_utils_build_expected_parts() -> None:
    sample = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))[0]
    skeleton = build_question_skeleton(sample)
    parts = flatten_skeleton_to_query_parts(skeleton)
    query, details = build_entity_relation_query(sample)
    assert parts["entities"] == sample.entities
    assert parts["relations"] == sample.relations
    assert parts["constraints"] == sample.constraints
    assert sample.question in query
    assert details["entities"] == sample.entities


def test_compare_retrievals_detects_improvement() -> None:
    sample = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))[0]
    index = PublicIndex.from_markdown_sections(Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    gold_section = sample.evidence[0].section_id
    some_other = next(doc.section_id for doc in index.documents if doc.section_id != gold_section)
    original_docs = [
        index.search("新华三服务", top_k=1)[0].model_copy(update={"section_id": some_other}),
    ]
    rewritten_docs = [
        index.search(sample.question, top_k=5)[0].model_copy(update={"section_id": gold_section}),
    ]
    comparison = compare_retrievals(sample, original_docs, rewritten_docs, k=1)
    assert comparison["original_hit"] is False
    assert comparison["rewritten_hit"] is True
    assert comparison["hit_improved"] is True
    assert comparison["coverage_improved"] is True


def test_rewrite_rag_pipeline_writes_rewritten_query_and_trace() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    index = PublicIndex.from_markdown_sections(Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    pipeline = RewriteRAGPipeline(
        index=index,
        config=RewriteRAGConfig(top_k=3, rerank=True, mode="entity_relation"),
    )
    record = pipeline.run(samples[0])
    assert record.rewritten_query
    assert record.trace["rewrite_mode"] == "entity_relation"
    assert "rewrite_details" in record.trace
    assert "retrieval_comparison" in record.trace
    output = record.to_output_dict()
    assert output["rewritten_query"] == record.rewritten_query


def test_summarize_rewrite_improvements() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    index = PublicIndex.from_markdown_sections(Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    pipeline = RewriteRAGPipeline(
        index=index,
        config=RewriteRAGConfig(top_k=2, rerank=False, mode="naive"),
    )
    records = [pipeline.run(sample) for sample in samples[:3]]
    summary = summarize_rewrite_improvements(records)
    assert set(summary) == {
        "hit_improvement_rate",
        "coverage_improvement_rate",
        "rewritten_hit_rate",
        "original_hit_rate",
    }


def test_rewrite_rag_pipeline_supports_dense_recall_and_rerank_trace() -> None:
    sample = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))[0]
    rewritten_query = "rewritten dense query"
    original_docs = [
        RetrievedDocument(
            source_id="doc_x",
            section_id="sec_x",
            content="无关内容",
            score=0.76,
            rank=1,
            metadata={"retriever": "qdrant_dense", "dense_score": 0.76, "rerank_score": 0.40},
        )
    ]
    rewritten_docs = [
        RetrievedDocument(
            source_id=sample.evidence[0].source_id,
            section_id=sample.evidence[0].section_id,
            content="命中文本",
            score=0.88,
            rank=1,
            metadata={"retriever": "qdrant_dense", "dense_score": 0.88, "rerank_score": 0.91},
        )
    ]
    retriever = _FakeRetriever(
        {
            sample.question: original_docs,
            rewritten_query: rewritten_docs,
        }
    )
    rewriter = QueryRewriter()
    rewriter.rewrite = lambda question, sample=None, mode="naive": type(  # type: ignore[method-assign]
        "RewriteOutputStub",
        (),
        {"rewritten_query": rewritten_query, "details": {"strategy": "stub_dense"}},
    )()

    pipeline = RewriteRAGPipeline(
        index=PublicIndex([]),
        retriever=retriever,
        rewriter=rewriter,
        reranker=_FakeReranker(),
        config=RewriteRAGConfig(top_k=1, rerank=True, mode="naive", retrieval_mode="dense"),
    )
    record = pipeline.run(sample)

    assert retriever.calls == [(sample.question, 1), (rewritten_query, 1)]
    assert record.trace["retrieval_mode"] == "dense"
    assert "initial_dense_candidates" in record.trace
    assert "final_reranked_candidates" in record.trace
    assert record.trace["original_dense_recall_scores"] == {"sec_x": 0.76}
    assert record.trace["rewritten_dense_recall_scores"] == {sample.evidence[0].section_id: 0.88}
    assert record.trace["original_rerank_scores"] == {"sec_x": 0.4}
    assert record.trace["rewritten_rerank_scores"] == {sample.evidence[0].section_id: 0.91}
    assert record.trace["final_kept_sections"] == [sample.evidence[0].section_id]
