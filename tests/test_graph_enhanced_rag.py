from __future__ import annotations

from pathlib import Path

from core.schema import RetrievedDocument
from dataio.loaders import load_benchmark_samples, load_graph_section_records
from modules.graph_expander import GraphExpander, GraphIndex
from modules.graph_organizer import GraphOrganizer
from pipelines.base import PublicIndex
from pipelines.graph_enhanced_rag import (
    GraphEnhancedRAGConfig,
    GraphEnhancedRAGPipeline,
    build_graph_retriever_config,
    summarize_graph_retrieval,
)
from retrievers.graph_retriever import GraphRetriever


class _FakeSeedRetriever:
    def __init__(self, docs: list[RetrievedDocument]) -> None:
        self.docs = docs
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        self.calls.append((query, top_k))
        return self.docs[:top_k]


class _FakeReranker:
    def rerank(self, query: str, documents: list[RetrievedDocument], top_n: int | None = None) -> list[RetrievedDocument]:
        ordered = sorted(documents, key=lambda item: float(item.metadata.get("rerank_score", 0.0)), reverse=True)
        if top_n is not None:
            ordered = ordered[:top_n]
        return [document.model_copy(update={"rank": rank}) for rank, document in enumerate(ordered, start=1)]


def test_load_graph_section_records_parses_relations() -> None:
    records = load_graph_section_records(Path("artifacts/two_file_demo/knowledge_extraction.jsonl"))
    assert records
    assert records[0].source_id
    assert records[0].section_id
    assert isinstance(records[0].entities, list)
    assert isinstance(records[0].relations, list)


def test_graph_index_builds_entity_and_relation_mappings() -> None:
    records = load_graph_section_records(Path("artifacts/two_file_demo/knowledge_extraction.jsonl"))
    graph_index = GraphIndex.build(records)
    assert graph_index.section_records
    assert graph_index.entity_to_sections
    assert graph_index.relation_to_sections


def test_graph_expander_returns_reasoned_candidates() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    text_index = PublicIndex.from_markdown_sections(Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    graph_index = GraphIndex.build(load_graph_section_records(Path("artifacts/two_file_demo/knowledge_extraction.jsonl")))
    seed_documents = text_index.search(samples[0].question, top_k=2)
    expander = GraphExpander(graph_index)
    candidates = expander.expand(samples[0], seed_documents, max_expand=5, use_gold_hints=True)
    assert isinstance(candidates, list)
    if candidates:
        assert candidates[0].section_id
        assert candidates[0].reasons


def test_graph_organizer_keeps_seed_and_expanded_candidates() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    text_index = PublicIndex.from_markdown_sections(Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    graph_index = GraphIndex.build(load_graph_section_records(Path("artifacts/two_file_demo/knowledge_extraction.jsonl")))
    seed_documents = text_index.search(samples[0].question, top_k=2)
    expander = GraphExpander(graph_index)
    organizer = GraphOrganizer(text_index, graph_index)
    candidates = expander.expand(samples[0], seed_documents, max_expand=5, use_gold_hints=True)
    result = organizer.organize(query=samples[0].question, seed_documents=seed_documents, expanded_candidates=candidates, keep_top_k=4)
    assert result.final_documents
    assert result.kept_section_ids
    assert "scores" in result.organization_details


def test_graph_retriever_runs_end_to_end() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    config = build_graph_retriever_config(GraphEnhancedRAGConfig(seed_top_k=2, expand_k=3, final_top_k=3))
    retriever = GraphRetriever.from_paths(
        sections_path=Path("artifacts/two_file_demo/markdown_sections.jsonl"),
        knowledge_path=Path("artifacts/two_file_demo/knowledge_extraction.jsonl"),
        config=config,
    )
    output = retriever.retrieve(samples[0])
    assert output.seed_documents
    assert output.organized.final_documents


def test_graph_enhanced_rag_pipeline_records_trace() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    config = GraphEnhancedRAGConfig(seed_top_k=2, expand_k=3, final_top_k=3, rerank=True, use_gold_hints=True)
    retriever = GraphRetriever.from_paths(
        sections_path=Path("artifacts/two_file_demo/markdown_sections.jsonl"),
        knowledge_path=Path("artifacts/two_file_demo/knowledge_extraction.jsonl"),
        config=build_graph_retriever_config(config),
    )
    pipeline = GraphEnhancedRAGPipeline(retriever, config=config)
    record = pipeline.run(samples[0])
    assert record.system_name == "graph_enhanced_rag"
    assert "seed_sections" in record.trace
    assert "expanded_sections" in record.trace
    assert "expansion_reasons" in record.trace
    assert "final_kept_sections" in record.trace
    assert "graph_expansion" in record.trace


def test_summarize_graph_retrieval_returns_expected_keys() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    config = GraphEnhancedRAGConfig(seed_top_k=2, expand_k=2, final_top_k=2, rerank=False, use_gold_hints=True)
    retriever = GraphRetriever.from_paths(
        sections_path=Path("artifacts/two_file_demo/markdown_sections.jsonl"),
        knowledge_path=Path("artifacts/two_file_demo/knowledge_extraction.jsonl"),
        config=build_graph_retriever_config(config),
    )
    pipeline = GraphEnhancedRAGPipeline(retriever, config=config)
    records = [pipeline.run(sample) for sample in samples[:3]]
    summary = summarize_graph_retrieval(records)
    assert set(summary) == {"expanded_count", "seed_hit_rate", "post_expand_hit_rate"}


def test_graph_enhanced_rag_pipeline_supports_dense_seed_and_rerank_trace() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    sample = samples[0]
    text_index = PublicIndex.from_markdown_sections(Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    graph_index = GraphIndex.build(load_graph_section_records(Path("artifacts/two_file_demo/knowledge_extraction.jsonl")))
    dense_seed_docs = [
        RetrievedDocument(
            source_id=sample.evidence[0].source_id,
            section_id=sample.evidence[0].section_id,
            content="dense seed",
            score=0.82,
            rank=1,
            metadata={"retriever": "qdrant_dense", "dense_score": 0.82, "rerank_score": 0.9},
        )
    ]
    retriever = GraphRetriever(
        text_index=text_index,
        graph_index=graph_index,
        seed_retriever=_FakeSeedRetriever(dense_seed_docs),
        config=build_graph_retriever_config(
            GraphEnhancedRAGConfig(seed_top_k=1, expand_k=1, final_top_k=1, rerank=True, retrieval_mode="dense")
        ),
    )
    pipeline = GraphEnhancedRAGPipeline(
        retriever,
        config=GraphEnhancedRAGConfig(seed_top_k=1, expand_k=1, final_top_k=1, rerank=True, retrieval_mode="dense"),
        reranker=_FakeReranker(),
    )

    record = pipeline.run(sample)

    assert record.trace["retrieval_mode"] == "dense"
    assert "initial_dense_candidates" in record.trace
    assert "final_reranked_candidates" in record.trace
    assert sample.evidence[0].section_id in record.trace["seed_dense_recall_scores"]
    assert "pre_rerank_sections" in record.trace
    assert "rerank_scores" in record.trace
