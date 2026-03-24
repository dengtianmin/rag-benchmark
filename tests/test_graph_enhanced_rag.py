from __future__ import annotations

from pathlib import Path

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
