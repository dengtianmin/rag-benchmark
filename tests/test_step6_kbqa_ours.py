from __future__ import annotations

from pathlib import Path

from core.schema import RetrievedDocument
from dataio.loaders import load_benchmark_samples, load_graph_section_records
from modules.entity_linker import EntityLinker
from modules.graph_expander import GraphIndex
from modules.kb_executor import KBExecutor
from modules.relation_driven_retriever import RelationDrivenRetriever
from modules.relation_matcher import RelationMatcher
from modules.skeleton_extractor import SkeletonExtractor
from modules.text_compensator import TextCompensator
from pipelines.base import PublicIndex
from pipelines.kbqa_baseline import KBQABaselineConfig, KBQABaselinePipeline
from pipelines.ours_ch4 import OursCh4Config, OursCh4Pipeline, summarize_ablation


class _FakeRetriever:
    def __init__(self, docs: list[RetrievedDocument]) -> None:
        self.docs = docs
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        self.calls.append((query, top_k))
        return [document.model_copy() for document in self.docs[:top_k]]


class _FakeReranker:
    def rerank(self, query: str, documents: list[RetrievedDocument], top_n: int | None = None) -> list[RetrievedDocument]:
        ordered = sorted(documents, key=lambda item: float(item.metadata.get("rerank_score", 0.0)), reverse=True)
        if top_n is not None:
            ordered = ordered[:top_n]
        return [document.model_copy(update={"rank": rank}) for rank, document in enumerate(ordered, start=1)]


def _build_indexes() -> tuple[PublicIndex, GraphIndex]:
    text_index = PublicIndex.from_markdown_sections(Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    graph_index = GraphIndex.build(load_graph_section_records(Path("artifacts/two_file_demo/knowledge_extraction.jsonl")))
    return text_index, graph_index


def test_entity_linker_and_relation_matcher_gold_modes() -> None:
    sample = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))[0]
    _, graph_index = _build_indexes()
    entity_result = EntityLinker(graph_index).link(sample.question, sample=sample, mode="gold")
    relation_result = RelationMatcher(graph_index).match(sample.question, sample=sample, mode="gold")
    assert entity_result.linked_entities == sample.entities
    assert relation_result.matched_relations == sample.relations


def test_kb_executor_returns_triples() -> None:
    sample = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))[0]
    text_index, graph_index = _build_indexes()
    entity_result = EntityLinker(graph_index).link(sample.question, sample=sample, mode="gold")
    relation_result = RelationMatcher(graph_index).match(sample.question, sample=sample, mode="gold")
    execution = KBExecutor(graph_index, text_index).execute(sample, entity_result, relation_result, top_k=3)
    assert execution.retrieved_triples is not None
    assert "linked_entities" in execution.execution_trace


def test_kbqa_pipeline_outputs_retrieved_triples() -> None:
    sample = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))[0]
    text_index, graph_index = _build_indexes()
    pipeline = KBQABaselinePipeline(
        EntityLinker(graph_index),
        RelationMatcher(graph_index),
        KBExecutor(graph_index, text_index),
        config=KBQABaselineConfig(top_k=3, entity_mode="gold", relation_mode="gold"),
    )
    record = pipeline.run(sample)
    output = record.to_output_dict()
    assert record.system_name == "kbqa_baseline"
    assert output["retrieved_triples"] is not None
    assert "execution_trace" in record.trace


def test_skeleton_extractor_oracle_and_stub_modes() -> None:
    sample = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))[0]
    _, graph_index = _build_indexes()
    extractor = SkeletonExtractor(graph_index)
    oracle = extractor.extract(sample, mode="oracle")
    stub = extractor.extract(sample, mode="stub_predicted")
    assert oracle.entities == sample.entities
    assert oracle.relations == sample.relations
    assert stub.mode == "stub_predicted"


def test_relation_driven_retriever_returns_documents() -> None:
    sample = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))[0]
    text_index, graph_index = _build_indexes()
    skeleton = SkeletonExtractor(graph_index).extract(sample, mode="oracle")
    result = RelationDrivenRetriever(text_index, graph_index).retrieve(sample, skeleton, top_k=3)
    assert result.documents
    assert result.details["mode"] == "relation_driven"


def test_text_compensator_activates_for_explanation() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    explanation = next(sample for sample in samples if sample.question_type.value == "explanation")
    text_index, graph_index = _build_indexes()
    skeleton = SkeletonExtractor(graph_index).extract(explanation, mode="oracle")
    base = RelationDrivenRetriever(text_index, graph_index).retrieve(explanation, skeleton, top_k=2)
    compensation = TextCompensator(text_index, graph_index).compensate(explanation, base.documents, top_k=3, enabled=True)
    assert compensation.activated is True
    assert compensation.documents


def test_ours_ch4_pipeline_and_ablation_summary() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    text_index, graph_index = _build_indexes()
    pipeline = OursCh4Pipeline(
        text_index,
        SkeletonExtractor(graph_index),
        RelationDrivenRetriever(text_index, graph_index),
        TextCompensator(text_index, graph_index),
        config=OursCh4Config.from_ablation("full", top_k=3, skeleton_mode="oracle"),
    )
    records = [pipeline.run(sample) for sample in samples[:3]]
    assert records[0].system_name == "ours_ch4"
    assert "skeleton" in records[0].trace
    assert "text_compensation_activated" in records[0].trace
    summary = summarize_ablation(records)
    assert "text_compensation_activation_rate" in summary


def test_ours_ch4_pipeline_supports_dense_relation_and_compensation_trace() -> None:
    samples = load_benchmark_samples(Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    sample = next(item for item in samples if item.question_type.value == "explanation")
    text_index, graph_index = _build_indexes()
    dense_docs = [
        RetrievedDocument(
            source_id=sample.evidence[0].source_id,
            section_id=sample.evidence[0].section_id,
            content="dense hit content",
            score=0.7,
            rank=1,
            metadata={"retriever": "qdrant_dense", "dense_score": 0.7, "rerank_score": 0.9},
        ),
        RetrievedDocument(
            source_id="doc_other",
            section_id="sec_other",
            content="other hit",
            score=0.4,
            rank=2,
            metadata={"retriever": "qdrant_dense", "dense_score": 0.4, "rerank_score": 0.5},
        ),
    ]
    fake_retriever = _FakeRetriever(dense_docs)
    pipeline = OursCh4Pipeline(
        text_index,
        SkeletonExtractor(graph_index),
        RelationDrivenRetriever(text_index, graph_index, text_retriever=fake_retriever),
        TextCompensator(text_index, graph_index, text_retriever=fake_retriever),
        config=OursCh4Config(top_k=2, skeleton_mode="oracle", rerank=True, retrieval_mode="dense"),
        reranker=_FakeReranker(),
    )

    record = pipeline.run(sample)

    assert record.trace["retrieval_mode"] == "dense"
    assert "initial_dense_candidates" in record.trace
    assert "final_reranked_candidates" in record.trace
    assert "dense_scores" in record.trace["relation_driven_details"]
    assert "relation_scores" in record.trace["relation_driven_details"]
    assert "constraint_scores" in record.trace["relation_driven_details"]
    assert "backfill_scores" in record.trace["text_compensation_details"]
    assert "rerank_scores" in record.trace
