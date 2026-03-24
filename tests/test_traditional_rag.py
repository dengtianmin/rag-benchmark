from __future__ import annotations

from pathlib import Path

from core.schema import (
    AnswerResult,
    BenchmarkSample,
    EvidenceItem,
    PipelineRunRecord,
    RetrievalResult,
    RetrievedDocument,
)
from dataio.loaders import load_benchmark_samples
from evaluation.answer_metrics import aggregate_answer_metrics, evaluate_answer_record
from evaluation.retrieval_metrics import (
    aggregate_retrieval_metrics,
    evaluate_retrieval_record,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from pipelines.base import PublicIndex
from pipelines.traditional_rag import TraditionalRAGConfig, TraditionalRAGPipeline


def _build_record(
    *,
    pred_answer: str = "北京",
    gold_answer: str = "北京",
    retrieved_section_ids: list[str] | None = None,
    gold_section_ids: list[str] | None = None,
) -> PipelineRunRecord:
    retrieved_section_ids = retrieved_section_ids or ["sec_gold", "sec_other"]
    gold_section_ids = gold_section_ids or ["sec_gold"]
    sample = BenchmarkSample(
        question_id="qa_test",
        question="北京总部在哪里？",
        answer_short=gold_answer,
        question_type="fact",
        source_scope="single_section",
        evidence=[EvidenceItem(source_id="doc_1", section_id=section_id, quote="北京总部") for section_id in gold_section_ids],
        source_id="doc_1",
        section_id=gold_section_ids[0],
    )
    retrieval = RetrievalResult(
        question_id="qa_test",
        query=sample.question,
        retrieved_documents=[
            RetrievedDocument(
                source_id="doc_1",
                section_id=section_id,
                content=f"content for {section_id}",
                score=1.0 / rank,
                rank=rank,
            )
            for rank, section_id in enumerate(retrieved_section_ids, start=1)
        ],
        retrieved_triples=[],
    )
    answer = AnswerResult(
        question_id="qa_test",
        answer_text=pred_answer,
        answer_short=pred_answer,
        supporting_evidence=[EvidenceItem(source_id="doc_1", section_id=gold_section_ids[0], quote="北京总部")],
    )
    return PipelineRunRecord(
        question_id="qa_test",
        system_name="traditional_rag",
        sample=sample,
        retrieval=retrieval,
        answer=answer,
        rewritten_query=None,
        trace={"unit_test": True},
    )


def test_pipeline_run_record_to_output_dict_matches_required_fields() -> None:
    record = _build_record()
    output = record.to_output_dict()
    assert output["qid"] == "qa_test"
    assert output["method_name"] == "traditional_rag"
    assert output["rewritten_query"] is None
    assert output["retrieved_doc_ids"] == ["doc_1", "doc_1"]
    assert output["retrieved_section_ids"] == ["sec_gold", "sec_other"]
    assert output["retrieved_triples"] == []
    assert output["supporting_evidence"][0]["section_id"] == "sec_gold"
    assert output["pred_answer"] == "北京"
    assert "trace" in output


def test_answer_metrics_for_exact_match_case() -> None:
    record = _build_record(pred_answer="北京", gold_answer="北京")
    metrics = evaluate_answer_record(record)
    assert metrics["em"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_retrieval_metrics_for_hit_case() -> None:
    record = _build_record(retrieved_section_ids=["sec_x", "sec_gold", "sec_y"], gold_section_ids=["sec_gold"])
    assert recall_at_k(record, 2) == 1.0
    assert precision_at_k(record, 2) == 0.5
    assert hit_at_k(record, 2) == 1.0
    assert reciprocal_rank(record) == 0.5
    metrics = evaluate_retrieval_record(record, 2)
    assert metrics["recall@2"] == 1.0
    assert metrics["precision@2"] == 0.5
    assert metrics["hit@2"] == 1.0
    assert metrics["mrr"] == 0.5


def test_metric_aggregation() -> None:
    records = [
        _build_record(pred_answer="北京", gold_answer="北京"),
        _build_record(pred_answer="上海", gold_answer="北京", retrieved_section_ids=["sec_other"], gold_section_ids=["sec_gold"]),
    ]
    answer_metrics = aggregate_answer_metrics(records)
    retrieval_metrics = aggregate_retrieval_metrics(records, k=1)
    assert answer_metrics["em"] == 0.5
    assert answer_metrics["accuracy"] == 0.5
    assert retrieval_metrics["hit@1"] == 0.5
    assert retrieval_metrics["mrr"] == 0.5


def test_traditional_rag_pipeline_runs_on_demo_subset() -> None:
    dataset_path = Path("outputs/two_file_demo/benchmark_dataset.jsonl")
    sections_path = Path("artifacts/two_file_demo/markdown_sections.jsonl")
    samples = load_benchmark_samples(dataset_path)
    index = PublicIndex.from_markdown_sections(sections_path)
    pipeline = TraditionalRAGPipeline(index=index, config=TraditionalRAGConfig(top_k=3, rerank=True))
    record = pipeline.run(samples[0])

    assert record.question_id == samples[0].question_id
    assert record.system_name == "traditional_rag"
    assert record.retrieval.query == samples[0].question
    assert len(record.retrieval.retrieved_documents) <= 3
    assert record.answer.answer_text is not None
    output = record.to_output_dict()
    assert output["qid"] == samples[0].question_id
    assert isinstance(output["retrieved_doc_ids"], list)
    assert isinstance(output["retrieved_section_ids"], list)
