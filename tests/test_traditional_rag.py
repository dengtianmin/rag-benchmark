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
from pipelines.base import NO_RETRIEVAL_ANSWER, PublicIndex, SectionDocument
from pipelines.traditional_rag import TraditionalRAGConfig, TraditionalRAGPipeline
from rerankers.tei_reranker import TEIReranker


class _FakeRetriever:
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


class _FakeResponse:
    def __init__(self, payload: list[dict], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code == 200
        self.text = str(payload)

    def json(self) -> list[dict]:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def post(self, url: str, *, json: dict, headers: dict, timeout: float) -> _FakeResponse:
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return self.responses.pop(0)


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


def test_answer_metrics_only_use_answer_text_not_supporting_evidence() -> None:
    record = _build_record(pred_answer="北京", gold_answer="北京")
    record.answer.supporting_evidence = [
        EvidenceItem(source_id="doc_1", section_id="sec_gold", quote='{"answer":"上海","supporting_evidence":[9]}')
    ]

    metrics = evaluate_answer_record(record)

    assert metrics["em"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["accuracy"] == 1.0


def test_answer_metrics_prefer_answer_text_even_if_answer_short_differs() -> None:
    record = _build_record(pred_answer="北京", gold_answer="北京")
    record.answer.answer_short = "上海"
    record.answer.answer_long = "深圳"

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


def test_traditional_rag_pipeline_returns_fallback_answer_when_no_document_is_retrieved() -> None:
    sample = BenchmarkSample(
        question_id="qa_no_hit",
        question="完全不存在的查询词",
        answer_short="标准答案",
        question_type="fact",
        source_scope="single_section",
        evidence=[EvidenceItem(source_id="doc_1", section_id="sec_1", quote="标准答案")],
        source_id="doc_1",
        section_id="sec_1",
    )
    index = PublicIndex(
        [
            SectionDocument(
                source_id="doc_1",
                section_id="sec_1",
                content="这里没有任何相关词汇",
                doc_title="示例文档",
                section_path=["示例章节"],
            )
        ]
    )
    pipeline = TraditionalRAGPipeline(index=index, config=TraditionalRAGConfig(top_k=3, rerank=True))

    record = pipeline.run(sample)

    assert record.retrieval.retrieved_documents == []
    assert record.answer.answer_text == NO_RETRIEVAL_ANSWER
    assert record.answer.supporting_evidence == []
    assert record.answer.metadata["no_retrieval"] is True


def test_traditional_rag_pipeline_supports_dense_retrieval_trace_and_rerank() -> None:
    sample = BenchmarkSample(
        question_id="qa_dense",
        question="数据库加密支持哪些能力？",
        answer_short="数据库加密与访问控制",
        question_type="fact",
        source_scope="single_section",
        evidence=[EvidenceItem(source_id="doc_1", section_id="sec_1", quote="数据库加密")],
        source_id="doc_1",
        section_id="sec_1",
    )
    index = PublicIndex([])
    docs = [
        RetrievedDocument(
            source_id="doc_1",
            section_id="sec_1",
            content="支持数据库加密与访问控制。",
            score=0.81,
            rank=1,
            metadata={"retriever": "qdrant_dense", "dense_score": 0.81, "rerank_score": 0.95},
        ),
        RetrievedDocument(
            source_id="doc_2",
            section_id="sec_2",
            content="提供统一日志审计能力。",
            score=0.85,
            rank=2,
            metadata={"retriever": "qdrant_dense", "dense_score": 0.85, "rerank_score": 0.60},
        ),
    ]
    retriever = _FakeRetriever(docs)
    pipeline = TraditionalRAGPipeline(
        index=index,
        retriever=retriever,
        reranker=_FakeReranker(),
        config=TraditionalRAGConfig(top_k=2, rerank=True, retrieval_mode="dense"),
    )

    record = pipeline.run(sample)

    assert retriever.calls == [(sample.question, 2)]
    assert record.trace["retrieval_mode"] == "dense"
    assert "initial_dense_candidates" in record.trace
    assert "final_reranked_candidates" in record.trace
    assert record.trace["dense_recall_scores"] == {"sec_1": 0.81, "sec_2": 0.85}
    assert record.trace["rerank_scores"] == {"sec_1": 0.95, "sec_2": 0.6}
    assert record.trace["final_kept_sections"] == ["sec_1", "sec_2"]


def test_traditional_rag_pipeline_uses_tei_reranker_and_records_trace() -> None:
    sample = BenchmarkSample(
        question_id="qa_tei",
        question="数据库加密支持哪些能力？",
        answer_short="数据库加密与访问控制",
        question_type="fact",
        source_scope="single_section",
        evidence=[EvidenceItem(source_id="doc_1", section_id="sec_1", quote="数据库加密")],
        source_id="doc_1",
        section_id="sec_1",
    )
    docs = [
        RetrievedDocument(source_id="doc_1", section_id="sec_1", content="支持数据库加密与访问控制。", score=0.81, rank=1),
        RetrievedDocument(source_id="doc_2", section_id="sec_2", content="提供统一日志审计能力。", score=0.85, rank=2),
    ]
    retriever = _FakeRetriever(docs)
    session = _FakeSession([_FakeResponse([{"index": 1, "score": 0.2}, {"index": 0, "score": 0.9}])])
    reranker = TEIReranker(base_url="http://127.0.0.1:8080", timeout=5, top_n=1, session=session)
    pipeline = TraditionalRAGPipeline(
        index=PublicIndex([]),
        retriever=retriever,
        reranker=reranker,
        config=TraditionalRAGConfig(top_k=2, rerank=True, rerank_top_n=1, retrieval_mode="dense"),
    )

    record = pipeline.run(sample)

    assert retriever.calls == [(sample.question, 2)]
    assert len(session.calls) == 1
    assert session.calls[0]["url"] == "http://127.0.0.1:8080/rerank"
    assert record.trace["rerank_backend"] == "tei"
    assert record.trace["rerank_url"] == "http://127.0.0.1:8080"
    assert record.trace["rerank_input_count"] == 2
    assert record.trace["rerank_top_n"] == 1
    assert record.trace["final_reranked_candidates"] == ["sec_1"]
    assert record.trace["rerank_scores"] == {"sec_1": 0.9}
