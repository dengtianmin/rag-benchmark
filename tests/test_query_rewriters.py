from __future__ import annotations

import json

from core.schema import BenchmarkSample
from core.types import QuestionType, SourceScope
from dataio.loaders import GraphRelationRecord, GraphSectionRecord
from clients.chat_llm_client import ChatLLMResponse
from modules.graph_expander import GraphIndex
from modules.query_rewriters import RetrievalLabQueryRewriter
from modules.skeleton_extractor import SkeletonExtractor


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


def _build_graph() -> GraphIndex:
    return GraphIndex.build(
        [
            GraphSectionRecord(
                source_id="doc_a",
                section_id="sec_a",
                content="2019年 Alpha 与 Beta 合作原因是为了降低成本。",
                entities=["Alpha", "Beta"],
                relations=[GraphRelationRecord(subject="Alpha", predicate="合作原因", object="降低成本")],
                constraints=["2019年"],
            )
        ]
    )


def test_retrieval_lab_query_rewriters_generate_queries() -> None:
    sample = _build_sample()
    skeleton = SkeletonExtractor(_build_graph()).extract(sample, mode="stub_predicted")
    rewriter = RetrievalLabQueryRewriter()

    original = rewriter.rewrite(sample, skeleton, mode="original")
    template = rewriter.rewrite(sample, skeleton, mode="template")
    rule_based = rewriter.rewrite(sample, skeleton, mode="rule_based")

    assert original.rewritten_query == sample.question
    assert template.rewritten_query
    assert rule_based.rewritten_query
    assert template.details["strategy"] == "deterministic_template"
    assert rule_based.details["strategy"] == "rule_based"


class _StubClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.api_key = "stub-key"
        self.model = "stub-model"

    def chat_completion(self, **kwargs) -> ChatLLMResponse:
        del kwargs
        return ChatLLMResponse(
            content=self.content,
            model_name=self.model,
            finish_reason="stop",
            raw={"choices": []},
            latency_ms=1,
        )


def test_llm_query_accepts_object_string_and_double_encoded_payloads() -> None:
    sample = _build_sample()
    skeleton = SkeletonExtractor(_build_graph()).extract(sample, mode="stub_predicted")

    object_rewriter = RetrievalLabQueryRewriter(llm_client=_StubClient('{"query":"alpha cause"}'))
    object_result = object_rewriter.rewrite(sample, skeleton, mode="llm")
    assert object_result.rewritten_query == "alpha cause"
    assert object_result.details["payload_kind"] == "object"

    string_rewriter = RetrievalLabQueryRewriter(llm_client=_StubClient('"alpha cause"'))
    string_result = string_rewriter.rewrite(sample, skeleton, mode="llm")
    assert string_result.rewritten_query == "alpha cause"
    assert string_result.details["payload_kind"] == "string"

    double_encoded = json.dumps('{"query":"alpha cause"}')
    double_rewriter = RetrievalLabQueryRewriter(llm_client=_StubClient(double_encoded))
    double_result = double_rewriter.rewrite(sample, skeleton, mode="llm")
    assert double_result.rewritten_query == "alpha cause"
    assert double_result.details["payload_kind"] == "double_encoded_object"


def test_llm_query_invalid_json_error_contains_raw_preview() -> None:
    sample = _build_sample()
    skeleton = SkeletonExtractor(_build_graph()).extract(sample, mode="stub_predicted")
    rewriter = RetrievalLabQueryRewriter(llm_client=_StubClient('{"query":"broken'))

    try:
        rewriter.rewrite(sample, skeleton, mode="llm")
    except ValueError as exc:
        message = str(exc)
        assert "LLM rewrite did not return valid JSON." in message
        assert "raw_output=" in message
    else:
        raise AssertionError("Expected ValueError for invalid JSON payload.")
