from __future__ import annotations

from clients.chat_llm_client import ChatLLMResponse
from core.schema import (
    BenchmarkSample,
    QuestionSkeletonLabel,
    SkeletonRewritePayload,
    StructuredSkeletonConstraint,
    StructuredSkeletonEntity,
    StructuredSkeletonRelation,
)
from core.types import QuestionType, SourceScope
from dataio.loaders import GraphRelationRecord, GraphSectionRecord
from modules.graph_expander import GraphIndex
from modules.llm_query_rewrite_prompt import parse_llm_query_rewrite_payload
from modules.query_rewriters import RetrievalLabQueryRewriter
from modules.skeleton_extractor import SkeletonExtractionResult, SkeletonExtractor
from retrievers.text_retriever import TextRetrieverQuery


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


class _CapturingStubClient(_StubClient):
    def __init__(self, content: str) -> None:
        super().__init__(content)
        self.calls: list[dict] = []

    def chat_completion(self, **kwargs) -> ChatLLMResponse:
        self.calls.append(kwargs)
        return super().chat_completion(**kwargs)


def test_parse_llm_rewrite_payload_accepts_object_and_double_encoded_string() -> None:
    payload = parse_llm_query_rewrite_payload(
        '{"must_keep_terms":["Alpha","合作原因"],"sparse_rewrite":"Alpha Beta 合作原因 2019年","dense_rewrite":"2019年 Alpha 与 Beta 合作原因"}'
    )
    assert payload.must_keep_terms == ["Alpha", "合作原因"]
    assert payload.sparse_rewrite == "Alpha Beta 合作原因 2019年"
    assert payload.dense_rewrite == "2019年 Alpha 与 Beta 合作原因"

    payload = parse_llm_query_rewrite_payload(
        '"{\\"must_keep_terms\\":[\\"Alpha\\"],\\"sparse_rewrite\\":\\"Alpha 合作原因\\",\\"dense_rewrite\\":\\"Alpha 合作原因 是什么\\"}"'
    )
    assert payload.must_keep_terms == ["Alpha"]
    assert payload.sparse_rewrite == "Alpha 合作原因"
    assert payload.dense_rewrite == "Alpha 合作原因 是什么"


def test_retrieval_lab_query_rewriters_keep_old_modes_and_add_splicing() -> None:
    sample = _build_sample()
    skeleton = SkeletonExtractor(_build_graph()).extract(sample, mode="stub_predicted")
    rewriter = RetrievalLabQueryRewriter()

    original = rewriter.rewrite(sample, skeleton, mode="original")
    template = rewriter.rewrite(sample, skeleton, mode="template")
    splicing = rewriter.rewrite(sample, skeleton, mode="splicing")
    rule_based = rewriter.rewrite(sample, skeleton, mode="rule_based")

    assert original.rewritten_query == sample.question
    assert template.rewritten_query
    assert splicing.rewritten_query == template.rewritten_query
    assert rule_based.rewritten_query
    assert template.details["strategy"] == "deterministic_splicing"
    assert rule_based.details["strategy"] == "rule_based"


def test_llm_rewrite_modes_select_expected_queries() -> None:
    sample = _build_sample()
    skeleton = SkeletonExtractor(_build_graph()).extract(sample, mode="stub_predicted")
    client = _StubClient(
        '{"must_keep_terms":["Alpha","Beta","合作原因","2019年"],"sparse_rewrite":"Alpha Beta 合作原因 2019年 原因","dense_rewrite":"2019年 Alpha 与 Beta 合作的原因和相关描述"}'
    )
    rewriter = RetrievalLabQueryRewriter(llm_client=client)

    sparse_result = rewriter.rewrite(sample, skeleton, mode="sparse_llm")
    dense_result = rewriter.rewrite(sample, skeleton, mode="dense_llm")
    hybrid_result = rewriter.rewrite(sample, skeleton, mode="hybrid_llm")

    assert sparse_result.rewritten_query == "Alpha Beta 合作原因 2019年 原因"
    assert sparse_result.lexical_query == "Alpha Beta 合作原因 2019年 原因"
    assert sparse_result.dense_query == "Alpha Beta 合作原因 2019年 原因"

    assert dense_result.rewritten_query == "2019年 Alpha 与 Beta 合作的原因和相关描述"
    assert dense_result.lexical_query == "2019年 Alpha 与 Beta 合作的原因和相关描述"
    assert dense_result.dense_query == "2019年 Alpha 与 Beta 合作的原因和相关描述"

    assert hybrid_result.rewritten_query == "2019年 Alpha 与 Beta 合作的原因和相关描述"
    assert hybrid_result.lexical_query == "Alpha Beta 合作原因 2019年 原因"
    assert hybrid_result.dense_query == "2019年 Alpha 与 Beta 合作的原因和相关描述"


def test_hybrid_llm_query_bundle_routes_by_retrieval_mode() -> None:
    sample = _build_sample()
    skeleton = SkeletonExtractor(_build_graph()).extract(sample, mode="stub_predicted")
    client = _StubClient(
        '{"must_keep_terms":["Alpha","合作原因"],"sparse_rewrite":"Alpha 合作原因","dense_rewrite":"Alpha 合作原因 的相关说明"}'
    )
    rewriter = RetrievalLabQueryRewriter(llm_client=client)
    result = rewriter.rewrite(sample, skeleton, mode="hybrid_llm")

    assert result.query_for_retrieval("lexical") == "Alpha 合作原因"
    assert result.query_for_retrieval("dense") == "Alpha 合作原因 的相关说明"
    hybrid_query = result.query_for_retrieval("hybrid")
    assert isinstance(hybrid_query, TextRetrieverQuery)
    assert hybrid_query.lexical_query == "Alpha 合作原因"
    assert hybrid_query.dense_query == "Alpha 合作原因 的相关说明"
    structured = result.structured_rewrite()
    assert structured["question_type"] in {"cause_explanation", "fallback_balanced"}
    assert structured["entities"]
    assert structured["relations"]


def test_select_llm_rewrite_terms_prefers_structured_anchor_target_and_clips_lengths() -> None:
    skeleton = SkeletonExtractionResult(
        entities=["fallback_a", "fallback_b", "fallback_c"],
        relations=["fallback_r1", "fallback_r2"],
        constraints=["fallback_c1", "fallback_c2"],
        skeleton=QuestionSkeletonLabel(entities=["fallback_a"], relations=["fallback_r1"], constraints=["fallback_c1"]),
        mode="stub_predicted",
        structured_entities=[
            StructuredSkeletonEntity(name="Alpha", normalized_name="alpha", role="anchor", surface="Alpha", confidence=1.0, source="test"),
            StructuredSkeletonEntity(name="Beta", normalized_name="beta", role="anchor", surface="Beta", confidence=1.0, source="test"),
            StructuredSkeletonEntity(name="Gamma", normalized_name="gamma", role="anchor", surface="Gamma", confidence=1.0, source="test"),
            StructuredSkeletonEntity(name="Delta", normalized_name="delta", role="anchor", surface="Delta", confidence=1.0, source="test"),
        ],
        structured_relations=[
            StructuredSkeletonRelation(name="合作原因", normalized_name="合作原因", relation_type="cause", role="target", surface="合作原因", confidence=1.0, source="test"),
            StructuredSkeletonRelation(name="签约流程", normalized_name="签约流程", relation_type="procedure", role="target", surface="签约流程", confidence=1.0, source="test"),
            StructuredSkeletonRelation(name="审批条件", normalized_name="审批条件", relation_type="condition", role="target", surface="审批条件", confidence=1.0, source="test"),
            StructuredSkeletonRelation(name="额外噪声", normalized_name="额外噪声", relation_type="relation", role="target", surface="额外噪声", confidence=1.0, source="test"),
        ],
        structured_constraints=[
            StructuredSkeletonConstraint(kind="time", value="2019年", normalized_value="2019年", confidence=1.0, source="test"),
            StructuredSkeletonConstraint(kind="scope", value="华东大区", normalized_value="华东大区", confidence=1.0, source="test"),
            StructuredSkeletonConstraint(kind="scope", value="应被裁剪", normalized_value="应被裁剪", confidence=1.0, source="test"),
        ],
        rewrite_payload=SkeletonRewritePayload(
            original_query="q",
            retrieval_query="q",
            compensation_query="q",
        ),
        details={},
    )
    rewriter = RetrievalLabQueryRewriter()

    anchor_entities, target_relations, constraints = rewriter._select_llm_rewrite_terms(skeleton)

    assert anchor_entities == ["Alpha", "Beta", "Gamma", "Delta"]
    assert target_relations == ["合作原因", "签约流程", "审批条件", "额外噪声"]
    assert constraints == ["2019年", "华东大区", "应被裁剪"]


def test_llm_rewrite_uses_conservative_prompt_and_json_only_fields() -> None:
    sample = _build_sample()
    skeleton = SkeletonExtractor(_build_graph()).extract(sample, mode="stub_predicted")
    client = _CapturingStubClient(
        '{"must_keep_terms":["Alpha","合作原因"],"sparse_rewrite":"Alpha 合作原因","dense_rewrite":"Alpha 合作原因 的相关说明"}'
    )
    rewriter = RetrievalLabQueryRewriter(llm_client=client)

    result = rewriter.rewrite(sample, skeleton, mode="hybrid_llm")

    assert result.lexical_query == "Alpha 合作原因"
    assert len(client.calls) == 1
    messages = client.calls[0]["messages"]
    assert len(messages) == 2
    system_prompt = messages[0]["content"]
    user_prompt = messages[1]["content"]

    assert "严禁回答问题" in system_prompt
    assert "must_keep_terms, sparse_rewrite, dense_rewrite" in system_prompt
    assert "示例输入" in user_prompt
    assert "示例输出" in user_prompt
    assert '"question": "2019年 Alpha 与 Beta 合作的原因是什么？"' in user_prompt
