from __future__ import annotations

from core.schema import BenchmarkSample, RetrievedDocument
from core.types import QuestionType, SourceScope
from generators import build_generator
from generators.llm_generator import LLMGenerator, STANDARD_INSUFFICIENT_ANSWER
from pipelines.base import MockGenerator
from prompts.rag_prompt_builder import RAGPromptBuilder
from runtime_config import RuntimeSettings


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.model_name = "test-model"
        self.finish_reason = "stop"
        self.latency_ms = 12
        self.raw = {}


class _FakeClient:
    def __init__(self, *, enabled: bool = True, content: str = "", error: Exception | None = None) -> None:
        self.enabled = enabled
        self.content = content
        self.error = error
        self.model = "test-model"
        self.base_url = "http://example.test/v1"

    def chat_completion(self, *, messages: list[dict[str, str]], temperature=None, max_tokens=None, json_mode=None):
        if self.error is not None:
            raise self.error
        return _FakeResponse(self.content)


def _sample() -> BenchmarkSample:
    return BenchmarkSample(
        question_id="q1",
        question="北京总部地址是什么？",
        answer_short="北京市朝阳区广顺南大街8号院利星行中心1号楼",
        answer_long="北京市朝阳区广顺南大街8号院利星行中心1号楼。",
        question_type=QuestionType.FACT,
        source_scope=SourceScope.SINGLE_SECTION,
        source_id="doc1",
        section_id="sec1",
        evidence=[],
    )


def _documents() -> list[RetrievedDocument]:
    return [
        RetrievedDocument(
            source_id="doc1",
            section_id="sec1",
            content="北京总部。北京市朝阳区广顺南大街8号院利星行中心1号楼。邮编100102。",
            score=1.0,
            rank=1,
            metadata={},
        ),
        RetrievedDocument(
            source_id="doc1",
            section_id="sec2",
            content="杭州总部。杭州市滨江区长河路466号。邮编310052。",
            score=0.8,
            rank=2,
            metadata={},
        ),
    ]


def test_build_generator_defaults_to_mock() -> None:
    settings = RuntimeSettings()
    generator = build_generator(settings)
    assert isinstance(generator, MockGenerator)


def test_llm_generator_builds_answer_result_from_json() -> None:
    generator = LLMGenerator(
        client=_FakeClient(
            content=(
                '{"answer":"北京市朝阳区广顺南大街8号院利星行中心1号楼",'
                '"supporting_evidence":[1]}'
            )
        )
    )

    result = generator.generate(_sample(), _documents())

    assert result.answer_text == "北京市朝阳区广顺南大街8号院利星行中心1号楼"
    assert result.answer_short == "北京市朝阳区广顺南大街8号院利星行中心1号楼"
    assert result.supporting_evidence[0].source_id == "doc1"
    assert result.supporting_evidence[0].section_id == "sec1"
    assert result.metadata["generator"] == "llm_generator"
    assert result.metadata["fallback_used"] is False
    assert result.metadata["selected_evidence_indices"] == [1]
    assert result.metadata["raw_llm_output"] == '{"answer":"北京市朝阳区广顺南大街8号院利星行中心1号楼","supporting_evidence":[1]}'
    assert result.metadata["evidence_index_map"][1]["section_id"] == "sec1"
    assert result.supporting_evidence[0].quote.startswith("[1]")


def test_llm_generator_falls_back_to_mock_on_invalid_json() -> None:
    generator = LLMGenerator(client=_FakeClient(content="not-json"))

    result = generator.generate(_sample(), _documents())

    assert result.metadata["generator"] == "llm_generator"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_generator"] == "json_safe_fallback"
    assert result.metadata["requested_generator"] == "llm_generator"
    assert result.metadata["fallback_reason"] == "LLM returned invalid JSON content."
    assert result.answer_text == STANDARD_INSUFFICIENT_ANSWER
    assert result.supporting_evidence == []
    assert result.metadata["raw_llm_output"] is None


def test_llm_generator_strips_code_fence_and_filters_duplicate_indices() -> None:
    generator = LLMGenerator(
        client=_FakeClient(
            content='```json\n{"answer":"北京市朝阳区广顺南大街8号院利星行中心1号楼","supporting_evidence":[1,1,3,"x",0,-1]}\n```'
        )
    )

    result = generator.generate(_sample(), _documents())

    assert result.answer_text == "北京市朝阳区广顺南大街8号院利星行中心1号楼"
    assert [item.section_id for item in result.supporting_evidence] == ["sec1"]
    assert result.metadata["selected_evidence_indices"] == [1]


def test_mock_generator_keeps_selected_evidence_indices_metadata() -> None:
    result = MockGenerator().generate(_sample(), _documents())

    assert "selected_evidence_indices" in result.metadata
    assert result.supporting_evidence


def test_rag_prompt_builder_uses_numbered_evidence_in_chinese() -> None:
    prompt = RAGPromptBuilder().build_prompt(question=_sample().question, retrieved_documents=_documents())

    assert "请仅依据给定的编号证据回答问题" in prompt
    assert "不要使用外部知识，不要猜测" in prompt
    assert "只能输出合法 JSON" in prompt
    assert "不要输出 markdown，不要输出代码块" in prompt
    assert '输出格式：{"answer":"...","supporting_evidence":[1,2]}' in prompt
    assert '{"answer":"无法根据已检索到的证据确定答案","supporting_evidence":[]}' in prompt
    assert "answer 中不要重复问题" in prompt
    assert "保留关键实体、数字、单位、型号、版本号的原始写法" in prompt
    assert "如果是是非题，优先输出“是”或“否”" in prompt
    assert "[1]" in prompt
    assert "[2]" in prompt
