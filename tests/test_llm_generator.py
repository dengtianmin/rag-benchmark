from __future__ import annotations

from core.schema import BenchmarkSample, RetrievedDocument
from core.types import QuestionType, SourceScope
from generators import build_generator
from generators.llm_generator import LLMGenerator
from pipelines.base import MockGenerator
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
                '{"answer_text":"北京市朝阳区广顺南大街8号院利星行中心1号楼",'
                '"answer_short":"北京市朝阳区广顺南大街8号院利星行中心1号楼",'
                '"answer_long":"",'
                '"used_evidence_indices":[0],'
                '"confidence":0.92,'
                '"insufficient":false}'
            )
        )
    )

    result = generator.generate(_sample(), _documents())

    assert result.answer_text == "北京市朝阳区广顺南大街8号院利星行中心1号楼"
    assert result.supporting_evidence[0].source_id == "doc1"
    assert result.supporting_evidence[0].section_id == "sec1"
    assert result.metadata["generator"] == "llm_generator"
    assert result.metadata["fallback_used"] is False


def test_llm_generator_falls_back_to_mock_on_invalid_json() -> None:
    generator = LLMGenerator(client=_FakeClient(content="not-json"))

    result = generator.generate(_sample(), _documents())

    assert result.metadata["generator"] == "mock_generator"
    assert result.metadata["fallback_used"] is True
    assert result.metadata["fallback_generator"] == "mock_generator"
    assert result.metadata["requested_generator"] == "llm_generator"
    assert result.metadata["fallback_reason"] == "LLM returned invalid JSON content."
