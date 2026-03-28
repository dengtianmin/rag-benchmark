from __future__ import annotations

import json
import re
from typing import Any

from clients.chat_llm_client import ChatLLMClient, ChatLLMClientError
from core.schema import AnswerResult, BenchmarkSample, EvidenceItem, RetrievedDocument
from pipelines.base import MockGenerator
from prompts.rag_prompt_builder import RAGPromptBuilder


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?;\n]+")
STANDARD_INSUFFICIENT_ANSWER = "无法根据已检索到的证据确定答案"


class LLMGenerator:
    """LLM-backed generator that falls back to MockGenerator when needed."""

    def __init__(
        self,
        client: ChatLLMClient,
        *,
        prompt_builder: RAGPromptBuilder | None = None,
        fallback_generator: MockGenerator | None = None,
        evidence_quote_chars: int = 240,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.fallback_generator = fallback_generator or MockGenerator(prompt_builder=self.prompt_builder)
        self.evidence_quote_chars = evidence_quote_chars

    def generate(self, sample: BenchmarkSample, documents: list[RetrievedDocument]) -> AnswerResult:
        if not documents:
            return self._fallback(sample, documents, reason="no_retrieval")

        if not self.client.enabled:
            return self._fallback(sample, documents, reason="llm_client_not_configured")

        user_prompt = self._build_user_prompt(sample, documents)
        prompt_chars = len(user_prompt)
        try:
            response = self.client.chat_completion(
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": user_prompt},
                ]
            )
            payload = self._parse_payload(response.content)
            answer = self._build_answer_result(sample, documents, payload)
        except (ChatLLMClientError, ValueError, KeyError, TypeError) as exc:
            return self._fallback(
                sample,
                documents,
                reason=str(exc),
                metadata_overrides={
                    "requested_generator": "llm_generator",
                    "model_name": self.client.model,
                    "base_url": self.client.base_url,
                    "prompt_chars": prompt_chars,
                },
            )

        answer.metadata.update(
            {
                "generator": "llm_generator",
                "model_name": response.model_name,
                "base_url": self.client.base_url,
                "latency_ms": response.latency_ms,
                "prompt_chars": prompt_chars,
                "completion_chars": len(response.content),
                "finish_reason": response.finish_reason,
                "fallback_used": False,
                "fallback_reason": None,
                "no_retrieval": False,
            }
        )
        return answer

    def _build_system_prompt(self) -> str:
        return (
            "你是企业产品文档问答模型。\n"
            "只能根据提供的编号证据回答。\n"
            "只返回合法 JSON，不要输出 Markdown、代码块、解释或额外文本。"
        )

    def _build_user_prompt(self, sample: BenchmarkSample, documents: list[RetrievedDocument]) -> str:
        return self.prompt_builder.build_prompt(question=sample.question, retrieved_documents=documents)

    def _parse_payload(self, content: str) -> dict[str, Any]:
        stripped = self._strip_code_fences(content.strip())
        candidates = [stripped]
        match = JSON_BLOCK_PATTERN.search(stripped)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise ValueError("LLM returned invalid JSON content.")

    def _strip_code_fences(self, content: str) -> str:
        return CODE_FENCE_PATTERN.sub("", content).strip()

    def _build_answer_result(
        self,
        sample: BenchmarkSample,
        documents: list[RetrievedDocument],
        payload: dict[str, Any],
    ) -> AnswerResult:
        answer_text = str(payload.get("answer", "")).strip() or STANDARD_INSUFFICIENT_ANSWER
        supporting_indices = self._normalize_indices(payload.get("supporting_evidence", []), len(documents))
        supporting_evidence = self._build_supporting_evidence(documents, supporting_indices)
        return AnswerResult(
            question_id=sample.question_id,
            answer_text=answer_text,
            answer_short=answer_text,
            answer_long="",
            supporting_evidence=supporting_evidence,
            metadata={
                "insufficient": answer_text == STANDARD_INSUFFICIENT_ANSWER,
                "selected_evidence_indices": supporting_indices,
                "parsed_json": payload,
            },
        )

    def _build_supporting_evidence(
        self,
        documents: list[RetrievedDocument],
        used_indices: Any,
    ) -> list[EvidenceItem]:
        indices = self._normalize_indices(used_indices, len(documents))
        evidence: list[EvidenceItem] = []
        for index in indices:
            document = documents[index - 1]
            evidence.append(
                EvidenceItem(
                    source_id=document.source_id,
                    section_id=document.section_id,
                    quote=self._extract_quote(document.content),
                )
            )
        return evidence

    def _normalize_indices(self, raw_indices: Any, limit: int) -> list[int]:
        if not isinstance(raw_indices, list):
            return []
        normalized: list[int] = []
        for item in raw_indices:
            try:
                index = int(item)
            except (TypeError, ValueError):
                continue
            if 1 <= index <= limit and index not in normalized:
                normalized.append(index)
        return normalized

    def _extract_quote(self, content: str) -> str:
        for sentence in SENTENCE_SPLIT_PATTERN.split(content):
            sentence = sentence.strip()
            if sentence:
                return sentence[: self.evidence_quote_chars]
        return content[: self.evidence_quote_chars]

    def _fallback(
        self,
        sample: BenchmarkSample,
        documents: list[RetrievedDocument],
        *,
        reason: str,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> AnswerResult:
        supporting_evidence = []
        if documents:
            supporting_evidence = self._build_supporting_evidence(documents, [])
        metadata = {
            "generator": "llm_generator",
            "model_name": getattr(self.client, "model", None),
            "base_url": getattr(self.client, "base_url", None),
            "latency_ms": 0,
            "prompt_chars": 0,
            "completion_chars": len(STANDARD_INSUFFICIENT_ANSWER),
            "finish_reason": "fallback",
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_generator": "json_safe_fallback",
            "requested_generator": "llm_generator",
            "parsed_json": {
                "answer": STANDARD_INSUFFICIENT_ANSWER,
                "supporting_evidence": [],
            },
            "selected_evidence_indices": [],
            "no_retrieval": not documents,
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)
        return AnswerResult(
            question_id=sample.question_id,
            answer_text=STANDARD_INSUFFICIENT_ANSWER,
            answer_short=STANDARD_INSUFFICIENT_ANSWER,
            answer_long="",
            supporting_evidence=[],
            metadata=metadata,
        )
