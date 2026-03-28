from __future__ import annotations

import json
import re
from typing import Any

from clients.chat_llm_client import ChatLLMClient, ChatLLMClientError
from core.schema import AnswerResult, BenchmarkSample, EvidenceItem, RetrievedDocument
from pipelines.base import MockGenerator
from prompts.rag_prompt_builder import RAGPromptBuilder


JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?;\n]+")
DEFAULT_INSUFFICIENT_ANSWER = "根据当前检索证据，无法确定答案。"


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
            "You answer enterprise product QA strictly from retrieved evidence.\n"
            "Return valid JSON only.\n"
            "Do not invent evidence, document ids, or section ids.\n"
            "If the evidence is insufficient, set insufficient=true and keep the answer concise."
        )

    def _build_user_prompt(self, sample: BenchmarkSample, documents: list[RetrievedDocument]) -> str:
        context = self.prompt_builder.build_context(documents)
        evidence_guide = "\n".join(
            f"[{index}] DOC={document.source_id} SECTION={document.section_id}"
            for index, document in enumerate(documents)
        )
        return (
            "Answer the question using only the retrieved context.\n"
            "Output JSON with keys: answer_text, answer_short, answer_long, used_evidence_indices, confidence, insufficient.\n"
            "Rules:\n"
            "- used_evidence_indices must be a JSON array of integer indices from the provided evidence list.\n"
            "- answer_text is required.\n"
            "- answer_short should be short and directly gradable when possible.\n"
            "- answer_long may be empty if not needed.\n"
            "- confidence must be a number between 0 and 1 when provided.\n"
            "- If the answer cannot be determined, set insufficient=true and explain briefly.\n\n"
            f"Question:\n{sample.question}\n\n"
            f"Evidence Index Map:\n{evidence_guide}\n\n"
            f"Retrieved Context:\n{context}\n"
        )

    def _parse_payload(self, content: str) -> dict[str, Any]:
        stripped = content.strip()
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

    def _build_answer_result(
        self,
        sample: BenchmarkSample,
        documents: list[RetrievedDocument],
        payload: dict[str, Any],
    ) -> AnswerResult:
        answer_text = str(payload.get("answer_text", "")).strip()
        answer_short = str(payload.get("answer_short", "")).strip()
        answer_long = str(payload.get("answer_long", "")).strip()
        insufficient = bool(payload.get("insufficient", False))

        if insufficient and not answer_text:
            answer_text = DEFAULT_INSUFFICIENT_ANSWER
        if not answer_text:
            answer_text = answer_short or answer_long
        if not answer_text:
            raise ValueError("LLM returned no usable answer fields.")
        if not answer_short:
            answer_short = answer_text

        confidence = payload.get("confidence")
        if confidence is not None:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))

        supporting_evidence = self._build_supporting_evidence(documents, payload.get("used_evidence_indices", []))
        return AnswerResult(
            question_id=sample.question_id,
            answer_text=answer_text,
            answer_short=answer_short,
            answer_long=answer_long,
            supporting_evidence=supporting_evidence,
            confidence=confidence,
            metadata={
                "insufficient": insufficient,
                "used_evidence_indices": self._normalize_indices(payload.get("used_evidence_indices", []), len(documents)),
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
            document = documents[index]
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
            if 0 <= index < limit and index not in normalized:
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
        fallback = self.fallback_generator.generate(sample, documents)
        metadata = {
            **fallback.metadata,
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_generator": "mock_generator",
            "requested_generator": "llm_generator",
        }
        if metadata_overrides:
            metadata.update(metadata_overrides)
        fallback.metadata = metadata
        return fallback
