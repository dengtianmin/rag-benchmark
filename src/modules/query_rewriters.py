from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

from clients.chat_llm_client import ChatLLMClient, ChatLLMClientError
from core.schema import BenchmarkSample
from modules.skeleton_extractor import SkeletonExtractionResult


RewriteMode = Literal["original", "template", "rule_based", "llm"]


@dataclass(slots=True)
class QueryRewriteResult:
    rewritten_query: str
    mode: RewriteMode
    details: dict


class RetrievalLabQueryRewriter:
    """Dedicated rewrite strategies for retrieval-only laboratory experiments."""

    def __init__(
        self,
        *,
        llm_client: ChatLLMClient | None = None,
        llm_max_tokens: int = 64,
        llm_temperature: float = 0.0,
    ) -> None:
        self.llm_client = llm_client
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature

    def rewrite(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        *,
        mode: RewriteMode,
    ) -> QueryRewriteResult:
        if mode == "original":
            return QueryRewriteResult(
                rewritten_query=sample.question,
                mode=mode,
                details={"strategy": "identity", "used_skeleton": False},
            )
        if mode == "template":
            query = self._template_query(sample, skeleton)
            return QueryRewriteResult(
                rewritten_query=query,
                mode=mode,
                details={"strategy": "deterministic_template", "used_skeleton": True},
            )
        if mode == "rule_based":
            query = self._rule_based_query(sample, skeleton)
            return QueryRewriteResult(
                rewritten_query=query,
                mode=mode,
                details={"strategy": "rule_based", "used_skeleton": True},
            )
        if mode == "llm":
            query, llm_details = self._llm_query(sample, skeleton)
            return QueryRewriteResult(
                rewritten_query=query,
                mode=mode,
                details={"strategy": "llm", "used_skeleton": True, **llm_details},
            )
        raise ValueError(f"Unsupported rewrite mode: {mode}")

    def _template_query(self, sample: BenchmarkSample, skeleton: SkeletonExtractionResult) -> str:
        anchors = [item.name for item in skeleton.structured_entities if item.role == "anchor"] or skeleton.entities
        relations = [
            item.relation_type if item.relation_type != "factoid" else item.name
            for item in skeleton.structured_relations
            if item.role == "target"
        ] or skeleton.relations
        constraints = [item.value for item in skeleton.structured_constraints] or skeleton.constraints

        segments = [sample.question.strip()]
        if anchors:
            segments.append("entities " + " ".join(anchors[:3]))
        if relations:
            segments.append("relations " + " ".join(relations[:3]))
        if constraints:
            segments.append("constraints " + " ".join(constraints[:3]))
        return " ; ".join(segment for segment in segments if segment).strip()

    def _rule_based_query(self, sample: BenchmarkSample, skeleton: SkeletonExtractionResult) -> str:
        anchors = [item.name for item in skeleton.structured_entities if item.role == "anchor"] or skeleton.entities
        context_entities = [
            item.name for item in skeleton.structured_entities if item.role != "anchor"
        ] or skeleton.entities[1:]
        target_relations = [
            item.relation_type if item.relation_type != "factoid" else item.name
            for item in skeleton.structured_relations
            if item.role == "target"
        ]
        other_relations = [item.name for item in skeleton.structured_relations if item.role != "target"] or skeleton.relations
        constraints = [item.value for item in skeleton.structured_constraints] or skeleton.constraints

        query_parts: list[str] = []
        if anchors:
            query_parts.extend(anchors[:2])
        if target_relations:
            query_parts.extend(target_relations[:2])
        elif other_relations:
            query_parts.extend(other_relations[:2])
        if constraints:
            query_parts.extend(constraints[:2])
        if sample.question_type.value == "explanation" and "原因" not in "".join(query_parts):
            query_parts.append("原因")
        if sample.question_type.value == "relation" and "关系" not in "".join(query_parts):
            query_parts.append("关系")
        if not query_parts and context_entities:
            query_parts.extend(context_entities[:2])
        query = " ".join(part.strip() for part in query_parts if part and part.strip())
        return query or sample.question

    def _llm_query(self, sample: BenchmarkSample, skeleton: SkeletonExtractionResult) -> tuple[str, dict]:
        client = self.llm_client
        if client is None:
            raise ValueError("LLM rewrite mode requires a configured ChatLLMClient.")
        if not client.api_key:
            raise ValueError("LLM rewrite mode requires GENERATOR_API_KEY.")
        if not client.model:
            raise ValueError("LLM rewrite mode requires GENERATOR_MODEL.")

        entity_terms = [item.name for item in skeleton.structured_entities]
        relation_terms = [
            item.relation_type if item.relation_type != "factoid" else item.name
            for item in skeleton.structured_relations
        ]
        constraint_terms = [item.value for item in skeleton.structured_constraints]
        user_prompt = (
            "把下面问题改写成一条更适合检索的短 query，只输出 JSON。"
            ' 返回格式: {"query":"..."}。'
            " 不要回答问题，不要解释。\n"
            f"question: {sample.question}\n"
            f"entities: {entity_terms}\n"
            f"relations: {relation_terms}\n"
            f"constraints: {constraint_terms}"
        )
        try:
            response = client.chat_completion(
                messages=[
                    {"role": "system", "content": "你是检索 query 重写器。只返回合法 JSON。"},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                json_mode=True,
            )
        except ChatLLMClientError as exc:
            raise ValueError(f"LLM rewrite request failed: {exc}") from exc

        raw_content = response.content
        try:
            payload = json.loads(raw_content)
        except json.JSONDecodeError as exc:
            preview = self._preview_raw_output(raw_content)
            raise ValueError(f"LLM rewrite did not return valid JSON. raw_output={preview}") from exc

        payload, payload_kind = self._normalize_llm_payload(payload, raw_content)
        if isinstance(payload, dict):
            query = str(payload.get("query", "")).strip()
        elif isinstance(payload, str):
            query = payload.strip()
        else:
            preview = self._preview_raw_output(raw_content)
            raise ValueError(
                f"LLM rewrite returned unsupported JSON payload type: {type(payload).__name__}. raw_output={preview}"
            )
        if not query:
            preview = self._preview_raw_output(raw_content)
            raise ValueError(f"LLM rewrite returned an empty query. raw_output={preview}")
        return query, {
            "model_name": response.model_name,
            "finish_reason": response.finish_reason,
            "latency_ms": response.latency_ms,
            "raw_response": raw_content,
            "payload_kind": payload_kind,
        }

    def _normalize_llm_payload(self, payload: object, raw_content: str) -> tuple[dict | str, str]:
        if isinstance(payload, dict):
            return payload, "object"
        if isinstance(payload, str):
            stripped = payload.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                try:
                    reparsed = json.loads(stripped)
                except json.JSONDecodeError:
                    preview = self._preview_raw_output(raw_content)
                    raise ValueError(
                        f"LLM rewrite returned a double-encoded payload that could not be parsed. raw_output={preview}"
                    )
                if isinstance(reparsed, dict):
                    return reparsed, "double_encoded_object"
                if isinstance(reparsed, str):
                    return reparsed, "double_encoded_string"
                preview = self._preview_raw_output(raw_content)
                raise ValueError(
                    "LLM rewrite returned a double-encoded payload with unsupported inner type: "
                    f"{type(reparsed).__name__}. raw_output={preview}"
                )
            return stripped, "string"
        preview = self._preview_raw_output(raw_content)
        raise ValueError(
            f"LLM rewrite returned unsupported top-level JSON type: {type(payload).__name__}. raw_output={preview}"
        )

    def _preview_raw_output(self, raw_content: str, max_chars: int = 200) -> str:
        compact = " ".join(raw_content.split())
        if len(compact) <= max_chars:
            return compact
        return compact[:max_chars] + "..."
