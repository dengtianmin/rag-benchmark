from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal

from clients.chat_llm_client import ChatLLMClient, ChatLLMClientError
from core.schema import BenchmarkSample
from modules.llm_query_rewrite_prompt import (
    LLMQueryRewritePayload,
    build_llm_query_rewrite_messages,
    parse_llm_query_rewrite_payload,
)
from modules.question_type_classifier import QuestionTypeClassifier
from modules.skeleton_extractor import SkeletonExtractionResult
from retrievers.text_retriever import TextRetrieverQuery


RewriteMode = Literal[
    "original",
    "template",
    "splicing",
    "rule_based",
    "llm",
    "sparse_llm",
    "dense_llm",
    "hybrid_llm",
]


@dataclass(slots=True)
class QueryRewriteRequest:
    question: str
    entities: list[str]
    relations: list[str]
    constraints: list[str]
    generation_settings: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QueryRewriteResult:
    rewritten_query: str
    mode: RewriteMode
    details: dict[str, Any]
    lexical_query: str
    dense_query: str
    question_type: str = "fallback_balanced"
    question_type_confidence: str = "low"
    question_type_evidence: dict[str, Any] = field(default_factory=dict)
    must_keep_terms: list[str] = field(default_factory=list)
    sparse_rewrite: str = ""
    dense_rewrite: str = ""

    def query_for_retrieval(self, retrieval_mode: str) -> str | TextRetrieverQuery:
        if retrieval_mode == "lexical":
            return self.lexical_query
        if retrieval_mode == "dense":
            return self.dense_query
        if retrieval_mode == "hybrid":
            return TextRetrieverQuery(
                lexical_query=self.lexical_query,
                dense_query=self.dense_query,
            )
        raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")

    def query_for_branch(self, branch: str) -> str:
        if branch == "lexical":
            return self.lexical_query
        if branch == "dense":
            return self.dense_query
        raise ValueError(f"Unsupported query branch: {branch}")

    def structured_rewrite(self) -> dict[str, object]:
        return {
            "must_keep_terms": list(self.must_keep_terms),
            "sparse_rewrite": self.sparse_rewrite or self.lexical_query,
            "dense_rewrite": self.dense_rewrite or self.dense_query,
        }


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
        self.question_type_classifier = QuestionTypeClassifier()

    def rewrite(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        *,
        mode: RewriteMode,
    ) -> QueryRewriteResult:
        normalized_mode = self._normalize_mode(mode)
        request = self.build_request(sample, skeleton)
        classification = self.question_type_classifier.classify(sample.question, skeleton)
        if normalized_mode == "original":
            return self._single_query_result(
                query=sample.question,
                mode=normalized_mode,
                details={"strategy": "identity", "used_skeleton": False},
                classification=classification,
            )
        if normalized_mode in {"template", "splicing"}:
            query = self._template_query(sample, skeleton)
            return self._single_query_result(
                query=query,
                mode=normalized_mode,
                details={"strategy": "deterministic_splicing", "used_skeleton": True},
                classification=classification,
            )
        if normalized_mode == "rule_based":
            query = self._rule_based_query(sample, skeleton)
            return self._single_query_result(
                query=query,
                mode=normalized_mode,
                details={"strategy": "rule_based", "used_skeleton": True},
                classification=classification,
            )
        if normalized_mode in {"llm", "sparse_llm", "dense_llm", "hybrid_llm"}:
            return self._llm_rewrite(sample, skeleton, request=request, mode=normalized_mode, classification=classification)
        raise ValueError(f"Unsupported rewrite mode: {mode}")

    def build_request(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        *,
        generation_settings: dict[str, Any] | None = None,
    ) -> QueryRewriteRequest:
        entities, relations, constraints = self._select_llm_rewrite_terms(skeleton)
        return QueryRewriteRequest(
            question=sample.question,
            entities=entities,
            relations=relations,
            constraints=constraints,
            generation_settings=generation_settings or {},
        )

    def _normalize_mode(self, mode: str) -> RewriteMode:
        if mode == "llm":
            return "dense_llm"
        if mode == "splicing":
            return "splicing"
        return mode  # type: ignore[return-value]

    def _single_query_result(
        self,
        *,
        query: str,
        mode: RewriteMode,
        details: dict[str, Any],
        classification,
        must_keep_terms: list[str] | None = None,
        sparse_rewrite: str | None = None,
        dense_rewrite: str | None = None,
    ) -> QueryRewriteResult:
        normalized_query = query.strip()
        return QueryRewriteResult(
            rewritten_query=normalized_query,
            mode=mode,
            details={
                **details,
                "question_type": classification.question_type,
                "question_type_confidence": classification.question_type_confidence,
                "question_type_evidence": classification.question_type_evidence,
            },
            lexical_query=normalized_query,
            dense_query=normalized_query,
            question_type=classification.question_type,
            question_type_confidence=classification.question_type_confidence,
            question_type_evidence=classification.question_type_evidence,
            must_keep_terms=list(must_keep_terms or []),
            sparse_rewrite=(sparse_rewrite or normalized_query).strip(),
            dense_rewrite=(dense_rewrite or normalized_query).strip(),
        )

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

    def _select_llm_rewrite_terms(self, skeleton: SkeletonExtractionResult) -> tuple[list[str], list[str], list[str]]:
        anchor_entities = self._clean_terms(
            [item.name for item in skeleton.structured_entities if item.role == "anchor"],
            limit=4,
        )
        if not anchor_entities:
            anchor_entities = self._clean_terms(skeleton.entities, limit=4)

        target_relations = self._clean_terms(
            [item.name for item in skeleton.structured_relations if item.role == "target"],
            limit=4,
        )
        if not target_relations:
            target_relations = self._clean_terms(skeleton.relations, limit=4)

        constraints = self._clean_terms(
            [item.value for item in skeleton.structured_constraints],
            limit=3,
        )
        if not constraints:
            constraints = self._clean_terms(skeleton.constraints, limit=3)

        return anchor_entities, target_relations, constraints

    def _llm_rewrite(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        *,
        request: QueryRewriteRequest,
        mode: RewriteMode,
        classification,
    ) -> QueryRewriteResult:
        client = self.llm_client
        if client is None:
            raise ValueError("LLM rewrite mode requires a configured ChatLLMClient.")
        if not client.api_key:
            raise ValueError("LLM rewrite mode requires GENERATOR_API_KEY.")
        if not client.model:
            raise ValueError("LLM rewrite mode requires GENERATOR_MODEL.")

        messages = build_llm_query_rewrite_messages(
            question=request.question,
            entities=request.entities,
            relations=request.relations,
            constraints=request.constraints,
        )
        try:
            response = client.chat_completion(
                messages=messages,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                json_mode=True,
            )
        except ChatLLMClientError as exc:
            raise ValueError(f"LLM rewrite request failed: {exc}") from exc

        fallback_result = self._fallback_llm_result(sample, skeleton, mode=mode, classification=classification)
        try:
            payload = parse_llm_query_rewrite_payload(response.content)
        except ValueError as exc:
            fallback_result.details.update(
                {
                    "model_name": response.model_name,
                    "finish_reason": response.finish_reason,
                    "latency_ms": response.latency_ms,
                    "raw_response": response.content,
                    "fallback_used": True,
                    "fallback_reason": str(exc),
                }
            )
            return fallback_result

        validated_payload = self._validate_llm_payload(payload, request)
        if validated_payload is None:
            fallback_result.details.update(
                {
                    "model_name": response.model_name,
                    "finish_reason": response.finish_reason,
                    "latency_ms": response.latency_ms,
                    "raw_response": response.content,
                    "fallback_used": True,
                    "fallback_reason": "llm_payload_failed_validation",
                }
            )
            return fallback_result

        result = self._result_from_llm_payload(validated_payload, mode=mode, classification=classification)
        result.details.update(
            {
                "model_name": response.model_name,
                "finish_reason": response.finish_reason,
                "latency_ms": response.latency_ms,
                "raw_response": response.content,
                "fallback_used": False,
            }
        )
        return result

    def _fallback_llm_result(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        *,
        mode: RewriteMode,
        classification,
    ) -> QueryRewriteResult:
        sparse_query = self._rule_based_query(sample, skeleton)
        dense_query = self._template_query(sample, skeleton)
        must_keep_terms = self._clean_terms(
            list(skeleton.entities) + list(skeleton.relations) + list(skeleton.constraints),
            limit=8,
        )
        return self._result_from_llm_payload(
            LLMQueryRewritePayload(
                must_keep_terms=must_keep_terms,
                sparse_rewrite=sparse_query,
                dense_rewrite=dense_query,
            ),
            mode=mode,
            strategy="llm_fallback",
            classification=classification,
        )

    def _result_from_llm_payload(
        self,
        payload: LLMQueryRewritePayload,
        *,
        mode: RewriteMode,
        strategy: str = "llm_structured",
        classification,
    ) -> QueryRewriteResult:
        if mode == "sparse_llm":
            return QueryRewriteResult(
                rewritten_query=payload.sparse_rewrite,
                mode=mode,
                details={
                    "strategy": strategy,
                    "used_skeleton": True,
                    "question_type": classification.question_type,
                    "question_type_confidence": classification.question_type_confidence,
                    "question_type_evidence": classification.question_type_evidence,
                },
                lexical_query=payload.sparse_rewrite,
                dense_query=payload.sparse_rewrite,
                question_type=classification.question_type,
                question_type_confidence=classification.question_type_confidence,
                question_type_evidence=classification.question_type_evidence,
                must_keep_terms=payload.must_keep_terms,
                sparse_rewrite=payload.sparse_rewrite,
                dense_rewrite=payload.dense_rewrite,
            )
        if mode in {"dense_llm", "llm"}:
            return QueryRewriteResult(
                rewritten_query=payload.dense_rewrite,
                mode=mode,
                details={
                    "strategy": strategy,
                    "used_skeleton": True,
                    "question_type": classification.question_type,
                    "question_type_confidence": classification.question_type_confidence,
                    "question_type_evidence": classification.question_type_evidence,
                },
                lexical_query=payload.dense_rewrite,
                dense_query=payload.dense_rewrite,
                question_type=classification.question_type,
                question_type_confidence=classification.question_type_confidence,
                question_type_evidence=classification.question_type_evidence,
                must_keep_terms=payload.must_keep_terms,
                sparse_rewrite=payload.sparse_rewrite,
                dense_rewrite=payload.dense_rewrite,
            )
        if mode == "hybrid_llm":
            return QueryRewriteResult(
                rewritten_query=payload.dense_rewrite,
                mode=mode,
                details={
                    "strategy": strategy,
                    "used_skeleton": True,
                    "question_type": classification.question_type,
                    "question_type_confidence": classification.question_type_confidence,
                    "question_type_evidence": classification.question_type_evidence,
                },
                lexical_query=payload.sparse_rewrite,
                dense_query=payload.dense_rewrite,
                question_type=classification.question_type,
                question_type_confidence=classification.question_type_confidence,
                question_type_evidence=classification.question_type_evidence,
                must_keep_terms=payload.must_keep_terms,
                sparse_rewrite=payload.sparse_rewrite,
                dense_rewrite=payload.dense_rewrite,
            )
        raise ValueError(f"Unsupported LLM rewrite mode: {mode}")

    def _validate_llm_payload(
        self,
        payload: LLMQueryRewritePayload,
        request: QueryRewriteRequest,
    ) -> LLMQueryRewritePayload | None:
        must_keep_terms = self._clean_terms(
            list(payload.must_keep_terms)
            + request.entities
            + request.relations
            + request.constraints,
            limit=10,
        )
        sparse_rewrite = self._normalize_plan_term(payload.sparse_rewrite)
        dense_rewrite = self._normalize_plan_term(payload.dense_rewrite)
        if not sparse_rewrite or not dense_rewrite:
            return None
        if self._contains_schema_label(sparse_rewrite) or self._contains_schema_label(dense_rewrite):
            return None
        if must_keep_terms and not self._contains_any_term(sparse_rewrite, must_keep_terms):
            return None
        if must_keep_terms and not self._contains_any_term(dense_rewrite, must_keep_terms):
            return None
        return LLMQueryRewritePayload(
            must_keep_terms=must_keep_terms,
            sparse_rewrite=sparse_rewrite,
            dense_rewrite=dense_rewrite,
        )

    def _contains_any_term(self, text: str, terms: list[str]) -> bool:
        normalized_text = self._normalize_plan_term(text)
        return any(term and term in normalized_text for term in terms[:6])

    def _clean_terms(self, terms: list[str], *, limit: int) -> list[str]:
        cleaned: list[str] = []
        for term in terms:
            normalized = self._normalize_plan_term(term)
            if not normalized or normalized in cleaned:
                continue
            cleaned.append(normalized)
            if len(cleaned) >= limit:
                break
        return cleaned

    def _normalize_plan_term(self, value: object) -> str:
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text.strip(";,，。；")

    def _contains_schema_label(self, text: str) -> bool:
        lowered = text.lower()
        return bool(
            re.search(
                r"\b(procedure|attribute|applicable_scope|comparison|condition|cause|factoid)\b",
                lowered,
            )
        )
