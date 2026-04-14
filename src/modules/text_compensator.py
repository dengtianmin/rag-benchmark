from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal

from core.schema import BenchmarkSample, RetrievedDocument
from modules.graph_expander import GraphIndex
from modules.query_rewriters import QueryRewriteResult
from modules.question_type_classifier import QuestionTypeClassification, QuestionTypeClassifier
from modules.skeleton_extractor import SkeletonExtractionResult
from pipelines.base import PublicIndex, SENTENCE_SPLIT_PATTERN, overlap_score, tokenize
from retrievers.text_retriever import LexicalTextRetriever, SupportsRetrieve, TextRetrieverQuery


TextCompensationStrategy = Literal["legacy_compensation", "new_compensation"]

_MODEL_ALIAS = {
    "g6": "gen6",
    "gen6": "gen6",
    "v2.5": "v2.5",
    "v2_5": "v2.5",
}
_CONDITION_TEMPLATE_MARKERS = {
    "trigger": ("告警", "触发", "出现", "发生", "检测到"),
    "precondition": ("如果", "若", "当", "在", "前提", "满足"),
    "exception": ("除非", "例外", "异常", "否则"),
    "post_action_failure": ("无效", "失败", "未恢复", "无法", "仍然", "则更换", "需要更换"),
    "threshold": ("以上", "超过", "至少", "低于", "高于", "阈值"),
}
_TIME_YEAR_PATTERN = re.compile(r"(20\d{2})")
_VERSION_PATTERN = re.compile(r"v?\s*(\d+(?:\.\d+)*)")


@dataclass(slots=True)
class TextCompensationResult:
    documents: list[RetrievedDocument]
    activated: bool
    reason: str
    details: dict


@dataclass(slots=True)
class QueryAnalysisPayload:
    question_type: str = "fallback_balanced"
    question_type_confidence: str = "low"
    question_type_evidence: dict[str, Any] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    primary_relation: str | None = None
    primary_constraint: str | None = None
    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_type": self.question_type,
            "question_type_confidence": self.question_type_confidence,
            "question_type_evidence": dict(self.question_type_evidence),
            "entities": list(self.entities),
            "relations": list(self.relations),
            "constraints": list(self.constraints),
            "primary_relation": self.primary_relation,
            "primary_constraint": self.primary_constraint,
            "source": self.source,
        }


@dataclass(slots=True)
class CompensationContext:
    sample: BenchmarkSample
    skeleton: SkeletonExtractionResult
    base_documents: list[RetrievedDocument]
    query_analysis: QueryAnalysisPayload
    strategy: TextCompensationStrategy


def normalize_entity(text: str) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"[\s\-_/]+", " ", lowered)
    lowered = re.sub(r"[^\w\u4e00-\u9fff. ]+", "", lowered)
    lowered = " ".join(_MODEL_ALIAS.get(token, token) for token in lowered.split())
    return lowered.strip()


def normalize_scope(text: str) -> str:
    return normalize_entity(text).replace("系列", "").replace("版本", "").strip()


def normalize_time_constraint(text: str) -> dict[str, Any]:
    normalized = str(text or "").strip().lower()
    years = _TIME_YEAR_PATTERN.findall(normalized)
    versions = _VERSION_PATTERN.findall(normalized)
    operator = "contains"
    if any(marker in normalized for marker in ("之后", "以上", "大于", ">= ")):
        operator = "gte"
    elif any(marker in normalized for marker in ("之前", "以下", "小于", "<= ")):
        operator = "lte"
    return {
        "raw": text,
        "normalized": normalized,
        "years": years,
        "versions": versions,
        "operator": operator,
    }


def build_query_analysis_payload(
    sample: BenchmarkSample,
    skeleton: SkeletonExtractionResult | None = None,
    *,
    query_analysis: QueryAnalysisPayload | dict[str, Any] | QuestionTypeClassification | None = None,
    rewrite_result: QueryRewriteResult | None = None,
    classifier: QuestionTypeClassifier | None = None,
) -> QueryAnalysisPayload:
    if isinstance(query_analysis, QueryAnalysisPayload):
        return query_analysis
    if isinstance(query_analysis, QuestionTypeClassification):
        payload = QueryAnalysisPayload(
            question_type=query_analysis.question_type,
            question_type_confidence=query_analysis.question_type_confidence,
            question_type_evidence=query_analysis.question_type_evidence,
            source="question_type_classifier",
        )
    elif isinstance(query_analysis, dict):
        payload = QueryAnalysisPayload(
            question_type=str(query_analysis.get("question_type", "fallback_balanced")),
            question_type_confidence=str(query_analysis.get("question_type_confidence", "low")),
            question_type_evidence=dict(query_analysis.get("question_type_evidence", {})),
            entities=[str(item) for item in query_analysis.get("entities", []) if str(item).strip()],
            relations=[str(item) for item in query_analysis.get("relations", []) if str(item).strip()],
            constraints=[str(item) for item in query_analysis.get("constraints", []) if str(item).strip()],
            primary_relation=query_analysis.get("primary_relation"),
            primary_constraint=query_analysis.get("primary_constraint"),
            source=str(query_analysis.get("source", "dict_payload")),
        )
    elif rewrite_result is not None:
        payload = QueryAnalysisPayload(
            question_type=rewrite_result.question_type,
            question_type_confidence=rewrite_result.question_type_confidence,
            question_type_evidence=rewrite_result.question_type_evidence,
            entities=list(rewrite_result.entities),
            relations=list(rewrite_result.relations),
            constraints=list(rewrite_result.constraints),
            primary_relation=rewrite_result.primary_relation,
            primary_constraint=rewrite_result.primary_constraint,
            source="rewrite_result",
        )
    else:
        if classifier is not None and skeleton is not None:
            classification = classifier.classify(sample.question, skeleton)
            payload = QueryAnalysisPayload(
                question_type=classification.question_type,
                question_type_confidence=classification.question_type_confidence,
                question_type_evidence=classification.question_type_evidence,
                source="question_type_classifier",
            )
        else:
            payload = QueryAnalysisPayload(
                question_type="fallback_balanced",
                question_type_confidence="low",
                source="fallback",
            )
    if skeleton is not None:
        if not payload.entities:
            payload.entities = [item.name for item in skeleton.structured_entities] or list(skeleton.entities)
        if not payload.relations:
            payload.relations = [item.name for item in skeleton.structured_relations] or list(skeleton.relations)
        if not payload.constraints:
            payload.constraints = [item.value for item in skeleton.structured_constraints] or list(skeleton.constraints)
        if payload.primary_relation is None:
            payload.primary_relation = next(
                (item.name for item in skeleton.structured_relations if item.role == "target"),
                payload.relations[0] if payload.relations else None,
            )
        if payload.primary_constraint is None:
            payload.primary_constraint = next(
                (item.value for item in skeleton.structured_constraints if item.kind != "llm"),
                payload.constraints[0] if payload.constraints else None,
            )
    return payload


def should_activate_text_compensation(
    sample: BenchmarkSample,
    base_documents: list[RetrievedDocument],
    skeleton: SkeletonExtractionResult | None = None,
    query_analysis: QueryAnalysisPayload | None = None,
    *,
    coverage_details: dict[str, Any] | None = None,
    coverage_threshold: float = 0.70,
    light_compensation_threshold: float = 0.80,
) -> tuple[bool, str, dict]:
    query_analysis = query_analysis or QueryAnalysisPayload()
    coverage_details = coverage_details or {}
    coverage = float(coverage_details.get("coverage", 0.0))
    primary_relation_covered = bool(coverage_details.get("primary_relation_covered"))
    primary_constraint_covered = bool(coverage_details.get("primary_constraint_covered"))
    fallback_low_confidence = (
        query_analysis.source in {"dict_payload", "rewrite_result", "fallback"}
        and (
            query_analysis.question_type == "fallback_balanced"
            or query_analysis.question_type_confidence == "low"
        )
    ) or (
        query_analysis.question_type_confidence == "low"
        and not (query_analysis.entities or query_analysis.relations or query_analysis.constraints)
    )
    old_rule_triggered = False
    fallback_reason = None
    if fallback_low_confidence:
        legacy_triggers = []
        if sample.requires_text_compensation:
            legacy_triggers.append("legacy_requires_text_compensation")
        if sample.question_type.value == "explanation":
            legacy_triggers.append("legacy_explanation_question_type")
        if len(base_documents) < 2:
            legacy_triggers.append("legacy_evidence_too_thin")
        old_rule_triggered = bool(legacy_triggers)
        fallback_reason = legacy_triggers[0] if legacy_triggers else "legacy_not_triggered"
        details = {
            "mode": "fallback",
            "fallback_used": True,
            "old_rule_triggered": old_rule_triggered,
            "old_rule_reasons": legacy_triggers,
            "coverage_snapshot": coverage_details,
        }
        return old_rule_triggered, fallback_reason, details

    hard_triggers: list[str] = []
    if sample.requires_text_compensation:
        hard_triggers.append("requires_text_compensation")
    if query_analysis.question_type in {"procedure_method", "cause_explanation"} and query_analysis.question_type_confidence != "low":
        hard_triggers.append(f"question_type:{query_analysis.question_type}")
    if query_analysis.primary_relation and not primary_relation_covered:
        hard_triggers.append("primary_relation_uncovered")
    if query_analysis.primary_constraint and not primary_constraint_covered:
        hard_triggers.append("primary_constraint_uncovered")
    if hard_triggers:
        return True, hard_triggers[0], {
            "mode": "hard_trigger",
            "hard_triggers": hard_triggers,
            "coverage_snapshot": coverage_details,
            "fallback_used": False,
            "old_rule_triggered": False,
        }

    if coverage < coverage_threshold:
        return True, "coverage_below_threshold", {
            "mode": "coverage_trigger",
            "coverage": coverage,
            "threshold": coverage_threshold,
            "compensation_tier": "full",
            "coverage_snapshot": coverage_details,
            "fallback_used": False,
            "old_rule_triggered": False,
        }
    if coverage < light_compensation_threshold:
        return True, "coverage_light_gap", {
            "mode": "coverage_trigger",
            "coverage": coverage,
            "threshold": light_compensation_threshold,
            "compensation_tier": "light",
            "coverage_snapshot": coverage_details,
            "fallback_used": False,
            "old_rule_triggered": False,
        }

    return False, "coverage_sufficient", {
        "mode": "coverage_skip",
        "coverage": coverage,
        "coverage_snapshot": coverage_details,
        "fallback_used": False,
        "old_rule_triggered": False,
    }


class TextCompensator:
    """Structure-guided text compensation with legacy and coverage-driven strategies."""

    def __init__(
        self,
        text_index: PublicIndex,
        graph_index: GraphIndex,
        text_retriever: SupportsRetrieve | None = None,
        *,
        default_strategy: TextCompensationStrategy = "legacy_compensation",
        coverage_threshold: float = 0.70,
        light_compensation_threshold: float = 0.80,
        max_targeted_queries: int = 3,
    ) -> None:
        self.text_index = text_index
        self.graph_index = graph_index
        self.text_retriever = text_retriever or LexicalTextRetriever(text_index)
        self.default_strategy = default_strategy
        self.coverage_threshold = coverage_threshold
        self.light_compensation_threshold = light_compensation_threshold
        self.max_targeted_queries = max_targeted_queries
        self.question_type_classifier = QuestionTypeClassifier()

    def compensate(
        self,
        sample: BenchmarkSample,
        skeleton_or_documents: SkeletonExtractionResult | list[RetrievedDocument],
        base_documents: list[RetrievedDocument] | None = None,
        *,
        top_k: int,
        enabled: bool = True,
        skeleton: SkeletonExtractionResult | None = None,
        query_analysis: QueryAnalysisPayload | dict[str, Any] | QuestionTypeClassification | None = None,
        rewrite_result: QueryRewriteResult | None = None,
        strategy: TextCompensationStrategy | None = None,
    ) -> TextCompensationResult:
        if isinstance(skeleton_or_documents, SkeletonExtractionResult):
            skeleton = skeleton_or_documents
            base_documents = list(base_documents or [])
        else:
            base_documents = list(skeleton_or_documents)
        if skeleton is None:
            raise ValueError("TextCompensator.compensate requires a skeleton.")
        strategy = strategy or self.default_strategy
        analysis_payload = build_query_analysis_payload(
            sample,
            skeleton,
            query_analysis=query_analysis,
            rewrite_result=rewrite_result,
            classifier=self.question_type_classifier,
        )
        context = CompensationContext(
            sample=sample,
            skeleton=skeleton,
            base_documents=base_documents,
            query_analysis=analysis_payload,
            strategy=strategy,
        )
        if not enabled:
            return TextCompensationResult(
                documents=base_documents[:top_k],
                activated=False,
                reason="disabled",
                details={
                    "strategy": strategy,
                    "compensated_count": 0,
                    "fallback_used": False,
                    "old_rule_triggered": False,
                    "query_analysis": analysis_payload.to_dict(),
                },
            )
        if strategy == "legacy_compensation":
            return self._legacy_compensate(context, top_k=top_k)
        return self._new_compensate(context, top_k=top_k)

    def _legacy_compensate(self, context: CompensationContext, *, top_k: int) -> TextCompensationResult:
        sample = context.sample
        skeleton = context.skeleton
        base_documents = context.base_documents
        gap_analysis = self._legacy_analyze_gaps(sample, skeleton, base_documents)
        if not gap_analysis["activate"]:
            return TextCompensationResult(
                documents=base_documents[:top_k],
                activated=False,
                reason="not_needed",
                details={
                    "strategy": "legacy_compensation",
                    "compensated_count": 0,
                    "gap_analysis": gap_analysis,
                    "query_analysis": context.query_analysis.to_dict(),
                    "fallback_used": False,
                    "old_rule_triggered": False,
                },
            )

        compensation_query = skeleton.compensation_query
        merged: dict[str, RetrievedDocument] = {document.section_id: document for document in base_documents}
        added_reasons: dict[str, str] = {}
        added_scores: dict[str, float] = {}
        backfill_scores: dict[str, float] = {}
        candidate_debug: list[dict[str, Any]] = []
        max_additions = 1 if top_k <= 3 else 2
        additions = 0

        for document in self.text_retriever.retrieve(compensation_query, top_k=max(top_k * 3, top_k)):
            if additions >= max_additions:
                break
            if document.section_id in merged:
                continue
            alignment = self._alignment_score(document, skeleton)
            if alignment["score"] <= 0.32:
                continue
            boosted_score = 0.35 * float(document.score) + 0.25 * alignment["score"]
            backfill_scores[document.section_id] = boosted_score
            merged[document.section_id] = document.model_copy(
                update={
                    "score": boosted_score,
                    "metadata": {
                        **document.metadata,
                        "retriever": "text_compensator",
                        "compensation_alignment_score": alignment["score"],
                        "compensation_alignment_details": alignment["details"],
                        "compensation_query": compensation_query,
                        "compensation_reason": "legacy_skeleton_guided_compensation",
                        "is_compensation_doc": True,
                    },
                }
            )
            added_reasons[document.section_id] = "skeleton_guided_compensation"
            added_scores[document.section_id] = boosted_score
            candidate_debug.append(
                {
                    "section_id": document.section_id,
                    "raw_score": float(document.score),
                    "alignment_score": alignment["score"],
                    "alignment_details": alignment["details"],
                }
            )
            additions += 1

        primary_documents = sorted(
            base_documents,
            key=lambda item: float(item.metadata.get("fusion_score", item.score)),
            reverse=True,
        )[:top_k]
        primary_keep = max(1, top_k - max_additions)
        guaranteed_primary = primary_documents[:primary_keep]
        guaranteed_ids = {document.section_id for document in guaranteed_primary}
        remaining_pool = [document for document in merged.values() if document.section_id not in guaranteed_ids]
        remaining_ranked = sorted(
            remaining_pool,
            key=lambda item: (
                float(item.metadata.get("fusion_score", item.score)),
                float(item.metadata.get("compensation_alignment_score", 0.0)),
                float(item.score),
            ),
            reverse=True,
        )[: max(0, top_k - len(guaranteed_primary))]
        reranked = [
            document.model_copy(update={"rank": rank})
            for rank, document in enumerate(guaranteed_primary + remaining_ranked, start=1)
        ]
        new_sections = [document.section_id for document in reranked if document.section_id in added_reasons]
        return TextCompensationResult(
            documents=reranked,
            activated=True,
            reason=gap_analysis["primary_reason"],
            details={
                "strategy": "legacy_compensation",
                "compensated_count": len(new_sections),
                "gap_analysis": gap_analysis,
                "compensation_query": compensation_query,
                "max_additions": max_additions,
                "guaranteed_primary": [document.section_id for document in guaranteed_primary],
                "added_sections": new_sections,
                "added_reasons": added_reasons,
                "added_scores": added_scores,
                "backfill_scores": backfill_scores,
                "candidate_debug": candidate_debug[: top_k * 2],
                "query_analysis": context.query_analysis.to_dict(),
                "fallback_used": False,
                "old_rule_triggered": False,
            },
        )

    def _new_compensate(self, context: CompensationContext, *, top_k: int) -> TextCompensationResult:
        coverage_before = self.compute_skeleton_coverage(
            context.skeleton,
            context.base_documents,
            query_analysis=context.query_analysis,
        )
        activated, reason, activation_details = should_activate_text_compensation(
            context.sample,
            context.base_documents,
            context.skeleton,
            context.query_analysis,
            coverage_details=coverage_before,
            coverage_threshold=self.coverage_threshold,
            light_compensation_threshold=self.light_compensation_threshold,
        )
        if not activated:
            return TextCompensationResult(
                documents=context.base_documents[:top_k],
                activated=False,
                reason="not_needed",
                details={
                    "strategy": "new_compensation",
                    "activation_reason": reason,
                    "coverage_before": coverage_before,
                    "coverage_after": coverage_before,
                    "cov_E": coverage_before["cov_E"],
                    "cov_R": coverage_before["cov_R"],
                    "cov_C": coverage_before["cov_C"],
                    "primary_relation_covered": coverage_before["primary_relation_covered"],
                    "primary_constraint_covered": coverage_before["primary_constraint_covered"],
                    "unmatched_before": coverage_before["unmatched_summary"],
                    "newly_covered_items": [],
                    "compensation_queries": [],
                    "added_docs_by_reason": {},
                    "fallback_used": activation_details.get("fallback_used", False),
                    "old_rule_triggered": activation_details.get("old_rule_triggered", False),
                    "query_analysis": context.query_analysis.to_dict(),
                    "compensated_count": 0,
                },
            )

        compensation_queries = self._build_targeted_compensation_queries(context, coverage_before)
        merged = {document.section_id: document for document in context.base_documents}
        added_docs_by_reason: dict[str, list[str]] = {}
        candidate_debug: list[dict[str, Any]] = []
        max_additions = 1 if activation_details.get("compensation_tier") == "light" else (2 if top_k <= 4 else 3)

        for query_item in compensation_queries:
            query = query_item["query"]
            query_reason = query_item["reason"]
            for document in self.text_retriever.retrieve(query, top_k=max(top_k * 3, top_k)):
                if document.section_id in merged:
                    continue
                gain = self._document_coverage_gain(document, context, coverage_before)
                if gain["coverage_gain"] <= 0.0 and gain["locality_support"] <= 0.0:
                    continue
                retriever_score = float(document.metadata.get("hybrid_score", document.metadata.get("dense_score", document.score)))
                final_comp_score = 0.45 * retriever_score + 0.40 * gain["coverage_gain"] + 0.15 * gain["locality_support"]
                matched_entities = gain["matched_entities"]
                matched_relations = gain["matched_relations"]
                matched_constraints = gain["matched_constraints"]
                merged[document.section_id] = document.model_copy(
                    update={
                        "score": final_comp_score,
                        "metadata": {
                            **document.metadata,
                            "retriever": "text_compensator_new",
                            "compensation_reason": query_reason,
                            "matched_entities": matched_entities,
                            "matched_relations": matched_relations,
                            "matched_constraints": matched_constraints,
                            "coverage_gain": gain["coverage_gain"],
                            "locality_support": gain["locality_support"],
                            "is_compensation_doc": True,
                            "compensation_query": query,
                            "compensation_query_reason": query_reason,
                            "compensation_query_targets": list(query_item.get("targets", [])),
                        },
                    }
                )
                added_docs_by_reason.setdefault(query_reason, []).append(document.section_id)
                candidate_debug.append(
                    {
                        "section_id": document.section_id,
                        "query": query,
                        "query_reason": query_reason,
                        "coverage_gain": gain["coverage_gain"],
                        "locality_support": gain["locality_support"],
                        "matched_entities": matched_entities,
                        "matched_relations": matched_relations,
                        "matched_constraints": matched_constraints,
                        "retriever_score": retriever_score,
                        "final_comp_score": final_comp_score,
                    }
                )
                if sum(len(items) for items in added_docs_by_reason.values()) >= max_additions:
                    break
            if sum(len(items) for items in added_docs_by_reason.values()) >= max_additions:
                break

        ranked = sorted(
            merged.values(),
            key=lambda item: (
                float(item.metadata.get("coverage_gain", 0.0)),
                float(item.metadata.get("locality_support", 0.0)),
                float(item.metadata.get("fusion_score", item.score)),
                float(item.score),
            ),
            reverse=True,
        )[:top_k]
        reranked = [document.model_copy(update={"rank": rank}) for rank, document in enumerate(ranked, start=1)]
        coverage_after = self.compute_skeleton_coverage(
            context.skeleton,
            reranked,
            query_analysis=context.query_analysis,
        )
        newly_covered_items = self._compute_newly_covered_items(coverage_before, coverage_after)
        return TextCompensationResult(
            documents=reranked,
            activated=True,
            reason=reason,
            details={
                "strategy": "new_compensation",
                "activation_reason": reason,
                "coverage_before": coverage_before,
                "coverage_after": coverage_after,
                "cov_E": coverage_before["cov_E"],
                "cov_R": coverage_before["cov_R"],
                "cov_C": coverage_before["cov_C"],
                "primary_relation_covered": coverage_before["primary_relation_covered"],
                "primary_constraint_covered": coverage_before["primary_constraint_covered"],
                "unmatched_before": coverage_before["unmatched_summary"],
                "newly_covered_items": newly_covered_items,
                "compensation_queries": compensation_queries,
                "added_docs_by_reason": added_docs_by_reason,
                "fallback_used": activation_details.get("fallback_used", False),
                "old_rule_triggered": activation_details.get("old_rule_triggered", False),
                "query_analysis": context.query_analysis.to_dict(),
                "candidate_debug": candidate_debug[: top_k * 3],
                "compensated_count": sum(len(items) for items in added_docs_by_reason.values()),
                "max_additions": max_additions,
                "guaranteed_primary": [document.section_id for document in context.base_documents[:top_k]],
                "added_sections": [
                    section_id
                    for section_ids in added_docs_by_reason.values()
                    for section_id in section_ids
                ],
                "added_reasons": {
                    section_id: reason_name
                    for reason_name, section_ids in added_docs_by_reason.items()
                    for section_id in section_ids
                },
                "backfill_scores": {
                    document.section_id: float(document.metadata.get("coverage_gain", 0.0))
                    for document in reranked
                    if document.metadata.get("is_compensation_doc")
                },
                "activation_details": activation_details,
            },
        )

    def compute_skeleton_coverage(
        self,
        skeleton: SkeletonExtractionResult,
        documents: list[RetrievedDocument],
        *,
        query_analysis: QueryAnalysisPayload | None = None,
    ) -> dict[str, Any]:
        query_analysis = query_analysis or QueryAnalysisPayload(
            entities=[item.name for item in skeleton.structured_entities] or list(skeleton.entities),
            relations=[item.name for item in skeleton.structured_relations] or list(skeleton.relations),
            constraints=[item.value for item in skeleton.structured_constraints] or list(skeleton.constraints),
            primary_relation=next(
                (item.name for item in skeleton.structured_relations if item.role == "target"),
                skeleton.relations[0] if skeleton.relations else None,
            ),
            primary_constraint=next(
                (item.value for item in skeleton.structured_constraints if item.kind != "llm"),
                skeleton.constraints[0] if skeleton.constraints else None,
            ),
            source="coverage_fallback",
        )
        entity_results = []
        for entity in query_analysis.entities:
            result = self.match_entity(entity, documents)
            if result["covered"]:
                entity_results.append(result)
            else:
                entity_results.append(result)
        relation_results = []
        for relation in query_analysis.relations:
            relation_results.append(self.match_relation(relation, skeleton, documents, query_analysis=query_analysis))
        constraint_results = []
        for constraint in query_analysis.constraints:
            constraint_results.append(self.match_constraint(constraint, skeleton, documents, query_analysis=query_analysis))

        matched_entities = [item for item in entity_results if item["covered"]]
        unmatched_entities = [item for item in entity_results if not item["covered"]]
        matched_relations = [item for item in relation_results if item["covered"]]
        unmatched_relations = [item for item in relation_results if not item["covered"]]
        matched_constraints = [item for item in constraint_results if item["covered"]]
        unmatched_constraints = [item for item in constraint_results if not item["covered"]]
        cov_E = len(matched_entities) / max(len(entity_results), 1) if entity_results else 1.0
        cov_R = len(matched_relations) / max(len(relation_results), 1) if relation_results else 1.0
        cov_C = len(matched_constraints) / max(len(constraint_results), 1) if constraint_results else 1.0
        coverage = 0.25 * cov_E + 0.35 * cov_R + 0.40 * cov_C
        primary_relation_covered = any(item["target"] == query_analysis.primary_relation and item["covered"] for item in relation_results) if query_analysis.primary_relation else True
        primary_constraint_covered = any(
            item["target"] == query_analysis.primary_constraint and item["covered"] and item.get("kind") != "llm"
            for item in constraint_results
        ) if query_analysis.primary_constraint else True
        return {
            "coverage": coverage,
            "cov_E": cov_E,
            "cov_R": cov_R,
            "cov_C": cov_C,
            "matched_entities": matched_entities,
            "unmatched_entities": unmatched_entities,
            "matched_relations": matched_relations,
            "unmatched_relations": unmatched_relations,
            "matched_constraints": matched_constraints,
            "unmatched_constraints": unmatched_constraints,
            "primary_relation_covered": primary_relation_covered,
            "primary_constraint_covered": primary_constraint_covered,
            "unmatched_summary": {
                "entities": [item["target"] for item in unmatched_entities],
                "relations": [item["target"] for item in unmatched_relations],
                "constraints": [item["target"] for item in unmatched_constraints],
            },
        }

    def match_entity(self, entity: str, documents: list[RetrievedDocument]) -> dict[str, Any]:
        normalized_target = normalize_entity(entity)
        aliases = {normalized_target, normalized_target.replace(" ", ""), normalized_target.replace("系列", "").strip()}
        for document in documents:
            section_entities = self.graph_index.section_to_entities.get(document.section_id, set())
            doc_blob = normalize_entity(document.content)
            if any(alias and (alias in section_entities or alias in doc_blob) for alias in aliases):
                return {
                    "target": entity,
                    "covered": True,
                    "match_type": "exact",
                    "section_id": document.section_id,
                    "aliases": sorted(aliases),
                }
        return {
            "target": entity,
            "covered": False,
            "match_type": "unsupported",
            "section_id": None,
            "aliases": sorted(aliases),
        }

    def match_relation(
        self,
        relation: str,
        skeleton: SkeletonExtractionResult,
        documents: list[RetrievedDocument],
        *,
        query_analysis: QueryAnalysisPayload,
    ) -> dict[str, Any]:
        relation_item = next((item for item in skeleton.structured_relations if item.name == relation), None)
        normalized_relation = normalize_entity(relation)
        best_match = {
            "target": relation,
            "covered": False,
            "status": "unsupported",
            "section_id": None,
            "details": {},
        }
        anchor_terms = query_analysis.entities[:2]
        for document in documents:
            section_triples = self.graph_index.section_to_triples.get(document.section_id, [])
            section_relations = self.graph_index.section_to_relations.get(document.section_id, set())
            section_entities = self.graph_index.section_to_entities.get(document.section_id, set())
            predicate_match = normalized_relation in section_relations
            relation_type_match = False
            directional_match = False
            endpoint_bonus = 0.0
            if relation_item is not None:
                relation_type_match = relation_item.relation_type != "factoid" and any(
                    relation_item.relation_type in self._infer_relation_types(predicate)
                    for _, _, predicate, _ in section_triples
                )
                if relation_item.subject_hint and relation_item.subject_hint.lower() in section_entities:
                    endpoint_bonus += 0.05
                if relation_item.object_hint and relation_item.object_hint.lower() in section_entities:
                    endpoint_bonus += 0.05
                for _, subject, predicate, obj in section_triples:
                    if normalize_entity(predicate) != normalized_relation:
                        continue
                    subject_ok = relation_item.subject_hint is None or normalize_entity(relation_item.subject_hint) == normalize_entity(subject)
                    object_ok = relation_item.object_hint is None or normalize_entity(relation_item.object_hint) == normalize_entity(obj)
                    if subject_ok or object_ok:
                        directional_match = True
                        endpoint_bonus += 0.15
                        break
            lexical = overlap_score(relation, document.content)
            local_windows = self.find_local_windows(document.content, anchor_terms, [relation], window_size=80)
            local_support = 1.0 if local_windows else 0.0
            alias_support = any(anchor and normalize_entity(anchor) in normalize_entity(document.content) for anchor in anchor_terms) and lexical > 0
            status = "unsupported"
            if predicate_match:
                status = "exact"
            elif relation_type_match and (directional_match or endpoint_bonus > 0 or local_support > 0):
                status = "semantic"
            elif alias_support and local_support > 0:
                status = "semantic"
            if status == "unsupported":
                continue
            best_match = {
                "target": relation,
                "covered": True,
                "status": status,
                "section_id": document.section_id,
                "details": {
                    "predicate_match": predicate_match,
                    "relation_type_match": relation_type_match,
                    "directional_match": directional_match,
                    "endpoint_bonus": endpoint_bonus,
                    "lexical": lexical,
                    "locality_support": local_support,
                },
            }
            if status == "exact":
                break
        return best_match

    def match_constraint(
        self,
        constraint: str,
        skeleton: SkeletonExtractionResult,
        documents: list[RetrievedDocument],
        *,
        query_analysis: QueryAnalysisPayload,
    ) -> dict[str, Any]:
        constraint_item = next((item for item in skeleton.structured_constraints if item.value == constraint), None)
        kind = self._infer_constraint_kind(constraint_item.value if constraint_item is not None else constraint)
        best = {
            "target": constraint,
            "kind": kind,
            "covered": False,
            "status": "unsupported",
            "section_id": None,
            "details": {},
        }
        for document in documents:
            if kind == "time":
                candidate = self._match_time_constraint(constraint, document)
            elif kind == "scope":
                candidate = self._match_scope_constraint(constraint, document, query_analysis=query_analysis)
            elif kind == "condition":
                candidate = self._match_condition_constraint(constraint, document, query_analysis=query_analysis)
            elif kind == "llm":
                candidate = self._match_llm_constraint(constraint, document)
            else:
                candidate = self._match_generic_constraint(constraint, document, query_analysis=query_analysis)
            if not candidate["covered"]:
                continue
            if kind == "llm" and candidate["status"] == "weak":
                candidate["covered"] = True
            best = {
                "target": constraint,
                "kind": kind,
                "covered": candidate["covered"],
                "status": candidate["status"],
                "section_id": document.section_id,
                "details": candidate["details"],
            }
            if candidate["status"] in {"exact", "semantic", "matched"}:
                break
        if kind == "llm" and best["covered"]:
            best["weak_evidence"] = True
        return best

    def find_local_windows(
        self,
        content: str,
        anchor_terms: list[str],
        relation_terms: list[str],
        *,
        window_size: int = 80,
    ) -> list[str]:
        normalized_content = str(content or "")
        windows: list[str] = []
        anchors = [term for term in anchor_terms if term]
        relations = [term for term in relation_terms if term]
        for sentence in SENTENCE_SPLIT_PATTERN.split(normalized_content):
            sentence = sentence.strip()
            if not sentence:
                continue
            if anchors and not any(term in sentence for term in anchors):
                continue
            if relations and not any(term in sentence for term in relations):
                continue
            windows.append(sentence[:window_size])
        return windows

    def _build_targeted_compensation_queries(self, context: CompensationContext, coverage_before: dict[str, Any]) -> list[dict[str, Any]]:
        primary_entity = context.query_analysis.entities[0] if context.query_analysis.entities else ""
        queries = [
            {
                "query": context.skeleton.compensation_query or context.sample.question,
                "reason": "original_backfill",
                "targets": ["question_backfill"],
            }
        ]
        if primary_entity:
            queries.append(
                {
                    "query": primary_entity,
                    "reason": "shared_entity_expansion",
                    "targets": [primary_entity],
                }
            )
        if coverage_before["unmatched_relations"]:
            target_relation = coverage_before["unmatched_relations"][0]["target"]
            queries.append(
                {
                    "query": " ".join(part for part in [primary_entity, target_relation] if part).strip(),
                    "reason": "missing_relation_targeted",
                    "targets": [target_relation],
                }
            )
        if coverage_before["unmatched_constraints"]:
            target_constraint = coverage_before["unmatched_constraints"][0]["target"]
            template = self._extract_condition_template(target_constraint)
            queries.append(
                {
                    "query": " ".join(part for part in [primary_entity, target_constraint, template or "条件"] if part).strip(),
                    "reason": "missing_constraint_targeted",
                    "targets": [target_constraint],
                }
            )
        deduped: list[dict[str, Any]] = []
        seen = set()
        for item in queries:
            key = item["query"]
            if not key or key in seen:
                continue
            deduped.append(item)
            seen.add(key)
            if len(deduped) >= self.max_targeted_queries:
                break
        return deduped

    def _document_coverage_gain(
        self,
        document: RetrievedDocument,
        context: CompensationContext,
        coverage_before: dict[str, Any],
    ) -> dict[str, Any]:
        missing_entities = [item["target"] for item in coverage_before["unmatched_entities"]]
        missing_relations = [item["target"] for item in coverage_before["unmatched_relations"]]
        missing_constraints = [item["target"] for item in coverage_before["unmatched_constraints"]]
        matched_entities = [entity for entity in missing_entities if self.match_entity(entity, [document])["covered"]]
        matched_relations = [
            relation
            for relation in missing_relations
            if self.match_relation(relation, context.skeleton, [document], query_analysis=context.query_analysis)["covered"]
        ]
        matched_constraints = [
            constraint
            for constraint in missing_constraints
            if self.match_constraint(constraint, context.skeleton, [document], query_analysis=context.query_analysis)["covered"]
        ]
        local_windows = self.find_local_windows(
            document.content,
            context.query_analysis.entities[:2],
            matched_relations or matched_constraints,
            window_size=80,
        )
        coverage_gain = (
            0.25 * (len(matched_entities) / max(len(missing_entities), 1) if missing_entities else 0.0)
            + 0.35 * (len(matched_relations) / max(len(missing_relations), 1) if missing_relations else 0.0)
            + 0.40 * (len(matched_constraints) / max(len(missing_constraints), 1) if missing_constraints else 0.0)
        )
        return {
            "matched_entities": matched_entities,
            "matched_relations": matched_relations,
            "matched_constraints": matched_constraints,
            "coverage_gain": coverage_gain,
            "locality_support": min(1.0, len(local_windows) / 2.0),
        }

    def _compute_newly_covered_items(self, before: dict[str, Any], after: dict[str, Any]) -> list[str]:
        before_items = set(before["unmatched_summary"]["entities"] + before["unmatched_summary"]["relations"] + before["unmatched_summary"]["constraints"])
        after_items = set(after["unmatched_summary"]["entities"] + after["unmatched_summary"]["relations"] + after["unmatched_summary"]["constraints"])
        return sorted(before_items - after_items)

    def _infer_constraint_kind(self, text: str) -> str:
        value = str(text or "")
        lowered = value.lower()
        if any(marker in value for marker in ("年", "月", "日", "近三个月", "之后", "之前", "以上", "以下")) or _TIME_YEAR_PATTERN.search(value):
            return "time"
        if any(marker in value for marker in ("型号", "系列", "版本", "产品线", "场景", "范围")):
            return "scope"
        if any(marker in value for marker in ("条件", "什么条件", "若", "如果", "当", "需要更换", "无效")):
            return "condition"
        if "llm" in lowered:
            return "llm"
        return "generic"

    def _match_time_constraint(self, constraint: str, document: RetrievedDocument) -> dict[str, Any]:
        expected = normalize_time_constraint(constraint)
        observed = normalize_time_constraint(document.content)
        if expected["years"] and any(year in observed["years"] for year in expected["years"]):
            return {"covered": True, "status": "matched", "details": {"time_relation": "equivalent"}}
        if expected["versions"] and any(version in observed["versions"] for version in expected["versions"]):
            return {"covered": True, "status": "matched", "details": {"time_relation": "contains"}}
        if expected["years"] and observed["years"] and not any(year in observed["years"] for year in expected["years"]):
            return {"covered": False, "status": "conflict", "details": {"conflict": True, "observed_years": observed["years"]}}
        return {"covered": False, "status": "unsupported", "details": {}}

    def _match_scope_constraint(self, constraint: str, document: RetrievedDocument, *, query_analysis: QueryAnalysisPayload) -> dict[str, Any]:
        normalized_scope = normalize_scope(constraint)
        normalized_content = normalize_scope(document.content)
        if normalized_scope and normalized_scope in normalized_content:
            anchor_hit = any(entity and normalize_scope(entity) in normalized_content for entity in query_analysis.entities[:2])
            local_windows = self.find_local_windows(document.content, query_analysis.entities[:2], query_analysis.relations[:2], window_size=80)
            if anchor_hit or local_windows or not query_analysis.entities:
                return {"covered": True, "status": "semantic", "details": {"locality_windows": local_windows, "anchor_hit": anchor_hit}}
        return {"covered": False, "status": "unsupported", "details": {}}

    def _match_condition_constraint(self, constraint: str, document: RetrievedDocument, *, query_analysis: QueryAnalysisPayload) -> dict[str, Any]:
        question_template = self._extract_condition_template(constraint)
        doc_template = self._extract_condition_template(document.content)
        if not question_template or not doc_template:
            return {"covered": False, "status": "unsupported", "details": {}}
        question_actions = self._condition_action_terms(constraint)
        doc_actions = self._condition_action_terms(document.content)
        shared_actions = sorted(set(question_actions) & set(doc_actions))
        action_overlap = overlap_score(" ".join(question_actions), " ".join(doc_actions)) if question_actions and doc_actions else overlap_score(constraint, document.content)
        local_windows = self.find_local_windows(
            document.content,
            query_analysis.entities[:2],
            shared_actions or query_analysis.relations[:2] or [constraint],
            window_size=120,
        )
        if question_template == doc_template and (action_overlap > 0.0 or local_windows or shared_actions):
            return {
                "covered": True,
                "status": "semantic",
                "details": {
                    "condition_template": question_template,
                    "action_overlap": action_overlap,
                    "shared_actions": shared_actions,
                    "locality_windows": local_windows,
                },
            }
        return {"covered": False, "status": "unsupported", "details": {}}

    def _match_llm_constraint(self, constraint: str, document: RetrievedDocument) -> dict[str, Any]:
        score = overlap_score(constraint, document.content)
        return {"covered": score > 0.0, "status": "weak" if score > 0.0 else "unsupported", "details": {"lexical": score}}

    def _match_generic_constraint(self, constraint: str, document: RetrievedDocument, *, query_analysis: QueryAnalysisPayload) -> dict[str, Any]:
        lexical = overlap_score(constraint, document.content)
        local_windows = self.find_local_windows(document.content, query_analysis.entities[:2], [constraint], window_size=80)
        if lexical > 0 and local_windows:
            return {"covered": True, "status": "semantic", "details": {"lexical": lexical, "locality_windows": local_windows}}
        return {"covered": False, "status": "unsupported", "details": {"lexical": lexical}}

    def _extract_condition_template(self, text: str) -> str | None:
        normalized = str(text or "")
        priority = ("post_action_failure", "precondition", "trigger", "exception", "threshold")
        for template in priority:
            markers = _CONDITION_TEMPLATE_MARKERS[template]
            if any(marker in normalized for marker in markers):
                return template
        if "条件" in normalized:
            return "precondition"
        return None

    def _condition_action_terms(self, text: str) -> list[str]:
        markers = ("更换", "复位", "无效", "失败", "告警", "恢复", "硬件")
        normalized = str(text or "")
        return [marker for marker in markers if marker in normalized]

    def _legacy_analyze_gaps(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        base_documents: list[RetrievedDocument],
    ) -> dict[str, Any]:
        if not base_documents:
            return {
                "activate": True,
                "primary_reason": "empty_evidence",
                "coverage_gap": 1.0,
                "constraint_gap": 1.0,
                "explanation_gap": 1.0,
                "doc_count": 0,
            }

        relation_scores = [float(document.metadata.get("relation_alignment_score", 0.0)) for document in base_documents]
        constraint_scores = [float(document.metadata.get("constraint_satisfaction_score", 0.0)) for document in base_documents]
        anchor_scores = [float(document.metadata.get("entity_alignment_score", 0.0)) for document in base_documents]
        explanation_scores = [
            max(
                overlap_score(sample.question, document.content),
                overlap_score(skeleton.compensation_query, document.content),
            )
            for document in base_documents
        ]
        coverage_gap = 1.0 - max(relation_scores, default=0.0)
        constraint_gap = 1.0 - max(constraint_scores, default=0.0) if skeleton.structured_constraints else 0.0
        explanation_gap = 1.0 - max(explanation_scores, default=0.0)
        anchor_gap = 1.0 - max(anchor_scores, default=0.0) if skeleton.structured_entities else 0.0

        triggers = []
        if len(base_documents) < 2:
            triggers.append("evidence_too_thin")
        if skeleton.structured_relations and coverage_gap > 0.72:
            triggers.append("relation_coverage_gap")
        if skeleton.structured_constraints and constraint_gap > 0.75:
            triggers.append("constraint_gap")
        if skeleton.structured_entities and anchor_gap > 0.8:
            triggers.append("anchor_gap")
        if explanation_gap > 0.9 and max(relation_scores, default=0.0) >= 0.45:
            triggers.append("explanation_gap")

        return {
            "activate": bool(triggers),
            "primary_reason": triggers[0] if triggers else "sufficient_evidence",
            "triggers": triggers,
            "coverage_gap": coverage_gap,
            "constraint_gap": constraint_gap,
            "explanation_gap": explanation_gap,
            "anchor_gap": anchor_gap,
            "thresholds": {
                "relation_coverage_gap": 0.72,
                "constraint_gap": 0.75,
                "anchor_gap": 0.8,
                "explanation_gap": 0.9,
            },
            "doc_count": len(base_documents),
        }

    def _alignment_score(self, document: RetrievedDocument, skeleton: SkeletonExtractionResult) -> dict[str, Any]:
        section_entities = self.graph_index.section_to_entities.get(document.section_id, set())
        section_relations = self.graph_index.section_to_relations.get(document.section_id, set())
        section_constraints = self.graph_index.section_to_constraints.get(document.section_id, set())
        entity_hits = [entity.name for entity in skeleton.structured_entities if entity.normalized_name in section_entities]
        relation_hits = [relation.name for relation in skeleton.structured_relations if relation.normalized_name in section_relations]
        constraint_hits = [
            constraint.value
            for constraint in skeleton.structured_constraints
            if any(constraint.normalized_value in item for item in section_constraints)
        ]
        lexical = overlap_score(skeleton.compensation_query, document.content)
        entity_score = len(entity_hits) / max(len(skeleton.structured_entities), 1) if skeleton.structured_entities else 0.0
        relation_score = len(relation_hits) / max(len(skeleton.structured_relations), 1) if skeleton.structured_relations else 0.0
        constraint_score = len(constraint_hits) / max(len(skeleton.structured_constraints), 1) if skeleton.structured_constraints else 0.0
        score = min(1.0, 0.15 * lexical + 0.3 * entity_score + 0.35 * relation_score + 0.2 * constraint_score)
        return {
            "score": score,
            "details": {
                "entity_hits": entity_hits,
                "relation_hits": relation_hits,
                "constraint_hits": constraint_hits,
                "lexical": lexical,
            },
        }

    def _infer_relation_types(self, text: str) -> set[str]:
        lowered = str(text or "").lower()
        relation_types = set()
        if any(marker in lowered for marker in ("原因", "为何", "为什么", "导致", "因为", "由于")):
            relation_types.add("cause")
        if any(marker in lowered for marker in ("步骤", "流程", "方法", "方式")):
            relation_types.add("procedure")
        if any(marker in lowered for marker in ("条件", "适用", "前提", "场景")):
            relation_types.add("condition")
        if any(marker in lowered for marker in ("范围", "对象", "地区", "部门")):
            relation_types.add("applicable_scope")
        if any(marker in lowered for marker in ("参数", "属性", "指标", "配置")):
            relation_types.add("attribute")
        if any(marker in lowered for marker in ("区别", "差异", "比较", "不同")):
            relation_types.add("comparison")
        if any(marker in lowered for marker in ("关系", "合作", "关联", "依赖")):
            relation_types.add("relation")
        return relation_types or {"factoid"}
