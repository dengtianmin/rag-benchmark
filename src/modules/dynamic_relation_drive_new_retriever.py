from __future__ import annotations

from dataclasses import dataclass

from core.schema import BenchmarkSample, RetrievedDocument
from modules.dynamic_relation_drive_retriever import DynamicWeightProfile
from modules.graph_expander import GraphIndex
from modules.question_type_classifier import (
    QuestionTypeClassification,
    QuestionTypeConfidence,
    QuestionTypeLabel,
)
from modules.relation_driven_retriever import RelationDrivenRetrieveResult, RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractionResult
from pipelines.base import PublicIndex
from retrievers.text_retriever import SupportsRetrieve, TextRetrieverQuery


_NEW_WEIGHT_PROFILES: dict[QuestionTypeLabel, DynamicWeightProfile] = {
    "fact_attribute": DynamicWeightProfile(semantic=0.64, entity=0.22, relation=0.04, constraint=0.10),
    "cause_explanation": DynamicWeightProfile(semantic=0.48, entity=0.14, relation=0.22, constraint=0.16),
    "procedure_method": DynamicWeightProfile(semantic=0.38, entity=0.16, relation=0.30, constraint=0.16),
    "relation_compare": DynamicWeightProfile(semantic=0.32, entity=0.24, relation=0.28, constraint=0.16),
    "fallback_balanced": DynamicWeightProfile(semantic=0.60, entity=0.20, relation=0.08, constraint=0.12),
}

_FACT_LIKE_ATTRIBUTE_TERMS = (
    "地址",
    "状态",
    "配置状态",
    "参数",
    "型号",
    "版本",
    "定义",
    "责任",
    "部署模式",
    "适用类型",
    "研发公司",
    "自定义接口",
    "时间阈值",
    "支持哪些应用",
    "支持哪些功能",
    "功能",
    "能力",
)
_STRICT_PROCEDURE_TERMS = ("步骤", "流程", "配置", "排查", "实现方法", "检测", "阻断", "处理方法", "部署步骤")
_DESCRIPTIVE_HOW_TERMS = ("如何满足", "如何影响", "如何提升", "如何实现", "如何通过", "如何支持")


@dataclass(frozen=True, slots=True)
class DynamicNewQuestionTypeDecision:
    adjusted_question_type: QuestionTypeLabel
    confidence: QuestionTypeConfidence
    evidence: dict[str, object]


class DynamicRelationDriveNewQuestionTypeAdjuster:
    """Conservative type adjustment used only by dynamic_relation_drive_new."""

    def adjust(
        self,
        question: str,
        classification: QuestionTypeClassification,
        skeleton: SkeletonExtractionResult,
    ) -> DynamicNewQuestionTypeDecision:
        entity_count = len(skeleton.structured_entities)
        question_lower = question.strip()
        original_type = classification.question_type
        adjusted_type = original_type
        reasons: list[str] = []

        if any(term in question_lower for term in _FACT_LIKE_ATTRIBUTE_TERMS):
            if entity_count <= 1 or "区别" not in question_lower:
                adjusted_type = "fact_attribute"
                reasons.append("attribute_slot_preferred")

        if adjusted_type == "procedure_method":
            has_strict_procedure = any(term in question_lower for term in _STRICT_PROCEDURE_TERMS)
            if not has_strict_procedure or any(term in question_lower for term in _DESCRIPTIVE_HOW_TERMS):
                adjusted_type = "fallback_balanced"
                reasons.append("procedure_scope_narrowed")

        if adjusted_type == "relation_compare":
            has_compare_signal = any(term in question_lower for term in ("区别", "关系", "对比", "兼容", "连接", "比较", "差异"))
            if entity_count < 2 or not has_compare_signal:
                adjusted_type = "fallback_balanced"
                reasons.append("compare_requires_two_entities_and_compare_intent")

        if original_type in {"procedure_method", "relation_compare"} and adjusted_type == original_type:
            if any(term in question_lower for term in _FACT_LIKE_ATTRIBUTE_TERMS) and entity_count <= 1:
                adjusted_type = "fact_attribute"
                reasons.append("single_entity_attribute_guard")

        if adjusted_type == "fallback_balanced" and original_type == "cause_explanation" and any(
            term in question_lower for term in ("为什么", "原因", "为何", "影响", "导致")
        ):
            adjusted_type = "cause_explanation"
            reasons.append("keep_explicit_cause_signal")

        evidence = {
            "original_question_type": original_type,
            "adjustment_reasons": reasons,
            "entity_count": entity_count,
            "relation_count": len(skeleton.structured_relations),
            "constraint_count": len(skeleton.structured_constraints),
        }
        return DynamicNewQuestionTypeDecision(
            adjusted_question_type=adjusted_type,
            confidence=classification.question_type_confidence,
            evidence=evidence,
        )


class DynamicRelationDriveNewRetriever(RelationDrivenRetriever):
    """More conservative dynamic scorer that prioritizes reducing false filtering."""

    def __init__(
        self,
        text_index: PublicIndex,
        graph_index: GraphIndex,
        text_retriever: SupportsRetrieve | None = None,
    ) -> None:
        super().__init__(text_index, graph_index, text_retriever=text_retriever)
        self.adjuster = DynamicRelationDriveNewQuestionTypeAdjuster()

    def retrieve(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        *,
        top_k: int,
        use_skeleton_rewrite: bool = True,
        query_override: str | TextRetrieverQuery | None = None,
        question_type_info: QuestionTypeClassification,
    ) -> RelationDrivenRetrieveResult:
        if query_override is not None:
            stage1_query = query_override
            query_source = "override"
        elif use_skeleton_rewrite:
            stage1_query = skeleton.rewritten_query(sample.question)
            query_source = "skeleton_rewrite"
        else:
            stage1_query = sample.question
            query_source = "original_question"

        candidate_pool = self.text_retriever.retrieve(stage1_query, top_k=max(top_k * 4, top_k))
        stage1_candidate_rows = [self._candidate_trace_row(document) for document in candidate_pool]
        query_trace = (
            {"lexical_query": stage1_query.lexical_query, "dense_query": stage1_query.dense_query}
            if isinstance(stage1_query, TextRetrieverQuery)
            else stage1_query
        )
        candidate_map = {document.section_id: document for document in candidate_pool}
        target_relations = [relation for relation in skeleton.structured_relations if relation.role == "target"] or skeleton.structured_relations
        anchor_entities = [entity for entity in skeleton.structured_entities if entity.role == "anchor"] or skeleton.structured_entities

        adjusted = self.adjuster.adjust(sample.question, question_type_info, skeleton)
        weight_profile = _NEW_WEIGHT_PROFILES[adjusted.adjusted_question_type]
        semantic_raw = {
            document.section_id: float(document.metadata.get("dense_score", document.score))
            for document in candidate_pool
        }
        semantic_scores = self._normalize_scores(semantic_raw)
        candidate_breakdown: dict[str, dict] = {}
        filtered_out: dict[str, dict] = {}

        for section_id, document in candidate_map.items():
            base_document = self.text_index.by_section_id.get(section_id)
            content = base_document.full_text if base_document is not None else document.content
            entity_alignment = self._entity_alignment(section_id, content, skeleton, focus_entities=anchor_entities)
            relation_alignment = self._relation_alignment(section_id, content, skeleton, focus_relations=target_relations)
            constraint_satisfaction = self._constraint_satisfaction(section_id, content, skeleton)
            high_semantic_protection = self._should_trigger_high_semantic_protection(
                semantic_score=semantic_scores.get(section_id, 0.0),
                entity_alignment_score=entity_alignment["score"],
            )
            conflict_penalty = self._conflict_penalty(section_id, content, skeleton)
            soft_penalty = self._soft_penalty(
                question_type=adjusted.adjusted_question_type,
                relation_score=relation_alignment["score"],
                entity_score=entity_alignment["score"],
                has_target_relation=bool(target_relations),
                has_anchor=bool(anchor_entities),
            )
            filter_decision = self._filter_decision(
                question_type=adjusted.adjusted_question_type,
                semantic_score=semantic_scores.get(section_id, 0.0),
                entity_score=entity_alignment["score"],
                relation_score=relation_alignment["score"],
                conflict_penalty=conflict_penalty["score"],
                high_semantic_protection=high_semantic_protection,
            )
            final_score = (
                weight_profile.semantic * semantic_scores.get(section_id, 0.0)
                + weight_profile.entity * entity_alignment["score"]
                + weight_profile.relation * relation_alignment["score"]
                + weight_profile.constraint * constraint_satisfaction["score"]
                - float(conflict_penalty["score"])
                - float(soft_penalty["score"])
            )
            breakdown = {
                "question_type": question_type_info.question_type,
                "question_type_confidence": question_type_info.question_type_confidence,
                "question_type_evidence": question_type_info.question_type_evidence,
                "adjusted_question_type_for_dynamic_new": adjusted.adjusted_question_type,
                "adjusted_question_type_evidence": adjusted.evidence,
                "dynamic_new_weight_profile": weight_profile.to_dict(),
                "semantic_score": semantic_scores.get(section_id, 0.0),
                "semantic_raw": semantic_raw.get(section_id, 0.0),
                "entity_alignment_score": entity_alignment["score"],
                "relation_alignment_score": relation_alignment["score"],
                "constraint_satisfaction_score": constraint_satisfaction["score"],
                "constraint_score": constraint_satisfaction["score"],
                "alignment_details": {
                    "entities": entity_alignment["matches"],
                    "relations": relation_alignment["matches"],
                    "constraints": constraint_satisfaction["matches"],
                },
                "soft_penalties": soft_penalty,
                "conflict_penalty": conflict_penalty,
                "high_semantic_protection_triggered": high_semantic_protection,
                "hard_drop_triggered": bool(filter_decision["drop"]),
                "filter_reasons": self._build_filter_reasons(
                    semantic_score=semantic_scores.get(section_id, 0.0),
                    entity_score=entity_alignment["score"],
                    relation_score=relation_alignment["score"],
                    constraint_score=constraint_satisfaction["score"],
                    soft_penalty=soft_penalty,
                    conflict_penalty=conflict_penalty,
                    filter_decision=filter_decision,
                ),
                "final_score": final_score,
                "final_fusion_score": final_score,
            }
            if filter_decision["drop"]:
                filtered_out[section_id] = breakdown
                continue
            candidate_breakdown[section_id] = breakdown

        recovery_ids: list[str] = []
        if len(candidate_breakdown) < top_k:
            recovery_ids = sorted(
                filtered_out,
                key=lambda section_id: filtered_out[section_id]["final_score"],
                reverse=True,
            )[: max(0, top_k - len(candidate_breakdown))]
            for section_id in recovery_ids:
                recovered = filtered_out.pop(section_id)
                recovered["hard_drop_triggered"] = False
                recovered["filter_reasons"].append("recovered_for_candidate_floor")
                candidate_breakdown[section_id] = recovered

        ranked_ids = sorted(candidate_breakdown, key=lambda section_id: candidate_breakdown[section_id]["final_score"], reverse=True)[:top_k]
        documents: list[RetrievedDocument] = []
        for rank, section_id in enumerate(ranked_ids, start=1):
            document = candidate_map[section_id]
            breakdown = candidate_breakdown[section_id]
            documents.append(
                document.model_copy(
                    update={
                        "score": breakdown["final_score"],
                        "rank": rank,
                        "metadata": {
                            **document.metadata,
                            "retriever": "dynamic_relation_drive_new",
                            "question_type": breakdown["question_type"],
                            "question_type_confidence": breakdown["question_type_confidence"],
                            "question_type_evidence": breakdown["question_type_evidence"],
                            "adjusted_question_type_for_dynamic_new": breakdown["adjusted_question_type_for_dynamic_new"],
                            "dynamic_new_weight_profile": breakdown["dynamic_new_weight_profile"],
                            "semantic_score": breakdown["semantic_score"],
                            "semantic_raw": breakdown["semantic_raw"],
                            "entity_alignment_score": breakdown["entity_alignment_score"],
                            "relation_alignment_score": breakdown["relation_alignment_score"],
                            "constraint_satisfaction_score": breakdown["constraint_satisfaction_score"],
                            "constraint_score": breakdown["constraint_score"],
                            "soft_penalties": breakdown["soft_penalties"],
                            "conflict_penalty": breakdown["conflict_penalty"]["score"],
                            "filter_reasons": breakdown["filter_reasons"],
                            "high_semantic_protection_triggered": breakdown["high_semantic_protection_triggered"],
                            "hard_drop_triggered": breakdown["hard_drop_triggered"],
                            "fusion_score": breakdown["final_score"],
                            "final_fusion_score": breakdown["final_fusion_score"],
                        },
                    }
                )
            )

        return RelationDrivenRetrieveResult(
            documents=documents,
            scores={section_id: candidate_breakdown[section_id]["final_score"] for section_id in ranked_ids},
            details={
                "mode": "dynamic_relation_drive_new",
                "stage1_query": query_trace,
                "query_source": query_source,
                "candidate_pool": stage1_candidate_rows,
                "skeleton_aware_scores": {section_id: candidate_breakdown[section_id] for section_id in ranked_ids},
                "filtered_out": {**filtered_out, **{section_id: candidate_breakdown[section_id] for section_id in candidate_breakdown if section_id not in ranked_ids}},
                "final_ranked_sections": ranked_ids,
                "kept_section_ids": list(candidate_breakdown.keys()),
                "filtered_out_section_ids": sorted(filtered_out.keys()),
                "filtered_out_reason_map": {
                    section_id: list(filtered_out[section_id].get("filter_reasons", []))
                    for section_id in filtered_out
                },
                "recovered_section_ids": recovery_ids,
                "fusion_weights": weight_profile.to_dict(),
                "question_type": question_type_info.question_type,
                "question_type_confidence": question_type_info.question_type_confidence,
                "question_type_evidence": question_type_info.question_type_evidence,
                "adjusted_question_type_for_dynamic_new": adjusted.adjusted_question_type,
                "adjusted_question_type_evidence": adjusted.evidence,
            },
        )

    def _should_trigger_high_semantic_protection(self, *, semantic_score: float, entity_alignment_score: float) -> bool:
        return semantic_score >= 0.60 and entity_alignment_score >= 0.10

    def _soft_penalty(
        self,
        *,
        question_type: QuestionTypeLabel,
        relation_score: float,
        entity_score: float,
        has_target_relation: bool,
        has_anchor: bool,
    ) -> dict[str, object]:
        score = 0.0
        reasons: list[str] = []
        if has_target_relation and relation_score < 0.10:
            score += {
                "fact_attribute": 0.03,
                "fallback_balanced": 0.04,
                "cause_explanation": 0.05,
                "procedure_method": 0.05,
                "relation_compare": 0.06,
            }[question_type]
            reasons.append("relation_alignment_missing_soft")
            reasons.append("target_relation_missing_soft")
        elif has_target_relation and relation_score < 0.20:
            score += 0.03
            reasons.append("target_relation_weak_soft")
        if has_anchor and entity_score < 0.10:
            score += 0.02 if question_type in {"fact_attribute", "fallback_balanced"} else 0.03
            reasons.append("anchor_missing_soft")
        return {"score": score, "reasons": reasons}

    def _conflict_penalty(self, section_id: str, content: str, skeleton: SkeletonExtractionResult) -> dict[str, object]:
        base_conflict = self._constraint_conflict(section_id, content, skeleton)
        penalty = float(base_conflict.get("score", 0.0))
        return {
            "score": penalty if skeleton.structured_constraints else 0.0,
            "reasons": list(base_conflict.get("reasons", [])),
        }

    def _filter_decision(
        self,
        *,
        question_type: QuestionTypeLabel,
        semantic_score: float,
        entity_score: float,
        relation_score: float,
        conflict_penalty: float,
        high_semantic_protection: bool,
    ) -> dict[str, object]:
        if conflict_penalty >= 0.45:
            return {"drop": True, "reasons": ["constraint_conflict_drop"]}
        if high_semantic_protection:
            return {"drop": False, "reasons": ["high_semantic_protection_active"]}
        if question_type in {"fact_attribute", "fallback_balanced"}:
            should_drop = semantic_score < 0.12 and entity_score < 0.05 and relation_score < 0.05
            return {"drop": should_drop, "reasons": ["weak_semantic_and_structure"] if should_drop else []}
        if question_type == "cause_explanation":
            should_drop = semantic_score < 0.18 and entity_score < 0.08 and relation_score < 0.06
            return {"drop": should_drop, "reasons": ["weak_explanation_alignment"] if should_drop else []}
        if question_type == "procedure_method":
            should_drop = semantic_score < 0.16 and entity_score < 0.08 and relation_score < 0.08
            return {"drop": should_drop, "reasons": ["weak_procedure_alignment"] if should_drop else []}
        should_drop = semantic_score < 0.35 and entity_score < 0.08 and relation_score < 0.08
        return {"drop": should_drop, "reasons": ["weak_compare_alignment"] if should_drop else []}

    def _build_filter_reasons(
        self,
        *,
        semantic_score: float,
        entity_score: float,
        relation_score: float,
        constraint_score: float,
        soft_penalty: dict[str, object],
        conflict_penalty: dict[str, object],
        filter_decision: dict[str, object],
    ) -> list[str]:
        reasons: list[str] = []
        if semantic_score >= 0.60:
            reasons.append("strong_semantic_match")
        elif semantic_score < 0.15:
            reasons.append("weak_semantic_match")
        if entity_score < 0.10:
            reasons.append("entity_alignment_missing")
        if relation_score < 0.10:
            reasons.append("relation_alignment_missing")
        if constraint_score < 0.10:
            reasons.append("constraint_gap")
        if float(conflict_penalty.get("score", 0.0)) > 0:
            reasons.append("constraint_conflict")
        reasons.extend(list(soft_penalty.get("reasons", [])))
        reasons.extend(list(filter_decision.get("reasons", [])))
        return reasons
