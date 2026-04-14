from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from modules.skeleton_extractor import SkeletonExtractionResult


QuestionTypeLabel = Literal[
    "fact_attribute",
    "cause_explanation",
    "procedure_method",
    "relation_compare",
    "fallback_balanced",
]
QuestionTypeConfidence = Literal["high", "medium", "low"]

QUESTION_TYPE_PRIORITY: tuple[QuestionTypeLabel, ...] = (
    "procedure_method",
    "cause_explanation",
    "fact_attribute",
    "relation_compare",
    "fallback_balanced",
)

_TRIGGER_RULES: dict[QuestionTypeLabel, tuple[str, ...]] = {
    "procedure_method": ("如何", "怎么", "怎样", "步骤", "方法", "配置", "排查", "实现", "检测", "阻断", "处理方法"),
    "cause_explanation": ("为什么", "原因", "影响", "导致", "为何", "为什么需要", "变得严重"),
    "fact_attribute": ("是什么", "哪个", "哪种", "地址", "状态", "配置状态", "参数", "型号", "版本", "定义", "责任", "部署模式", "支持哪种"),
    "relation_compare": ("区别", "关系", "对比", "兼容", "连接", "与", "和"),
}
_RELATION_COMPARE_MARKERS = ("区别", "关系", "对比", "兼容", "连接", "比较", "差异", "A与B", "A和B")
_CAUSE_RELATION_TYPES = frozenset({"cause", "reason", "impact"})


@dataclass(slots=True)
class QuestionTypeClassification:
    question_type: QuestionTypeLabel
    question_type_confidence: QuestionTypeConfidence
    question_type_evidence: dict[str, object] = field(default_factory=dict)


class QuestionTypeClassifier:
    """Rule-first classifier used by dynamic_relation_drive and experiment analysis."""

    def classify(self, question: str, skeleton: SkeletonExtractionResult) -> QuestionTypeClassification:
        normalized_question = question.strip()
        rule_hits = self._collect_rule_hits(normalized_question)
        entity_count = len(skeleton.structured_entities)
        relation_count = len(skeleton.structured_relations)
        constraint_count = len(skeleton.structured_constraints)

        candidate_type = self._select_rule_type(rule_hits)
        corrected_type = self._apply_skeleton_correction(
            question=normalized_question,
            skeleton=skeleton,
            candidate_type=candidate_type,
        )
        confidence = self._estimate_confidence(
            candidate_type=corrected_type,
            rule_hits=rule_hits,
            skeleton=skeleton,
        )
        if confidence == "low":
            corrected_type = "fallback_balanced"

        evidence = {
            "trigger_terms": list(rule_hits.get(corrected_type, [])),
            "rule_hits": {label: list(values) for label, values in rule_hits.items() if values},
            "entity_count": entity_count,
            "relation_count": relation_count,
            "constraint_count": constraint_count,
            "skeleton_relation_types": [item.relation_type for item in skeleton.structured_relations if item.relation_type],
        }
        return QuestionTypeClassification(
            question_type=corrected_type,
            question_type_confidence=confidence,
            question_type_evidence=evidence,
        )

    def _collect_rule_hits(self, question: str) -> dict[QuestionTypeLabel, list[str]]:
        hits: dict[QuestionTypeLabel, list[str]] = {label: [] for label in QUESTION_TYPE_PRIORITY}
        for label, trigger_terms in _TRIGGER_RULES.items():
            for term in trigger_terms:
                if not term:
                    continue
                if label == "relation_compare" and term in {"与", "和"}:
                    if any(marker in question for marker in _RELATION_COMPARE_MARKERS):
                        hits[label].append(term)
                    continue
                if term in question:
                    hits[label].append(term)
        return hits

    def _select_rule_type(self, rule_hits: dict[QuestionTypeLabel, list[str]]) -> QuestionTypeLabel:
        for question_type in QUESTION_TYPE_PRIORITY:
            if question_type == "fallback_balanced":
                continue
            if rule_hits.get(question_type):
                return question_type
        return "fallback_balanced"

    def _apply_skeleton_correction(
        self,
        *,
        question: str,
        skeleton: SkeletonExtractionResult,
        candidate_type: QuestionTypeLabel,
    ) -> QuestionTypeLabel:
        entity_count = len(skeleton.structured_entities)
        relation_types = {item.relation_type for item in skeleton.structured_relations if item.relation_type}
        has_compare_signal = any(marker in question for marker in _RELATION_COMPARE_MARKERS)
        single_entity_attribute = (
            entity_count <= 1
            and candidate_type == "fact_attribute"
            and any(item.relation_type in {"attribute", "factoid"} for item in skeleton.structured_relations)
        )

        if relation_types & _CAUSE_RELATION_TYPES:
            return "cause_explanation"
        if entity_count >= 2 and has_compare_signal and not single_entity_attribute:
            return "relation_compare"
        return candidate_type

    def _estimate_confidence(
        self,
        *,
        candidate_type: QuestionTypeLabel,
        rule_hits: dict[QuestionTypeLabel, list[str]],
        skeleton: SkeletonExtractionResult,
    ) -> QuestionTypeConfidence:
        entity_count = len(skeleton.structured_entities)
        relation_count = len(skeleton.structured_relations)
        if candidate_type == "fallback_balanced":
            return "low"
        trigger_count = len(rule_hits.get(candidate_type, []))
        if trigger_count >= 2:
            return "high"
        if candidate_type == "relation_compare" and entity_count >= 2 and relation_count >= 1:
            return "high"
        if candidate_type == "cause_explanation" and any(
            item.relation_type in _CAUSE_RELATION_TYPES for item in skeleton.structured_relations
        ):
            return "high"
        if trigger_count >= 1 or entity_count > 0 or relation_count > 0:
            return "medium"
        return "low"
