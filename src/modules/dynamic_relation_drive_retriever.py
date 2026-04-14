from __future__ import annotations

from dataclasses import dataclass

from core.schema import BenchmarkSample, RetrievedDocument
from modules.graph_expander import GraphIndex
from modules.question_type_classifier import (
    QuestionTypeClassification,
    QuestionTypeClassifier,
    QuestionTypeLabel,
)
from modules.relation_driven_retriever import RelationDrivenRetrieveResult, RelationDrivenRetriever
from modules.skeleton_extractor import SkeletonExtractionResult
from pipelines.base import PublicIndex
from retrievers.text_retriever import SupportsRetrieve, TextRetrieverQuery


@dataclass(frozen=True, slots=True)
class DynamicWeightProfile:
    semantic: float
    entity: float
    relation: float
    constraint: float

    def to_dict(self) -> dict[str, float]:
        return {
            "w_sem": self.semantic,
            "w_ent": self.entity,
            "w_rel": self.relation,
            "w_con": self.constraint,
        }


_WEIGHT_PROFILES: dict[QuestionTypeLabel, DynamicWeightProfile] = {
    "fact_attribute": DynamicWeightProfile(semantic=0.58, entity=0.22, relation=0.08, constraint=0.12),
    "cause_explanation": DynamicWeightProfile(semantic=0.42, entity=0.14, relation=0.28, constraint=0.16),
    "procedure_method": DynamicWeightProfile(semantic=0.32, entity=0.16, relation=0.36, constraint=0.16),
    "relation_compare": DynamicWeightProfile(semantic=0.28, entity=0.24, relation=0.32, constraint=0.16),
    "fallback_balanced": DynamicWeightProfile(semantic=0.50, entity=0.20, relation=0.15, constraint=0.15),
}


class DynamicRelationDriveRetriever(RelationDrivenRetriever):
    """Parallel scorer that preserves stage1 retrieval but changes stage2 fusion and filtering."""

    def __init__(
        self,
        text_index: PublicIndex,
        graph_index: GraphIndex,
        text_retriever: SupportsRetrieve | None = None,
    ) -> None:
        super().__init__(text_index, graph_index, text_retriever=text_retriever)
        self.question_type_classifier = QuestionTypeClassifier()

    def retrieve(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        *,
        top_k: int,
        use_skeleton_rewrite: bool = True,
        query_override: str | TextRetrieverQuery | None = None,
        question_type_info: QuestionTypeClassification | None = None,
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
        if isinstance(stage1_query, TextRetrieverQuery):
            query_trace = {
                "lexical_query": stage1_query.lexical_query,
                "dense_query": stage1_query.dense_query,
            }
        else:
            query_trace = stage1_query

        candidate_map = {document.section_id: document for document in candidate_pool}
        target_relations = [relation for relation in skeleton.structured_relations if relation.role == "target"] or skeleton.structured_relations
        anchor_entities = [entity for entity in skeleton.structured_entities if entity.role == "anchor"] or skeleton.structured_entities
        classification = question_type_info or self.question_type_classifier.classify(sample.question, skeleton)
        weight_profile = _WEIGHT_PROFILES[classification.question_type]

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
            filter_decision = self._dynamic_filter_decision(
                question_type=classification.question_type,
                semantic_score=semantic_scores.get(section_id, 0.0),
                entity_score=entity_alignment["score"],
                relation_score=relation_alignment["score"],
                constraint_score=constraint_satisfaction["score"],
                conflict_penalty=conflict_penalty["score"],
                has_anchor=bool(anchor_entities),
                has_target_relation=bool(target_relations),
                has_constraints=bool(skeleton.structured_constraints),
                high_semantic_protection=high_semantic_protection,
            )
            soft_penalty = self._soft_penalty(
                question_type=classification.question_type,
                semantic_score=semantic_scores.get(section_id, 0.0),
                entity_score=entity_alignment["score"],
                relation_score=relation_alignment["score"],
                constraint_score=constraint_satisfaction["score"],
                has_anchor=bool(anchor_entities),
                has_target_relation=bool(target_relations),
                has_constraints=bool(skeleton.structured_constraints),
                high_semantic_protection=high_semantic_protection,
            )
            final_score = (
                weight_profile.semantic * semantic_scores.get(section_id, 0.0)
                + weight_profile.entity * entity_alignment["score"]
                + weight_profile.relation * relation_alignment["score"]
                + weight_profile.constraint * constraint_satisfaction["score"]
                - conflict_penalty["score"]
                - soft_penalty["score"]
            )
            breakdown = {
                "question_type": classification.question_type,
                "question_type_confidence": classification.question_type_confidence,
                "question_type_evidence": classification.question_type_evidence,
                "dynamic_weight_profile": weight_profile.to_dict(),
                "semantic_score": semantic_scores.get(section_id, 0.0),
                "semantic_raw": semantic_raw.get(section_id, 0.0),
                "entity_alignment_score": entity_alignment["score"],
                "relation_alignment_score": relation_alignment["score"],
                "entity_relation_alignment_score": (
                    0.35 * entity_alignment["score"] + 0.45 * relation_alignment["score"] + 0.20 * constraint_satisfaction["score"]
                ),
                "constraint_satisfaction_score": constraint_satisfaction["score"],
                "constraint_score": constraint_satisfaction["score"],
                "alignment_details": {
                    "entities": entity_alignment["matches"],
                    "relations": relation_alignment["matches"],
                    "constraints": constraint_satisfaction["matches"],
                },
                "soft_penalties": soft_penalty,
                "conflict_penalty": conflict_penalty,
                "whether_high_semantic_protection_triggered": high_semantic_protection,
                "gate": filter_decision,
                "filter_reasons": self._build_dynamic_filter_reasons(
                    semantic_score=semantic_scores.get(section_id, 0.0),
                    entity_score=entity_alignment["score"],
                    relation_score=relation_alignment["score"],
                    constraint_score=constraint_satisfaction["score"],
                    conflict_penalty=conflict_penalty,
                    soft_penalty=soft_penalty,
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
                recovered["gate"]["drop"] = False
                recovered["gate"]["recovered"] = True
                recovered["filter_reasons"].append("recovered_for_candidate_floor")
                candidate_breakdown[section_id] = recovered

        ranked_ids = sorted(
            candidate_breakdown,
            key=lambda section_id: candidate_breakdown[section_id]["final_score"],
            reverse=True,
        )[:top_k]
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
                            "retriever": "dynamic_relation_drive",
                            "question_type": breakdown["question_type"],
                            "question_type_confidence": breakdown["question_type_confidence"],
                            "question_type_evidence": breakdown["question_type_evidence"],
                            "dynamic_weight_profile": breakdown["dynamic_weight_profile"],
                            "semantic_score": breakdown["semantic_score"],
                            "semantic_raw": breakdown["semantic_raw"],
                            "entity_alignment_score": breakdown["entity_alignment_score"],
                            "relation_alignment_score": breakdown["relation_alignment_score"],
                            "entity_relation_alignment_score": breakdown["entity_relation_alignment_score"],
                            "constraint_satisfaction_score": breakdown["constraint_satisfaction_score"],
                            "constraint_score": breakdown["constraint_score"],
                            "alignment_details": breakdown["alignment_details"],
                            "soft_penalties": breakdown["soft_penalties"],
                            "conflict_penalty": breakdown["conflict_penalty"]["score"],
                            "whether_high_semantic_protection_triggered": breakdown["whether_high_semantic_protection_triggered"],
                            "filter_reasons": breakdown["filter_reasons"],
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
                "mode": "dynamic_relation_drive",
                "stage1_query": query_trace,
                "query_source": query_source,
                "candidate_pool": stage1_candidate_rows,
                "skeleton_aware_scores": {
                    section_id: candidate_breakdown[section_id]
                    for section_id in ranked_ids
                },
                "filtered_out": {
                    **{section_id: filtered_out[section_id] for section_id in filtered_out},
                    **{
                        section_id: candidate_breakdown[section_id]
                        for section_id in candidate_breakdown
                        if section_id not in ranked_ids
                    },
                },
                "final_ranked_sections": ranked_ids,
                "kept_section_ids": list(candidate_breakdown.keys()),
                "filtered_out_section_ids": sorted(filtered_out.keys()),
                "filtered_out_reason_map": {
                    section_id: list(filtered_out[section_id].get("filter_reasons", []))
                    for section_id in filtered_out
                },
                "recovered_section_ids": recovery_ids,
                "fusion_weights": weight_profile.to_dict(),
                "structural_focus": {
                    "anchor_entities": [entity.name for entity in anchor_entities],
                    "target_relations": [relation.name for relation in target_relations],
                },
                "question_type": classification.question_type,
                "question_type_confidence": classification.question_type_confidence,
                "question_type_evidence": classification.question_type_evidence,
            },
        )

    def _should_trigger_high_semantic_protection(
        self,
        *,
        semantic_score: float,
        entity_alignment_score: float,
    ) -> bool:
        # High-semantic candidates with a minimum entity anchor often contain the gold text,
        # and relation-side graph coverage can be incomplete. These cases should be softened,
        # not hard-dropped, to avoid killing true positives already in the stage1 pool.
        return semantic_score >= 0.75 and entity_alignment_score >= 0.15

    def _soft_penalty(
        self,
        *,
        question_type: QuestionTypeLabel,
        semantic_score: float,
        entity_score: float,
        relation_score: float,
        constraint_score: float,
        has_anchor: bool,
        has_target_relation: bool,
        has_constraints: bool,
        high_semantic_protection: bool,
    ) -> dict[str, object]:
        reasons: list[str] = []
        penalty = 0.0
        relation_missing = has_target_relation and relation_score < 0.10
        relation_weak = has_target_relation and 0.10 <= relation_score < 0.25
        anchor_missing = has_anchor and entity_score < 0.10
        constraint_gap = has_constraints and constraint_score < 0.12

        if relation_missing:
            base_penalty = {
                "fact_attribute": 0.04,
                "cause_explanation": 0.08,
                "procedure_method": 0.14,
                "relation_compare": 0.12,
                "fallback_balanced": 0.09,
            }[question_type]
            if high_semantic_protection:
                base_penalty *= 0.5
                reasons.append("high_semantic_relation_softened")
            penalty += base_penalty
            reasons.append("relation_alignment_missing_soft")
        elif relation_weak:
            penalty += {
                "fact_attribute": 0.02,
                "cause_explanation": 0.04,
                "procedure_method": 0.08,
                "relation_compare": 0.06,
                "fallback_balanced": 0.05,
            }[question_type]
            reasons.append("relation_alignment_weak_soft")

        if anchor_missing:
            penalty += 0.03 if question_type == "fact_attribute" else 0.08
            reasons.append("entity_alignment_missing_soft")
        if constraint_gap and semantic_score < 0.85:
            penalty += 0.05 if question_type != "fact_attribute" else 0.03
            reasons.append("constraint_gap_soft")
        return {"score": penalty, "reasons": reasons}

    def _conflict_penalty(self, section_id: str, content: str, skeleton: SkeletonExtractionResult) -> dict[str, object]:
        base_conflict = self._constraint_conflict(section_id, content, skeleton)
        if not skeleton.structured_constraints:
            return {"score": 0.0, "reasons": [], "matched_target_constraints": [], "conflicting_constraints": []}
        matched_target_constraints: list[str] = []
        conflicting_constraints: list[str] = []
        text_blob = f"{content}\n" + "\n".join(self.graph_index.section_to_constraints.get(section_id, set()))
        lowered_blob = text_blob.lower()
        penalty = float(base_conflict.get("score", 0.0))
        for constraint in skeleton.structured_constraints:
            if constraint.normalized_value in lowered_blob:
                matched_target_constraints.append(constraint.value)
                continue
            mutually_exclusive_hits = [
                item
                for item in self.graph_index.section_to_constraints.get(section_id, set())
                if item.lower() != constraint.normalized_value and any(char.isdigit() for char in item)
            ]
            if mutually_exclusive_hits:
                conflicting_constraints.append(mutually_exclusive_hits[0])
                penalty = max(penalty, 0.45)
        return {
            "score": min(0.8, penalty),
            "reasons": list(base_conflict.get("reasons", [])) + (["explicit_constraint_conflict"] if conflicting_constraints else []),
            "matched_target_constraints": matched_target_constraints,
            "conflicting_constraints": conflicting_constraints,
        }

    def _dynamic_filter_decision(
        self,
        *,
        question_type: QuestionTypeLabel,
        semantic_score: float,
        entity_score: float,
        relation_score: float,
        constraint_score: float,
        conflict_penalty: float,
        has_anchor: bool,
        has_target_relation: bool,
        has_constraints: bool,
        high_semantic_protection: bool,
    ) -> dict[str, object]:
        reasons: list[str] = []
        drop = False
        if conflict_penalty >= 0.45 and has_constraints and constraint_score < 0.15:
            return {"drop": True, "reasons": ["constraint_conflict_drop"], "recovered": False}

        if question_type == "fact_attribute":
            if has_anchor and entity_score < 0.05 and semantic_score < 0.35:
                drop = True
                reasons.append("entity_alignment_too_weak")
        elif question_type == "cause_explanation":
            if has_anchor and entity_score < 0.05 and semantic_score < 0.30 and not high_semantic_protection:
                drop = True
                reasons.append("entity_alignment_too_weak")
        elif question_type == "procedure_method":
            if has_anchor and entity_score < 0.08 and relation_score < 0.08 and semantic_score < 0.35 and not high_semantic_protection:
                drop = True
                reasons.append("structure_filter_drop")
        elif question_type == "relation_compare":
            if has_target_relation and has_anchor and entity_score < 0.10 and relation_score < 0.10 and not high_semantic_protection:
                drop = True
                reasons.append("structure_filter_drop")
        else:
            if has_anchor and entity_score < 0.06 and relation_score < 0.06 and semantic_score < 0.30 and not high_semantic_protection:
                drop = True
                reasons.append("structure_filter_drop")
        if has_target_relation and relation_score < 0.10:
            reasons.append("target_relation_missing")
        elif has_target_relation and relation_score < 0.25:
            reasons.append("target_relation_weak")
        if has_anchor and entity_score < 0.10:
            reasons.append("anchor_missing")
        if has_constraints and constraint_score < 0.12:
            reasons.append("constraint_not_supported")
        return {"drop": drop, "reasons": reasons, "recovered": False}

    def _build_dynamic_filter_reasons(
        self,
        *,
        semantic_score: float,
        entity_score: float,
        relation_score: float,
        constraint_score: float,
        conflict_penalty: dict[str, object],
        soft_penalty: dict[str, object],
        filter_decision: dict[str, object],
    ) -> list[str]:
        reasons: list[str] = []
        if semantic_score >= 0.75:
            reasons.append("strong_semantic_match")
        elif semantic_score < 0.15:
            reasons.append("weak_semantic_match")
        if entity_score >= 0.4:
            reasons.append("entity_aligned")
        elif entity_score < 0.1:
            reasons.append("entity_alignment_missing")
        if relation_score >= 0.4:
            reasons.append("relation_aligned")
        elif relation_score < 0.1:
            reasons.append("relation_alignment_missing")
        if constraint_score >= 0.4:
            reasons.append("constraint_satisfied")
        elif constraint_score < 0.1:
            reasons.append("constraint_gap")
        if float(conflict_penalty["score"]) > 0:
            reasons.append("constraint_conflict")
        reasons.extend(list(soft_penalty.get("reasons", [])))
        reasons.extend(list(filter_decision.get("reasons", [])))
        return reasons
