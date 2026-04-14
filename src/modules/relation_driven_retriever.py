from __future__ import annotations

from dataclasses import dataclass

from core.schema import BenchmarkSample, RetrievedDocument
from modules.graph_expander import GraphIndex
from modules.skeleton_extractor import SkeletonExtractionResult
from pipelines.base import PublicIndex, overlap_score
from retrievers.text_retriever import LexicalTextRetriever, SupportsRetrieve, TextRetrieverQuery


@dataclass(slots=True)
class RelationDrivenRetrieveResult:
    documents: list[RetrievedDocument]
    scores: dict[str, float]
    details: dict


class RelationDrivenRetriever:
    """Skeleton-aware stage2 retrieval over a text-generated candidate pool."""

    def __init__(
        self,
        text_index: PublicIndex,
        graph_index: GraphIndex,
        text_retriever: SupportsRetrieve | None = None,
        *,
        alpha: float = 0.35,
        beta: float = 0.4,
        gamma: float = 0.25,
    ) -> None:
        self.text_index = text_index
        self.graph_index = graph_index
        self.text_retriever = text_retriever or LexicalTextRetriever(text_index)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def retrieve(
        self,
        sample: BenchmarkSample,
        skeleton: SkeletonExtractionResult,
        *,
        top_k: int,
        use_relation_driven: bool = True,
        use_skeleton_rewrite: bool = True,
        query_override: str | TextRetrieverQuery | None = None,
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

        if not use_relation_driven:
            documents = [
                document.model_copy(
                    update={
                        "rank": rank,
                        "metadata": {
                            **document.metadata,
                            "retriever": "text_stage1_only",
                            "semantic_score": float(document.metadata.get("dense_score", document.score)),
                            "entity_relation_alignment_score": 0.0,
                            "constraint_satisfaction_score": 0.0,
                            "fusion_score": float(document.score),
                        },
                    }
                )
                for rank, document in enumerate(candidate_pool[:top_k], start=1)
            ]
            return RelationDrivenRetrieveResult(
                documents=documents,
                scores={document.section_id: document.score for document in documents},
                details={
                    "mode": "text_only",
                    "stage1_query": query_trace,
                    "query_source": query_source,
                    "candidate_pool": stage1_candidate_rows,
                    "kept_section_ids": [document.section_id for document in documents],
                    "filtered_out_section_ids": [],
                    "filtered_out_reason_map": {},
                    "recovered_section_ids": [],
                    "kept_reasons": {document.section_id: ["semantic_rank"] for document in documents},
                    "fusion_weights": {"semantic": 1.0, "alignment": 0.0, "constraint": 0.0},
                },
            )

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
            conflict = self._constraint_conflict(section_id, content, skeleton)
            structural_alignment = 0.25 * entity_alignment["score"] + 0.55 * relation_alignment["score"] + 0.20 * constraint_satisfaction["score"]
            final_score = (
                self.alpha * semantic_scores.get(section_id, 0.0)
                + self.beta * structural_alignment
                + self.gamma * constraint_satisfaction["score"]
            )
            gate = self._structural_gate(
                anchor_score=entity_alignment["score"],
                relation_score=relation_alignment["score"],
                constraint_score=constraint_satisfaction["score"],
                conflict=conflict,
                has_anchor=bool(anchor_entities),
                has_target_relation=bool(target_relations),
                has_constraints=bool(skeleton.structured_constraints),
            )
            final_score *= gate["multiplier"]
            breakdown = {
                "semantic_score": semantic_scores.get(section_id, 0.0),
                "semantic_raw": semantic_raw.get(section_id, 0.0),
                "entity_alignment_score": entity_alignment["score"],
                "relation_alignment_score": relation_alignment["score"],
                "entity_relation_alignment_score": structural_alignment,
                "constraint_satisfaction_score": constraint_satisfaction["score"],
                "constraint_conflict": conflict,
                "alignment_details": {
                    "entities": entity_alignment["matches"],
                    "relations": relation_alignment["matches"],
                    "constraints": constraint_satisfaction["matches"],
                },
                "gate": gate,
                "filter_reasons": self._build_filter_reasons(
                    semantic_scores.get(section_id, 0.0),
                    entity_alignment["score"],
                    relation_alignment["score"],
                    constraint_satisfaction["score"],
                    conflict,
                    gate,
                    relation_alignment["matches"],
                ),
                "final_score": final_score,
            }
            if gate["drop"]:
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
                            "retriever": "relation_driven",
                            "semantic_score": breakdown["semantic_score"],
                            "semantic_raw": breakdown["semantic_raw"],
                            "entity_alignment_score": breakdown["entity_alignment_score"],
                            "relation_alignment_score": breakdown["relation_alignment_score"],
                            "entity_relation_alignment_score": breakdown["entity_relation_alignment_score"],
                            "constraint_satisfaction_score": breakdown["constraint_satisfaction_score"],
                            "alignment_details": breakdown["alignment_details"],
                            "filter_reasons": breakdown["filter_reasons"],
                            "fusion_score": breakdown["final_score"],
                        },
                    }
                )
            )
        return RelationDrivenRetrieveResult(
            documents=documents,
            scores={section_id: candidate_breakdown[section_id]["final_score"] for section_id in ranked_ids},
            details={
                "mode": "relation_driven",
                "stage1_query": query_trace,
                "query_source": query_source,
                "candidate_pool": stage1_candidate_rows,
                "skeleton_aware_scores": {
                    section_id: candidate_breakdown[section_id]
                    for section_id in ranked_ids
                },
                "filtered_out": {
                    **{
                        section_id: filtered_out[section_id]
                        for section_id in filtered_out
                    },
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
                "dense_scores": {
                    section_id: candidate_breakdown[section_id]["semantic_score"]
                    for section_id in ranked_ids
                },
                "relation_scores": {
                    section_id: candidate_breakdown[section_id]["relation_alignment_score"]
                    for section_id in ranked_ids
                },
                "constraint_scores": {
                    section_id: candidate_breakdown[section_id]["constraint_satisfaction_score"]
                    for section_id in ranked_ids
                },
                "fusion_weights": {"semantic": self.alpha, "alignment": self.beta, "constraint": self.gamma},
                "structural_focus": {
                    "anchor_entities": [entity.name for entity in anchor_entities],
                    "target_relations": [relation.name for relation in target_relations],
                },
            },
        )

    def _candidate_trace_row(self, document: RetrievedDocument) -> dict:
        branch_trace = document.metadata.get("hybrid_branch_trace", {})
        return {
            "section_id": document.section_id,
            "stage1_rank": document.rank,
            "stage1_score": float(document.score),
            "lexical_score": float(document.metadata.get("lexical_score", document.score)),
            "dense_score": float(document.metadata.get("dense_score", document.score)),
            "hybrid_score": float(document.metadata.get("hybrid_score", document.score)),
            "hybrid_sources": list(document.metadata.get("hybrid_sources", [])),
            "retriever": document.metadata.get("retriever"),
            "branch_trace": branch_trace if isinstance(branch_trace, dict) else {},
        }

    def _normalize_scores(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values()) or 1.0
        return {section_id: score / max_score for section_id, score in scores.items()}

    def _entity_alignment(
        self,
        section_id: str,
        content: str,
        skeleton: SkeletonExtractionResult,
        *,
        focus_entities: list | None = None,
    ) -> dict:
        entities = focus_entities if focus_entities is not None else skeleton.structured_entities
        if not entities:
            return {"score": 0.0, "matches": []}
        section_entities = self.graph_index.section_to_entities.get(section_id, set())
        matches = []
        total = 0.0
        for entity in entities:
            matched = entity.normalized_name in section_entities
            lexical = overlap_score(entity.name, content)
            role_weight = 1.2 if entity.role == "anchor" else 1.0
            contribution = role_weight * (1.0 if matched else lexical)
            total += contribution
            if matched or lexical > 0:
                matches.append({"entity": entity.name, "role": entity.role, "matched": matched, "lexical": lexical})
        return {"score": min(1.0, total / max(len(entities), 1)), "matches": matches}

    def _relation_alignment(
        self,
        section_id: str,
        content: str,
        skeleton: SkeletonExtractionResult,
        *,
        focus_relations: list | None = None,
    ) -> dict:
        relations = focus_relations if focus_relations is not None else skeleton.structured_relations
        if not relations:
            return {"score": 0.0, "matches": []}
        matches = []
        total = 0.0
        section_triples = self.graph_index.section_to_triples.get(section_id, [])
        section_relations = self.graph_index.section_to_relations.get(section_id, set())
        section_entities = self.graph_index.section_to_entities.get(section_id, set())
        for relation in relations:
            generic_relation = self._is_generic_relation_term(relation.name)
            predicate_match = relation.normalized_name in section_relations
            relation_type_match = relation.relation_type != "factoid" and any(
                relation.relation_type in self._infer_relation_types(predicate)
                for _, _, predicate, _ in section_triples
            )
            endpoint_bonus = 0.0
            if relation.subject_hint and relation.subject_hint.lower() in section_entities:
                endpoint_bonus += 0.05
            if relation.object_hint and relation.object_hint.lower() in section_entities:
                endpoint_bonus += 0.05
            directional_match = False
            for _, subject, predicate, obj in section_triples:
                if predicate.lower() != relation.normalized_name:
                    continue
                subject_ok = relation.subject_hint is None or relation.subject_hint.lower() == subject.lower()
                object_ok = relation.object_hint is None or relation.object_hint.lower() == obj.lower()
                if subject_ok or object_ok:
                    directional_match = True
                    endpoint_bonus += 0.15
                    break
            lexical = overlap_score(relation.name, content)
            type_bonus = 0.35 if relation_type_match else 0.0
            role_weight = 1.25 if relation.role == "target" else 1.0
            generic_penalty = 0.4 if generic_relation else 1.0
            if generic_relation:
                lexical *= 0.35
                endpoint_bonus *= 0.5
                type_bonus *= 0.35
            contribution = role_weight * generic_penalty * ((1.0 if predicate_match else lexical) + endpoint_bonus + type_bonus)
            total += min(1.0, contribution)
            if predicate_match or lexical > 0 or endpoint_bonus > 0 or relation_type_match:
                matches.append(
                    {
                        "relation": relation.name,
                        "relation_type": relation.relation_type,
                        "role": relation.role,
                        "predicate_match": predicate_match,
                        "relation_type_match": relation_type_match,
                        "directional_match": directional_match,
                        "lexical": lexical,
                        "endpoint_bonus": endpoint_bonus,
                        "generic_relation": generic_relation,
                        "generic_penalty": generic_penalty,
                    }
                )
        return {"score": min(1.0, total / max(len(relations), 1)), "matches": matches}

    def _constraint_satisfaction(self, section_id: str, content: str, skeleton: SkeletonExtractionResult) -> dict:
        if not skeleton.structured_constraints:
            return {"score": 0.0, "matches": []}
        section_constraints = self.graph_index.section_to_constraints.get(section_id, set())
        matches = []
        total = 0.0
        for constraint in skeleton.structured_constraints:
            graph_hit = any(constraint.normalized_value in item for item in section_constraints)
            lexical = overlap_score(constraint.value, content)
            contribution = 1.0 if graph_hit else lexical
            total += contribution
            if graph_hit or lexical > 0:
                matches.append(
                    {
                        "kind": constraint.kind,
                        "value": constraint.value,
                        "graph_hit": graph_hit,
                        "lexical": lexical,
                    }
                )
        return {"score": min(1.0, total / len(skeleton.structured_constraints)), "matches": matches}

    def _constraint_conflict(self, section_id: str, content: str, skeleton: SkeletonExtractionResult) -> dict:
        if not skeleton.structured_constraints:
            return {"score": 0.0, "reasons": []}
        section_constraints = self.graph_index.section_to_constraints.get(section_id, set())
        text_blob = " ".join(section_constraints) + " " + content
        reasons = []
        penalty = 0.0
        for constraint in skeleton.structured_constraints:
            if constraint.kind != "time":
                continue
            expected = constraint.normalized_value
            if expected in text_blob.lower():
                continue
            conflicting_times = [token for token in section_constraints if any(char.isdigit() for char in token)]
            if conflicting_times:
                penalty += 0.6
                reasons.append(f"time_conflict:{constraint.value}!={conflicting_times[0]}")
        return {"score": min(1.0, penalty), "reasons": reasons}

    def _structural_gate(
        self,
        *,
        anchor_score: float,
        relation_score: float,
        constraint_score: float,
        conflict: dict,
        has_anchor: bool,
        has_target_relation: bool,
        has_constraints: bool,
    ) -> dict:
        reasons = []
        multiplier = 1.0
        drop = False
        if has_target_relation and relation_score < 0.12:
            multiplier *= 0.1
            reasons.append("target_relation_missing")
        elif has_target_relation and relation_score < 0.3:
            multiplier *= 0.25
            reasons.append("target_relation_weak")
        if has_anchor and anchor_score < 0.08:
            multiplier *= 0.45
            reasons.append("anchor_missing")
        if has_constraints and constraint_score < 0.12:
            multiplier *= 0.55
            reasons.append("constraint_not_supported")
        if conflict["score"] > 0:
            multiplier *= max(0.1, 1.0 - conflict["score"])
            reasons.extend(conflict["reasons"])
        if has_target_relation and relation_score < 0.08 and has_anchor and anchor_score < 0.08:
            drop = True
            reasons.append("structure_filter_drop")
        if conflict["score"] >= 0.6 and constraint_score < 0.2:
            drop = True
            reasons.append("constraint_conflict_drop")
        return {"multiplier": multiplier, "drop": drop, "reasons": reasons, "recovered": False}

    def _infer_relation_types(self, text: str) -> set[str]:
        text = text.lower()
        relation_types = set()
        if any(marker in text for marker in ("原因", "为何", "为什么", "导致", "因为", "由于")):
            relation_types.add("cause")
        if any(marker in text for marker in ("步骤", "流程", "方法", "方式")):
            relation_types.add("procedure")
        if any(marker in text for marker in ("条件", "适用", "前提", "场景")):
            relation_types.add("condition")
        if any(marker in text for marker in ("范围", "对象", "地区", "部门")):
            relation_types.add("applicable_scope")
        if any(marker in text for marker in ("参数", "属性", "指标", "配置")):
            relation_types.add("attribute")
        if any(marker in text for marker in ("区别", "差异", "比较", "不同")):
            relation_types.add("comparison")
        if any(marker in text for marker in ("关系", "合作", "关联", "依赖")):
            relation_types.add("relation")
        return relation_types or {"factoid"}

    def _is_generic_relation_term(self, text: str) -> bool:
        normalized = str(text).strip().lower()
        if not normalized:
            return False
        generic_terms = {
            "支持",
            "提供",
            "实现",
            "具备",
            "承担",
            "采用",
            "用于",
            "进行",
            "拥有",
            "包含",
            "包括",
            "涉及",
            "相关",
            "作用",
            "方式",
            "功能",
            "特点",
            "说明",
            "介绍",
            "达到",
        }
        if normalized in generic_terms:
            return True
        return len(normalized) <= 2 and normalized in {"有", "是", "为", "将", "把"}

    def _build_filter_reasons(
        self,
        semantic_score: float,
        entity_score: float,
        relation_score: float,
        constraint_score: float,
        conflict: dict,
        gate: dict,
        relation_matches: list[dict],
    ) -> list[str]:
        reasons = []
        if semantic_score >= 0.4:
            reasons.append("strong_semantic_match")
        elif semantic_score < 0.15:
            reasons.append("weak_semantic_match")
        if any(match.get("generic_relation") for match in relation_matches):
            reasons.append("generic_relation_match")
        if relation_score >= 0.4:
            reasons.append("relation_aligned")
        elif relation_score < 0.1:
            reasons.append("relation_alignment_missing")
        if entity_score >= 0.4:
            reasons.append("entity_aligned")
        elif entity_score < 0.1:
            reasons.append("entity_alignment_missing")
        if constraint_score >= 0.4:
            reasons.append("constraint_satisfied")
        elif constraint_score < 0.1:
            reasons.append("constraint_gap")
        if conflict["score"] > 0:
            reasons.append("constraint_conflict")
        reasons.extend(gate["reasons"])
        return reasons
