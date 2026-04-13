from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from core.schema import (
    BenchmarkSample,
    QuestionSkeletonLabel,
    SkeletonRewritePayload,
    StructuredSkeletonConstraint,
    StructuredSkeletonEntity,
    StructuredSkeletonRelation,
)
from modules.graph_expander import GraphIndex
from modules.llm_skeleton_prompt import build_llm_skeleton_messages, parse_llm_skeleton_payload
from pipelines.base import overlap_score, tokenize

if TYPE_CHECKING:
    from clients.chat_llm_client import ChatLLMClient


SkeletonMode = Literal["oracle", "predicted", "stub_predicted", "llm_predicted"]

_TIME_PATTERN = re.compile(
    r"(?:\d{4}年\d{1,2}月\d{1,2}日|\d{4}年\d{1,2}月|\d{4}年|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}月\d{1,2}日|"
    r"当前|目前|如今|当时|之前|之后|以来|期间|同时|最早|最晚|首次|最后)"
)
_SCOPE_MARKERS = ("哪些", "哪种", "哪类", "哪个", "范围", "领域", "地区", "国家", "部门", "阶段", "部分")
_CONDITION_MARKERS = ("如果", "若", "当", "在", "由于", "因为", "导致", "为了", "通过", "依赖", "是否", "需要")
_STOPWORDS = {
    "什么",
    "为何",
    "为什么",
    "如何",
    "哪些",
    "哪种",
    "哪个",
    "的是",
    "以及",
    "并且",
    "相关",
    "情况",
    "影响",
    "进行",
    "是否",
}
_RELATION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("cause", ("原因", "为何", "为什么", "导致", "引发", "造成", "因为", "由于")),
    ("procedure", ("步骤", "流程", "如何", "怎么", "怎样", "方式", "方法")),
    ("condition", ("条件", "前提", "适用", "场景", "情况下", "什么时候", "何时")),
    ("applicable_scope", ("范围", "适用范围", "对象", "哪些", "哪类", "哪个", "地区", "部门", "阶段")),
    ("attribute", ("参数", "属性", "特征", "指标", "配置", "数值", "值")),
    ("comparison", ("区别", "差异", "比较", "相比", "优于", "不同", "变化")),
    ("relation", ("关系", "关联", "联系", "合作", "依赖", "属于")),
]
_RELATION_ALIAS = {
    "合作原因": "cause",
    "原因": "cause",
    "导致": "cause",
    "引发": "cause",
    "步骤": "procedure",
    "流程": "procedure",
    "条件": "condition",
    "适用": "condition",
    "范围": "applicable_scope",
    "参数": "attribute",
    "属性": "attribute",
    "区别": "comparison",
    "比较": "comparison",
    "关系": "relation",
    "合作": "relation",
}


@dataclass(slots=True)
class SkeletonExtractionResult:
    entities: list[str]
    relations: list[str]
    constraints: list[str]
    skeleton: QuestionSkeletonLabel
    mode: SkeletonMode
    structured_entities: list[StructuredSkeletonEntity]
    structured_relations: list[StructuredSkeletonRelation]
    structured_constraints: list[StructuredSkeletonConstraint]
    rewrite_payload: SkeletonRewritePayload
    details: dict

    def rewritten_query(self, question: str | None = None) -> str:
        return self.rewrite_payload.retrieval_query or question or self.rewrite_payload.original_query

    @property
    def compensation_query(self) -> str:
        return self.rewrite_payload.compensation_query or self.rewrite_payload.retrieval_query

    def to_trace_dict(self) -> dict:
        anchor_entities = [item.model_dump() for item in self.structured_entities if item.role == "anchor"]
        target_relations = [item.model_dump() for item in self.structured_relations if item.role == "target"]
        return {
            "anchor_entities": anchor_entities,
            "target_relations": target_relations,
            "entities": [item.model_dump() for item in self.structured_entities],
            "relations": [item.model_dump() for item in self.structured_relations],
            "constraints": [item.model_dump() for item in self.structured_constraints],
            "rewrite_payload": self.rewrite_payload.model_dump(),
            "mode": self.mode,
            "details": self.details,
        }


class SkeletonExtractor:
    """Lightweight but usable Chapter-4 skeleton extractor."""

    def __init__(
        self,
        graph_index: GraphIndex | None = None,
        *,
        llm_client: ChatLLMClient | None = None,
        llm_max_tokens: int = 256,
        llm_temperature: float = 0.0,
        llm_entity_candidate_limit: int = 10,
        llm_relation_candidate_limit: int = 10,
    ) -> None:
        self.graph_index = graph_index
        self.llm_client = llm_client
        self.llm_max_tokens = llm_max_tokens
        self.llm_temperature = llm_temperature
        self.llm_entity_candidate_limit = llm_entity_candidate_limit
        self.llm_relation_candidate_limit = llm_relation_candidate_limit

    def extract(self, sample: BenchmarkSample, mode: SkeletonMode = "oracle") -> SkeletonExtractionResult:
        normalized_mode: SkeletonMode = "stub_predicted" if mode == "predicted" else mode
        if normalized_mode == "oracle":
            return self._build_oracle_result(sample)
        if normalized_mode == "stub_predicted":
            return self._build_stub_predicted_result(sample, requested_mode=normalized_mode)
        if normalized_mode == "llm_predicted":
            return self._build_llm_predicted_result(sample)
        raise ValueError(f"Unsupported skeleton mode: {mode}")

    def _build_oracle_result(self, sample: BenchmarkSample) -> SkeletonExtractionResult:
        structured_entities = [
            StructuredSkeletonEntity(
                name=item,
                normalized_name=item.lower(),
                role="anchor" if index == 0 else "context",
                surface=item,
                confidence=1.0,
                source="oracle_label",
            )
            for index, item in enumerate(sample.entities)
            if item
        ]
        structured_relations = [
            StructuredSkeletonRelation(
                name=item,
                normalized_name=item.lower(),
                relation_type=self._canonical_relation_type(item, sample.question),
                role="target" if index == 0 else "context",
                surface=item,
                confidence=1.0,
                source="oracle_label",
            )
            for index, item in enumerate(sample.relations)
            if item
        ]
        structured_constraints = [
            StructuredSkeletonConstraint(
                kind="oracle",
                value=item,
                normalized_value=item.lower(),
                confidence=1.0,
                source="oracle_label",
            )
            for item in sample.constraints
            if item
        ]
        rewrite_payload = self._build_rewrite_payload(sample.question, structured_entities, structured_relations, structured_constraints)
        skeleton = sample.skeleton_label
        return SkeletonExtractionResult(
            entities=list(skeleton.entities),
            relations=list(skeleton.relations),
            constraints=list(skeleton.constraints),
            skeleton=skeleton,
            mode="oracle",
            structured_entities=structured_entities,
            structured_relations=structured_relations,
            structured_constraints=structured_constraints,
            rewrite_payload=rewrite_payload,
            details={"mode": "oracle", "source": "benchmark_fields"},
        )

    def _build_stub_predicted_result(
        self,
        sample: BenchmarkSample,
        *,
        requested_mode: SkeletonMode,
    ) -> SkeletonExtractionResult:
        question = sample.question.strip()
        question_tokens = tokenize(question)
        entity_candidates, relation_candidates, _ = self._collect_candidates(
            question,
            question_tokens,
            entity_limit=5,
            relation_limit=5,
        )

        constraint_candidates = self._extract_constraints(question, question_tokens)
        entity_candidates = self._assign_entity_roles(entity_candidates, question)
        relation_candidates = self._assign_relation_roles(relation_candidates, question, entity_candidates)
        entity_candidates = self._prune_entities(entity_candidates, relation_candidates)
        relation_candidates = self._prune_relations(relation_candidates)
        rewrite_payload = self._build_rewrite_payload(question, entity_candidates, relation_candidates, constraint_candidates)
        skeleton = QuestionSkeletonLabel(
            entities=[item.name for item in entity_candidates],
            relations=[item.name for item in relation_candidates],
            constraints=[item.value for item in constraint_candidates],
        )
        return SkeletonExtractionResult(
            entities=skeleton.entities,
            relations=skeleton.relations,
            constraints=skeleton.constraints,
            skeleton=skeleton,
            mode=requested_mode,
            structured_entities=entity_candidates,
            structured_relations=relation_candidates,
            structured_constraints=constraint_candidates,
            rewrite_payload=rewrite_payload,
            details={
                "mode": "predicted",
                "requested_mode": requested_mode,
                "source": "heuristic_graph_aligned_parser",
                "question_tokens": question_tokens,
                "raw_relation_cues": self._collect_relation_cues(question),
                "entity_count": len(entity_candidates),
                "relation_count": len(relation_candidates),
                "constraint_count": len(constraint_candidates),
            },
        )

    def _build_llm_predicted_result(self, sample: BenchmarkSample) -> SkeletonExtractionResult:
        question = sample.question.strip()
        question_tokens = tokenize(question)
        entity_candidates, relation_candidates, _ = self._collect_candidates(
            question,
            question_tokens,
            entity_limit=self.llm_entity_candidate_limit,
            relation_limit=self.llm_relation_candidate_limit,
        )
        details = {
            "mode": "llm_predicted",
            "requested_mode": "llm_predicted",
            "source": "llm_graph_aligned_parser",
            "question_tokens": question_tokens,
            "entity_candidates": [item.name for item in entity_candidates],
            "relation_candidates": [item.name for item in relation_candidates],
            "raw_relation_cues": self._collect_relation_cues(question),
            "parsed_ok": False,
            "raw_llm_output": None,
            "anchored_entities": [],
            "anchored_relations": [],
            "unmatched_entities": [],
            "unmatched_relations": [],
            "normalized_constraints": [],
        }

        client = self.llm_client
        if client is None:
            details["error"] = "llm_client_not_configured"
            return self._build_empty_result(question, details)
        if not client.api_key:
            details["error"] = "GENERATOR_API_KEY is not configured."
            return self._build_empty_result(question, details)
        if not client.model:
            details["error"] = "GENERATOR_MODEL is not configured."
            return self._build_empty_result(question, details)

        entity_candidate_names = [item.name for item in entity_candidates]
        relation_candidate_names = [item.name for item in relation_candidates]
        messages = build_llm_skeleton_messages(
            question=question,
            entity_candidates=entity_candidate_names,
            relation_candidates=relation_candidate_names,
        )

        try:
            response = client.chat_completion(
                messages=messages,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                json_mode=True,
            )
            details["raw_llm_output"] = response.content
            details["model_name"] = response.model_name
            details["finish_reason"] = response.finish_reason
            details["latency_ms"] = response.latency_ms
            payload = parse_llm_skeleton_payload(response.content)
        except Exception as exc:
            details["error"] = str(exc)
            return self._build_empty_result(question, details)

        details["parsed_ok"] = True
        normalized_entities = self._normalize_string_list(payload.get("entities", []))
        normalized_relations = self._normalize_string_list(payload.get("relations", []))
        normalized_constraints = self._normalize_string_list(payload.get("constraints", []))

        anchored_entity_names, unmatched_entities = self._anchor_terms(
            normalized_entities,
            entity_candidate_names,
        )
        anchored_relation_names, unmatched_relations = self._anchor_terms(
            normalized_relations,
            relation_candidate_names,
        )
        details["anchored_entities"] = anchored_entity_names
        details["anchored_relations"] = anchored_relation_names
        details["unmatched_entities"] = unmatched_entities
        details["unmatched_relations"] = unmatched_relations
        details["normalized_constraints"] = normalized_constraints

        structured_entities = self._build_structured_entities_for_llm(
            anchored_entity_names,
            entity_candidates,
            question,
        )
        structured_relations = self._build_structured_relations_for_llm(
            anchored_relation_names,
            relation_candidates,
            question,
            structured_entities,
        )
        structured_constraints = self._build_structured_constraints_for_llm(normalized_constraints)
        rewrite_payload = self._build_rewrite_payload(question, structured_entities, structured_relations, structured_constraints)
        skeleton = QuestionSkeletonLabel(
            entities=[item.name for item in structured_entities],
            relations=[item.name for item in structured_relations],
            constraints=[item.value for item in structured_constraints],
        )
        details["entity_count"] = len(structured_entities)
        details["relation_count"] = len(structured_relations)
        details["constraint_count"] = len(structured_constraints)
        return SkeletonExtractionResult(
            entities=skeleton.entities,
            relations=skeleton.relations,
            constraints=skeleton.constraints,
            skeleton=skeleton,
            mode="llm_predicted",
            structured_entities=structured_entities,
            structured_relations=structured_relations,
            structured_constraints=structured_constraints,
            rewrite_payload=rewrite_payload,
            details=details,
        )

    def _build_empty_result(self, question: str, details: dict) -> SkeletonExtractionResult:
        rewrite_payload = self._build_rewrite_payload(question, [], [], [])
        skeleton = QuestionSkeletonLabel(entities=[], relations=[], constraints=[])
        return SkeletonExtractionResult(
            entities=[],
            relations=[],
            constraints=[],
            skeleton=skeleton,
            mode="llm_predicted",
            structured_entities=[],
            structured_relations=[],
            structured_constraints=[],
            rewrite_payload=rewrite_payload,
            details=details,
        )

    def _collect_candidates(
        self,
        question: str,
        question_tokens: list[str],
        *,
        entity_limit: int,
        relation_limit: int,
    ) -> tuple[list[StructuredSkeletonEntity], list[StructuredSkeletonRelation], set[str]]:
        question_token_set = set(question_tokens)
        graph_index = self.graph_index

        entity_candidates: list[StructuredSkeletonEntity] = []
        relation_candidates: list[StructuredSkeletonRelation] = []
        if graph_index is not None:
            entity_candidates = self._match_entities(question, question_token_set, graph_index, limit=entity_limit)
            relation_candidates = self._match_relations(question, question_token_set, graph_index, limit=relation_limit)
        return entity_candidates, relation_candidates, question_token_set

    def _match_entities(
        self,
        question: str,
        question_token_set: set[str],
        graph_index: GraphIndex,
        *,
        limit: int = 5,
    ) -> list[StructuredSkeletonEntity]:
        candidates: list[tuple[str, float, dict]] = []
        for entity, section_ids in graph_index.entity_to_sections.items():
            if not entity:
                continue
            score = self._lexical_alignment_score(question, question_token_set, entity)
            if score <= 0:
                continue
            candidates.append(
                (
                    entity,
                    score,
                    {
                        "matched_sections": min(len(section_ids), 20),
                        "token_overlap": overlap_score(question, entity),
                    },
                )
            )
        ranked = sorted(candidates, key=lambda item: (item[1], item[2]["matched_sections"]), reverse=True)[:limit]
        return [
            StructuredSkeletonEntity(
                name=entity,
                normalized_name=entity.lower(),
                role="context",
                surface=entity,
                confidence=min(1.0, 0.45 + score),
                source="graph_vocab_overlap",
                alignment=alignment,
            )
            for entity, score, alignment in ranked
        ]

    def _match_relations(
        self,
        question: str,
        question_token_set: set[str],
        graph_index: GraphIndex,
        *,
        limit: int = 5,
    ) -> list[StructuredSkeletonRelation]:
        relation_candidates: list[tuple[str, float, dict]] = []
        for relation, triples in graph_index.relation_to_triples.items():
            if not relation:
                continue
            score = self._lexical_alignment_score(question, question_token_set, relation)
            if score <= 0:
                continue
            subject_hints = [triple[1] for triple in triples[:3] if triple[1]]
            object_hints = [triple[3] for triple in triples[:3] if triple[3]]
            relation_candidates.append(
                (
                    relation,
                    score,
                    {
                        "matched_triples": min(len(triples), 20),
                        "token_overlap": overlap_score(question, relation),
                        "subject_hints": subject_hints,
                        "object_hints": object_hints,
                    },
                )
            )
        ranked = sorted(relation_candidates, key=lambda item: (item[1], item[2]["matched_triples"]), reverse=True)[:limit]
        structured: list[StructuredSkeletonRelation] = []
        for relation, score, alignment in ranked:
            structured.append(
                StructuredSkeletonRelation(
                    name=relation,
                    normalized_name=relation.lower(),
                    relation_type=self._canonical_relation_type(relation, question),
                    role="context",
                    surface=relation,
                    confidence=min(1.0, 0.4 + score),
                    source="graph_relation_overlap",
                    subject_hint=alignment["subject_hints"][0] if alignment["subject_hints"] else None,
                    object_hint=alignment["object_hints"][0] if alignment["object_hints"] else None,
                    alignment=alignment,
                )
            )
        return structured

    def _extract_constraints(self, question: str, question_tokens: list[str]) -> list[StructuredSkeletonConstraint]:
        constraints: list[StructuredSkeletonConstraint] = []
        seen: set[tuple[str, str]] = set()
        for match in _TIME_PATTERN.finditer(question):
            value = match.group(0).strip()
            if not value:
                continue
            key = ("time", value.lower())
            if key in seen:
                continue
            seen.add(key)
            constraints.append(
                StructuredSkeletonConstraint(
                    kind="time",
                    value=value,
                    normalized_value=value.lower(),
                    confidence=0.9,
                    source="regex_time",
                    alignment={"pattern": "time", "span": [match.start(), match.end()]},
                )
            )

        for token in question_tokens:
            if token in _SCOPE_MARKERS:
                window = self._build_marker_window(question_tokens, token)
                if window:
                    key = ("scope", window.lower())
                    if key not in seen:
                        seen.add(key)
                        constraints.append(
                            StructuredSkeletonConstraint(
                                kind="scope",
                                value=window,
                                normalized_value=window.lower(),
                                confidence=0.65,
                                source="marker_scope",
                                alignment={"marker": token},
                            )
                        )

        for token in question_tokens:
            if token in _CONDITION_MARKERS:
                window = self._build_marker_window(question_tokens, token)
                if window:
                    key = ("condition", window.lower())
                    if key not in seen:
                        seen.add(key)
                        constraints.append(
                            StructuredSkeletonConstraint(
                                kind="condition",
                                value=window,
                                normalized_value=window.lower(),
                                confidence=0.6,
                                source="marker_condition",
                                alignment={"marker": token},
                            )
                        )
        return constraints[:6]

    def _assign_entity_roles(self, entities: list[StructuredSkeletonEntity], question: str) -> list[StructuredSkeletonEntity]:
        if not entities:
            return entities
        ranked = sorted(
            entities,
            key=lambda item: (
                item.confidence,
                1.0 if item.name.lower() in question.lower() else 0.0,
                len(item.name),
            ),
            reverse=True,
        )
        anchor_name = ranked[0].normalized_name
        assigned: list[StructuredSkeletonEntity] = []
        for entity in entities:
            role = "anchor" if entity.normalized_name == anchor_name else "context"
            assigned.append(entity.model_copy(update={"role": role}))
        return assigned

    def _assign_relation_roles(
        self,
        relations: list[StructuredSkeletonRelation],
        question: str,
        entities: list[StructuredSkeletonEntity],
    ) -> list[StructuredSkeletonRelation]:
        question_relation_type = self._canonical_relation_type("", question)
        if not relations:
            fallback_type = question_relation_type
            return [
                StructuredSkeletonRelation(
                    name=fallback_type,
                    normalized_name=fallback_type,
                    relation_type=fallback_type,
                    role="target",
                    surface="",
                    confidence=0.35 if fallback_type != "factoid" else 0.0,
                    source="question_pattern",
                    subject_hint=entities[0].name if entities else None,
                    alignment={"raw_cues": self._collect_relation_cues(question)},
                )
            ] if fallback_type != "factoid" else []
        ranked = sorted(
            relations,
            key=lambda item: (
                1.0 if item.relation_type == question_relation_type and question_relation_type != "factoid" else 0.0,
                1.0 if item.relation_type != "factoid" else 0.0,
                item.confidence,
                1.0 if item.name.lower() in question.lower() else 0.0,
            ),
            reverse=True,
        )
        target_name = ranked[0].normalized_name
        assigned: list[StructuredSkeletonRelation] = []
        for relation in relations:
            role = "target" if relation.normalized_name == target_name else "context"
            subject_hint = relation.subject_hint or (entities[0].name if entities else None)
            assigned.append(relation.model_copy(update={"role": role, "subject_hint": subject_hint}))
        return assigned

    def _prune_entities(
        self,
        entities: list[StructuredSkeletonEntity],
        relations: list[StructuredSkeletonRelation],
    ) -> list[StructuredSkeletonEntity]:
        if not entities:
            return []
        max_count = 2 if relations else 3
        ranked = sorted(
            entities,
            key=lambda item: (
                1.0 if item.role == "anchor" else 0.0,
                item.confidence,
            ),
            reverse=True,
        )
        return ranked[:max_count]

    def _prune_relations(self, relations: list[StructuredSkeletonRelation]) -> list[StructuredSkeletonRelation]:
        if not relations:
            return []
        ranked = sorted(
            relations,
            key=lambda item: (
                1.0 if item.role == "target" else 0.0,
                1.0 if item.relation_type != "factoid" else 0.0,
                item.confidence,
            ),
            reverse=True,
        )
        return ranked[:2]

    def _build_structured_entities_for_llm(
        self,
        entities: list[str],
        candidates: list[StructuredSkeletonEntity],
        question: str,
    ) -> list[StructuredSkeletonEntity]:
        if not entities:
            return []
        candidate_map = {item.name: item for item in candidates}
        structured: list[StructuredSkeletonEntity] = []
        for entity in entities:
            candidate = candidate_map.get(entity)
            if candidate is not None:
                structured.append(candidate.model_copy())
                continue
            structured.append(
                StructuredSkeletonEntity(
                    name=entity,
                    normalized_name=entity.lower(),
                    role="context",
                    surface=entity,
                    confidence=0.45 if entity.lower() in question.lower() else 0.3,
                    source="llm_output_unmatched",
                )
            )
        return self._assign_entity_roles(structured, question)

    def _build_structured_relations_for_llm(
        self,
        relations: list[str],
        candidates: list[StructuredSkeletonRelation],
        question: str,
        entities: list[StructuredSkeletonEntity],
    ) -> list[StructuredSkeletonRelation]:
        if not relations:
            return []
        candidate_map = {item.name: item for item in candidates}
        structured: list[StructuredSkeletonRelation] = []
        for relation in relations:
            candidate = candidate_map.get(relation)
            if candidate is not None:
                structured.append(candidate.model_copy())
                continue
            structured.append(
                StructuredSkeletonRelation(
                    name=relation,
                    normalized_name=relation.lower(),
                    relation_type=self._canonical_relation_type(relation, question),
                    role="context",
                    surface=relation,
                    confidence=0.35 if relation.lower() in question.lower() else 0.25,
                    source="llm_output_unmatched",
                    subject_hint=entities[0].name if entities else None,
                )
            )
        assigned = self._assign_relation_roles(structured, question, entities)
        return self._prune_relations(assigned)

    def _build_structured_constraints_for_llm(self, constraints: list[str]) -> list[StructuredSkeletonConstraint]:
        structured: list[StructuredSkeletonConstraint] = []
        for constraint in constraints:
            kind = "time" if _TIME_PATTERN.search(constraint) else "llm"
            structured.append(
                StructuredSkeletonConstraint(
                    kind=kind,
                    value=constraint,
                    normalized_value=constraint.lower(),
                    confidence=0.5,
                    source="llm_output",
                )
            )
        return structured

    def _normalize_string_list(self, values: object) -> list[str]:
        if not isinstance(values, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            text = re.sub(r"\s+", " ", str(value).strip())
            if not text:
                continue
            lowered = text.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            cleaned.append(text)
        return cleaned

    def _anchor_terms(self, predicted: list[str], candidates: list[str]) -> tuple[list[str], list[str]]:
        candidate_map = {candidate.lower(): candidate for candidate in candidates}
        anchored: list[str] = []
        unmatched: list[str] = []
        for term in predicted:
            term_key = term.lower()
            candidate = candidate_map.get(term_key)
            if candidate is None:
                best_candidate = ""
                best_score = 0.0
                for option in candidates:
                    score = max(overlap_score(term, option), overlap_score(option, term))
                    if score > best_score:
                        best_candidate = option
                        best_score = score
                if best_candidate and best_score >= 0.75:
                    candidate = best_candidate
                else:
                    unmatched.append(term)
                    candidate = term
            if candidate not in anchored:
                anchored.append(candidate)
        return anchored, unmatched

    def _build_marker_window(self, question_tokens: list[str], marker: str) -> str:
        try:
            index = question_tokens.index(marker)
        except ValueError:
            return ""
        window = [marker]
        for token in question_tokens[index + 1 : index + 4]:
            if token in _STOPWORDS:
                continue
            window.append(token)
        return " ".join(window).strip()

    def _lexical_alignment_score(self, question: str, question_token_set: set[str], candidate: str) -> float:
        candidate_tokens = [token for token in tokenize(candidate) if token not in _STOPWORDS]
        if not candidate_tokens:
            candidate_tokens = [candidate.lower()]
        token_overlap = len(question_token_set & set(candidate_tokens)) / len(set(candidate_tokens))
        surface_bonus = 1.0 if candidate.lower() in question.lower() else 0.0
        fuzzy_bonus = overlap_score(question, candidate)
        char_overlap = 0.0
        candidate_chars = {char for char in candidate.lower() if not char.isspace()}
        question_chars = {char for char in question.lower() if not char.isspace()}
        if candidate_chars and question_chars:
            char_overlap = len(candidate_chars & question_chars) / len(candidate_chars)
        score = 0.35 * token_overlap + 0.2 * fuzzy_bonus + 0.25 * surface_bonus + 0.2 * char_overlap
        return score if score >= 0.14 else 0.0

    def _collect_relation_cues(self, question: str) -> list[str]:
        cues = []
        for relation_type, markers in _RELATION_PATTERNS:
            for marker in markers:
                if marker in question:
                    cues.append(f"{relation_type}:{marker}")
        return cues

    def _canonical_relation_type(self, relation: str, question: str) -> str:
        relation_lower = relation.lower()
        for alias, relation_type in _RELATION_ALIAS.items():
            if alias and alias.lower() in relation_lower:
                return relation_type
        for relation_type, markers in _RELATION_PATTERNS:
            if any(marker in relation for marker in markers):
                return relation_type
        for relation_type, markers in _RELATION_PATTERNS:
            if any(marker in question for marker in markers):
                return relation_type
        return "factoid"

    def _build_rewrite_payload(
        self,
        question: str,
        entities: list[StructuredSkeletonEntity],
        relations: list[StructuredSkeletonRelation],
        constraints: list[StructuredSkeletonConstraint],
    ) -> SkeletonRewritePayload:
        entity_terms = [item.name for item in entities]
        anchor_terms = [item.name for item in entities if item.role == "anchor"]
        relation_terms = [item.name for item in relations]
        target_relation_terms = [
            item.relation_type if item.relation_type != "factoid" else item.name
            for item in relations
            if item.role == "target"
        ]
        constraint_terms = [f"{item.kind}:{item.value}" for item in constraints]
        structure_segments = []
        if anchor_terms:
            structure_segments.append("anchor " + " ".join(anchor_terms))
        if target_relation_terms:
            structure_segments.append("target_relation " + " ".join(target_relation_terms))
        elif relation_terms:
            structure_segments.append("relations " + " ".join(relation_terms))
        if entity_terms:
            structure_segments.append("entities " + " ".join(entity_terms))
        if constraint_terms:
            structure_segments.append("constraints " + " ".join(constraint_terms))
        structure_query = " ; ".join(structure_segments)
        retrieval_query = " ; ".join(segment for segment in [question, structure_query] if segment)
        compensation_segments = [question]
        if anchor_terms:
            compensation_segments.append("context about " + " ".join(anchor_terms[:2]))
        if target_relation_terms:
            compensation_segments.append("details for " + " ".join(target_relation_terms[:1]))
        elif relation_terms:
            compensation_segments.append("details for " + " ".join(relation_terms[:1]))
        if constraint_terms:
            compensation_segments.append("under " + " ".join(constraint_terms[:2]))
        return SkeletonRewritePayload(
            original_query=question,
            retrieval_query=retrieval_query,
            structure_query=structure_query,
            compensation_query=" ; ".join(segment for segment in compensation_segments if segment),
            entity_query=" ".join(entity_terms),
            relation_query=" ".join(relation_terms),
            constraint_query=" ".join(constraint_terms),
        )
