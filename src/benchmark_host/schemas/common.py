from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


QuestionType = Literal["fact", "relation", "multi_evidence", "explanation"]
SourceScope = Literal["single_section", "single_doc", "cross_doc"]


@dataclass(slots=True)
class Evidence:
    doc_id: str
    section_id: str
    quote: str


@dataclass(slots=True)
class BenchmarkSample:
    qid: str
    question: str
    answer_short: str
    answer_long: str
    question_type: QuestionType
    entities: list[str] = field(default_factory=list)
    relations: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    source_scope: SourceScope = "single_section"
    requires_text_compensation: bool = False
    doc_id: str = ""
    section_id: str = ""
    support_score: float = 0.0
    completeness_score: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkSample":
        return cls(
            qid=data["qid"],
            question=data["question"],
            answer_short=data.get("answer_short", ""),
            answer_long=data.get("answer_long", ""),
            question_type=data.get("question_type", "fact"),
            entities=list(data.get("entities", [])),
            relations=list(data.get("relations", [])),
            constraints=list(data.get("constraints", [])),
            evidence=[Evidence(**item) for item in data.get("evidence", [])],
            source_scope=data.get("source_scope", "single_section"),
            requires_text_compensation=bool(data.get("requires_text_compensation", False)),
            doc_id=data.get("doc_id", ""),
            section_id=data.get("section_id", ""),
            support_score=float(data.get("support_score", 0.0)),
            completeness_score=float(data.get("completeness_score", 0.0)),
        )


@dataclass(slots=True)
class MarkdownSection:
    doc_id: str
    file_path: str
    doc_title: str
    section_id: str
    block_id: str
    section_path: list[str]
    content: str
    block_type: str
    char_len: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarkdownSection":
        return cls(**data)


@dataclass(slots=True)
class ExtractedNode:
    name: str
    type: str = "unknown"
    normalized_name: str = ""


@dataclass(slots=True)
class ExtractedRelation:
    subject: str
    predicate: str
    object: str


@dataclass(slots=True)
class KnowledgeExtraction:
    doc_id: str
    section_id: str
    section_path: list[str]
    content: str
    entities: list[ExtractedNode] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    attributes: list[dict[str, Any]] = field(default_factory=list)
    procedures: list[str] = field(default_factory=list)
    evidence_spans: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeExtraction":
        extraction = data.get("extraction", {}) or {}
        entities = []
        for item in extraction.get("entities", []):
            if isinstance(item, dict):
                entities.append(
                    ExtractedNode(
                        name=item.get("name", ""),
                        type=item.get("type", "unknown"),
                        normalized_name=item.get("normalized_name", item.get("name", "")),
                    )
                )
            elif isinstance(item, str):
                entities.append(ExtractedNode(name=item, normalized_name=item))
        relations = []
        for item in extraction.get("relations", []):
            if isinstance(item, dict):
                relations.append(
                    ExtractedRelation(
                        subject=item.get("subject", ""),
                        predicate=item.get("predicate", ""),
                        object=item.get("object", ""),
                    )
                )
        constraints = [str(item) for item in extraction.get("constraints", [])]
        return cls(
            doc_id=data["doc_id"],
            section_id=data["section_id"],
            section_path=list(data.get("section_path", [])),
            content=data.get("content", ""),
            entities=entities,
            relations=relations,
            constraints=constraints,
            attributes=list(extraction.get("attributes", [])),
            procedures=[str(item) for item in extraction.get("procedures", [])],
            evidence_spans=[str(item) for item in extraction.get("evidence_spans", [])],
        )


@dataclass(slots=True)
class RetrievalCandidate:
    section_id: str
    doc_id: str
    score: float
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationResult:
    answer: str
    evidence: list[Evidence]
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExperimentPrediction:
    qid: str
    system_name: str
    question: str
    answer: str
    gold_answer: str
    retrieved: list[RetrievalCandidate] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "system_name": self.system_name,
            "question": self.question,
            "answer": self.answer,
            "gold_answer": self.gold_answer,
            "retrieved": [asdict(candidate) for candidate in self.retrieved],
            "evidence": [asdict(evidence) for evidence in self.evidence],
            "trace": self.trace,
        }


@dataclass(slots=True)
class ExperimentPaths:
    benchmark_dataset: Path
    markdown_sections: Path
    knowledge_extraction: Path
    output_dir: Path
