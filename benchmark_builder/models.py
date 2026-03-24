from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


QuestionType = Literal["fact", "relation", "multi_evidence", "explanation"]
SourceScope = Literal["single_section", "single_doc", "cross_doc"]
BlockType = Literal["paragraph", "list", "table", "mixed"]


class MarkdownSection(BaseModel):
    doc_id: str
    file_path: str
    doc_title: str
    section_id: str
    block_id: str
    section_path: list[str] = Field(default_factory=list)
    content: str
    block_type: BlockType
    char_len: int


class Entity(BaseModel):
    name: str
    type: str
    normalized_name: str


class Relation(BaseModel):
    head: str
    relation: str
    tail: str
    description: str = ""


class Constraint(BaseModel):
    target: str
    type: str
    value: str


class Attribute(BaseModel):
    entity: str
    attribute: str
    value: str


class Procedure(BaseModel):
    name: str
    steps: list[str] = Field(default_factory=list)


class EvidenceSpan(BaseModel):
    quote: str
    reason: str


class KnowledgeExtractionPayload(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    attributes: list[Attribute] = Field(default_factory=list)
    procedures: list[Procedure] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)


class KnowledgeExtractionResult(BaseModel):
    doc_id: str
    section_id: str
    section_path: list[str] = Field(default_factory=list)
    content: str
    extraction: KnowledgeExtractionPayload = Field(default_factory=KnowledgeExtractionPayload)
    raw_response_text: str = ""
    parse_error: str | None = None
    success: bool = True


class QAEvidence(BaseModel):
    doc_id: str
    section_id: str
    quote: str


class QACandidate(BaseModel):
    qid: str
    question: str
    answer_short: str
    answer_long: str
    question_type: QuestionType
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    requires_text_compensation: bool
    evidence: list[QAEvidence] = Field(default_factory=list)
    source_scope: SourceScope
    doc_id: str
    section_id: str

    @field_validator("question", "answer_short", "answer_long")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class QAValidationResult(BaseModel):
    qid: str
    is_valid: bool
    support_score: float = 0.0
    completeness_score: float = 0.0
    type_consistent: bool = False
    is_nontrivial: bool = False
    has_hallucination: bool = False
    reject_reasons: list[str] = Field(default_factory=list)
    raw_response_text: str = ""
    parse_error: str | None = None


class FinalBenchmarkRecord(BaseModel):
    qid: str
    question: str
    answer_short: str
    answer_long: str
    question_type: QuestionType
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    requires_text_compensation: bool
    evidence: list[QAEvidence] = Field(default_factory=list)
    source_scope: SourceScope
    doc_id: str
    section_id: str
    support_score: float
    completeness_score: float
