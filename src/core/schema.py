from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.types import DocumentId, QuestionId, QuestionType, RetrievalMode, SectionId, SourceScope, SplitName


class EvidenceItem(BaseModel):
    """Normalized evidence unit aligned to source document and section."""

    model_config = ConfigDict(extra="ignore")

    source_id: DocumentId
    section_id: SectionId
    quote: str = ""

    @classmethod
    def from_raw(cls, payload: dict) -> "EvidenceItem":
        source_id = payload.get("source_id") or payload.get("doc_id") or payload.get("document_id")
        section_id = payload.get("section_id") or payload.get("chunk_id") or payload.get("block_id")
        return cls(source_id=str(source_id or ""), section_id=str(section_id or ""), quote=str(payload.get("quote", "")))

    @model_validator(mode="after")
    def validate_required_fields(self) -> "EvidenceItem":
        if not self.source_id:
            raise ValueError("EvidenceItem.source_id is required.")
        if not self.section_id:
            raise ValueError("EvidenceItem.section_id is required.")
        return self


class QuestionSkeletonLabel(BaseModel):
    """Question-side structural hints shared by rewrite/KBQA/Ours pipelines."""

    model_config = ConfigDict(extra="ignore")

    question_type: QuestionType | None = None
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    requires_text_compensation: bool = False


class BenchmarkSample(BaseModel):
    """Unified input protocol shared by all baselines and the evaluator."""

    model_config = ConfigDict(extra="ignore")

    question_id: QuestionId
    question: str
    answer_short: str = ""
    answer_long: str = ""
    question_type: QuestionType
    entities: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    source_scope: SourceScope
    requires_text_compensation: bool = False
    source_id: DocumentId = ""
    section_id: SectionId = ""
    support_score: float | None = None
    completeness_score: float | None = None
    raw_record: dict = Field(default_factory=dict)

    @property
    def skeleton_label(self) -> QuestionSkeletonLabel:
        return QuestionSkeletonLabel(
            question_type=self.question_type,
            entities=self.entities,
            relations=self.relations,
            constraints=self.constraints,
            requires_text_compensation=self.requires_text_compensation,
        )

    @model_validator(mode="after")
    def validate_answers_and_evidence(self) -> "BenchmarkSample":
        if not self.question_id:
            raise ValueError("BenchmarkSample.question_id is required.")
        if not self.question.strip():
            raise ValueError("BenchmarkSample.question must be non-empty.")
        if not (self.answer_short.strip() or self.answer_long.strip()):
            raise ValueError("At least one of answer_short or answer_long must be available.")
        if self.source_id and not self.section_id and self.source_scope == SourceScope.SINGLE_SECTION:
            raise ValueError("single_section sample with source_id must also include section_id.")
        return self


class DatasetSplit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    split_name: SplitName
    samples: list[BenchmarkSample] = Field(default_factory=list)


class DatasetBundle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dataset_name: str
    splits: dict[SplitName, DatasetSplit] = Field(default_factory=dict)

    @property
    def total_samples(self) -> int:
        return sum(len(split.samples) for split in self.splits.values())


class RetrievedDocument(BaseModel):
    """Unified retrieval output for text sections/chunks."""

    model_config = ConfigDict(extra="ignore")

    source_id: DocumentId
    section_id: SectionId
    content: str
    score: float = 0.0
    rank: int = 0
    metadata: dict = Field(default_factory=dict)


class RetrievedTriple(BaseModel):
    """Unified retrieval output for graph/KB structured items."""

    model_config = ConfigDict(extra="ignore")

    source_id: DocumentId
    section_id: SectionId
    subject: str
    predicate: str
    object: str
    score: float = 0.0
    rank: int = 0
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: QuestionId
    query: str
    mode: RetrievalMode = RetrievalMode.DOCUMENT
    retrieved_documents: list[RetrievedDocument] = Field(default_factory=list)
    retrieved_triples: list[RetrievedTriple] = Field(default_factory=list)
    debug_info: dict = Field(default_factory=dict)


class AnswerResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question_id: QuestionId
    answer_text: str
    answer_short: str = ""
    answer_long: str = ""
    supporting_evidence: list[EvidenceItem] = Field(default_factory=list)
    confidence: float | None = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_answer(self) -> "AnswerResult":
        if not self.answer_text.strip() and not self.answer_short.strip() and not self.answer_long.strip():
            raise ValueError("AnswerResult must contain at least one answer field.")
        return self


class PipelineRunRecord(BaseModel):
    """Single pipeline execution record; evaluator can consume this directly later."""

    model_config = ConfigDict(extra="ignore")

    question_id: QuestionId
    system_name: str
    sample: BenchmarkSample
    retrieval: RetrievalResult
    answer: AnswerResult
    rewritten_query: str | None = None
    trace: dict = Field(default_factory=dict)

    @property
    def qid(self) -> str:
        return self.question_id

    @property
    def method_name(self) -> str:
        return self.system_name

    def to_output_dict(self) -> dict:
        return {
            "qid": self.question_id,
            "method_name": self.system_name,
            "rewritten_query": self.rewritten_query,
            "retrieved_doc_ids": [item.source_id for item in self.retrieval.retrieved_documents],
            "retrieved_section_ids": [item.section_id for item in self.retrieval.retrieved_documents],
            "retrieved_triples": [item.model_dump() for item in self.retrieval.retrieved_triples],
            "supporting_evidence": [item.model_dump() for item in self.answer.supporting_evidence],
            "pred_answer": self.answer.answer_text or self.answer.answer_short or self.answer.answer_long,
            "trace": self.trace,
        }
