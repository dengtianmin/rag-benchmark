from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.schema import AnswerResult, BenchmarkSample, PipelineRunRecord, RetrievalResult, RetrievedDocument
from core.types import RetrievalMode
from dataio.loaders import iter_jsonl
from prompts.rag_prompt_builder import RAGPromptBuilder


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9_./:-]*")
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?;\n]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


def normalized_text(text: str) -> str:
    return " ".join(tokenize(text))


def overlap_score(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)


@dataclass(slots=True)
class SectionDocument:
    source_id: str
    section_id: str
    content: str
    doc_title: str = ""
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        title = " / ".join(self.section_path)
        return "\n".join(part for part in [self.doc_title, title, self.content] if part)


class PublicIndex:
    """Minimal shared lexical index over markdown sections for baseline pipelines."""

    def __init__(self, documents: list[SectionDocument]) -> None:
        self.documents = documents
        self.by_section_id = {doc.section_id: doc for doc in documents}

    @classmethod
    def from_markdown_sections(cls, path: str | Path) -> "PublicIndex":
        documents = []
        for row in iter_jsonl(path):
            documents.append(
                SectionDocument(
                    source_id=str(row.get("doc_id", "")),
                    section_id=str(row.get("section_id", "")),
                    content=str(row.get("content", "")),
                    doc_title=str(row.get("doc_title", "")),
                    section_path=[str(item) for item in row.get("section_path", [])],
                    metadata={
                        "file_path": row.get("file_path", ""),
                        "block_type": row.get("block_type", ""),
                        "char_len": row.get("char_len", 0),
                    },
                )
            )
        return cls(documents)

    def search(self, query: str, top_k: int) -> list[RetrievedDocument]:
        scored: list[tuple[SectionDocument, float]] = []
        for rank, document in enumerate(self.documents, start=1):
            score = overlap_score(query, document.full_text)
            if score <= 0:
                continue
            scored.append((document, score))
        ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RetrievedDocument(
                source_id=document.source_id,
                section_id=document.section_id,
                content=document.content,
                score=score,
                rank=rank,
                metadata={
                    **document.metadata,
                    "doc_title": document.doc_title,
                    "section_path": document.section_path,
                    "retriever": "mock_lexical",
                },
            )
            for rank, (document, score) in enumerate(ranked, start=1)
        ]


class MockEmbedder:
    """Placeholder embedder; currently maps query-document relevance to lexical overlap."""

    def score(self, query: str, document: SectionDocument) -> float:
        return overlap_score(query, document.full_text)


class MockReranker:
    """Optional reranker that re-scores retrieved documents with the same lightweight heuristic."""

    def rerank(self, query: str, documents: list[RetrievedDocument]) -> list[RetrievedDocument]:
        reranked = sorted(documents, key=lambda item: overlap_score(query, item.content), reverse=True)
        return [
            document.model_copy(update={"rank": rank, "metadata": {**document.metadata, "reranked": True}})
            for rank, document in enumerate(reranked, start=1)
        ]


class MockGenerator:
    """
    Minimal local generator.
    TODO: replace with real LLM while keeping input/output contract unchanged.
    """

    def __init__(self, prompt_builder: RAGPromptBuilder | None = None, max_sentences: int = 3) -> None:
        self.prompt_builder = prompt_builder or RAGPromptBuilder()
        self.max_sentences = max_sentences

    def generate(self, sample: BenchmarkSample, documents: list[RetrievedDocument]) -> AnswerResult:
        prompt = self.prompt_builder.build_prompt(question=sample.question, retrieved_documents=documents)
        sentences: list[tuple[float, str, RetrievedDocument]] = []
        for document in documents:
            for sentence in SENTENCE_SPLIT_PATTERN.split(document.content):
                sentence = sentence.strip()
                if not sentence:
                    continue
                score = overlap_score(sample.question, sentence)
                if score > 0:
                    sentences.append((score, sentence, document))
        top_sentences = sorted(sentences, key=lambda item: item[0], reverse=True)[: self.max_sentences]
        if top_sentences:
            answer_text = "。".join(sentence for _, sentence, _ in top_sentences)
            supporting_evidence = [
                {
                    "source_id": document.source_id,
                    "section_id": document.section_id,
                    "quote": sentence[:240],
                }
                for _, sentence, document in top_sentences
            ]
        elif documents:
            answer_text = documents[0].content[:240]
            supporting_evidence = [
                {
                    "source_id": documents[0].source_id,
                    "section_id": documents[0].section_id,
                    "quote": documents[0].content[:240],
                }
            ]
        else:
            answer_text = ""
            supporting_evidence = []
        return AnswerResult(
            question_id=sample.question_id,
            answer_text=answer_text,
            answer_short=answer_text,
            supporting_evidence=supporting_evidence,
            metadata={
                "generator": "mock_generator",
                "prompt_preview": prompt[:400],
            },
        )


class BasePipeline:
    method_name: str = "base"

    def run(self, sample: BenchmarkSample) -> PipelineRunRecord:
        raise NotImplementedError

    @staticmethod
    def build_retrieval_result(question_id: str, query: str, documents: list[RetrievedDocument]) -> RetrievalResult:
        return RetrievalResult(
            question_id=question_id,
            query=query,
            mode=RetrievalMode.DOCUMENT,
            retrieved_documents=documents,
            retrieved_triples=[],
            debug_info={"retrieved_count": len(documents)},
        )
