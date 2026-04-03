from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from core.schema import AnswerResult, BenchmarkSample, PipelineRunRecord, RetrievalResult, RetrievedDocument
from core.types import RetrievalMode
from dataio.loaders import iter_jsonl
from prompts.rag_prompt_builder import RAGPromptBuilder


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9_./:-]*")
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？!?;\n]+")
NO_RETRIEVAL_ANSWER = "未检索到相关证据。"


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
    """Shared BM25 lexical index over markdown sections for baseline pipelines."""

    def __init__(self, documents: list[SectionDocument]) -> None:
        self.documents = documents
        self.by_section_id = {doc.section_id: doc for doc in documents}
        self._doc_tokens: list[list[str]] = [tokenize(doc.full_text) for doc in documents]
        self._doc_term_freqs: list[dict[str, int]] = []
        self._doc_freqs: dict[str, int] = {}
        self._doc_lengths: list[int] = []
        self._avg_doc_length = 0.0
        self._bm25_k1 = 1.5
        self._bm25_b = 0.75
        self._build_bm25_index()

    def _build_bm25_index(self) -> None:
        total_length = 0
        for tokens in self._doc_tokens:
            term_freqs: dict[str, int] = {}
            for token in tokens:
                term_freqs[token] = term_freqs.get(token, 0) + 1
            self._doc_term_freqs.append(term_freqs)
            self._doc_lengths.append(len(tokens))
            total_length += len(tokens)
            for token in term_freqs:
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1
        if self.documents:
            self._avg_doc_length = total_length / len(self.documents)

    def _bm25_idf(self, token: str) -> float:
        doc_freq = self._doc_freqs.get(token, 0)
        if doc_freq == 0 or not self.documents:
            return 0.0
        doc_count = len(self.documents)
        # Standard BM25 idf with +1 smoothing to keep scores non-negative on common tokens.
        return math.log(1.0 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))

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
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        unique_query_tokens = set(query_tokens)
        avg_doc_length = self._avg_doc_length or 1.0
        for document, term_freqs, doc_length in zip(self.documents, self._doc_term_freqs, self._doc_lengths):
            score = 0.0
            norm = self._bm25_k1 * (1.0 - self._bm25_b + self._bm25_b * (doc_length / avg_doc_length))
            for token in unique_query_tokens:
                term_freq = term_freqs.get(token, 0)
                if term_freq <= 0:
                    continue
                idf = self._bm25_idf(token)
                score += idf * (term_freq * (self._bm25_k1 + 1.0)) / (term_freq + norm)
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

    backend_name = "mock"
    base_url = None

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_n: int | None = None,
    ) -> list[RetrievedDocument]:
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be positive when provided.")
        scored_documents = [
            (document, overlap_score(query, document.content))
            for document in documents
        ]
        reranked = sorted(scored_documents, key=lambda item: item[1], reverse=True)
        if top_n is not None:
            reranked = reranked[:top_n]
        return [
            document.model_copy(
                update={
                    "rank": rank,
                    "metadata": {
                        **document.metadata,
                        "reranked": True,
                        "rerank_score": score,
                        "rank_after_rerank": rank,
                        "reranker": self.backend_name,
                    },
                }
            )
            for rank, (document, score) in enumerate(reranked, start=1)
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
        numbered_evidence = self.prompt_builder.build_numbered_evidence(documents)
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
            selected_indices: list[int] = []
            for _, _, document in top_sentences:
                for item in numbered_evidence:
                    if item.document.section_id == document.section_id and item.index not in selected_indices:
                        selected_indices.append(item.index)
                        break
            supporting_evidence = [
                {
                    "source_id": item.document.source_id,
                    "section_id": item.document.section_id,
                    "quote": item.text[:240],
                }
                for item in numbered_evidence
                if item.index in selected_indices
            ]
        elif documents:
            answer_text = documents[0].content[:240]
            selected_indices = [1]
            supporting_evidence = [
                {
                    "source_id": documents[0].source_id,
                    "section_id": documents[0].section_id,
                    "quote": numbered_evidence[0].text[:240],
                }
            ]
        else:
            answer_text = NO_RETRIEVAL_ANSWER
            selected_indices = []
            supporting_evidence = []
        return AnswerResult(
            question_id=sample.question_id,
            answer_text=answer_text,
            answer_short=answer_text,
            supporting_evidence=supporting_evidence,
            metadata={
                "generator": "mock_generator",
                "model_name": None,
                "base_url": None,
                "latency_ms": 0,
                "prompt_chars": len(prompt),
                "completion_chars": len(answer_text),
                "finish_reason": "mock",
                "fallback_used": False,
                "fallback_reason": None,
                "no_retrieval": not documents,
                "prompt_preview": prompt[:400],
                "selected_evidence_indices": selected_indices,
            },
        )


class SupportsGenerate(Protocol):
    def generate(self, sample: BenchmarkSample, documents: list[RetrievedDocument]) -> AnswerResult:
        ...


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

    @staticmethod
    def resolve_rerank_backend(reranker: Any | None) -> str:
        if reranker is None:
            return "disabled"
        return str(
            getattr(
                reranker,
                "last_backend_used",
                getattr(reranker, "backend_name", reranker.__class__.__name__.lower()),
            )
        )

    @staticmethod
    def resolve_rerank_url(reranker: Any | None) -> str | None:
        if reranker is None:
            return None
        base_url = getattr(reranker, "base_url", None)
        if not base_url and hasattr(reranker, "primary"):
            base_url = getattr(getattr(reranker, "primary"), "base_url", None)
        return str(base_url) if base_url else None

    @classmethod
    def build_rerank_trace(
        cls,
        *,
        reranker: Any | None,
        before_documents: list[RetrievedDocument],
        after_documents: list[RetrievedDocument],
        top_n: int | None,
    ) -> dict[str, Any]:
        before_sections = [document.section_id for document in before_documents]
        after_sections = [document.section_id for document in after_documents]
        return {
            "rerank_backend": cls.resolve_rerank_backend(reranker),
            "rerank_url": cls.resolve_rerank_url(reranker),
            "rerank_input_count": len(before_documents),
            "rerank_top_n": top_n,
            "rerank_fallback_used": bool(
                reranker is not None and getattr(reranker, "last_error", None) is not None
            ),
            "rerank_error": getattr(reranker, "last_error", None) if reranker is not None else None,
            "rerank_order_changed": before_sections != after_sections,
            "pre_rerank_sections": before_sections,
            "post_rerank_sections": after_sections,
            "rerank_score_list": [
                {
                    "section_id": document.section_id,
                    "score": float(document.metadata.get("rerank_score")),
                }
                for document in after_documents
                if "rerank_score" in document.metadata
            ],
        }
