from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol

from core.schema import RetrievedDocument
from embedders.zhipu_embedder import ZhipuEmbedder
from pipelines.base import MockReranker, PublicIndex
from retrievers.qdrant_store import QdrantStore
from rerankers.bge_reranker import BGEReranker
from rerankers.fallback_reranker import FallbackReranker
from rerankers.tei_reranker import TEIReranker
from runtime_config import RuntimeSettings


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TextRetrieverQuery:
    lexical_query: str
    dense_query: str

    def for_mode(self, retrieval_mode: str) -> str:
        if retrieval_mode == "lexical":
            return self.lexical_query or self.dense_query
        if retrieval_mode == "dense":
            return self.dense_query or self.lexical_query
        raise ValueError(f"Unsupported retrieval mode for single-query fallback: {retrieval_mode}")


class SupportsRetrieve(Protocol):
    def retrieve(self, query: str | TextRetrieverQuery, top_k: int) -> list[RetrievedDocument]:
        ...


@dataclass(slots=True)
class LexicalTextRetriever:
    index: PublicIndex

    def retrieve(self, query: str | TextRetrieverQuery, top_k: int) -> list[RetrievedDocument]:
        resolved_query = query.lexical_query if isinstance(query, TextRetrieverQuery) else query
        return self.index.search(resolved_query, top_k=top_k)


@dataclass(slots=True)
class DenseTextRetriever:
    store: QdrantStore

    def retrieve(self, query: str | TextRetrieverQuery, top_k: int) -> list[RetrievedDocument]:
        resolved_query = query.dense_query if isinstance(query, TextRetrieverQuery) else query
        return self.store.search_by_query_vector(resolved_query, top_k=top_k)


@dataclass(slots=True)
class HybridTextRetriever:
    lexical: LexicalTextRetriever
    dense: DenseTextRetriever

    def retrieve(self, query: str | TextRetrieverQuery, top_k: int) -> list[RetrievedDocument]:
        lexical_docs = self.lexical.retrieve(query, top_k=top_k)
        dense_docs = self.dense.retrieve(query, top_k=top_k)
        lexical_scores = {document.section_id: float(document.score) for document in lexical_docs}
        dense_scores = {document.section_id: float(document.score) for document in dense_docs}

        merged: dict[str, RetrievedDocument] = {}
        combined_scores: dict[str, float] = {}

        for document in lexical_docs:
            merged[document.section_id] = document.model_copy(
                update={
                    "metadata": {
                        **document.metadata,
                        "lexical_score": float(document.score),
                        "hybrid_sources": ["lexical"],
                    }
                }
            )
            combined_scores[document.section_id] = combined_scores.get(document.section_id, 0.0) + float(document.score)

        for document in dense_docs:
            existing = merged.get(document.section_id)
            if existing is None:
                merged[document.section_id] = document.model_copy(
                    update={
                        "metadata": {
                            **document.metadata,
                            "dense_score": float(document.score),
                            "hybrid_sources": ["dense"],
                        }
                    }
                )
            else:
                merged[document.section_id] = existing.model_copy(
                    update={
                        "score": max(float(existing.score), float(document.score)),
                        "metadata": {
                            **existing.metadata,
                            "lexical_score": float(existing.metadata.get("lexical_score", existing.score)),
                            "dense_score": float(document.score),
                            "hybrid_sources": ["lexical", "dense"],
                        },
                    }
                )
            combined_scores[document.section_id] = combined_scores.get(document.section_id, 0.0) + float(document.score)

        ranked_ids = sorted(combined_scores, key=lambda section_id: combined_scores[section_id], reverse=True)[:top_k]
        results = []
        for rank, section_id in enumerate(ranked_ids, start=1):
            document = merged[section_id]
            metadata = {
                **document.metadata,
                "hybrid_score": combined_scores[section_id],
                "retriever": "hybrid_retriever",
                "hybrid_branch_trace": {
                    "lexical_query": query.lexical_query if isinstance(query, TextRetrieverQuery) else query,
                    "dense_query": query.dense_query if isinstance(query, TextRetrieverQuery) else query,
                    "lexical_candidate_ids": [item.section_id for item in lexical_docs],
                    "dense_candidate_ids": [item.section_id for item in dense_docs],
                    "lexical_candidate_scores": lexical_scores,
                    "dense_candidate_scores": dense_scores,
                },
            }
            if "hybrid_sources" not in metadata:
                metadata["hybrid_sources"] = ["lexical"]
            results.append(
                document.model_copy(
                    update={
                        "rank": rank,
                        "score": combined_scores[section_id],
                        "metadata": metadata,
                    }
                )
            )
        return results


def build_text_retriever(
    *,
    index: PublicIndex,
    settings: RuntimeSettings,
    logger: logging.Logger | None = None,
) -> SupportsRetrieve:
    logger = logger or LOGGER
    retrieval_mode = settings.retrieval.mode
    lexical = LexicalTextRetriever(index)

    if retrieval_mode == "lexical":
        logger.info("Using lexical retrieval mode.")
        return lexical

    embedder = ZhipuEmbedder(
        api_key=settings.embedding.api_key,
        model=settings.embedding.model,
        dimensions=settings.embedding.dimension,
        batch_size=settings.embedding.batch_size,
    )
    store = QdrantStore(
        collection_name=settings.retrieval.qdrant_collection,
        vector_size=settings.embedding.dimension,
        url=settings.retrieval.qdrant_url,
        path=settings.retrieval.qdrant_path,
        use_local=settings.retrieval.qdrant_use_local,
        embedder=embedder,
    )
    dense = DenseTextRetriever(store)

    if retrieval_mode == "dense":
        logger.info("Using dense retrieval mode.")
        return dense
    if retrieval_mode == "hybrid":
        logger.info("Using hybrid retrieval mode.")
        return HybridTextRetriever(lexical=lexical, dense=dense)
    raise ValueError(f"Unsupported retrieval mode: {retrieval_mode}")


def build_reranker(settings: RuntimeSettings, logger: logging.Logger | None = None) -> object | None:
    if not settings.rerank.enabled:
        return None
    logger = logger or LOGGER
    backend = settings.rerank.backend.lower()

    if backend == "mock":
        logger.info("Using mock reranker backend.")
        return MockReranker()

    if backend == "tei":
        logger.info("Using TEI reranker backend at %s.", settings.rerank.tei_url)
        primary: object = TEIReranker(
            base_url=settings.rerank.tei_url,
            timeout=settings.rerank.timeout,
            api_key=settings.rerank.api_key,
            max_retries=settings.rerank.max_retries,
            top_n=settings.rerank.top_n,
            logger=logger,
        )
    elif backend in {"bge", "bge_local"}:
        logger.info("Using local BGE reranker backend from %s.", settings.rerank.model_path)
        primary = BGEReranker(
            model_path=settings.rerank.model_path,
            device=settings.rerank.device,
            use_fp16=settings.rerank.use_fp16,
            query_max_length=settings.rerank.query_max_length,
            passage_max_length=settings.rerank.passage_max_length,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported reranker backend: {settings.rerank.backend}")

    if settings.rerank.allow_mock_fallback:
        logger.info("Reranker mock fallback is enabled for backend=%s.", backend)
        return FallbackReranker(primary=primary, fallback=MockReranker(), logger=logger)
    return primary
