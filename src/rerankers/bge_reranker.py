from __future__ import annotations

import logging
from typing import Any

from core.schema import RetrievedDocument


LOGGER = logging.getLogger(__name__)


class BGEReranker:
    """Local BGE reranker wrapper based on FlagEmbedding."""

    backend_name = "bge_local"

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_fp16: bool = False,
        query_max_length: int = 256,
        passage_max_length: int = 512,
        *,
        model: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not model_path:
            raise ValueError("model_path is required.")
        if query_max_length <= 0:
            raise ValueError("query_max_length must be positive.")
        if passage_max_length <= 0:
            raise ValueError("passage_max_length must be positive.")

        self.model_path = model_path
        self.device = device
        self.use_fp16 = use_fp16
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length
        self.logger = logger or LOGGER
        self.model = model or self._load_model()

    def score_pairs(self, query: str, docs: list[RetrievedDocument]) -> list[float]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not docs:
            self.logger.info("Rerank scoring skipped because docs list is empty.")
            return []

        pairs = [(query, self._truncate_passage(document.content)) for document in docs]
        self.logger.info(
            "Scoring %s query-document pair(s) with local BGE reranker on device=%s.",
            len(pairs),
            self.device,
        )
        try:
            scores = self.model.compute_score(
                pairs,
                max_query_length=self.query_max_length,
                max_passage_length=self.passage_max_length,
            )
        except Exception as exc:
            self.logger.exception("BGE reranker scoring failed.")
            raise RuntimeError(f"BGE reranker scoring failed: {exc}") from exc

        if isinstance(scores, (int, float)):
            normalized_scores = [float(scores)]
        else:
            normalized_scores = [float(score) for score in scores]

        if len(normalized_scores) != len(docs):
            raise ValueError(
                f"Reranker score count mismatch: expected {len(docs)}, got {len(normalized_scores)}."
            )
        return normalized_scores

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDocument],
        top_n: int | None = None,
    ) -> list[RetrievedDocument]:
        if not docs:
            self.logger.info("Rerank skipped because docs list is empty.")
            return []
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be positive when provided.")

        scores = self.score_pairs(query, docs)
        reranked = [
            document.model_copy(
                update={
                    "rank": rank,
                    "metadata": {
                        **document.metadata,
                        "rerank_score": score,
                        "rank_after_rerank": rank,
                        "reranker": self.backend_name,
                    },
                }
            )
            for rank, (document, score) in enumerate(
                sorted(
                    zip(docs, scores, strict=True),
                    key=lambda item: item[1],
                    reverse=True,
                ),
                start=1,
            )
        ]
        if top_n is not None:
            reranked = reranked[:top_n]
            reranked = [
                document.model_copy(
                    update={
                        "rank": rank,
                        "metadata": {
                            **document.metadata,
                            "rank_after_rerank": rank,
                        },
                    }
                )
                for rank, document in enumerate(reranked, start=1)
            ]
        return reranked

    def _load_model(self) -> Any:
        try:
            from FlagEmbedding import FlagReranker
        except ImportError as exc:
            raise ImportError(
                "FlagEmbedding is required to load the local BGE reranker. Install dependencies first."
            ) from exc

        try:
            self.logger.info(
                "Loading local BGE reranker from %s on device=%s use_fp16=%s.",
                self.model_path,
                self.device,
                self.use_fp16,
            )
            return FlagReranker(
                self.model_path,
                use_fp16=self.use_fp16,
                devices=self.device,
            )
        except Exception as exc:
            self.logger.exception("Failed to load local BGE reranker from %s.", self.model_path)
            raise RuntimeError(
                f"Failed to load local BGE reranker from {self.model_path}: {exc}"
            ) from exc

    def _truncate_passage(self, content: str) -> str:
        return content or ""
