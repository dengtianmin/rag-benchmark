from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from tenacity import Retrying, before_sleep_log, retry_if_exception_type, stop_after_attempt, wait_exponential


LOGGER = logging.getLogger(__name__)
MAX_API_BATCH_SIZE = 64


class ZhipuEmbedder:
    """Reusable Embedding-3 wrapper for document indexing and online query embedding."""

    def __init__(
        self,
        api_key: str | None,
        model: str = "embedding-3",
        dimensions: int = 1024,
        batch_size: int = 32,
        *,
        client: Any | None = None,
        max_retries: int = 3,
        logger: logging.Logger | None = None,
    ) -> None:
        if not model:
            raise ValueError("model is required.")
        if dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive.")
        if client is None and not api_key:
            raise ValueError("api_key is required when client is not provided.")

        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.batch_size = min(batch_size, MAX_API_BATCH_SIZE)
        self.max_retries = max_retries
        self.logger = logger or LOGGER
        self.client = client or self._build_client(api_key)

        if batch_size > MAX_API_BATCH_SIZE:
            self.logger.warning(
                "Configured batch_size=%s exceeds API max batch size %s; clamped to %s.",
                batch_size,
                MAX_API_BATCH_SIZE,
                self.batch_size,
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list[str].")
        if not texts:
            self.logger.info("Embedding documents skipped because input list is empty.")
            return []

        normalized_texts = [self._normalize_text(text) for text in texts]
        results: list[list[float] | None] = [None] * len(normalized_texts)

        non_empty_pairs = [(index, text) for index, text in enumerate(normalized_texts) if text]
        empty_count = len(normalized_texts) - len(non_empty_pairs)
        if empty_count:
            self.logger.info("Filtered %s empty texts before embedding request.", empty_count)
            zero_vector = self._zero_vector()
            for index, text in enumerate(normalized_texts):
                if not text:
                    results[index] = zero_vector.copy()

        if not non_empty_pairs:
            return [vector if vector is not None else self._zero_vector() for vector in results]

        self.logger.info(
            "Embedding %s documents in %s batch(es) with model=%s dimensions=%s batch_size=%s.",
            len(non_empty_pairs),
            self._num_batches(len(non_empty_pairs)),
            self.model,
            self.dimensions,
            self.batch_size,
        )

        for batch_number, batch_pairs in enumerate(self._chunked(non_empty_pairs, self.batch_size), start=1):
            batch_indices = [index for index, _ in batch_pairs]
            batch_texts = [text for _, text in batch_pairs]
            self.logger.debug(
                "Submitting embedding batch %s with %s text(s).",
                batch_number,
                len(batch_texts),
            )
            batch_embeddings = self._embed_batch(batch_texts)
            if len(batch_embeddings) != len(batch_texts):
                raise ValueError(
                    f"Embedding batch result count mismatch: expected {len(batch_texts)}, got {len(batch_embeddings)}."
                )
            for index, embedding in zip(batch_indices, batch_embeddings, strict=True):
                results[index] = embedding

        final_embeddings = [vector if vector is not None else self._zero_vector() for vector in results]
        if len(final_embeddings) != len(texts):
            raise ValueError(
                f"Embedding result count mismatch: expected {len(texts)}, got {len(final_embeddings)}."
            )
        return final_embeddings

    def embed_query(self, text: str) -> list[float]:
        normalized = self._normalize_text(text)
        if not normalized:
            self.logger.info("Embedding query received empty text; returning zero vector.")
            return self._zero_vector()
        embeddings = self.embed_documents([normalized])
        if len(embeddings) != 1:
            raise ValueError(f"Expected a single query embedding, got {len(embeddings)}.")
        return embeddings[0]

    def _build_client(self, api_key: str | None) -> Any:
        try:
            from zhipuai import ZhipuAI
        except ImportError as exc:
            raise ImportError(
                "zhipuai package is required to build ZhipuEmbedder client. Install dependencies first."
            ) from exc
        return ZhipuAI(api_key=api_key)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        last_exception: Exception | None = None
        for attempt in Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(Exception),
            before_sleep=before_sleep_log(self.logger, logging.WARNING),
            reraise=True,
        ):
            with attempt:
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=texts,
                        dimensions=self.dimensions,
                    )
                    embeddings = self._parse_embeddings_response(response)
                    self._validate_embedding_batch(texts, embeddings)
                    return embeddings
                except Exception as exc:
                    last_exception = exc
                    self.logger.warning(
                        "Embedding batch request failed on attempt %s/%s: %s",
                        attempt.retry_state.attempt_number,
                        self.max_retries,
                        exc,
                    )
                    raise
        if last_exception is not None:
            raise last_exception
        raise RuntimeError("Embedding request failed without a captured exception.")

    def _parse_embeddings_response(self, response: Any) -> list[list[float]]:
        data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
        if data is None:
            raise ValueError("Embedding response missing data field.")
        if not isinstance(data, Iterable):
            raise TypeError("Embedding response data field must be iterable.")

        parsed: list[tuple[int | None, list[float]]] = []
        for item in data:
            if isinstance(item, dict):
                index = item.get("index")
                embedding = item.get("embedding")
            else:
                index = getattr(item, "index", None)
                embedding = getattr(item, "embedding", None)
            if embedding is None:
                raise ValueError("Embedding item missing embedding field.")
            vector = [float(value) for value in embedding]
            parsed.append((int(index) if index is not None else None, vector))

        if not parsed:
            return []

        if all(index is not None for index, _ in parsed):
            parsed.sort(key=lambda item: item[0])
        return [vector for _, vector in parsed]

    def _validate_embedding_batch(self, texts: list[str], embeddings: list[list[float]]) -> None:
        if len(embeddings) != len(texts):
            raise ValueError(f"Embedding response count mismatch: expected {len(texts)}, got {len(embeddings)}.")
        for position, embedding in enumerate(embeddings):
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Embedding dimension mismatch at position {position}: "
                    f"expected {self.dimensions}, got {len(embedding)}."
                )

    def _normalize_text(self, text: str | None) -> str:
        if text is None:
            return ""
        if not isinstance(text, str):
            raise TypeError(f"Embedding input must be str or None, got {type(text)!r}.")
        return text.strip()

    def _num_batches(self, item_count: int) -> int:
        if item_count <= 0:
            return 0
        return (item_count + self.batch_size - 1) // self.batch_size

    def _zero_vector(self) -> list[float]:
        return [0.0] * self.dimensions

    @staticmethod
    def _chunked(items: list[tuple[int, str]], chunk_size: int) -> Iterable[list[tuple[int, str]]]:
        for offset in range(0, len(items), chunk_size):
            yield items[offset : offset + chunk_size]
