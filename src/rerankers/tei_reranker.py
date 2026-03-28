from __future__ import annotations

import logging
from typing import Any

import requests

from core.schema import RetrievedDocument


LOGGER = logging.getLogger(__name__)


class TEIReranker:
    """HTTP client for a local TEI rerank service."""

    backend_name = "tei"

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        api_key: str | None = None,
        max_retries: int = 0,
        top_n: int | None = None,
        *,
        session: requests.Session | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required.")
        if timeout <= 0:
            raise ValueError("timeout must be positive.")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative.")
        if top_n is not None and top_n <= 0:
            raise ValueError("top_n must be positive when provided.")

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.max_retries = max_retries
        self.top_n = top_n
        self.logger = logger or LOGGER
        self.session = session or requests.Session()

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDocument],
        top_n: int | None = None,
    ) -> list[RetrievedDocument]:
        if not isinstance(query, str):
            raise TypeError("query must be a string.")
        if not docs:
            self.logger.info("TEI rerank skipped because docs list is empty.")
            return []

        effective_top_n = top_n if top_n is not None else self.top_n
        if effective_top_n is not None and effective_top_n <= 0:
            raise ValueError("top_n must be positive when provided.")

        payload = {
            "query": query,
            "texts": [document.content or "" for document in docs],
        }
        if effective_top_n is not None:
            payload["top_n"] = effective_top_n

        self.logger.info(
            "Calling TEI rerank at %s/rerank with %s document(s), top_n=%s.",
            self.base_url,
            len(docs),
            effective_top_n,
        )
        data = self._post_rerank(payload)
        scored_results = self._extract_results(data, expected_count=len(docs))

        reranked = []
        for rank, result in enumerate(scored_results, start=1):
            source_document = docs[result["index"]]
            reranked.append(
                source_document.model_copy(
                    update={
                        "rank": rank,
                        "metadata": {
                            **source_document.metadata,
                            "rerank_score": result["score"],
                            "rank_after_rerank": rank,
                            "reranker": self.backend_name,
                        },
                    }
                )
            )
        if effective_top_n is not None:
            reranked = reranked[:effective_top_n]
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

    def _post_rerank(self, payload: dict[str, Any]) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error: Exception | None = None
        url = f"{self.base_url}/rerank"
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                last_error = exc
                self.logger.warning("TEI rerank request timed out on attempt %s/%s.", attempt, attempts)
                if attempt == attempts:
                    raise RuntimeError(
                        f"TEI rerank request timed out after {attempts} attempt(s): {exc}"
                    ) from exc
                continue
            except requests.RequestException as exc:
                last_error = exc
                self.logger.warning("TEI rerank request failed on attempt %s/%s: %s", attempt, attempts, exc)
                if attempt == attempts:
                    raise RuntimeError(
                        f"TEI rerank request failed after {attempts} attempt(s): {exc}"
                    ) from exc
                continue

            if response.ok:
                try:
                    return response.json()
                except ValueError as exc:
                    self.logger.exception("TEI rerank response is not valid JSON.")
                    raise RuntimeError(f"TEI rerank returned invalid JSON: {exc}") from exc

            body_preview = response.text[:400]
            message = (
                f"TEI rerank request failed with status {response.status_code}: {body_preview or '<empty body>'}"
            )
            self.logger.warning("%s", message)
            if response.status_code < 500 or attempt == attempts:
                raise RuntimeError(message)
        raise RuntimeError(f"TEI rerank request failed: {last_error}")

    def _extract_results(self, payload: Any, *, expected_count: int) -> list[dict[str, float | int]]:
        if isinstance(payload, list):
            raw_results = payload
        elif isinstance(payload, dict):
            raw_results = payload.get("results")
            if raw_results is None:
                raw_results = payload.get("data")
        else:
            raw_results = None

        if not isinstance(raw_results, list):
            raise RuntimeError(f"Unexpected TEI rerank response format: {payload!r}")

        parsed_results: list[dict[str, float | int]] = []
        for fallback_index, item in enumerate(raw_results):
            if not isinstance(item, dict):
                raise RuntimeError(f"Unexpected TEI rerank item format: {item!r}")
            index = item.get("index", item.get("document_index", fallback_index))
            score = item.get("score", item.get("relevance_score"))
            if index is None or score is None:
                raise RuntimeError(f"TEI rerank response item missing index/score: {item!r}")
            parsed_results.append({"index": int(index), "score": float(score)})

        if any(result["index"] < 0 or result["index"] >= expected_count for result in parsed_results):
            raise RuntimeError(f"TEI rerank response contains invalid document index: {parsed_results!r}")

        parsed_results.sort(key=lambda item: float(item["score"]), reverse=True)
        return parsed_results
