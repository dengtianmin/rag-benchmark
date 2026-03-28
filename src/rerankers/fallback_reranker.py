from __future__ import annotations

import logging
from typing import Any

from core.schema import RetrievedDocument


LOGGER = logging.getLogger(__name__)


class FallbackReranker:
    """Use a primary reranker and fall back to a secondary reranker on failure."""

    def __init__(
        self,
        primary: Any,
        fallback: Any,
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.logger = logger or LOGGER
        self.backend_name = getattr(primary, "backend_name", primary.__class__.__name__.lower())
        self.base_url = getattr(primary, "base_url", None)
        self.last_backend_used = self.backend_name
        self.last_error: str | None = None

    def rerank(
        self,
        query: str,
        docs: list[RetrievedDocument],
        top_n: int | None = None,
    ) -> list[RetrievedDocument]:
        try:
            result = self.primary.rerank(query, docs, top_n=top_n)
        except Exception as exc:
            self.last_error = str(exc)
            self.last_backend_used = getattr(self.fallback, "backend_name", self.fallback.__class__.__name__.lower())
            self.logger.warning(
                "Primary reranker backend=%s failed, falling back to %s: %s",
                self.backend_name,
                self.last_backend_used,
                exc,
            )
            return self.fallback.rerank(query, docs, top_n=top_n)

        self.last_error = None
        self.last_backend_used = getattr(self.primary, "backend_name", self.backend_name)
        return result
