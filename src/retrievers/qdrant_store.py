from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from core.schema import RetrievedDocument
from qdrant_client import QdrantClient
from qdrant_client.http import models


LOGGER = logging.getLogger(__name__)


DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}


class QdrantStore:
    """Qdrant-backed vector store for section-level retrieval."""

    def __init__(
        self,
        *,
        collection_name: str,
        vector_size: int,
        distance: str = "cosine",
        url: str | None = None,
        path: str | Path | None = None,
        use_local: bool = False,
        content_key: str = "content",
        content_max_chars: int | None = None,
        client: QdrantClient | None = None,
        embedder: Any | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if not collection_name:
            raise ValueError("collection_name is required.")
        if vector_size <= 0:
            raise ValueError("vector_size must be positive.")
        if distance.lower() not in DISTANCE_MAP:
            raise ValueError(f"Unsupported distance: {distance}.")

        self.collection_name = collection_name
        self.vector_size = vector_size
        self.distance = distance.lower()
        self.url = url
        self.path = Path(path).expanduser() if path is not None else None
        self.use_local = use_local
        self.content_key = content_key
        self.content_max_chars = content_max_chars
        self.embedder = embedder
        self.logger = logger or LOGGER
        self.client = client or self._build_client()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def ensure_collection(self) -> None:
        """Create collection if missing; validate vector schema if it already exists."""

        try:
            if self.client.collection_exists(self.collection_name):
                collection = self.client.get_collection(self.collection_name)
                self._validate_collection_config(collection.config.params.vectors)
                self.logger.info(
                    "Qdrant collection %s already exists and matches vector_size=%s distance=%s.",
                    self.collection_name,
                    self.vector_size,
                    self.distance,
                )
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=DISTANCE_MAP[self.distance],
                ),
            )
            self.logger.info(
                "Created Qdrant collection %s with vector_size=%s distance=%s.",
                self.collection_name,
                self.vector_size,
                self.distance,
            )
        except Exception as exc:
            self.logger.exception("Failed to ensure Qdrant collection %s.", self.collection_name)
            raise RuntimeError(f"Failed to ensure collection {self.collection_name}.") from exc

    def recreate_collection(self) -> None:
        """Drop and recreate collection with the configured vector schema."""

        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=DISTANCE_MAP[self.distance],
                ),
            )
            self.logger.info(
                "Recreated Qdrant collection %s with vector_size=%s distance=%s.",
                self.collection_name,
                self.vector_size,
                self.distance,
            )
        except Exception as exc:
            self.logger.exception("Failed to recreate Qdrant collection %s.", self.collection_name)
            raise RuntimeError(f"Failed to recreate collection {self.collection_name}.") from exc

    def upsert_sections(self, sections: list[Any], vectors: list[list[float]]) -> None:
        """Upsert section documents and embeddings into Qdrant."""

        if len(sections) != len(vectors):
            raise ValueError(f"Section/vector count mismatch: {len(sections)} sections vs {len(vectors)} vectors.")
        if not sections:
            self.logger.info("Upsert skipped because section list is empty.")
            return

        points = []
        for section, vector in zip(sections, vectors, strict=True):
            self._validate_vector(vector)
            payload = self._build_payload(section)
            section_id = str(payload["section_id"]).strip()
            if not section_id:
                raise ValueError("Section payload must contain non-empty section_id.")
            points.append(
                models.PointStruct(
                    id=self._make_point_id(section_id),
                    vector=vector,
                    payload=payload,
                )
            )

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
            self.logger.info("Upserted %s section vectors into collection %s.", len(points), self.collection_name)
        except Exception as exc:
            self.logger.exception("Failed to upsert sections into collection %s.", self.collection_name)
            raise RuntimeError(f"Failed to upsert sections into collection {self.collection_name}.") from exc

    def search_by_vector(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> list[RetrievedDocument]:
        """Search the collection using a precomputed query vector."""

        self._validate_vector(query_vector)
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        try:
            hits = self._query_points(
                query_vector,
                top_k=top_k,
                score_threshold=score_threshold,
                with_payload=with_payload,
            )
            documents = [
                self._scored_point_to_document(point, rank=rank)
                for rank, point in enumerate(hits, start=1)
            ]
            self.logger.info(
                "Qdrant search returned %s document(s) from collection %s.",
                len(documents),
                self.collection_name,
            )
            return documents
        except Exception as exc:
            self.logger.exception("Failed to search collection %s by vector.", self.collection_name)
            raise RuntimeError(f"Failed to search collection {self.collection_name}.") from exc

    def search_by_query_vector(
        self,
        query: str | list[float],
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[RetrievedDocument]:
        """
        Search by raw query text or by query vector.

        If `query` is a string, an embedder with `embed_query()` must be configured.
        If `query` is a vector, it is passed through directly.
        """

        if isinstance(query, str):
            if self.embedder is None:
                raise ValueError("embedder is required for text query search.")
            query_vector = self.embedder.embed_query(query)
        elif isinstance(query, list):
            query_vector = [float(value) for value in query]
        else:
            raise TypeError(f"query must be str or list[float], got {type(query)!r}.")
        return self.search_by_vector(
            query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    def _build_client(self) -> QdrantClient:
        try:
            if self.use_local:
                if self.path is None:
                    raise ValueError("path is required when use_local=True.")
                self.path.mkdir(parents=True, exist_ok=True)
                self.logger.info("Initializing local Qdrant client at %s.", self.path)
                return QdrantClient(path=str(self.path))
            if not self.url:
                raise ValueError("url is required when use_local=False.")
            self.logger.info("Initializing remote Qdrant client at %s.", self.url)
            return QdrantClient(url=self.url)
        except Exception as exc:
            self.logger.exception("Failed to initialize Qdrant client.")
            raise RuntimeError("Failed to initialize Qdrant client.") from exc

    def _validate_collection_config(self, vectors_config: Any) -> None:
        if not isinstance(vectors_config, models.VectorParams):
            raise ValueError("Named vectors are not supported by this store implementation.")
        actual_size = int(vectors_config.size)
        actual_distance = str(vectors_config.distance).split(".")[-1].lower()
        if actual_size != self.vector_size:
            raise ValueError(
                f"Collection vector size mismatch: expected {self.vector_size}, got {actual_size}."
            )
        if actual_distance != self.distance:
            raise ValueError(
                f"Collection distance mismatch: expected {self.distance}, got {actual_distance}."
            )

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.vector_size:
            raise ValueError(f"Vector dimension mismatch: expected {self.vector_size}, got {len(vector)}.")

    def _build_payload(self, section: Any) -> dict[str, Any]:
        source_id = self._read_section_field(section, "source_id", fallback="doc_id")
        section_id = self._read_section_field(section, "section_id")
        doc_title = self._read_section_field(section, "doc_title")
        section_path = self._read_section_field(section, "section_path", default=[])
        block_type = self._read_section_field(section, "block_type", default="")
        char_len = self._read_section_field(section, "char_len", default=0)
        content = self._read_section_field(section, "content", default="")
        metadata = self._read_section_field(section, "metadata", default={})

        if not block_type and isinstance(metadata, dict):
            block_type = metadata.get("block_type", "")
        if not char_len and isinstance(metadata, dict):
            char_len = metadata.get("char_len", 0)

        payload = {
            "source_id": str(source_id or ""),
            "section_id": str(section_id or ""),
            "doc_title": str(doc_title or ""),
            "section_path": [str(item) for item in (section_path or [])],
            "block_type": str(block_type or ""),
            "char_len": int(char_len or 0),
        }
        if self.content_max_chars is None:
            payload[self.content_key] = str(content or "")
        else:
            payload[self.content_key] = str(content or "")[: self.content_max_chars]
        return payload

    def _read_section_field(self, section: Any, name: str, *, fallback: str | None = None, default: Any = "") -> Any:
        if isinstance(section, dict):
            if name in section:
                return section[name]
            if fallback is not None and fallback in section:
                return section[fallback]
            return default
        if hasattr(section, name):
            return getattr(section, name)
        if fallback is not None and hasattr(section, fallback):
            return getattr(section, fallback)
        return default

    def _scored_point_to_document(self, point: Any, *, rank: int) -> RetrievedDocument:
        payload = dict(getattr(point, "payload", None) or {})
        source_id = str(payload.get("source_id", ""))
        section_id = str(payload.get("section_id", getattr(point, "id", "")))
        content = str(payload.get(self.content_key, ""))
        return RetrievedDocument(
            source_id=source_id,
            section_id=section_id,
            content=content,
            score=float(getattr(point, "score", 0.0) or 0.0),
            rank=rank,
            metadata={
                "doc_title": payload.get("doc_title", ""),
                "section_path": payload.get("section_path", []),
                "block_type": payload.get("block_type", ""),
                "char_len": payload.get("char_len", 0),
                "retriever": "qdrant_dense",
                "collection_name": self.collection_name,
                "distance": self.distance,
            },
        )

    def _make_point_id(self, section_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection_name}:{section_id}"))

    def _query_points(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None,
        with_payload: bool,
    ) -> list[Any]:
        if hasattr(self.client, "search"):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=with_payload,
                with_vectors=False,
            )
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=False,
        )
        return list(getattr(result, "points", []))
