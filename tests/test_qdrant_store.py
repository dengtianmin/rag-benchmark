from __future__ import annotations

from pathlib import Path

import pytest

from retrievers.qdrant_store import QdrantStore


class _FakeEmbedder:
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed_query(self, text: str) -> list[float]:
        if not text:
            raise ValueError("empty query")
        return list(self.vector)


def _build_section(
    *,
    source_id: str,
    section_id: str,
    content: str,
    doc_title: str = "Demo Doc",
    section_path: list[str] | None = None,
    block_type: str = "paragraph",
    char_len: int | None = None,
) -> dict:
    return {
        "source_id": source_id,
        "section_id": section_id,
        "content": content,
        "doc_title": doc_title,
        "section_path": section_path or ["Intro"],
        "block_type": block_type,
        "char_len": char_len if char_len is not None else len(content),
    }


def test_ensure_collection_creates_qdrant_collection(tmp_path: Path) -> None:
    store = QdrantStore(
        collection_name="sections",
        vector_size=4,
        use_local=True,
        path=tmp_path / "qdrant",
    )

    try:
        store.ensure_collection()

        collection = store.client.get_collection("sections")
        assert collection.config.params.vectors.size == 4
    finally:
        store.close()


def test_upsert_sections_and_search_round_trip(tmp_path: Path) -> None:
    store = QdrantStore(
        collection_name="sections",
        vector_size=4,
        use_local=True,
        path=tmp_path / "qdrant",
        embedder=_FakeEmbedder([1.0, 0.0, 0.0, 0.0]),
    )
    try:
        store.ensure_collection()

        sections = [
            _build_section(source_id="doc-1", section_id="sec-1", content="alpha content", section_path=["A"]),
            _build_section(source_id="doc-2", section_id="sec-2", content="beta content", section_path=["B"]),
        ]
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]

        store.upsert_sections(sections, vectors)
        by_vector = store.search_by_vector([1.0, 0.0, 0.0, 0.0], top_k=2)
        by_query = store.search_by_query_vector("alpha query", top_k=2)

        assert by_vector
        assert by_vector[0].section_id == "sec-1"
        assert by_vector[0].source_id == "doc-1"
        assert by_vector[0].content == "alpha content"
        assert by_vector[0].metadata["doc_title"] == "Demo Doc"
        assert by_vector[0].metadata["section_path"] == ["A"]
        assert by_vector[0].metadata["retriever"] == "qdrant_dense"
        assert by_query[0].section_id == "sec-1"
    finally:
        store.close()


def test_upsert_sections_raises_on_dimension_mismatch(tmp_path: Path) -> None:
    store = QdrantStore(
        collection_name="sections",
        vector_size=4,
        use_local=True,
        path=tmp_path / "qdrant",
    )
    try:
        store.ensure_collection()

        with pytest.raises(ValueError, match="Vector dimension mismatch"):
            store.upsert_sections(
                [_build_section(source_id="doc-1", section_id="sec-1", content="alpha")],
                [[1.0, 0.0]],
            )
    finally:
        store.close()


def test_ensure_collection_raises_when_existing_dimension_differs(tmp_path: Path) -> None:
    store_a = QdrantStore(
        collection_name="sections",
        vector_size=4,
        use_local=True,
        path=tmp_path / "qdrant",
    )
    try:
        store_a.ensure_collection()

        store_b = QdrantStore(
            collection_name="sections",
            vector_size=8,
            use_local=True,
            path=tmp_path / "qdrant",
            client=store_a.client,
        )

        with pytest.raises(RuntimeError):
            store_b.ensure_collection()
    finally:
        store_a.close()
