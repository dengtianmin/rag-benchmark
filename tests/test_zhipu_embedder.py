from __future__ import annotations

from dataclasses import dataclass

import pytest

from embedders.zhipu_embedder import ZhipuEmbedder


@dataclass
class _FakeEmbeddingItem:
    index: int
    embedding: list[float]


@dataclass
class _FakeEmbeddingResponse:
    data: list[_FakeEmbeddingItem]


class _FakeEmbeddingsAPI:
    def __init__(self, owner: "_FakeClient") -> None:
        self.owner = owner

    def create(self, *, model: str, input: list[str], dimensions: int) -> _FakeEmbeddingResponse:
        return self.owner.handle_create(model=model, input=input, dimensions=dimensions)


class _FakeClient:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict] = []
        self.embeddings = _FakeEmbeddingsAPI(self)

    def handle_create(self, *, model: str, input: list[str], dimensions: int) -> _FakeEmbeddingResponse:
        self.calls.append({"model": model, "input": list(input), "dimensions": dimensions})
        return self.handler(model=model, input=input, dimensions=dimensions)


def _make_response(texts: list[str], dimensions: int) -> _FakeEmbeddingResponse:
    return _FakeEmbeddingResponse(
        data=[
            _FakeEmbeddingItem(index=index, embedding=[float(index + 1)] * dimensions)
            for index, _ in enumerate(texts)
        ]
    )


def test_embed_documents_splits_batches_and_preserves_order() -> None:
    client = _FakeClient(lambda **kwargs: _make_response(kwargs["input"], kwargs["dimensions"]))
    embedder = ZhipuEmbedder(
        api_key="test-key",
        model="embedding-3",
        dimensions=4,
        batch_size=2,
        client=client,
    )

    vectors = embedder.embed_documents(["doc-a", "doc-b", "doc-c", "doc-d", "doc-e"])

    assert len(client.calls) == 3
    assert [len(call["input"]) for call in client.calls] == [2, 2, 1]
    assert len(vectors) == 5
    assert vectors[0] == [1.0, 1.0, 1.0, 1.0]
    assert vectors[2] == [1.0, 1.0, 1.0, 1.0]
    assert all(len(vector) == 4 for vector in vectors)


def test_embed_documents_filters_empty_texts_and_fills_zero_vectors() -> None:
    client = _FakeClient(lambda **kwargs: _make_response(kwargs["input"], kwargs["dimensions"]))
    embedder = ZhipuEmbedder(
        api_key="test-key",
        model="embedding-3",
        dimensions=3,
        batch_size=4,
        client=client,
    )

    vectors = embedder.embed_documents(["  ", "alpha", "", "beta", None])  # type: ignore[list-item]

    assert len(client.calls) == 1
    assert client.calls[0]["input"] == ["alpha", "beta"]
    assert vectors[0] == [0.0, 0.0, 0.0]
    assert vectors[1] == [1.0, 1.0, 1.0]
    assert vectors[2] == [0.0, 0.0, 0.0]
    assert vectors[3] == [2.0, 2.0, 2.0]
    assert vectors[4] == [0.0, 0.0, 0.0]


def test_embed_query_returns_single_vector() -> None:
    client = _FakeClient(lambda **kwargs: _make_response(kwargs["input"], kwargs["dimensions"]))
    embedder = ZhipuEmbedder(
        api_key="test-key",
        model="embedding-3",
        dimensions=2,
        batch_size=8,
        client=client,
    )

    vector = embedder.embed_query("query text")

    assert vector == [1.0, 1.0]
    assert len(client.calls) == 1
    assert client.calls[0]["input"] == ["query text"]


def test_embed_query_returns_zero_vector_for_empty_query() -> None:
    client = _FakeClient(lambda **kwargs: _make_response(kwargs["input"], kwargs["dimensions"]))
    embedder = ZhipuEmbedder(
        api_key="test-key",
        model="embedding-3",
        dimensions=5,
        batch_size=8,
        client=client,
    )

    vector = embedder.embed_query("   ")

    assert vector == [0.0] * 5
    assert client.calls == []


def test_embedder_retries_on_transient_failure() -> None:
    state = {"count": 0}

    def handler(**kwargs):
        state["count"] += 1
        if state["count"] < 2:
            raise RuntimeError("temporary upstream error")
        return _make_response(kwargs["input"], kwargs["dimensions"])

    client = _FakeClient(handler)
    embedder = ZhipuEmbedder(
        api_key="test-key",
        model="embedding-3",
        dimensions=3,
        batch_size=8,
        client=client,
        max_retries=2,
    )

    vectors = embedder.embed_documents(["alpha"])

    assert state["count"] == 2
    assert vectors == [[1.0, 1.0, 1.0]]


def test_embed_documents_raises_on_invalid_input_type() -> None:
    client = _FakeClient(lambda **kwargs: _make_response(kwargs["input"], kwargs["dimensions"]))
    embedder = ZhipuEmbedder(
        api_key="test-key",
        model="embedding-3",
        dimensions=3,
        batch_size=8,
        client=client,
    )

    with pytest.raises(TypeError):
        embedder.embed_documents("not-a-list")  # type: ignore[arg-type]


def test_embed_documents_raises_on_dimension_mismatch() -> None:
    client = _FakeClient(
        lambda **kwargs: _FakeEmbeddingResponse(
            data=[_FakeEmbeddingItem(index=0, embedding=[1.0, 2.0])]
        )
    )
    embedder = ZhipuEmbedder(
        api_key="test-key",
        model="embedding-3",
        dimensions=3,
        batch_size=8,
        client=client,
    )

    with pytest.raises(ValueError, match="dimension mismatch"):
        embedder.embed_documents(["alpha"])
