from __future__ import annotations

import pytest
import requests

from core.schema import RetrievedDocument
from rerankers.tei_reranker import TEIReranker


class _FakeResponse:
    def __init__(self, payload, *, status_code: int = 200, text: str | None = None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = text or str(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls: list[dict] = []

    def post(self, url: str, *, json: dict, headers: dict, timeout: float):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _doc(section_id: str, content: str) -> RetrievedDocument:
    return RetrievedDocument(
        source_id=f"doc_{section_id}",
        section_id=section_id,
        content=content,
        score=0.0,
        rank=0,
        metadata={},
    )


def test_tei_reranker_sorts_documents_by_score_desc() -> None:
    session = _FakeSession([_FakeResponse({"results": [{"index": 1, "score": 0.35}, {"index": 0, "score": 0.92}]})])
    reranker = TEIReranker(base_url="http://127.0.0.1:8080", timeout=5, session=session)

    reranked = reranker.rerank("query", [_doc("a", "alpha"), _doc("b", "beta")])

    assert [document.section_id for document in reranked] == ["a", "b"]
    assert [document.rank for document in reranked] == [1, 2]
    assert reranked[0].metadata["rerank_score"] == 0.92
    assert reranked[0].metadata["rank_after_rerank"] == 1
    assert reranked[0].metadata["reranker"] == "tei"
    assert session.calls[0]["url"] == "http://127.0.0.1:8080/rerank"
    assert session.calls[0]["json"]["texts"] == ["alpha", "beta"]


def test_tei_reranker_returns_empty_list_for_empty_docs() -> None:
    session = _FakeSession([])
    reranker = TEIReranker(base_url="http://127.0.0.1:8080", timeout=5, session=session)

    assert reranker.rerank("query", []) == []
    assert session.calls == []


def test_tei_reranker_applies_top_n_cutoff() -> None:
    session = _FakeSession(
        [_FakeResponse({"results": [{"index": 0, "score": 0.1}, {"index": 2, "score": 0.9}, {"index": 1, "score": 0.5}]})]
    )
    reranker = TEIReranker(base_url="http://127.0.0.1:8080", timeout=5, session=session)

    reranked = reranker.rerank("query", [_doc("a", "alpha"), _doc("b", "beta"), _doc("c", "gamma")], top_n=2)

    assert [document.section_id for document in reranked] == ["c", "b"]
    assert [document.rank for document in reranked] == [1, 2]
    assert session.calls[0]["json"]["top_n"] == 2


def test_tei_reranker_raises_readable_error_on_http_failure() -> None:
    session = _FakeSession([_FakeResponse({"detail": "bad request"}, status_code=503, text="service unavailable")])
    reranker = TEIReranker(base_url="http://127.0.0.1:8080", timeout=5, max_retries=0, session=session)

    with pytest.raises(RuntimeError, match="status 503"):
        reranker.rerank("query", [_doc("a", "alpha")])


def test_tei_reranker_retries_timeout_and_raises_readable_error() -> None:
    session = _FakeSession([requests.Timeout("timed out"), requests.Timeout("timed out again")])
    reranker = TEIReranker(base_url="http://127.0.0.1:8080", timeout=1, max_retries=1, session=session)

    with pytest.raises(RuntimeError, match="timed out"):
        reranker.rerank("query", [_doc("a", "alpha")])
