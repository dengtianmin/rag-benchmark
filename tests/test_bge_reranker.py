from __future__ import annotations

import pytest

from core.schema import RetrievedDocument
from rerankers.bge_reranker import BGEReranker


class _FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[dict] = []

    def compute_score(self, pairs, *, max_query_length: int, max_passage_length: int):
        self.calls.append(
            {
                "pairs": list(pairs),
                "max_query_length": max_query_length,
                "max_passage_length": max_passage_length,
            }
        )
        return list(self.scores)


def _doc(section_id: str, content: str, score: float = 0.0, rank: int = 0) -> RetrievedDocument:
    return RetrievedDocument(
        source_id=f"doc-{section_id}",
        section_id=section_id,
        content=content,
        score=score,
        rank=rank,
        metadata={"origin": "unit_test"},
    )


def test_score_pairs_returns_scores_in_input_order() -> None:
    model = _FakeModel([0.2, 0.9, 0.4])
    reranker = BGEReranker(
        model_path="/models/bge-reranker",
        device="cpu",
        use_fp16=False,
        query_max_length=32,
        passage_max_length=64,
        model=model,
    )

    docs = [_doc("a", "alpha"), _doc("b", "beta"), _doc("c", "gamma")]
    scores = reranker.score_pairs("query", docs)

    assert scores == [0.2, 0.9, 0.4]
    assert len(model.calls) == 1
    assert model.calls[0]["pairs"][1] == ("query", "beta")


def test_rerank_sorts_documents_by_descending_score() -> None:
    model = _FakeModel([0.3, 0.8, 0.5])
    reranker = BGEReranker(
        model_path="/models/bge-reranker",
        device="cpu",
        use_fp16=False,
        query_max_length=32,
        passage_max_length=64,
        model=model,
    )

    docs = [_doc("a", "alpha"), _doc("b", "beta"), _doc("c", "gamma")]
    reranked = reranker.rerank("query", docs)

    assert [document.section_id for document in reranked] == ["b", "c", "a"]
    assert [document.rank for document in reranked] == [1, 2, 3]
    assert reranked[0].metadata["rerank_score"] == 0.8
    assert reranked[0].metadata["reranker"] == "bge_local"
    assert reranked[0].content == "beta"


def test_rerank_applies_top_n_cutoff() -> None:
    model = _FakeModel([0.7, 0.2, 0.9])
    reranker = BGEReranker(
        model_path="/models/bge-reranker",
        device="cpu",
        use_fp16=False,
        query_max_length=32,
        passage_max_length=64,
        model=model,
    )

    docs = [_doc("a", "alpha"), _doc("b", "beta"), _doc("c", "gamma")]
    reranked = reranker.rerank("query", docs, top_n=2)

    assert [document.section_id for document in reranked] == ["c", "a"]
    assert [document.rank for document in reranked] == [1, 2]


def test_rerank_returns_empty_list_for_empty_docs() -> None:
    model = _FakeModel([])
    reranker = BGEReranker(
        model_path="/models/bge-reranker",
        device="cpu",
        use_fp16=False,
        query_max_length=32,
        passage_max_length=64,
        model=model,
    )

    assert reranker.rerank("query", []) == []
    assert reranker.score_pairs("query", []) == []


def test_score_pairs_raises_on_score_count_mismatch() -> None:
    model = _FakeModel([0.1])
    reranker = BGEReranker(
        model_path="/models/bge-reranker",
        device="cpu",
        use_fp16=False,
        query_max_length=32,
        passage_max_length=64,
        model=model,
    )

    with pytest.raises(ValueError, match="score count mismatch"):
        reranker.score_pairs("query", [_doc("a", "alpha"), _doc("b", "beta")])


def test_model_load_failure_has_readable_error(monkeypatch) -> None:
    def _raise(*args, **kwargs):
        raise RuntimeError("model files missing")

    monkeypatch.setattr("rerankers.bge_reranker.BGEReranker._load_model", _raise)

    with pytest.raises(RuntimeError, match="model files missing"):
        BGEReranker(
            model_path="/models/bge-reranker",
            device="cpu",
            use_fp16=False,
            query_max_length=32,
            passage_max_length=64,
        )
