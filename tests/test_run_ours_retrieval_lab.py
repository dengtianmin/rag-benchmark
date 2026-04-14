from __future__ import annotations

import json
from pathlib import Path

from clients.chat_llm_client import ChatLLMResponse
from scripts import run_ours_retrieval_lab as retrieval_lab_module
from scripts.run_ours_retrieval_lab import run_retrieval_lab


class _StubClient:
    api_key = "stub-key"
    model = "stub-model"

    def chat_completion(self, **kwargs) -> ChatLLMResponse:
        del kwargs
        return ChatLLMResponse(
            content='{"must_keep_terms":["Alpha","合作原因"],"sparse_rewrite":"Alpha 合作原因 2019年","dense_rewrite":"2019年 Alpha 与 Beta 合作原因"}',
            model_name=self.model,
            finish_reason="stop",
            raw={"choices": []},
            latency_ms=1,
        )


def test_run_ours_retrieval_lab_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "retrieval_lab"
    result = run_retrieval_lab(
        dataset="outputs/full_run/benchmark_dataset.jsonl",
        sections="artifacts/full_run/markdown_sections.jsonl",
        knowledge="artifacts/full_run/knowledge_extraction.jsonl",
        top_k=3,
        limit=2,
        max_workers=1,
        skeleton_mode="oracle",
        rewrite_modes=["original", "splicing"],
        retrieval_modes=["lexical"],
        scorer_modes=["plain", "relation_driven"],
        output_dir=output_dir,
        overwrite=True,
    )

    assert result.exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "matrix.csv").exists()
    assert (output_dir / "question_type_breakdown.json").exists()

    plain_predictions = output_dir / "combos" / "original__lexical__plain" / "predictions.jsonl"
    relation_predictions = output_dir / "combos" / "splicing__lexical__relation_driven" / "predictions.jsonl"
    assert plain_predictions.exists()
    assert relation_predictions.exists()

    plain_row = json.loads(plain_predictions.read_text(encoding="utf-8").splitlines()[0])
    relation_row = json.loads(relation_predictions.read_text(encoding="utf-8").splitlines()[0])

    assert plain_row["scorer_mode"] == "plain"
    assert "dense_scores" not in plain_row
    assert plain_row["trace_metadata"]["retrieval_trace"]["mode"] == "plain"
    assert plain_row["lexical_query"] == plain_row["rewritten_query"]
    assert plain_row["dense_query"] == plain_row["rewritten_query"]
    assert "structured_rewrite" in plain_row
    assert isinstance(plain_row["stage1_query"], str)
    assert isinstance(plain_row["stage1_candidate_ids"], list)
    assert "gold_in_stage1_pool" in plain_row
    assert plain_row["miss_type"] in {
        "in_final_topk",
        "not_in_stage1_pool",
        "in_stage1_but_filtered",
        "in_final_candidates_but_rank_too_low",
        "unknown",
    }

    assert relation_row["scorer_mode"] == "relation_driven"
    assert "dense_scores" in relation_row
    assert relation_row["trace_metadata"]["retrieval_trace"]["mode"] == "relation_driven"
    assert "gold_semantic_score" in relation_row
    assert "gold_relation_score" in relation_row
    assert "filtered_out_reason_map" in relation_row


def test_run_ours_retrieval_lab_supports_dynamic_new_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(retrieval_lab_module, "_build_llm_client", lambda max_tokens, temperature: _StubClient())
    output_dir = tmp_path / "retrieval_lab_dynamic_new"
    run_retrieval_lab(
        dataset="outputs/full_run/benchmark_dataset.jsonl",
        sections="artifacts/full_run/markdown_sections.jsonl",
        knowledge="artifacts/full_run/knowledge_extraction.jsonl",
        top_k=3,
        limit=1,
        max_workers=1,
        skeleton_mode="oracle",
        rewrite_modes=["hybrid_llm"],
        retrieval_modes=["hybrid"],
        scorer_modes=["relation_driven", "dynamic_relation_drive", "dynamic_relation_drive_new"],
        output_dir=output_dir,
        overwrite=True,
        llm_rewrite_max_tokens=128,
        llm_rewrite_temperature=0.0,
    )

    assert (output_dir / "dynamic_compare_report.json").exists()
    assert (output_dir / "question_type_breakdown.json").exists()
    assert (output_dir / "regression_cases.json").exists()
    prediction_path = output_dir / "combos" / "hybrid_llm__hybrid__dynamic_relation_drive_new" / "predictions.jsonl"
    row = json.loads(prediction_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["scorer_mode"] == "dynamic_relation_drive_new"
    assert "adjusted_question_type_for_dynamic_new" in row
    assert "dynamic_new_weight_profile" in row


def test_run_ours_retrieval_lab_supports_hybrid_llm_predictions(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(retrieval_lab_module, "_build_llm_client", lambda max_tokens, temperature: _StubClient())
    output_dir = tmp_path / "retrieval_lab_hybrid"
    result = run_retrieval_lab(
        dataset="outputs/full_run/benchmark_dataset.jsonl",
        sections="artifacts/full_run/markdown_sections.jsonl",
        knowledge="artifacts/full_run/knowledge_extraction.jsonl",
        top_k=3,
        limit=1,
        max_workers=1,
        skeleton_mode="oracle",
        rewrite_modes=["hybrid_llm"],
        retrieval_modes=["lexical"],
        scorer_modes=["plain"],
        output_dir=output_dir,
        overwrite=True,
        llm_rewrite_max_tokens=128,
        llm_rewrite_temperature=0.0,
    )

    assert result.exists()
    prediction_path = output_dir / "combos" / "hybrid_llm__lexical__plain" / "predictions.jsonl"
    row = json.loads(prediction_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["rewrite_mode"] == "hybrid_llm"
    assert row["lexical_query"] == "Alpha 合作原因 2019年"
    assert row["dense_query"] == "2019年 Alpha 与 Beta 合作原因"
    assert row["trace_metadata"]["retrieval_trace"]["stage1_query"] == "Alpha 合作原因 2019年"


def test_run_ours_retrieval_lab_records_hybrid_branch_trace(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(retrieval_lab_module, "_build_llm_client", lambda max_tokens, temperature: _StubClient())
    output_dir = tmp_path / "retrieval_lab_hybrid_dense"
    run_retrieval_lab(
        dataset="outputs/full_run/benchmark_dataset.jsonl",
        sections="artifacts/full_run/markdown_sections.jsonl",
        knowledge="artifacts/full_run/knowledge_extraction.jsonl",
        top_k=3,
        limit=1,
        max_workers=1,
        skeleton_mode="oracle",
        rewrite_modes=["hybrid_llm"],
        retrieval_modes=["hybrid"],
        scorer_modes=["plain"],
        output_dir=output_dir,
        overwrite=True,
        llm_rewrite_max_tokens=128,
        llm_rewrite_temperature=0.0,
    )

    prediction_path = output_dir / "combos" / "hybrid_llm__hybrid__plain" / "predictions.jsonl"
    row = json.loads(prediction_path.read_text(encoding="utf-8").splitlines()[0])
    assert "lexical_stage1_candidate_ids" in row
    assert "dense_stage1_candidate_ids" in row
    assert "final_topk_source_breakdown" in row
