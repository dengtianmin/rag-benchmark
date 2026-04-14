from __future__ import annotations

import json
from pathlib import Path

from clients.chat_llm_client import ChatLLMResponse
from scripts import run_ours_retrieval_lab as retrieval_lab_module
from scripts.run_dynamic_relation_drive_compare import main as compare_main


class _StubClient:
    api_key = "stub-key"
    model = "stub-model"

    def chat_completion(self, **kwargs) -> ChatLLMResponse:
        del kwargs
        return ChatLLMResponse(
            content='{"must_keep_terms":["Alpha"],"sparse_rewrite":"Alpha 功能","dense_rewrite":"Alpha 的功能和说明"}',
            model_name=self.model,
            finish_reason="stop",
            raw={"choices": []},
            latency_ms=1,
        )


def test_compare_script_outputs_three_scorers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(retrieval_lab_module, "_build_llm_client", lambda max_tokens, temperature: _StubClient())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_dynamic_relation_drive_compare.py",
            "--dataset",
            "outputs/full_run/benchmark_dataset.jsonl",
            "--sections",
            "artifacts/full_run/markdown_sections.jsonl",
            "--knowledge",
            "artifacts/full_run/knowledge_extraction.jsonl",
            "--limit",
            "1",
            "--max-workers",
            "1",
            "--output-dir",
            str(tmp_path / "compare"),
            "--overwrite",
        ],
    )

    compare_main()

    output_dir = tmp_path / "compare"
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "dynamic_compare_report.json").exists()
    assert (output_dir / "question_type_breakdown.json").exists()
    assert (output_dir / "regression_cases.json").exists()
    payload = json.loads((output_dir / "dynamic_compare_report.json").read_text(encoding="utf-8"))
    combo = payload["hybrid_llm__hybrid"]
    assert "relation_driven" in combo["scorers"]
    assert "dynamic_relation_drive" in combo["scorers"]
    assert "dynamic_relation_drive_new" in combo["scorers"]
