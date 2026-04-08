from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_run_all_baselines_script_generates_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "all_baselines"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_all_baselines.py",
            "--dataset",
            "outputs/two_file_demo/benchmark_dataset.jsonl",
            "--sections",
            "artifacts/two_file_demo/markdown_sections.jsonl",
            "--knowledge",
            "artifacts/two_file_demo/knowledge_extraction.jsonl",
            "--limit",
            "3",
            "--top-k",
            "3",
            "--seed-top-k",
            "2",
            "--expand-k",
            "2",
            "--max-workers",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "summary:" in result.stdout

    summary_path = output_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 3
    assert summary["max_workers"] == 2

    expected = {
        "traditional_rag",
        "graph_enhanced_rag",
        "kbqa_baseline",
    }
    assert set(summary["pipelines"]) == expected

    for name in expected:
        pipeline_summary = summary["pipelines"][name]
        assert Path(pipeline_summary["metrics"]).exists()
        assert Path(pipeline_summary["predictions"]).exists()
        metrics = json.loads(Path(pipeline_summary["metrics"]).read_text(encoding="utf-8"))
        assert metrics["sample_count"] == 3
        assert metrics["max_workers"] == 2


def test_run_all_baselines_script_reports_missing_inputs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_all_baselines.py",
            "--dataset",
            "outputs/not_exists/benchmark_dataset.jsonl",
            "--sections",
            "artifacts/two_file_demo/markdown_sections.jsonl",
            "--knowledge",
            "artifacts/two_file_demo/knowledge_extraction.jsonl",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Dataset file not found" in result.stderr
    assert "outputs/full_run/benchmark_dataset.jsonl" in result.stderr
