from __future__ import annotations

import json
from pathlib import Path

from scripts.run_rewrite_lab import run_rewrite_lab


def test_run_rewrite_lab_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "rewrite_lab"
    result = run_rewrite_lab(
        dataset="outputs/two_file_demo/benchmark_dataset.jsonl",
        sections="artifacts/two_file_demo/markdown_sections.jsonl",
        knowledge="artifacts/two_file_demo/knowledge_extraction.jsonl",
        rewriters=["skip_baseline", "rule_based_v1"],
        top_k=3,
        use_rerank=False,
        output_dir=str(output_dir),
        max_samples=2,
        question_types=None,
        write_html_report=False,
    )
    assert result.exists()
    assert (output_dir / "rewrites.jsonl").exists()
    assert (output_dir / "pairwise_eval.jsonl").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "bucket_summary.json").exists()
    assert (output_dir / "report.md").exists()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "skip_baseline" in metrics
    assert "rule_based_v1" in metrics
