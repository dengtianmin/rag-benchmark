from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_ablation_script_supports_runtime_retrieval_flags(tmp_path: Path) -> None:
    output_dir = tmp_path / "ablation"
    env = os.environ.copy()
    env["RETRIEVAL_MODE"] = "lexical"
    env["USE_RERANK"] = "false"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_ablation.py",
            "--dataset",
            "outputs/two_file_demo/benchmark_dataset.jsonl",
            "--sections",
            "artifacts/two_file_demo/markdown_sections.jsonl",
            "--knowledge",
            "artifacts/two_file_demo/knowledge_extraction.jsonl",
            "--limit",
            "2",
            "--top-k",
            "2",
            "--max-workers",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    assert "comparison:" in result.stdout

    comparison_path = output_dir / "comparison.json"
    assert comparison_path.exists()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert len(comparison) == 4
    assert all(item["retrieval_mode"] == "lexical" for item in comparison)
    assert all(item["use_rerank"] is False for item in comparison)
