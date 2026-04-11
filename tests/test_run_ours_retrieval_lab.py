from __future__ import annotations

import json
from pathlib import Path

from scripts.run_ours_retrieval_lab import run_retrieval_lab


def test_run_ours_retrieval_lab_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "retrieval_lab"
    result = run_retrieval_lab(
        dataset="outputs/two_file_demo/benchmark_dataset.jsonl",
        sections="artifacts/two_file_demo/markdown_sections.jsonl",
        knowledge="artifacts/two_file_demo/knowledge_extraction.jsonl",
        top_k=3,
        limit=2,
        max_workers=1,
        skeleton_mode="oracle",
        rewrite_modes=["original", "template"],
        retrieval_modes=["lexical"],
        scorer_modes=["plain", "relation_driven"],
        output_dir=output_dir,
        overwrite=True,
    )

    assert result.exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "comparison.json").exists()
    assert (output_dir / "matrix.csv").exists()

    plain_predictions = output_dir / "combos" / "original__lexical__plain" / "predictions.jsonl"
    relation_predictions = output_dir / "combos" / "template__lexical__relation_driven" / "predictions.jsonl"
    assert plain_predictions.exists()
    assert relation_predictions.exists()

    plain_row = json.loads(plain_predictions.read_text(encoding="utf-8").splitlines()[0])
    relation_row = json.loads(relation_predictions.read_text(encoding="utf-8").splitlines()[0])

    assert plain_row["scorer_mode"] == "plain"
    assert "dense_scores" not in plain_row
    assert plain_row["trace_metadata"]["retrieval_trace"]["mode"] == "plain"

    assert relation_row["scorer_mode"] == "relation_driven"
    assert "dense_scores" in relation_row
    assert relation_row["trace_metadata"]["retrieval_trace"]["mode"] == "relation_driven"
