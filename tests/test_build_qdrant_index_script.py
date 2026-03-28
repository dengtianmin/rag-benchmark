from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_qdrant_index_script_dry_run_writes_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "qdrant_index_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_qdrant_index.py",
            "--sections",
            "artifacts/two_file_demo/markdown_sections.jsonl",
            "--dry-run",
            "--limit",
            "3",
            "--manifest-path",
            str(manifest_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "collection_name:" in result.stdout
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["indexed_count"] == 3
    assert manifest["dry_run"] is True
