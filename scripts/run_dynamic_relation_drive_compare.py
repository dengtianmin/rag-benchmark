from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.run_ours_retrieval_lab import _parse_csv_arg, run_retrieval_lab


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare relation_driven, dynamic_relation_drive, and dynamic_relation_drive_new.")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/full_run/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/full_run/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/full_run/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/dynamic_relation_drive_compare"))
    parser.add_argument("--skeleton-mode", choices=["oracle", "stub_predicted", "llm_predicted"], default="oracle")
    parser.add_argument("--rewrite-modes", default="hybrid_llm")
    parser.add_argument("--retrieval-modes", default="hybrid")
    parser.add_argument("--scorer-modes", default="relation_driven,dynamic_relation_drive,dynamic_relation_drive_new")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--llm-rewrite-max-tokens", type=int, default=64)
    parser.add_argument("--llm-rewrite-temperature", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_retrieval_lab(
        dataset=args.dataset,
        sections=args.sections,
        knowledge=args.knowledge,
        top_k=args.top_k,
        limit=args.limit,
        max_workers=args.max_workers,
        skeleton_mode=args.skeleton_mode,
        rewrite_modes=_parse_csv_arg(args.rewrite_modes),
        retrieval_modes=_parse_csv_arg(args.retrieval_modes),
        scorer_modes=_parse_csv_arg(args.scorer_modes),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        llm_rewrite_max_tokens=args.llm_rewrite_max_tokens,
        llm_rewrite_temperature=args.llm_rewrite_temperature,
    )
    print(f"outputs: {output_path}")


if __name__ == "__main__":
    main()
