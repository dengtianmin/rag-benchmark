from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataio.loaders import load_benchmark_samples


def inspect_dataset(dataset_path: Path, sample_size: int = 3, seed: int = 42) -> None:
    samples = load_benchmark_samples(dataset_path)
    print(f"dataset_path: {dataset_path}")
    print(f"total_samples: {len(samples)}")

    question_type_dist = Counter(sample.question_type.value for sample in samples)
    source_scope_dist = Counter(sample.source_scope.value for sample in samples)
    text_comp_dist = Counter(sample.requires_text_compensation for sample in samples)

    print("\nquestion_type distribution:")
    for key, value in question_type_dist.most_common():
        print(f"  {key}: {value}")

    print("\nsource_scope distribution:")
    for key, value in source_scope_dist.most_common():
        print(f"  {key}: {value}")

    print("\nrequires_text_compensation distribution:")
    for key, value in text_comp_dist.items():
        print(f"  {key}: {value}")

    rng = random.Random(seed)
    preview = rng.sample(samples, k=min(sample_size, len(samples)))
    print(f"\nrandom_samples: {len(preview)}")
    for idx, sample in enumerate(preview, start=1):
        print(f"\n[{idx}] question_id={sample.question_id}")
        print(f"question: {sample.question}")
        print(f"question_type: {sample.question_type.value}")
        print(f"source_scope: {sample.source_scope.value}")
        print(f"entities: {sample.entities}")
        print(f"relations: {sample.relations}")
        print(f"constraints: {sample.constraints}")
        print(f"answer_short: {sample.answer_short}")
        print(f"evidence_count: {len(sample.evidence)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect benchmark_dataset.jsonl")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/two_file_demo/benchmark_dataset.jsonl"),
        help="Path to benchmark_dataset.jsonl",
    )
    parser.add_argument("--sample-size", type=int, default=3, help="Number of random samples to preview.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample preview.")
    args = parser.parse_args()
    inspect_dataset(args.dataset, sample_size=args.sample_size, seed=args.seed)


if __name__ == "__main__":
    main()
