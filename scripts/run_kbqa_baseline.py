from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataio.loaders import load_benchmark_samples, load_graph_section_records
from evaluation.answer_metrics import aggregate_answer_metrics
from evaluation.retrieval_metrics import aggregate_retrieval_metrics
from modules.entity_linker import EntityLinker
from modules.graph_expander import GraphIndex
from modules.kb_executor import KBExecutor
from modules.relation_matcher import RelationMatcher
from pipelines.base import PublicIndex
from pipelines.kbqa_baseline import KBQABaselineConfig, KBQABaselinePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KBQA baseline.")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/two_file_demo/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/two_file_demo/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/two_file_demo/knowledge_extraction.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--entity-mode", choices=["gold", "heuristic"], default="gold")
    parser.add_argument("--relation-mode", choices=["gold", "heuristic"], default="gold")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/kbqa_baseline"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_benchmark_samples(args.dataset)
    if args.limit is not None:
        samples = samples[: args.limit]
    text_index = PublicIndex.from_markdown_sections(args.sections)
    graph_index = GraphIndex.build(load_graph_section_records(args.knowledge))
    pipeline = KBQABaselinePipeline(
        EntityLinker(graph_index),
        RelationMatcher(graph_index),
        KBExecutor(graph_index, text_index),
        config=KBQABaselineConfig(top_k=args.top_k, entity_mode=args.entity_mode, relation_mode=args.relation_mode),
    )
    records = [pipeline.run(sample) for sample in samples]
    metrics = {
        "method_name": pipeline.method_name,
        "sample_count": len(records),
        "top_k": args.top_k,
        "entity_mode": args.entity_mode,
        "relation_mode": args.relation_mode,
        "answer": aggregate_answer_metrics(records),
        "retrieval": aggregate_retrieval_metrics(records, k=args.top_k),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "predictions.jsonl"
    metrics_path = args.output_dir / "metrics.json"
    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_output_dict(), ensure_ascii=False) + "\n")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"predictions: {prediction_path}")
    print(f"metrics: {metrics_path}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
