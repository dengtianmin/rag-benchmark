from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataio.loaders import load_benchmark_samples, load_graph_section_records
from modules.graph_expander import GraphIndex
from modules.skeleton_extractor import SkeletonExtractor
from modules.skeleton_metrics import aggregate_by_question_type, aggregate_micro, compute_sample_metrics
from runtime_config import load_runtime_settings
from tqdm import tqdm


DEFAULT_MODES = ["oracle", "stub_predicted", "llm_predicted"]
DEFAULT_KNOWLEDGE_CANDIDATES = [
    Path("artifacts/full_run/knowledge_extraction.jsonl"),
    Path("artifacts/knowledge_extraction.jsonl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate skeleton predictors against sample.skeleton_label.")
    parser.add_argument("--dataset", type=Path, default=Path("outputs/full_run/benchmark_dataset.jsonl"))
    parser.add_argument("--sections", type=Path, default=Path("artifacts/full_run/markdown_sections.jsonl"))
    parser.add_argument("--knowledge", type=Path, default=Path("artifacts/full_run/knowledge_extraction.jsonl"))
    parser.add_argument("--modes", default="oracle,stub_predicted,llm_predicted")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--question-types", default="")
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=256)
    parser.add_argument("--llm-temperature", type=float, default=0.0)
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments/skeleton_eval"))
    return parser.parse_args()


def _parse_csv_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _question_type_label(sample) -> str:
    return sample.question_type.value if sample.question_type is not None else "unknown"


def _resolve_existing_path(path: Path, *, label: str, fallbacks: list[Path] | None = None) -> Path:
    if path.exists():
        return path
    for candidate in fallbacks or []:
        if candidate.exists():
            print(f"{label} not found at {path}, fallback to {candidate}")
            return candidate
    fallback_text = ", ".join(str(item) for item in (fallbacks or [])) or "no fallback candidates"
    raise FileNotFoundError(f"{label} file not found: {path}. tried fallbacks: {fallback_text}")


def _build_llm_client(args: argparse.Namespace):
    from clients.chat_llm_client import ChatLLMClient

    settings = load_runtime_settings()
    model_name = args.llm_model if args.llm_model is not None else os.getenv("DASHSCOPE_MODEL") or settings.generator.model
    api_key = os.getenv("DASHSCOPE_API_KEY") or settings.generator.api_key
    base_url = os.getenv("DASHSCOPE_BASE_URL") or settings.generator.base_url
    return ChatLLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model_name,
        timeout=settings.generator.timeout,
        max_tokens=args.llm_max_tokens,
        temperature=args.llm_temperature,
        json_mode=True,
    )


def _build_diagnostic_row(mode: str, details: dict, prediction) -> dict[str, object]:
    entity_total = len(prediction.entities)
    relation_total = len(prediction.relations)
    anchored_entities = len(details.get("anchored_entities", []))
    anchored_relations = len(details.get("anchored_relations", []))
    anchor_total = entity_total + relation_total
    anchored_total = anchored_entities + anchored_relations
    parsed_ok = details.get("parsed_ok")
    return {
        "mode": mode,
        "parsed_ok": bool(parsed_ok) if parsed_ok is not None else None,
        "is_empty_prediction": not (prediction.entities or prediction.relations or prediction.constraints),
        "entity_anchor_denominator": entity_total,
        "entity_anchor_numerator": anchored_entities,
        "relation_anchor_denominator": relation_total,
        "relation_anchor_numerator": anchored_relations,
        "anchor_denominator": anchor_total,
        "anchor_numerator": anchored_total,
        "has_error": bool(details.get("error")),
    }


def _aggregate_diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    sample_count = len(rows)
    if sample_count == 0:
        return {
            "sample_count": 0,
            "json_parse_success_rate": None,
            "empty_prediction_rate": 0.0,
            "anchor_success_rate": None,
            "entity_anchor_success_rate": None,
            "relation_anchor_success_rate": None,
            "api_error_rate": 0.0,
        }

    parse_rows = [row for row in rows if row["parsed_ok"] is not None]
    json_parse_success_rate = (
        sum(1 for row in parse_rows if row["parsed_ok"]) / len(parse_rows)
        if parse_rows
        else None
    )

    anchor_denominator = sum(int(row["anchor_denominator"]) for row in rows)
    anchor_numerator = sum(int(row["anchor_numerator"]) for row in rows)
    entity_anchor_denominator = sum(int(row["entity_anchor_denominator"]) for row in rows)
    entity_anchor_numerator = sum(int(row["entity_anchor_numerator"]) for row in rows)
    relation_anchor_denominator = sum(int(row["relation_anchor_denominator"]) for row in rows)
    relation_anchor_numerator = sum(int(row["relation_anchor_numerator"]) for row in rows)

    return {
        "sample_count": sample_count,
        "json_parse_success_rate": json_parse_success_rate,
        "empty_prediction_rate": sum(1 for row in rows if row["is_empty_prediction"]) / sample_count,
        "anchor_success_rate": anchor_numerator / anchor_denominator if anchor_denominator else None,
        "entity_anchor_success_rate": (
            entity_anchor_numerator / entity_anchor_denominator if entity_anchor_denominator else None
        ),
        "relation_anchor_success_rate": (
            relation_anchor_numerator / relation_anchor_denominator if relation_anchor_denominator else None
        ),
        "api_error_rate": sum(1 for row in rows if row["has_error"]) / sample_count,
    }


def _print_summary(summary: dict[str, object], modes: list[str]) -> None:
    headers = ["mode", "samples", "entity_f1", "relation_f1", "constraint_f1", "exact_match", "parse_ok"]
    rows = []
    for mode in modes:
        overall = summary["overall"][mode]
        diagnostics = summary["diagnostics"][mode]
        parse_value = diagnostics["json_parse_success_rate"]
        rows.append(
            {
                "mode": mode,
                "samples": str(overall["sample_count"]),
                "entity_f1": f'{overall["entities"]["f1"]:.4f}',
                "relation_f1": f'{overall["relations"]["f1"]:.4f}',
                "constraint_f1": f'{overall["constraints"]["f1"]:.4f}',
                "exact_match": f'{overall["skeleton_exact_match"]:.4f}',
                "parse_ok": "-" if parse_value is None else f"{parse_value:.4f}",
            }
        )
    widths = {header: max(len(header), max(len(row[header]) for row in rows)) for header in headers}
    print("Skeleton Evaluation Summary")
    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(row[header].ljust(widths[header]) for header in headers))


def _evaluate_sample(sample, modes: list[str], extractor: SkeletonExtractor) -> list[dict[str, object]]:
    gold = sample.skeleton_label
    rows: list[dict[str, object]] = []
    for mode in modes:
        prediction = extractor.extract(sample, mode=mode)
        sample_metrics = compute_sample_metrics(prediction.skeleton, gold)
        rows.append(
            {
                "question_id": sample.question_id,
                "question_type": _question_type_label(sample),
                "mode": mode,
                "question": sample.question,
                "gold": gold.model_dump(),
                "prediction": prediction.skeleton.model_dump(),
                "metrics": sample_metrics,
                "details": prediction.details,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    modes = _parse_csv_arg(args.modes) or list(DEFAULT_MODES)
    dataset_path = _resolve_existing_path(args.dataset, label="dataset")
    knowledge_path = _resolve_existing_path(
        args.knowledge,
        label="knowledge",
        fallbacks=DEFAULT_KNOWLEDGE_CANDIDATES,
    )
    samples = load_benchmark_samples(dataset_path)
    if args.question_types:
        allowed_question_types = set(_parse_csv_arg(args.question_types))
        samples = [sample for sample in samples if _question_type_label(sample) in allowed_question_types]
    if args.sample_count is not None:
        samples = samples[: args.sample_count]

    graph_index = GraphIndex.build(load_graph_section_records(knowledge_path))
    llm_client = _build_llm_client(args) if "llm_predicted" in modes else None
    extractor = SkeletonExtractor(
        graph_index,
        llm_client=llm_client,
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
    )

    per_mode_metric_rows: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    per_mode_diagnostic_rows: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    per_sample_rows: list[dict[str, object]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = {
            executor.submit(_evaluate_sample, sample, modes, extractor): sample
            for sample in samples
        }
        progress = tqdm(total=len(futures), desc="Evaluating skeleton predictors", unit="sample")
        for future in as_completed(futures):
            sample = futures[future]
            rows = future.result()
            for row in rows:
                mode = str(row["mode"])
                metric_row = {
                    **row["metrics"],
                    "question_type": row["question_type"],
                }
                per_mode_metric_rows[mode].append(metric_row)
                details = dict(row["details"])
                prediction_payload = row["prediction"]
                prediction_view = type(
                    "PredictionView",
                    (),
                    {
                        "entities": prediction_payload.get("entities", []),
                        "relations": prediction_payload.get("relations", []),
                        "constraints": prediction_payload.get("constraints", []),
                    },
                )()
                per_mode_diagnostic_rows[mode].append(_build_diagnostic_row(mode, details, prediction_view))
                per_sample_rows.append(row)
            progress.update(1)
            progress.set_postfix({"qid": sample.question_id, "modes": len(modes)})
        progress.close()

    summary = {
        "overall": {mode: aggregate_micro(rows) for mode, rows in per_mode_metric_rows.items()},
        "by_question_type": {mode: aggregate_by_question_type(rows) for mode, rows in per_mode_metric_rows.items()},
        "diagnostics": {mode: _aggregate_diagnostics(rows) for mode, rows in per_mode_diagnostic_rows.items()},
        "config": {
            "dataset": str(dataset_path),
            "sections": str(args.sections),
            "knowledge": str(knowledge_path),
            "modes": modes,
            "sample_count": len(samples),
            "question_types": _parse_csv_arg(args.question_types),
            "llm_model": args.llm_model if args.llm_model is not None else (llm_client.model if llm_client is not None else None),
            "llm_max_tokens": args.llm_max_tokens,
            "llm_temperature": args.llm_temperature,
            "max_workers": args.max_workers,
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    prediction_path = args.output_dir / "per_sample_predictions.jsonl"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in per_sample_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    _print_summary(summary, modes)
    print(f"\nsummary: {summary_path}")
    print(f"per-sample: {prediction_path}")


if __name__ == "__main__":
    main()
