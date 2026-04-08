from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from benchmark_host.utils.jsonl import write_jsonl

from rewrite_lab.reporting.html_report import write_html_report
from rewrite_lab.schema import RewriteCandidate, RewriteLabRecord, RewriteLabSummary


def _format_table(rows: list[dict[str, object]], keys: list[str]) -> str:
    if not rows:
        return "_No data._"
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join(["---"] * len(keys)) + " |"
    body = ["| " + " | ".join(str(row.get(key, "")) for key in keys) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _representative_samples(records: list[RewriteLabRecord], limit: int = 2) -> dict[str, list[RewriteLabRecord]]:
    grouped: dict[str, list[RewriteLabRecord]] = defaultdict(list)
    for record in records:
        for tag in record.bucket_tags:
            if len(grouped[tag]) < limit:
                grouped[tag].append(record)
    return grouped


def write_report_bundle(
    *,
    output_dir: Path,
    candidates: list[RewriteCandidate],
    records: list[RewriteLabRecord],
    summaries: list[RewriteLabSummary],
    write_html: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "rewrites.jsonl", [candidate.to_dict() for candidate in candidates])
    write_jsonl(output_dir / "pairwise_eval.jsonl", [record.to_dict() for record in records])

    metrics_payload = {summary.strategy_name: summary.metrics for summary in summaries}
    bucket_payload = {summary.strategy_name: summary.bucket_summary for summary in summaries}
    (output_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "bucket_summary.json").write_text(
        json.dumps(bucket_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    overview_rows = []
    question_type_rows = []
    for summary in summaries:
        overview_rows.append(
            {
                "rewriter": summary.strategy_name,
                "samples": summary.sample_count,
                "rewritten_hit": round(summary.metrics.get("rewritten_hit@5", summary.metrics.get("rewritten_hit@3", 0.0)), 4),
                "win_rate": round(summary.metrics.get("win_rate", 0.0), 4),
                "loss_rate": round(summary.metrics.get("loss_rate", 0.0), 4),
                "non_inferiority": round(summary.metrics.get("non_inferiority_rate", 0.0), 4),
                "entity_preservation": round(summary.metrics.get("entity_preservation_rate", 0.0), 4),
                "noise_rate": round(summary.metrics.get("noise_rate", 0.0), 4),
            }
        )
        for question_type, payload in summary.by_question_type.items():
            question_type_rows.append(
                {
                    "rewriter": summary.strategy_name,
                    "question_type": question_type,
                    "samples": payload.get("sample_count", 0),
                    "rewritten_hit": round(float(payload.get("rewritten_hit_rate", 0.0)), 4),
                    "win_rate": round(float(payload.get("win_rate", 0.0)), 4),
                    "loss_rate": round(float(payload.get("loss_rate", 0.0)), 4),
                }
            )

    representative = _representative_samples(records)
    recommendation = max(
        summaries,
        key=lambda item: (item.metrics.get("non_inferiority_rate", 0.0), item.metrics.get("win_rate", 0.0), -item.metrics.get("loss_rate", 0.0)),
        default=None,
    )

    lines = [
        "# Rewrite Lab Report",
        "",
        "## Overall Metrics",
        _format_table(
            overview_rows,
            ["rewriter", "samples", "rewritten_hit", "win_rate", "loss_rate", "non_inferiority", "entity_preservation", "noise_rate"],
        ),
        "",
        "## By Question Type",
        _format_table(question_type_rows, ["rewriter", "question_type", "samples", "rewritten_hit", "win_rate", "loss_rate"]),
        "",
        "## Bucket Summary",
    ]
    for summary in summaries:
        lines.append(f"### {summary.strategy_name}")
        lines.append("")
        lines.append(json.dumps(summary.bucket_summary, ensure_ascii=False, indent=2))
        lines.append("")

    lines.append("## Representative Samples")
    for tag, examples in representative.items():
        lines.append(f"### {tag}")
        lines.append("")
        for record in examples:
            lines.append(f"- `{record.strategy_name}` `{record.question_id}`")
            lines.append(f"  original: {record.original_question}")
            lines.append(f"  rewrite: {record.rewritten_query}")
            lines.append(f"  retrieval: {json.dumps(record.retrieval_metrics, ensure_ascii=False)}")
        lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    if recommendation is None:
        lines.append("No samples were evaluated.")
    else:
        lines.append(
            f"Recommended for e2e integration candidate: `{recommendation.strategy_name}`. "
            f"It has the best non-inferiority and win/loss balance in this offline rewrite-only lab."
        )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    if write_html:
        write_html_report(report_path, output_dir / "report.html")
