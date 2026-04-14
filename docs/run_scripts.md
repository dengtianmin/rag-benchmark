评估提取骨架的方法
/home/huang/miniconda3/envs/paper_benchmark/bin/python scripts/eval_skeleton_predictors.py \
  --dataset outputs/two_file_demo/benchmark_dataset.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --modes oracle,stub_predicted,llm_predicted \
  --sample-count 100 \
  --output-dir /tmp/skeleton_eval_smoke


评估rewrite的方法
python scripts/run_ours_retrieval_lab.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --limit 100 \
  --max-workers 10 \
  --skeleton-mode oracle \
  --rewrite-modes original,splicing,sparse_llm,dense_llm,hybrid_llm \
  --retrieval-modes hybrid \
  --scorer-modes plain \
  --output-dir outputs/experiments/ours_rewrite_compare_5 \
  --overwrite \
  --llm-rewrite-max-tokens 128 

分析 rewrite 在 relation_driven 下的失败原因
python scripts/run_ours_retrieval_lab.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --limit 100 \
  --max-workers 10 \
  --skeleton-mode oracle \
  --rewrite-modes original,splicing,sparse_llm,dense_llm,hybrid_llm \
  --retrieval-modes hybrid \
  --scorer-modes plain,relation_driven \
  --output-dir outputs/experiments/ours_rewrite_compare_trace \
  --overwrite \
  --llm-rewrite-max-tokens 128 \
  --llm-rewrite-temperature 0.0


聚合 stage1 / filtering / hybrid branch trace
python scripts/analyze_rewrite_failure_modes.py \
  --input-dir outputs/experiments/ours_rewrite_compare_trace \
  --output-dir outputs/experiments/ours_rewrite_compare_trace_analysis


重点看这些输出
- outputs/experiments/ours_rewrite_compare_trace/combos/*/predictions.jsonl
- outputs/experiments/ours_rewrite_compare_trace_analysis/per_combo_analysis.json
- outputs/experiments/ours_rewrite_compare_trace_analysis/focus_comparisons.json
- outputs/experiments/ours_rewrite_compare_trace_analysis/report.md


抽样查看 dense / sparse / hybrid 在 relation_driven 下的失败样本
python3 - <<'PY'
import json
from collections import defaultdict
from pathlib import Path

base = Path("outputs/experiments/ours_rewrite_compare_trace/combos")
combos = [
    "dense_llm__hybrid__relation_driven",
    "sparse_llm__hybrid__relation_driven",
    "hybrid_llm__hybrid__relation_driven",
]
out_dir = Path("outputs/experiments/ours_rewrite_compare_trace_analysis/failure_samples")
out_dir.mkdir(parents=True, exist_ok=True)

for combo in combos:
    path = base / combo / "predictions.jsonl"
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]
    failures = [row for row in rows if not row.get("gold_in_final_topk")]
    buckets = defaultdict(list)
    for row in failures:
        buckets[str(row.get("miss_type") or "unknown")].append(row)
    sampled = []
    miss_types = sorted(buckets)
    while len(sampled) < min(20, len(failures)) and any(buckets.values()):
        for miss_type in miss_types:
            if buckets[miss_type] and len(sampled) < 20:
                sampled.append(buckets[miss_type].pop(0))
    out_path = out_dir / f"{combo}.failure_sample_20.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(combo, len(sampled), out_path)
PY
