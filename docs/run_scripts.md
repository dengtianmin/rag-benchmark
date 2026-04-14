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

分析实验结果
python scripts/analyze_rewrite_failure_modes.py \
  --input-dir outputs/experiments/ours_rewrite_compare_trace \
  --output-dir outputs/experiments/ours_rewrite_compare_trace_analysis


运行 Ours-Ch4（支持关闭补偿或切换补偿策略）
python scripts/run_ours_ch4.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --limit 100 \
  --max-workers 10 \
  --output-dir outputs/experiments/ours_ch4_full_run

关闭文本补偿
python scripts/run_ours_ch4.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --limit 100 \
  --max-workers 10 \
  --disable-text-compensation \
  --output-dir outputs/experiments/ours_ch4_no_comp

使用新式文本补偿
python scripts/run_ours_ch4.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --top-k 5 \
  --limit 100 \
  --max-workers 10 \
  --text-compensation-strategy new_compensation \
  --output-dir outputs/experiments/ours_ch4_new_comp

三组文本补偿对比实验
python scripts/run_text_compensation_compare.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --sections artifacts/full_run/markdown_sections.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --limit 100 \
  --max-workers 10 \
  --generator-provider dashscope \
  --generator-model qwen2.5-7b-instruct-1m \
  --output-dir outputs/experiments/text_compensation_compare

说明
- 对比脚本固定使用 `hybrid` 检索与 `relation_driven` 检索器打分。
- 对比脚本会输出 `no_text_compensation`、`legacy_compensation`、`new_compensation` 三组结果。
- 生成模型默认强制设置为 `qwen2.5-7b-instruct-1m`，其 API Key 和 Base URL 从 `.env` 读取。

