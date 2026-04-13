# Skeleton Evaluation

本次改动为项目增加了新的骨架抽取模式 `llm_predicted`，用于和现有 `oracle`、`stub_predicted` 一起做 skeleton-level 离线评测。

## 范围

- 仅评测 question skeleton 抽取质量
- 不做端到端问答评测
- 不做 token cost 评估
- 不改 retrieval / generation 主逻辑

## 支持的骨架模式

- `oracle`
- `stub_predicted`
- `llm_predicted`

其中：

- `oracle` 直接读取标注骨架
- `stub_predicted` 继续沿用现有 heuristic / graph-aligned 逻辑
- `llm_predicted` 先基于图索引召回候选实体和关系，再调用 LLM 输出简单 JSON schema

## llm_predicted 输出 schema

模型必须输出严格 JSON：

```json
{
  "entities": ["..."],
  "relations": ["..."],
  "constraints": ["..."]
}
```

当前实现会对结果做：

- `strip`
- 去空串
- 去重
- entity / relation 轻量 anchor 到候选项

无法 anchor 的项会保留原值，并记录到 `details.unmatched_entities` / `details.unmatched_relations`。

## 离线评测脚本

脚本路径：

`scripts/eval_skeleton_predictors.py`

主要能力：

- 支持 `oracle,stub_predicted,llm_predicted`
- 支持 `--sample-count`
- 支持 `--question-types`
- 支持 `--max-workers`
- 终端显示 `tqdm` 进度条
- 输出 `summary.json`
- 输出 `per_sample_predictions.jsonl`

## 默认模型配置

`llm_predicted` 默认优先读取 `.env` 中的 DashScope 配置：

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `DASHSCOPE_MODEL`

当前推荐模型：

`qwen2.5-7b-instruct-1m`

如果显式传入 `--llm-model`，则以命令行为准。

## 运行示例

评测 `full_run` 的前 100 个样本，并发 10 worker：

```bash
/home/huang/miniconda3/envs/paper_benchmark/bin/python scripts/eval_skeleton_predictors.py \
  --dataset outputs/full_run/benchmark_dataset.jsonl \
  --knowledge artifacts/full_run/knowledge_extraction.jsonl \
  --modes oracle,stub_predicted,llm_predicted \
  --sample-count 100 \
  --max-workers 10 \
  --output-dir /tmp/skeleton_eval_full_run_100
```

如果传入的 `--knowledge` 路径不存在，脚本会自动尝试 fallback 到仓库中可用的知识抽取文件。

## 输出指标

`summary.json` 包含：

- `overall`
- `by_question_type`
- `diagnostics`

当前已实现的核心指标：

- entity precision / recall / f1
- relation precision / recall / f1
- constraint precision / recall / f1
- skeleton exact match
- `json_parse_success_rate`
- `empty_prediction_rate`
- `anchor_success_rate`

当前聚合方式为 `micro`。

## 已知限制

- 当前只做单轮 LLM 调用，不做 self-refine
- anchor 规则仍是轻量 overlap 映射，不含复杂别名词典
- constraints 暂时只保留字符串列表
- 若运行环境缺少项目基础依赖，脚本本身仍可能因依赖缺失而无法启动
