# Ours 方法与背景文档差距对照

本文档用于对照论文背景文档中的第 4 章方法设想，与当前项目中 `Ours-Ch4` 的实际实现之间的差异，并给出后续可演进方向。

相关参考：

- [project_background.md](/home/paper/Benchmark/docs/project_background.md)
- [ours_ch4.py](/home/paper/Benchmark/src/pipelines/ours_ch4.py)
- [skeleton_extractor.py](/home/paper/Benchmark/src/modules/skeleton_extractor.py)
- [relation_driven_retriever.py](/home/paper/Benchmark/src/modules/relation_driven_retriever.py)
- [text_compensator.py](/home/paper/Benchmark/src/modules/text_compensator.py)

## 三列表对照

| 文档设想 | 当前实现 | 可改进点 |
|---|---|---|
| 知识图谱前移到“查询理解与检索目标建模”阶段，作为结构先验参与检索。 | `Ours-Ch4` 已按“骨架抽取 -> 关系驱动检索 -> 文本补偿”串联成完整 pipeline。 | 让图谱不只是提供 section 命中提示，而是真正参与 query understanding、候选生成和过滤决策。 |
| 构建问题骨架 `S(q)={E(q),R(q),C(q)}`，实体、关系、约束共同建模。 | `oracle` 模式直接读取数据集标注；`stub_predicted` 主要依赖字符串 overlap 从图索引中找实体和关系，约束基本缺失。 | 增加真实 skeleton predictor，尤其补上约束抽取与标准化，而不是只靠 oracle 或启发式 stub。 |
| 检索重写应是“关系感知”的结构化重写，而不是普通扩展。 | 当前重写只是把原问题与 `entities`、`relations`、`constraints` 拼接成一个文本查询。 | 改成模板化或学习式 rewrite，区分主实体、关系、限定条件，并针对不同检索器优化表达。 |
| 检索目标从“相似文本召回”转向“关系一致证据定位”。 | 当前通过 `dense_score + relation_score + constraint_score` 的线性融合实现。 | 从启发式加权走向更强的结构匹配，例如 relation path、一致性约束、学习排序或图路径打分。 |
| 关系一致性要体现在实体关系链、约束条件、事实链条层面。 | 实体命中 section 加分，关系命中 section 加分，约束依赖文本 overlap。 | 把“关系一致性”从 section-level hit 升级为 chain/path-level consistency，避免仅靠词面命中。 |
| 文本证据补偿是在结构先验引导下补足图谱未显式表达的细节。 | 当前补偿触发依赖规则：`requires_text_compensation`、`explanation` 题型、或候选太少。 | 改成更细的补偿判定，例如基于候选覆盖度、约束缺失、答案置信度或多阶段校验触发。 |
| 文本补偿与结构约束应协同，而不是简单并列。 | 当前补偿主要是原问题再检索加共享实体扩展，再统一按分数排序。 | 让补偿显式围绕缺失关系、缺失约束、缺失解释片段来回填，而不是泛化 backfill。 |
| 方法应体现为统一框架，而不是模块堆叠。 | 工程上已经形成统一 pipeline，并输出 skeleton、relation-driven、compensation、rerank 等 trace。 | 继续统一模块接口，使 skeleton、relation retrieval、compensation 能单独替换为真实模型。 |
| 创新点是“图谱真正成为检索先验”。 | 当前项目文档已明确说明：`Ours-Ch4` 目前是“骨架版本实现”，便于后续替换真实模块。 | 从 prototype 过渡到真实可学习、可泛化模块，这是当前最核心的演进方向。 |
| 文档本身更偏方法论，不强调工程实验设计。 | 当前项目已补齐消融、rerank、trace、dense/hybrid 检索接口等实验能力。 | 这些工程能力应保留，并反向服务论文实验，把“方法差距”量化成可验证结论。 |

## 总结判断

当前项目中的 `Ours-Ch4` 与背景文档在整体框架上是一致的，已经覆盖了：

- 骨架抽取
- 检索重写
- 关系驱动检索
- 文本证据补偿

但它更准确地说是一个“可运行的工程化骨架版实现”，而不是背景文档中完整的论文方法。当前版本的主要特点是：

- 有统一 pipeline，可跑通实验与消融；
- 已具备图提示融合、文本补偿、rerank、trace 等工程能力；
- 关键模块仍以 oracle、启发式规则、线性加权为主。

因此，当前 `Ours-Ch4` 更适合作为论文方法的实验原型与验证底座。若要进一步贴近背景文档中的方法设想，优先级最高的工作包括：

1. 将 skeleton 从 oracle/stub 升级为真实预测模块。
2. 将 query rewrite 从拼接式增强升级为结构化关系感知重写。
3. 将 relation-driven retrieval 从启发式融合升级为更强的结构一致性建模。
4. 将 text compensation 从规则触发升级为面向“缺失证据类型”的定向补偿机制。

## 建议使用方式

这个文档适合用于以下场景：

- 向导师或评审解释“当前实现离论文目标还有多远”；
- 规划后续模块迭代优先级；
- 在实验章节中说明 prototype 与 full method 的边界；
- 作为后续补全文档或实现真实模块时的 checklist。
