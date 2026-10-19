# ZVEP-CIR 下一阶段可执行方案

## 1. 文档目的

本文档将 [CIR_ZVEP_CIR_new_route_20260418.docx](C:\Users\qh\Desktop\CIR\next_plan\CIR_ZVEP_CIR_new_route_20260418.docx) 转成后续可直接执行的工程方案。目标不是复述概念，而是明确：

- 当前真实起点是什么
- 新路线到底要做什么
- 每个阶段具体改哪些代码、产出哪些文件、跑哪些实验
- 每一阶段什么情况下继续，什么情况下停止回修

后续训练、监控、复盘、迭代，都以这份 markdown 为唯一执行口径。

## 2. 当前起点

### 2.1 已有最好结果

当前我们已经跑完的关键结果如下，后续全部新实验都以它们为起点，不再回到旧 token-diff pseudo 路线做小修小补。

| 实验 | mean R@10 | mean R@50 | 结论 |
| --- | ---: | ---: | --- |
| Stage A `DQU baseline` | 61.0292 | 80.1985 | 固定基线 |
| Stage C `DQU + all pseudo` | 59.9681 | 80.3474 | `all pseudo` 拉低 `R@10` |
| Stage C `DQU + filtered pseudo` | 58.9209 | 79.8498 | 旧 pseudo 过滤后仍无净增益 |
| Stage C `DQU + filtered pseudo + curriculum` | 60.9817 | 80.4854 | curriculum 只能减伤 |
| Stage D `DQU-PFP(new) human-only` | 61.1446 | 80.8781 | 当前最优主线 |
| Stage E `Stage D + filtered pseudo + curriculum` | 60.9779 | 80.4781 | 旧 pseudo 接到最强模型仍拖累整体 |
| N4 `v4 highq curriculum` | 61.0318 | 80.6649 | 目前最好的 structured pseudo mixing 版本，仍未超过 Stage D |

### 2.2 当前最硬结论

1. 当前真正成立的是 `human-only` 模型线，不是旧 pseudo 数据线。
2. 旧 pseudo 的主要问题不在阈值，而在 supervision 单位和训练接入方式。
3. 下一阶段不能再围绕 `token diff + quality score + delayed curriculum` 继续调。
4. 新路线必须先证明 `human-only` 的 query 建模更强，再把新 pseudo 接回训练。

## 3. 新路线定义

### 3.1 方法名

推荐正式使用：

`ZVEP-CIR`

全称：

`Zero-shot Verified Edit Programs for Composed Image Retrieval`

### 3.2 一句话定义

先用 zero-shot MLLM 把候选图像对转成结构化 edit program，并通过多重验证得到 clean/soft/ambiguity 三层 pseudo bank；再用 keep/change-aware query encoder 和 relevance-aware training，把高质量 pseudo 转成对 supervised CIR 的稳定增益。

### 3.3 与旧路线的本质区别

| 维度 | 旧路线 | 新路线 |
| --- | --- | --- |
| pseudo 单位 | caption token diff | verified edit program |
| teacher 起点 | 当前 supervised model | zero-shot candidate mining + MLLM reasoning |
| 数据分层 | 保留 / 丢弃 | clean / soft / ambiguity |
| 训练接入 | pseudo 基本都当硬正样本 | clean 强监督，soft 弱监督，ambiguity 作 hard negatives |
| 目标 | 让 pseudo 看起来更干净 | 让 pseudo 真正变成有训练价值的数据结构 |

## 4. 工程前提

### 4.1 保持不变的部分

- backbone：`ViT-H-14 + laion2B-s32B-b79K`
- 主底座：当前最优 `DQU_PFP_REFCOND_CIR`
- benchmark：先固定 `FashionIQ val-split`
- 训练协议：单次训练，不做 5-seed

### 4.2 必需依赖

这条路线有一个硬前提：`Z2-Z4` 需要可用的 MLLM 生成/验证通道。没有这一项，就只能先做 `H` 系列 human-only 改造，不能真正完成 ZVEP 全链路。

当前执行时必须满足以下二选一：

1. 本地可调用的远程 API
2. 可离线部署并可批量推理的本地/集群多模态模型

如果这两项都没有，则：

- 可以先实现所有代码接口和缓存协议
- 可以先跑 `human-only` 与 `candidate mining`
- 但不能声称 ZVEP 全路线已经完成

## 5. 目录与文件约定

### 5.1 代码目录

主工程目录固定为：

`C:\Users\qh\Desktop\CIR\project_stage1_dqu_baseline`

远端目录固定为：

`/data/user/hqi881/CIR/project_stage1_dqu_baseline`

### 5.2 新增文件建议

| 文件 | 作用 | 是否第一批必须 |
| --- | --- | --- |
| `scripts/mine_zs_candidates.py` | zero-shot pair mining | 是 |
| `scripts/generate_edit_programs.py` | 生成结构化 edit program | 是 |
| `scripts/verify_edit_programs.py` | schema / purity / cycle / ambiguity / direction 验证 | 是 |
| `scripts/build_pseudo_banks.py` | clean / soft / ambiguity 三层 bank 构建 | 是 |
| `scripts/run_fashioniq_zvep_hpc.sh` | HPC 训练入口 | 是 |
| `src/model_zvep.py` | keep/change-aware 模型逻辑 | 是 |
| `src/losses_zvep.py` | clean/soft/rank loss | 是 |
| `src/data_mixer_zvep.py` | 分层 pseudo mixer | 是 |
| `configs/zvep/*.env` | 每轮实验配置 | 是 |
| `scripts/run_reasoning_rerank.py` | top-K rerank | 否，最后冲顶再做 |

### 5.3 结果目录

本地结果固定放在：

`C:\Users\qh\Desktop\CIR\results\zvep_*`

远端结果固定放在：

`/data/user/hqi881/CIR/project_stage1_dqu_baseline/checkpoints/zvep_*`

每轮实验必须落下：

- `metrics_best.json`
- `history.jsonl`
- `train.log`
- `config.env`
- `report.md`

## 6. 阶段总览

新路线按下面顺序推进，不允许跳步：

1. `Z0`：冻结当前最佳基线
2. `H1`：先做人类标注数据上的 keep/change-aware human-only 改造
3. `Z2`：candidate mining
4. `Z3`：MLLM 生成 edit programs
5. `Z4`：verification + pseudo bank 构建
6. `P1-P3`：新 pseudo 接回训练
7. `R1-R3`：relevance-aware ranking
8. `T1-T2`：可选 rerank

核心纪律：

- human-only 没站稳，不进 pseudo
- new pseudo 没超过 old pseudo，不进 relevance-aware loss
- retriever 没站稳，不上 reranker

## 7. Z0：冻结当前最佳基线

### 目标

把当前 `Stage D = DQU-PFP(new) human-only` 固定成唯一 parent baseline。

### 输入

- 当前最优结果：`61.1446 / 80.8781`
- 现有模型变体：`dqu_pfp_refcond`

### 动作

1. 固定一份 `zvep_base.env`
2. 固定一份 `baseline_snapshot.json`
3. 固定一份 `baseline_report.md`

### 输出

- `configs/zvep/zvep_base.env`
- `results/zvep_z0_base/baseline_snapshot.json`
- `results/zvep_z0_base/baseline_report.md`

### 完成标准

- 后续所有实验都从这份 parent config 派生
- 指标口径不再变化

## 8. H1：human-only keep/change-aware query modeling

### 目标

先在完全不依赖 pseudo 的条件下，证明新的 query 建模本身能站稳。

### 核心思想

把当前文本输入从：

- `reference + modification + visual`

扩展成：

- `reference + keep + change + visual`

也就是让模型显式知道：

- 什么应该保留
- 什么才是变化核心

### H 系列实验

| 实验 ID | 配置 | 目的 |
| --- | --- | --- |
| `H0` | 当前 `Stage D` | 固定起点 |
| `H1` | `H0 + keep/change decomposition` | 验证拆分语义本身是否有用 |
| `H2` | `H1 + four-token ref/keep/change/visual encoder` | 验证新 query encoder |
| `H3` | `H2 + keep consistency / conflict regularization` | 验证显式保留约束 |

### 代码改造

1. `src/datasets.py`
   - 额外返回：
     - `reference_caption`
     - `keep_instruction`
     - `change_instruction`

2. `src/query_decomposition.py`
   - 从当前 structured parser 升级为 keep/change-oriented parser
   - 先支持 `FashionIQ`

3. `src/model_zvep.py`
   - 新增 `dqu_zvep_cir`
   - conditioning token 固定为：
     - `reference_text`
     - `keep_text`
     - `change_text`
     - `visual_feature`

4. `src/train.py`
   - 支持 `model_variant=dqu_zvep_cir`

### 成功标准

- `H2` 或 `H3` 至少有一条稳定超过 `61.1446 / 80.8781`

### Stop / Go

- `Stop`：如果 `H2/H3` 都不超过 `H0`，不要继续 pseudo，先回修人类数据建模
- `Go`：只有 `H-best > H0`，才进入 `Z2-Z4`

## 9. Z2：zero-shot candidate mining

### 目标

先找出值得交给 MLLM 的 ref-target pair，压缩推理预算。

### 原则

- 只在同一类别内配对
- 不做全量两两枚举
- 候选数量受控

### 第一版策略

1. 用 OpenCLIP / 当前 backbone 图像编码器建立 index
2. 每张 reference 检索 `top-40` 或 `top-50` 邻居
3. 只保留 image similarity 落在中间窗口的 pair
4. 结合 caption/粗属性做一次轻量筛选
5. 每个 reference 最多保留 `6-10` 个 pair

### 输出

- `data/FashionIQ/zvep_candidates/dress.jsonl`
- `data/FashionIQ/zvep_candidates/shirt.jsonl`
- `data/FashionIQ/zvep_candidates/toptee.jsonl`

### 成功标准

- 候选规模可控
- 近重复与明显无关 pair 被明显压下
- 可解释地覆盖多种属性变化

## 10. Z3：MLLM 生成 structured edit programs

### 目标

把旧 pseudo 的 supervision 单位从 token diff 升级成结构化 edit program。

### 三步生成协议

1. 对 reference 图做 `object-only description`
2. 对 target 图做 `object-only description`
3. 用两张图加两份描述输出严格 JSON

### JSON schema

至少包含：

- `pair_id`
- `category`
- `reference_id`
- `target_id`
- `reference_desc`
- `target_desc`
- `keep`
- `change`
- `add`
- `remove`
- `conflict_slots`
- `short_instruction`
- `full_instruction`
- `ambiguous`
- `generator_conf`

### category-aware ontology

`FashionIQ` 第一版 slot 固定如下：

- `dress`
  - `color_primary`
  - `color_secondary`
  - `pattern`
  - `sleeve_length`
  - `sleeve_style`
  - `neckline`
  - `fit`
  - `length`
  - `material`
  - `decoration`
- `shirt/toptee`
  - `color_primary`
  - `color_secondary`
  - `pattern`
  - `sleeve_length`
  - `neckline`
  - `collar`
  - `hood`
  - `graphic`
  - `pocket`
  - `fit`
  - `material`

### 输出

- `data/FashionIQ/zvep_edit_programs/raw/*.jsonl`

### 成功标准

- 生成结果是可解析 JSON，不是自由文本
- 基本不再出现 `camera/wearing/jeans/person/background` 一类污染词

## 11. Z4：verification 与 pseudo bank 构建

### 目标

不再做“生成完就信”，而是把伪样本分成不同训练角色。

### 必做验证

1. `Schema check`
2. `Object purity check`
3. `Self-consistency check`
4. `Reverse-cycle check`
5. `Target verification check`
6. `Conflict check`
7. `Direction / retrieval check`
8. `Ambiguity check`

### 综合质量分数

第一版统一记为：

`Q = 0.15*S_schema + 0.15*S_object_purity + 0.15*S_self_consistency + 0.15*S_reverse_cycle + 0.10*S_conflict_free + 0.15*S_direction_margin + 0.15*S_ambiguity_uniqueness`

### 三层 bank

- `clean`: `Q >= 0.80`
- `soft`: `0.65 <= Q < 0.80`
- `reject`: 其余

另外单独维护：

- `ambiguity / hard-negative bank`

### 输出

- `data/FashionIQ/zvep_banks/clean/*.jsonl`
- `data/FashionIQ/zvep_banks/soft/*.jsonl`
- `data/FashionIQ/zvep_banks/ambiguity/*.jsonl`
- `results/zvep_z4/verification_report.md`
- `results/zvep_z4/bank_stats.json`

### 成功标准

- clean 样本纯度明显高于旧 pseudo
- soft 样本不是硬丢弃，而是保留为弱监督
- ambiguity 样本可单独分析和训练

## 12. P 系列：新 pseudo 接回训练

### 目标

验证新 pseudo 的生成单位是否真的比旧 pseudo 更有训练价值。

### P 系列实验

| 实验 ID | 配置 | 目的 |
| --- | --- | --- |
| `P0` | `H-best + 旧 token-diff pseudo` | 旧路线对照 |
| `P1` | `H-best + ZVEP clean pseudo` | 新 pseudo 单位是否更强 |
| `P2` | `P1 + soft pseudo` | soft pseudo 是否提供额外监督 |
| `P3` | `P2 + ambiguity bank` | 分层 bank 是否更稳定 |

### 训练接入原则

- human：保持当前最稳 ranking loss
- clean：作为强监督，允许较高权重
- soft：不当唯一硬正样本，做弱监督或蒸馏
- ambiguity：不当正样本，可作 hard negative / mask / relevance 建模

### 输出

- `results/zvep_p1/*.json`
- `results/zvep_p2/*.json`
- `results/zvep_p3/*.json`

### 成功标准

- `P1` 至少超过 `P0`
- `P2/P3` 不能再出现 dress/shirt 明显拖垮

## 13. R 系列：relevance-aware ranking

### 目标

解决 false negatives 和局部相似样本误杀的问题，主要改善 `R@10`。

### R 系列实验

| 实验 ID | 配置 | 要回答的问题 |
| --- | --- | --- |
| `R0` | `P-best + 原始 ranking` | 旧目标上限 |
| `R1` | `R0 + clean/soft 分权` | 分层 pseudo 是否必要 |
| `R2` | `R1 + hard-negative relevance loss` | false negative 是否是关键瓶颈 |
| `R3` | `R2 + ambiguous negative masking` | 邻域错杀是否被修复 |

### 第一版损失

`L_total = L_human + lambda_clean * L_clean + lambda_soft * L_soft + lambda_rank * L_rank`

建议起点：

- `lambda_clean = 1.0`
- `lambda_soft = 0.25`
- `lambda_rank = 0.20`

### 成功标准

- `R-best` 超过 `P-best`
- 提升主要体现在 `R@10`

## 14. T 系列：可选 reasoning reranker

### 目标

只在 retriever 已经站稳后，冲最后一截。

### 实验

| 实验 ID | 配置 | 作用 |
| --- | --- | --- |
| `T0` | 最佳 retriever | 主方法 |
| `T1` | `T0 + top-50 reasoning rerank` | 看是否还有一截可拉 |
| `T2` | `T0 + top-100 reasoning rerank` | 看 top-K 与成本关系 |

### 纪律

- reranker 只能是加分项
- 不能用 reranker 掩盖 retriever 没站稳的问题

## 15. 首轮超参数建议

| 项目 | 起点 | 建议 sweep |
| --- | --- | --- |
| backbone | 与 Stage D 一致 | 第一版不换 |
| pseudo warmup | 2-4 epoch | `2 / 3 / 4` |
| human:clean:soft | `3:1:1` | `2:1:1 / 3:1:1` |
| `lambda_clean` | `1.0` | `0.7 / 1.0 / 1.2` |
| `lambda_soft` | `0.25` | `0.15 / 0.25 / 0.40` |
| `lambda_rank` | `0.20` | `0.10 / 0.20 / 0.30` |
| clean threshold | `0.80` | `0.78 / 0.80 / 0.82` |
| soft threshold | `0.65-0.80` | `0.60-0.80` |

## 16. 运行顺序

真正开跑时，严格按这个顺序执行：

1. 固定 `Z0`
2. 跑 `H0/H1/H2/H3`
3. 只要 `H-best > H0`，开始 `Z2`
4. 完成 `Z3/Z4` 后，先跑 `P0/P1`
5. 只有 `P1 > P0`，再上 `P2/P3`
6. 只有 `P-best` 已稳定优于旧 pseudo 路线，才上 `R1-R3`
7. 最后视情况跑 `T1/T2`

## 17. Stop / Go 规则

### Stop

1. 如果 `human-only` 新模型都不能超过 `Stage D`，停止进入 pseudo。
2. 如果 `clean pseudo` 接入后 dress/shirt 明显掉点，停止调 curriculum，回到 verification 和 bank 构建。
3. 如果 reranker 有效但 retriever 不稳，停止 reranker，回修训练主线。

### Go

1. `H-best > H0`
2. `P1 > P0`
3. `R-best > P-best`
4. 只有主模型已形成净提升，才考虑迁移到 `Shoes` 或 `CIRR`

## 18. 第一批立刻开工项

按当前工程状态，第一批必须先做下面这些，不要一开始就试图完成整条路线：

1. 冻结 `Z0` 基线配置
2. 新增 `dqu_zvep_cir`，先跑 `H1/H2`
3. 把 `datasets.py` 扩展到支持 `keep_instruction` / `change_instruction`
4. 写 `mine_zs_candidates.py`
5. 定义 edit program JSON schema 和缓存目录
6. 确认 MLLM 生成/验证通道是否可用

## 19. 完成标准

本路线“完成”不等于只是把代码写出来，而是满足下面四条：

1. `human-only` 新模型稳定超过当前 `Stage D`
2. 新 pseudo 不再像旧 pseudo 那样一接入就拖垮 dress/shirt
3. 最终最好模型相对 `Stage D` 有净提升
4. 所有阶段的日志、history、metrics、报告都已落盘并拉回本地

## 20. 当前执行判断

按现状，下一步最合理的真正起点不是直接跑 pseudo，而是：

1. 先冻结 `Stage D`
2. 立即实现并验证 `H1/H2`
3. 同时核实 MLLM 可用性

如果 `H1/H2` 站不稳，ZVEP 后半条链不应继续推进。

