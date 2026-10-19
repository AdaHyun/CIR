# 本地资产审计

审计范围为当前可访问的 `/workspace/CIR`、相关 workspace 验证材料及可见备份目录；未访问 Windows `D:\Study\cjf` 或历史 HPC 路径。两个源包独立审计，源码无 AGENTS.md。完整发现记录见 `source_discovery.json`；来源与复制哈希见 `manifests/source_mapping.csv`。

## 两个来源

| 来源 | 实际发现 | 审计结论 |
|---|---|---|
| CIR-minimal-reproduction-20260919 | 170 文件，135,508,654 字节；9 个 val 资产、3 个 RR checkpoint、6 份完整查询排名 JSONL、51 张案例 PNG、源码及旧结果 | 缓存特征和旧 RR 可用；不具备端到端重训材料 |
| phase_packages | 6 阶段代码及轻量结果，含日志/err/out/history/指标/summary | 没有 .pt 权重/特征/Top-K；摘要不能当作实际训练资产 |

扫描包括非标准扩展名和压缩容器 magic。发现的 ZIP 容器为 12 个 PyTorch 文件本身，无额外可恢复训练资产的备份压缩包；没有发现源包外可见的 CIR 训练备份。未读的 Windows 磁盘不能据此判定不存在资产。所有源文件均复制到 archive，另将真实 val/checkpoint 复制到主动工作目录；相同名称跨包不覆盖。

## 实测 schema 和内容

| 类别 | val query | val gallery | Top-K | dtype | 状态 |
|---|---|---|---|---|---|
| dress | [2017,1024] | [2628,1024] | 2017×100 ID/score | float32 | 已验证可用 |
| shirt | [2038,1024] | [3089,1024] | 2038×100 ID/score | float32 | 已验证可用 |
| toptee | [1961,1024] | [2902,1024] | 1961×100 ID/score | float32 | 已验证可用 |

每个 .pt 是 dict：

- query：query_features Tensor、source_ids、target_ids；没有原始 modification text 或原生 query_id，新增稳定 query ID 包含类别/split/序号/source/target，明确区别于源数据原生 ID。
- gallery：features Tensor、image_ids；唯一显式 ID 定义特征行号，没有按相近向量猜映射。
- topk：sorted_ids、sorted_scores、source_ids、target_ids、target_ranks、target_in_topk、coverage；ID/score 为等长嵌套列表。
- checkpoint：model_state_dict、args、best_metrics；旧权重不含 optimizer/RNG/cursor，不能完整续训旧实验。

逐行检查有限值、单位范数、同维度、ID 唯一性、source/target 完整、query/topk 同顺序、候选唯一性、图库成员关系、参考排除、target 标记与排名一致；缓存分数与显式 ID 对应的真实点积最大误差均 <5e-7。三类图库 ID 集合严格等于该类全部 query 的 reference/target 并集，符合历史 `datasets.FashionIQ.get_test_data()` 的 val-split 规则。

详细 keys/shape/dtype/norm、checkpoint 每个张量 shape、全部 arguments、rank 分歧见 `runs/delivery-final/reports/audit-<category>.json`。12 个模型/特征文件均通过原始 SHA-256 清单校验。

## checkpoint 与训练规则

三类 RR 都 strict 加载到 FeatureReranker：query_proj/cand_proj 为 1024→768；pair 输入 4×768+1；MLP 768→1；包含可学习 base_scale。不是 CN/DQU backbone 权重。参数直接从 checkpoint args 确认：seed42、batch64、AdamW weight_decay .0001、无 distillation；dress epochs36/lr.001/dropout.2/k16/pool64，shirt/toptee epochs40/lr.0007/dropout.1/k8/pool16；val_k100。固定融合 .05/.04/.03 另由打包评估和历史 blend 记录核对。

旧 `train_fashioniq_feature_reranker.py` 在正目标插入处 `candidate_scores_final.append(1.0)`，其他负样本读缓存分数；推理所有候选用真实分数。新增 `src/cir_full/model.py::training_item` 对所有候选计算真实点积，并保留原模型结构。未改原代码或旧权重。

## 来源兼容性边界

- 原最小包的 query/gallery/topk 内部映射已验证；旧 RR 与该包冻结 val 资产的实际推理也验证通过。
- phase_packages 中 RR 类定义可与新模型结构对照，但没有该来源的大型资产可供跨包 feature-space 对齐。
- phase StageR asset_summary 指向 StageP v3_safe 的 `dqu_pfp_refcond`、ViT-H-14 backbone 和历史 open_clip 权重路径；同时列出 train coverage。这是**历史摘要证据**，没有 producer checkpoint/hash，不能确认最小包的冻结特征一定由同一字节版本生成。
- train 三件套缺失，train/val 是否同源仍未确认；程序要求 provenance 文件，不能以 shape/相同 summary 解除。
- 原始标注、完整图库图片、query 描述/关键词缓存未发现；README 中“已含标注”的旧描述与实际包内容不符，以实物为准。
- 51 张案例 PNG 的文件名是 case_a/b/c + 方法/名次，没有足够的图片 ID、query ID、文本和协议对照。归档保留，不把它们作为完整可复现 DQU 预测。

## 已知问题与实际评估

原始清单有 5 个源码文件 checksum 不符：code/datasets.py、code/scripts/run_fashioniq_mix_hpc.sh、code/src/datasets.py、code/src/test.py、code/src/utils.py。原始 bytes 和清单都保留，详见 `original_integrity.json`。原 run_verify.sh 还先生成 reproduced_metrics 再检查其旧 hash，导致第六项失败。没有改 hash 或关闭校验；新模型只在对应资产 hash 通过后加载。

旧 RR 重算：R@10=62.040561059683235，R@50=81.09148556455976，mR=71.5660233121215。CN 零融合路径：61.36592095949684 / 80.8951757783617；dress R@10 是1087/2017，原 coverage 为1086/2017。归档与当前路径的逐查询 rank 差异：CN dress14/shirt6/toptee9；旧RR dress1/shirt0/toptee0（整体旧 RR 命中数相同）。完整对照在 `archived_rankings_comparison.json`，没有强制对齐旧排名或删除查询。

## 状态分类

| 资产/能力 | 标记 | 说明 |
|---|---|---|
| 三类 val 特征/Top100/旧 RR | 已验证可用 | SHA + schema + 实际全查询评估 |
| 原包 5 个源码校验项 | 存在但有问题 | 原样保存，不用于主动入口 |
| train query/image/topk | 缺失 | 不是 val 的别名 |
| train Top-K（若恢复了同源特征/映射） | 可由现有格式重建 | 目前缺训练特征，尚不能重建 |
| CN/DQU producer checkpoint、完整图片/标注/查询缓存 | 必须另行索取 | 不从摘要重构 |
| CN/旧 RR 全查询 ID+score 导出 | 已验证可用 | 新生成，完整分母6016 |
| 完整 DQU→CN→新RR 图片链路 | 缺失/受阻 | DQU预测与原始图片/文本、新RR权重缺失 |

后续核查：所有29处CN查询rank差异的重排区间，缓存相似度跨度均为0（精确相等分数的重排），证据已加入archived_rankings_comparison.json；不因此改写归档或当前排序。
