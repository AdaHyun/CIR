# 缺失材料与最小恢复格式

没有从远端找回或下载材料；下述路径均为工程内的目标位置。历史服务器路径只在 source archive 中作查找线索，不能用作运行默认值。

## RR 重训必需

每个 category=dress/shirt/toptee 需要：

1. `assets/<category>/<category>_train_query_features.pt`：dict，`query_features`=[N,1024] float Tensor，已L2归一化，`source_ids`/`target_ids` 各 N 个字符串，按 query 行对应；可选唯一 `query_ids`。每个 target/source 必须能在该 split 图库映射中找到。
2. `..._train_image_features.pt`：dict，`features`=[M,1024] float Tensor、`image_ids` M 个唯一字符串，行号唯一对应。train 与 val 图库允许不同。
3. `..._train_topk.pt`：dict，`sorted_ids`、`sorted_scores`=[N,K] 同顺序列表；source_ids/target_ids 与 query **逐项相等**；候选属于本图库、排除参考图、按真实点积分数降序；pool 至少 dress64 / shirt16 / toptee16，且足够不同负样本。建议保存 query_ids、target_in_topk；可选完整图库 target_ranks，不能用截断结果猜完整 rank。
4. `assets/<category>/provenance.json`：train 与 val 的真实 producer checkpoint SHA-256、相同 feature_definition（模型/backbone/预处理/查询构造/归一化），各自每个输入文件的 SHA-256。样例在 configs，未知字段不能用推测填满。图库协议可分别记录，不要求 train/val 图库相同。现有 val 的 producer hash 也缺证据，只有历史路径摘要。

若仅缺 Top-K，现有 build-topk 可分块重建，显式 ID 排除参考图；新增产物在独立 run 下，配置 asset_files 引用，不覆盖已有冻结特征。若 train query/image 特征也缺，则不能重建真实资产。

## 重导出训练特征需要的材料

匹配现有 val 特征的 **CN/StageP 实际 checkpoint 完整权重和 SHA**、模型配置/代码版本、OpenCLIP ViT-H-14 权重、FashionIQ 授权图片/标注/划分、生成 image captions 和修正文本/关键词/视觉查询写入缓存、完全一致的图像预处理和 query 构造。历史线索：StageR summary 指向 `stageP_bankdistill_20260423/v3_safe/<category>/...best_model.pt` 和 `stageR_feature_rerank_20260423/assets/`。这些只证明当时的配置记录，不能保证源文件仍存在。

原始 exporter 已完整复制在 `archive/phase_packages/05_stageR_feature_rerank/code/scripts/build_fashioniq_rerank_assets.py`，对应 datasets/model 也同目录保留。它调用 .cuda()、依赖 dataset 私有查询构造/缓存、使用历史宽松 load_state_dict；**不作为已验证的主动导出入口**，不以 checkpoint+图片足够为前提。材料找回后先单独校验完整模型加载/预处理，建立明确的 schema 和 producer manifest，再运行昂贵导出。本次没有修改冻结特征或重训 DQU/RC/CN。

## 可视化最小材料

- DQU：优先找回全查询真实排名，包含 query 身份、reference/target、Top-K ID/score、图库协议及 checkpoint/feature 来源。可通过 dqu-export 导入并与同一 query-union 协议核对。仅 aggregate metrics/history/旧案例 PNG 不够。
- 若 DQU 无预测但有特征，找回 matching query/gallery features 与显式标签映射，用相同图库协议重新生成排名；若仅有 checkpoint，则仍需模型/backbone、数据/查询缓存和预处理。
- CN/旧 Proposed：完整 ID/score 排名已由本次导出；缺原始修改文本和完整真实图片。需要 `data/fashioniq/captions/cap.<category>.val.json`（candidate、target、captions）及 `resized_image/<category>/<ID>.jpg`。同 reference/target 对存在多个 query 时必须提供显式逐查询映射，不猜行序。
- 新 Proposed：依赖真实 train 特征、同源证据、完成 true-similarity RR 训练后的 best.pt；旧 RR 权重不能冒充新模型。
- Top100 外目标仅能写 outside Top-100/下界101，不能从 coverage 或其它模型的排序补全。

## 影响

缺训练材料阻断 true-similarity 训练、新模型评估和训练成本测量；不影响旧 RR/CN 缓存特征评估。缺 DQU 和原图/文本阻断完整新图片可视化；不影响保存 CN/旧RR 全查询 ID/score 记录。当前代码、检查器、配置、恢复测试与 CPU RR 推理成本记录已交付，未把“受阻”改写为“已完成”。

# CIR-FULL分支新增路线补充（以此为准）

上文说明的是精确恢复历史冻结资产所缺材料；**现在完整新路线不依赖这些丢失旧权重/训练feature**。已实现 `scripts/full.py pipeline`，从公开ViT-H-14及真实FashionIQ训练新的DQU/RC/teacher bank/CN，再由同一CN best导出train/val并训练新RR。数据、caption替代策略、模型权威下载校验与人工导入步骤见docs/A100_RUNBOOK.md。当前云端真正缺的是授权图片、公开大模型文件和作者原缓存（可显式用免费本地生成替代），而不是“只能等待找回旧CN才有代码可跑”。没有把实现代码等同于A100完成结果。
