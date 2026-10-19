# 阶段 05：Stage R 特征级候选重排

## 阶段目标

这一轮不再继续强化训练数据，而是切到检索后处理：先保留已有强 student，再在 top-K 候选上训练轻量 feature reranker，最后用 residual blend 做安全融合。

## 方法与创新点

### 1. reranker 输入

在 [train_fashioniq_feature_reranker.py](C:\Users\qh\Desktop\CIR\phase_packages\05_stageR_feature_rerank\code\scripts\train_fashioniq_feature_reranker.py) 中，输入由五部分拼接：

```text
[q, c, q ⊙ c, |q - c|, s_base]
```

其中：

- `q`：query feature
- `c`：candidate feature
- `s_base`：原始检索分数

### 2. reranker 打分

模型先预测一个 residual，再与 base score 结合：

```text
r = MLP([q, c, q ⊙ c, |q - c|, s_base])
s_model = r + beta * s_base
```

实现位置：

- `return residual + self.base_scale * base_scores`

### 3. residual blend

验证时不直接用 reranker 分数替换 base score，而是做 blend：

```text
s_blend = s_base + lambda (s_model - s_base)
```

实现位置：

- [eval_fashioniq_feature_reranker.py](C:\Users\qh\Desktop\CIR\phase_packages\05_stageR_feature_rerank\code\scripts\eval_fashioniq_feature_reranker.py)
- `blended = base_scores + lam * (model_scores - base_scores)`

### 4. 训练损失

主损失是候选分类交叉熵：

```text
L_ce = CE(logits, y)
```

可选蒸馏项：

```text
L_distill = KL(softmax(s_model / T), softmax(s_base / T))
L_total = L_ce + lambda_distill * L_distill
```

## 代码入口

- asset 构建：
  - [build_fashioniq_rerank_assets.py](C:\Users\qh\Desktop\CIR\phase_packages\05_stageR_feature_rerank\code\scripts\build_fashioniq_rerank_assets.py)
  - [run_stageR_build_assets_hpc.sh](C:\Users\qh\Desktop\CIR\phase_packages\05_stageR_feature_rerank\code\scripts\run_stageR_build_assets_hpc.sh)
- reranker 训练：
  - [train_fashioniq_feature_reranker.py](C:\Users\qh\Desktop\CIR\phase_packages\05_stageR_feature_rerank\code\scripts\train_fashioniq_feature_reranker.py)
  - [run_stageR_feature_rerank_hpc.sh](C:\Users\qh\Desktop\CIR\phase_packages\05_stageR_feature_rerank\code\scripts\run_stageR_feature_rerank_hpc.sh)
- blend 评测：
  - [eval_fashioniq_feature_reranker.py](C:\Users\qh\Desktop\CIR\phase_packages\05_stageR_feature_rerank\code\scripts\eval_fashioniq_feature_reranker.py)

## 结果

结果汇总文件：

- [STAGER_FINAL_SUMMARY.json](C:\Users\qh\Desktop\CIR\phase_packages\05_stageR_feature_rerank\results\stageR_feature_rerank_20260423_final\STAGER_FINAL_SUMMARY.json)
- [STAGER_FINAL_REPORT.md](C:\Users\qh\Desktop\CIR\phase_packages\05_stageR_feature_rerank\results\stageR_feature_rerank_20260423_final\STAGER_FINAL_REPORT.md)

### 最佳 `mean R@10` 组合

| 类别 | 最优配置 |
| --- | --- |
| dress | `dress_toprank_v6 blend=0.07` |
| shirt | `shirt_toprank_v2 blend=0.04` |
| toptee | `toptee_toprank_v2 blend=0.03` |

最终：

- `mean R@10 = 62.1067`
- `mean R@50 = 80.9262`

### 最佳综合组合

| 类别 | 最优配置 |
| --- | --- |
| dress | `human_wide_v3 blend=0.05` |
| shirt | `shirt_toprank_v2 blend=0.04` |
| toptee | `toptee_toprank_v2 blend=0.03` |

最终：

- `mean R@10 = 62.0406`
- `mean R@50 = 81.0915`

### 相对上一轮最优 `StageP`

`StageP` 最优是 `61.3494 / 80.8952`。

因此：

- 最佳 `R@10` 组合：`+0.7573 / +0.0310`
- 最佳综合组合：`+0.6912 / +0.1963`

## 阶段结论

- 这是目前整个项目最强的一轮。
- 有效增益来自“轻量 rerank + 小 lambda 安全融合”，而不是直接替换 base score。
- 该阶段证明：在训练主干接近平台期时，top-K feature reranking 是更稳的提升手段。

