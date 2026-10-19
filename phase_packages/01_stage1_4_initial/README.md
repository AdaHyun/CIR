# 阶段 01：初始 Stage 1-4

## 阶段目标

这是最早的完整闭环，目标是先把 `DQU` 跑通，再尝试把 `PFP` 和第一版 pseudo-triplet 增强接进来。

对应四个子阶段：

1. `Stage 1`：`DQU` baseline。
2. `Stage 2`：`DQU-PFP` 首版。
3. `Stage 3`：pseudo-triplet 构造与离线质量检查。
4. `Stage 4`：`human + filtered pseudo` 混训。

## 方法与创新点

### 1. 基础主干：DQU 融合查询

核心在 [model.py](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\code\src\model.py)。

设：

- 文本特征为 `q_t = normalize(f_text(t))`
- 参考图像特征为 `q_v = normalize(f_img(v))`

模型用动态门控做融合：

```text
c = W_c [q_t ; q_v] + b_c
s = sigma(W_s Dropout(c) + b_s)
q = s ⊙ q_t + (1 - s) ⊙ q_v
```

实现位置：

- `combined_feature = self.combiner_fc(...)`
- `dynamic_scaler = self.scaler_fc(...)`
- `query = dynamic_scaler * textual_query + (1 - dynamic_scaler) * visual_query`

### 2. 排序损失

训练损失是 batch 内的 ranking NCE：

```text
X = q · t^T
L_rank = CE(alpha X, y)
```

其中：

- `q` 是查询特征
- `t` 是目标图像特征
- `alpha = self.loss_weight`
- `y` 是对角标签

实现位置：

- `ranking_nce_loss()` in [model.py](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\code\src\model.py)

### 3. Stage 2：PFP 首版

这一版在 `DQU` 上叠加了 prompt-style 文本分支。它仍然沿用 DQU 的图文融合主干，但文本特征会先经过 prompt 构造和 routing，再参与最终融合。

关键类同样在 [model.py](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\code\src\model.py)：

- `DQU_PFP_CIR`

### 4. Stage 3：第一版 pseudo-triplet

通过 [generate_fashioniq_pseudo.py](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\code\scripts\generate_fashioniq_pseudo.py) 构造伪三元组，再依据分数过滤，形成 `Stage 4` 的训练池。

## 代码入口

- 训练入口：
  - [run_fashioniq_stage1.sh](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\code\scripts\run_fashioniq_stage1.sh)
  - [run_fashioniq_stage2.sh](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\code\scripts\run_fashioniq_stage2.sh)
  - [run_fashioniq_stage4.sh](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\code\scripts\run_fashioniq_stage4.sh)
- pseudo 生成：
  - [generate_fashioniq_pseudo.py](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\code\scripts\generate_fashioniq_pseudo.py)
- 汇总：
  - [aggregate_fashioniq_metrics.py](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\code\scripts\aggregate_fashioniq_metrics.py)

## 结果

### Stage 1：DQU baseline

结果文件：

- [fashioniq_stage1_summary.json](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\results\stage1\fashioniq_stage1_summary.json)

| 类别 | R@10 | R@50 |
| --- | ---: | ---: |
| dress | 54.3381 | 76.1527 |
| shirt | 61.5800 | 79.6369 |
| toptee | 65.6298 | 85.2116 |
| mean | 60.5160 | 80.3337 |

### Stage 2：DQU-PFP 首版

结果文件：

- [fashioniq_stage2_summary.json](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\results\stage2\fashioniq_stage2_summary.json)

| 类别 | R@10 | R@50 |
| --- | ---: | ---: |
| dress | 54.0902 | 75.4090 |
| shirt | 61.7764 | 79.7350 |
| toptee | 63.9084 | 84.8491 |
| mean | 59.9250 | 79.9977 |

### Stage 3：pseudo-triplet 离线结果

结果文件：

- [pseudo_dress_summary.json](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\results\stage3\pseudo_dress_summary.json)
- [pseudo_shirt_summary.json](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\results\stage3\pseudo_shirt_summary.json)
- [pseudo_toptee_summary.json](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\results\stage3\pseudo_toptee_summary.json)

| 类别 | 保留数 / 原始数 |
| --- | --- |
| dress | 2604 / 61001 |
| shirt | 2694 / 59194 |
| toptee | 1631 / 61018 |

### Stage 4：human + filtered pseudo

结果文件：

- [fashioniq_stage4_summary.json](C:\Users\qh\Desktop\CIR\phase_packages\01_stage1_4_initial\results\stage4\fashioniq_stage4_summary.json)

| 类别 | R@10 | R@50 |
| --- | ---: | ---: |
| dress | 54.8339 | 76.1527 |
| shirt | 62.3150 | 80.3729 |
| toptee | 64.7474 | 85.3666 |
| mean | 60.6321 | 80.6308 |

## 阶段结论

- `Stage 2` 相对 `Stage 1` 下降，说明首版 `PFP` 接法没有形成有效增益。
- `Stage 4` 相对 `Stage 2` 有恢复，但提升不大，说明第一版 pseudo 质量和混训方式都还不够强。
- 这个阶段最大的价值是把 `DQU -> prompt -> pseudo -> 混训` 的基本实验链跑通了。

