# CIR 分阶段代码与结果归档

本目录按阶段整理了本项目已经完成的主要实验闭环。每个阶段文件夹都包含两部分：

- `code/`：该阶段对应的完整可执行代码子集。公共核心源码 `src/` 在每个阶段都完整保留；阶段特有的 `scripts/`、`configs/`、`slurm/` 单独收录。
- `results/`：该阶段的轻量结果资产，只保留日志、指标、history、汇总、伪样本与报告，不包含大模型权重和大型特征文件。

## 阶段索引

| 阶段目录 | 阶段主题 | 该阶段最优结果 |
| --- | --- | --- |
| `01_stage1_4_initial` | 初始 `Stage 1-4`：DQU baseline、PFP 首版、pseudo-triplet 首版 | `Stage 4 mean R@10 = 60.6321`, `mean R@50 = 80.6308` |
| `02_stageA_E_mainline` | 老师主线 `Stage A-E`：强基线、refcond 文本分支、关键对照实验 | `Stage D mean R@10 = 61.1446`, `mean R@50 = 80.8781` |
| `03_stageN_structured_pseudo` | 第二轮结构化 pseudo 扩展：retain/modify、curriculum/static/dynamic mixing | `N4 v4 mean R@10 = 61.0318`, `mean R@50 = 80.6649` |
| `04_stageP_bank_distill` | multi-teacher bank + soft/conflict distillation | `StageP v3 mean R@10 = 61.3494`, `mean R@50 = 80.8952` |
| `05_stageR_feature_rerank` | feature-level reranking + residual blend | 最佳 `R@10` 组合：`62.1067 / 80.9262`；最佳综合组合：`62.0406 / 81.0915` |
| `06_zvep_exploration` | ZVEP keep/change 路线探索 | `H3 mean R@10 = 51.4689`, `mean R@50 = 74.0671` |

## 远端回拉说明

打包前已经核对过 `HPC3` 上的代码目录，并补回了本地缺失的远端代码文件：

- `configs/zvep_base.env`
- `scripts/run_fashioniq_stage1_hpc.sh`
- `scripts/stageR_build_assets.slurm`
- `scripts/stageR_train_rerank.slurm`
- `slurm/train_fashioniq_stage1_hpc.slurm`

另外，`ZVEP` 的正式 `H2/H3` 指标文件也已从 `HPC3` 回拉到：

- `06_zvep_exploration/results/zvep_h_full/`

## 当前总览

如果只看已经完成的全部阶段，当前全项目最强结果来自 `StageR`：

- 最佳 `mean R@10` 组合：`62.1067 / 80.9262`
- 最佳综合 `mean R@10 + mean R@50` 组合：`62.0406 / 81.0915`

也就是说，按目前这套代码和结果归档，后续继续推进时，优先参考：

1. `02_stageA_E_mainline` 的 `Stage D` 主干；
2. `04_stageP_bank_distill` 的 teacher-bank 蒸馏经验；
3. `05_stageR_feature_rerank` 的 feature rerank 与 residual blend。

