# FashionIQ 下一阶段推进方案执行结果

## 1. 执行范围

本次已经在 `HPC3` 上完成并拉回本地的阶段包括：

1. `Stage A`：干净的 `DQU baseline`
2. `Stage B`：`pseudo-triplet` teacher、过滤与重构
3. `Stage C`：`DQU + pseudo` 四组关键对照及两轮迭代
4. `Stage D`：`DQU-PFP(new)` human-only
5. `Stage E`：`DQU-PFP(new) + filtered pseudo + curriculum`

本地总结果汇总文件：

- `artifacts/final_summary_20260418.json`

## 2. 当前最佳结果

当前整条路线里最好的结果是：

- `Stage D = DQU-PFP(new) human-only`

对应指标：

| 方法 | mean R@10 | mean R@50 |
| --- | ---: | ---: |
| Stage A DQU baseline | 61.0292 | 80.1985 |
| Stage C best: E4 filtered pseudo + curriculum | 60.9817 | 80.4854 |
| Stage D DQU-PFP(new) human-only | 61.1446 | 80.8781 |
| Stage E DQU-PFP(new) + filtered pseudo + curriculum | 60.9779 | 80.4781 |

结论：

- `Stage D` 是当前全流程里的最佳配置。
- `Stage D` 相比 `Stage A baseline`：
  - `mean R@10 +0.1154`
  - `mean R@50 +0.6796`
- `Stage E` 没有超过 `Stage D`，说明当前这版 `pseudo` 接到 `DQU-PFP(new)` 上以后仍然会带来拖累。

## 3. Stage C 结论

`Stage C` 的四组关键对照结果如下：

| 实验 | mean R@10 | mean R@50 |
| --- | ---: | ---: |
| E1 DQU human-only | 61.0292 | 80.1985 |
| E2 DQU + all pseudo | 59.9681 | 80.3474 |
| E3 DQU + filtered pseudo | 58.9209 | 79.8498 |
| E4 DQU + filtered pseudo + curriculum | 60.9817 | 80.4854 |

直接结论：

1. `all pseudo` 没有带来提升。
2. `filtered pseudo` 单独接入也没有超过 baseline。
3. `curriculum` 明显有价值。
4. 但当前 `pseudo` 路线整体还没有在 `mean R@10` 上稳定超过 baseline。

两轮附加迭代：

- `E4 curriculum start=5`：`mean R@10 = 60.6608`
- `E4 simple pseudo + curriculum`：`mean R@10 = 60.4438`

它们都没有超过 `E4`。

## 4. 关键诊断

从 `history.jsonl` 可见：

- `E4 dress` 的最佳 epoch 出现在 pseudo 注入前
- `E4 shirt` 的最佳 epoch 也出现在 pseudo 注入前
- 只有 `E4 toptee` 在 pseudo 注入后出现了最优点

这说明：

1. 当前 `pseudo` 对 `toptee` 有帮助
2. 当前 `pseudo` 对 `dress / shirt` 仍有干扰
3. `curriculum` 的主要作用是“减轻 pseudo 伤害”，而不是让 pseudo 在所有类别上都稳定变成正增益

## 5. Stage D 结论

`DQU-PFP(new)` 的 human-only 结果：

| 类别 | R@10 | R@50 |
| --- | ---: | ---: |
| dress | 54.7348 | 76.7476 |
| shirt | 62.6104 | 80.4711 |
| toptee | 66.0887 | 85.4156 |
| average | 61.1446 | 80.8781 |

这说明模型线成立：

- `reference-conditioned flexible prompt` 这条线本身是正增益
- 它比当前 `pseudo` 线更稳定

## 6. Stage E 结论

`DQU-PFP(new) + filtered pseudo + curriculum` 的结果：

| 类别 | R@10 | R@50 |
| --- | ---: | ---: |
| dress | 54.1894 | 76.4502 |
| shirt | 62.7576 | 80.0785 |
| toptee | 65.9867 | 84.9057 |
| average | 60.9779 | 80.4781 |

这说明：

- 组合实验没有超过 `Stage D`
- 当前最强的模型分支一旦接上当前 `pseudo`，整体仍会被拖慢

## 7. 当前最合理的项目结论

按现在这批完整实验，最稳妥的结论是：

1. `DQU baseline` 已经被干净复现并固定下来。
2. `DQU-PFP(new)` 这条模型线成立，并且是当前最优主线。
3. `pseudo-triplet` 不是完全没用，但当前只在部分品类上体现出正增益。
4. `curriculum` 是必要的，但它还不足以让当前 `pseudo` 路线整体超过强 baseline。
5. 因此，下一轮如果继续推进，主线应该是：
   - 以 `Stage D` 作为当前最佳模型
   - 继续专门修 `dress / shirt` 的 pseudo 质量，而不是再重复当前组合训练

## 8. 本地结果位置

关键文件已经拉回本地：

- `artifacts/stageA_baseline/`
- `artifacts/stageB_reports/`
- `artifacts/stageC_results/`
- `artifacts/stageD_results/`
- `artifacts/stageE_results/`
- `artifacts/history_csv/`
- `checkpoints/stageC_*/`
- `checkpoints/stageD_refcond/`
- `checkpoints/stageE_combo/`
- `slurm/`

建议直接先看：

1. `artifacts/final_summary_20260418.json`
2. `artifacts/stageC_results/dqu_pseudo_ablation_table.md`
3. `artifacts/stageD_results/dqupfp_new_human_only_metrics.json`
4. `artifacts/stageE_results/stageE_combo_summary.json`
