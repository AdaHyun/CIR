# 阶段 02：老师主线 Stage A-E

## 阶段目标

这是按照老师意见重构后的主线。核心目标不是继续堆系统，而是把问题收缩成一条可验证的强基线闭环：

1. `Stage A`：冻结并验证强 baseline。
2. `Stage B`：做 pseudo 预算、过滤和样本分析。
3. `Stage C`：补齐关键对照实验。
4. `Stage D`：重做文本分支，形成更可信的 `PFP(new)`。
5. `Stage E`：把最优文本分支与 pseudo 方案组合。

## 方法与创新点

### 1. DQU 基线保留

基础公式仍然是 DQU 的动态融合：

```text
q = s ⊙ q_t + (1 - s) ⊙ q_v
```

### 2. Stage D：reference-conditioned 文本分支

这一轮的关键变化是：不再用第一版那种模板式 prompt mixture，而是做更靠近真实 CIR 条件建模的 reference-conditioned 文本分支。其核心思想是：

- 文本特征不再独立看待；
- 文本表示显式受参考图像上下文影响；
- 最终仍接回 DQU 主干做融合与排序。

对应源码在 [model.py](C:\Users\qh\Desktop\CIR\phase_packages\02_stageA_E_mainline\code\src\model.py)。

### 3. Stage C/E：pseudo mixing 策略

这一阶段开始显式引入 mixing ratio 控制，并把 pseudo 注入时机从“直接混合”改为“可配置比例和课程式注入”，为后续 `Stage N` 奠定基础。

## 代码入口

- 主实验驱动：
  - [run_fashioniq_experiment.sh](C:\Users\qh\Desktop\CIR\phase_packages\02_stageA_E_mainline\code\scripts\run_fashioniq_experiment.sh)
  - [run_fashioniq_experiment_hpc.sh](C:\Users\qh\Desktop\CIR\phase_packages\02_stageA_E_mainline\code\scripts\run_fashioniq_experiment_hpc.sh)
- 基线冻结：
  - [build_baseline_snapshot.py](C:\Users\qh\Desktop\CIR\phase_packages\02_stageA_E_mainline\code\scripts\build_baseline_snapshot.py)
  - [freeze_next_stage_base.py](C:\Users\qh\Desktop\CIR\phase_packages\02_stageA_E_mainline\code\scripts\freeze_next_stage_base.py)
- pseudo 分析：
  - [prepare_pseudo_budget.py](C:\Users\qh\Desktop\CIR\phase_packages\02_stageA_E_mainline\code\scripts\prepare_pseudo_budget.py)
  - [build_pseudo_reports.py](C:\Users\qh\Desktop\CIR\phase_packages\02_stageA_E_mainline\code\scripts\build_pseudo_reports.py)

## 结果

结果汇总文件：

- [final_summary_20260418.json](C:\Users\qh\Desktop\CIR\phase_packages\02_stageA_E_mainline\results\local_sync_20260418\artifacts\final_summary_20260418.json)
- [FINAL_PIPELINE_REPORT.md](C:\Users\qh\Desktop\CIR\phase_packages\02_stageA_E_mainline\results\local_sync_20260418\FINAL_PIPELINE_REPORT.md)

| 子阶段 | 配置 | mean R@10 | mean R@50 |
| --- | --- | ---: | ---: |
| Stage A | DQU baseline | 61.0292 | 80.1985 |
| Stage C | E2 all pseudo | 59.9681 | 80.3474 |
| Stage C | E3 filtered pseudo | 58.9209 | 79.8498 |
| Stage C | E4 filtered pseudo + curriculum | 60.9817 | 80.4854 |
| Stage D | `DQU-PFP(new) human-only` | 61.1446 | 80.8781 |
| Stage E | `DQU-PFP(new) + filtered pseudo + curriculum` | 60.9779 | 80.4781 |

## 阶段结论

- 这一轮真正成立的是 `Stage D`，它第一次把内部最好结果推到了 `61.1446 / 80.8781`。
- `all pseudo` 和 `filtered pseudo` 仍然没有形成稳定正增益。
- 说明问题不在“有没有 pseudo”，而在 pseudo 的训练价值和混训方式。
- 从后续阶段看，这一轮最重要的遗产就是：确定了 `Stage D` 是后续所有增强路线的 student 主干。

