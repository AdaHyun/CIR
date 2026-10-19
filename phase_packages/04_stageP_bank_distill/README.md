# 阶段 04：Stage P 多教师 bank 蒸馏

## 阶段目标

这一轮的目标是把 pseudo 路线进一步升级为 multi-teacher bank 监督，而不是只把 pseudo 当成额外训练样本。

核心思路：

1. 构建多个 teacher 的候选 bank。
2. 对 student 引入软目标分布监督。
3. 对明显冲突的样本引入 conflict margin 惩罚。
4. 在 `Stage D` 主干上做蒸馏增强。

## 方法与创新点

### 1. soft bank distillation

在 [train.py](C:\Users\qh\Desktop\CIR\phase_packages\04_stageP_bank_distill\code\src\train.py) 中，soft bank 的核心损失是：

```text
z_k = <q, t_k> / tau
p_teacher = normalize(w_k)
L_soft = KL(log_softmax(z), p_teacher)
```

实现位置：

- `kl_per_sample = F.kl_div(log_probs, target_probs, reduction="none").sum(dim=-1)`

### 2. conflict bank loss

对于标记为 conflict 的样本，引入 margin loss：

```text
L_conflict = max(0, m - sim(q, t_pos) + sim(q, t_neg))
```

实现位置：

- `raw_loss = F.relu(conflict_margin - pos_sim + neg_sim)`

### 3. 总损失

```text
L_total = L_rank + lambda_soft * L_soft + lambda_conflict * L_conflict
```

实现位置：

- `total_loss = sum(loss_dict.values())`

## 代码入口

- bank 构建：
  - [build_multiteacher_bank.py](C:\Users\qh\Desktop\CIR\phase_packages\04_stageP_bank_distill\code\scripts\build_multiteacher_bank.py)
  - [run_build_multiteacher_bank_hpc.sh](C:\Users\qh\Desktop\CIR\phase_packages\04_stageP_bank_distill\code\scripts\run_build_multiteacher_bank_hpc.sh)
- bank distill 训练：
  - [run_fashioniq_bankdistill_hpc.sh](C:\Users\qh\Desktop\CIR\phase_packages\04_stageP_bank_distill\code\scripts\run_fashioniq_bankdistill_hpc.sh)
  - [train_fashioniq_bankdistill.slurm](C:\Users\qh\Desktop\CIR\phase_packages\04_stageP_bank_distill\code\slurm\train_fashioniq_bankdistill.slurm)

## 结果

结果汇总文件：

- [FINAL_STAGEP_SUMMARY.json](C:\Users\qh\Desktop\CIR\phase_packages\04_stageP_bank_distill\results\stageP_bankdistill_20260423\FINAL_STAGEP_SUMMARY.json)
- [FINAL_STAGEP_SUMMARY.md](C:\Users\qh\Desktop\CIR\phase_packages\04_stageP_bank_distill\results\stageP_bankdistill_20260423\FINAL_STAGEP_SUMMARY.md)

### 主要结果

| 配置 | dress R@10 / R@50 | shirt R@10 / R@50 | toptee R@10 / R@50 | mean R@10 | mean R@50 |
| --- | --- | --- | --- | ---: | ---: |
| Stage D human-only | - | - | - | 61.1446 | 80.8781 |
| StageP v3 | 53.8423 / 77.1443 | 63.1992 / 80.1766 | 67.0066 / 85.3646 | 61.3494 | 80.8952 |
| StageP v4b | 53.8919 / 76.2023 | 61.7272 / 80.0785 | 65.7828 / 84.7527 | 60.4673 | 80.3445 |
| StageP v5 dress-only | 53.0491 / 76.2519 | - | - | - | - |

### 相对 Stage D

`StageP v3` 相比 `Stage D`：

- `mean R@10 +0.2048`
- `mean R@50 +0.0171`

## 阶段结论

- 这是一条真正比 `Stage D` 更强的训练线。
- 说明把 pseudo 从“样本扩充”升级成“teacher bank 监督”是有效的。
- 这个阶段是后续 `StageR` 之前最有价值的一次增益。

