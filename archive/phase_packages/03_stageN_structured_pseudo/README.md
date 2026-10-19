# 阶段 03：Stage N 结构化 pseudo 扩展

## 阶段目标

这一轮的目标是把 pseudo 从“简单筛样本”升级成“结构化 retain/modify 数据增强”，再系统比较不同 mixing 策略。

核心动作：

1. 把修改文本拆成更结构化的 retain/modify 形式。
2. 生成 structured pseudo pool。
3. 比较 `static / curriculum / dynamic_weighted` 三种混训方式。
4. 做 high-quality 子池再训练。

## 方法与创新点

### 1. 结构化 pseudo

生成逻辑入口：

- [generate_fashioniq_structured_pseudo.py](C:\Users\qh\Desktop\CIR\phase_packages\03_stageN_structured_pseudo\code\scripts\generate_fashioniq_structured_pseudo.py)
- [generate_structured_queries.py](C:\Users\qh\Desktop\CIR\phase_packages\03_stageN_structured_pseudo\code\scripts\generate_structured_queries.py)

这一阶段不再只保留自然语言字符串，而是显式保留：

- retain 信息
- modify 信息
- 对应的结构化属性与方向

### 2. mixing 比例调度

调度逻辑在 [data_mixer.py](C:\Users\qh\Desktop\CIR\phase_packages\03_stageN_structured_pseudo\code\src\data_mixer.py)。

三种主要策略：

```text
static:
  r(e) = r_max

curriculum:
  r(e) = r_max * progress(e)

dynamic_weighted:
  r(e) = r_max * (0.5 + 0.5 * progress(e))
```

其中 `progress(e)` 是训练进度，`r_max = pseudo_max_ratio`。

同时使用 `WeightedRandomSampler`，让高质量 pseudo 在采样中有更高概率。

## 代码入口

- [run_fashioniq_mix_hpc.sh](C:\Users\qh\Desktop\CIR\phase_packages\03_stageN_structured_pseudo\code\scripts\run_fashioniq_mix_hpc.sh)
- [train.py](C:\Users\qh\Desktop\CIR\phase_packages\03_stageN_structured_pseudo\code\src\train.py)
- [data_mixer.py](C:\Users\qh\Desktop\CIR\phase_packages\03_stageN_structured_pseudo\code\src\data_mixer.py)

## 结果

结果汇总文件：

- [stageN_summary_20260419.json](C:\Users\qh\Desktop\CIR\phase_packages\03_stageN_structured_pseudo\results\stageN_sync_20260419\stageN_summary_20260419.json)
- [stageN_hpc_progress_20260419.md](C:\Users\qh\Desktop\CIR\phase_packages\03_stageN_structured_pseudo\results\stageN_hpc_progress_20260419.md)

### pseudo 池规模

| 池 | dress | shirt | toptee |
| --- | ---: | ---: | ---: |
| raw structured pseudo | 2412 | 2757 | 2245 |
| high-quality 子集 | 652 | 599 | 510 |

### N4 混训结果

| 配置 | mean R@10 | mean R@50 |
| --- | ---: | ---: |
| Stage D human-only | 61.1446 | 80.8781 |
| N4 v2 static | 60.8782 | 80.3623 |
| N4 v2 curriculum | 60.8776 | 80.4613 |
| N4 v2 dynamic | 60.6965 | 80.5931 |
| N4 v3 curriculum | 60.9030 | 80.8291 |
| N4 v4 highq curriculum | 61.0318 | 80.6649 |

## 阶段结论

- 结构化 pseudo 的数据组织比第一版更干净、更可控。
- `curriculum` 和 `high-quality` 子集确实比直接混训更合理。
- 但最终仍未超过 `Stage D human-only`。
- 这一轮的贡献是把“pseudo 质量”和“mixing 调度”拆开验证清楚，而不是单纯堆更多样本。

