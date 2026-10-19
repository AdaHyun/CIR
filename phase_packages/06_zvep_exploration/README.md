# 阶段 06：ZVEP keep/change 路线探索

## 阶段目标

这是一条探索性路线，目标是把修改文本拆成更接近编辑语义的 keep/change 结构，再测试这种表达能否优于现有主干。

## 方法与创新点

### 1. keep/change 建模

相关实现位于：

- [model_structured.py](C:\Users\qh\Desktop\CIR\phase_packages\06_zvep_exploration\code\src\model_structured.py)
- [query_decomposition.py](C:\Users\qh\Desktop\CIR\phase_packages\06_zvep_exploration\code\src\query_decomposition.py)

思想是把文本分解为：

- keep / retain：应保留的语义
- change / modify：应发生的编辑

再把这两部分同参考图像上下文一起编码。

### 2. H2 / H3 两种变体

- `H2 = dqu_zvep_cir`
- `H3 = dqu_zvep_cir_reg`

它们都基于 keep/change 语义构造新的文本查询，只是正则与约束不同。

## 代码入口

- [run_fashioniq_zvep_hpc.sh](C:\Users\qh\Desktop\CIR\phase_packages\06_zvep_exploration\code\scripts\run_fashioniq_zvep_hpc.sh)
- [train_fashioniq_zvep_generic.slurm](C:\Users\qh\Desktop\CIR\phase_packages\06_zvep_exploration\code\slurm\train_fashioniq_zvep_generic.slurm)

## 结果

### smoke 结果

目录：

- [zvep_smoke2](C:\Users\qh\Desktop\CIR\phase_packages\06_zvep_exploration\results\zvep_smoke2)

### 正式 H2 / H3 结果

目录：

- [zvep_h_full](C:\Users\qh\Desktop\CIR\phase_packages\06_zvep_exploration\results\zvep_h_full)

| 配置 | dress R@10 / R@50 | shirt R@10 / R@50 | toptee R@10 / R@50 | mean R@10 | mean R@50 |
| --- | --- | --- | --- | ---: | ---: |
| H2 = `dqu_zvep_cir` | 43.6787 / 70.2033 | 52.8459 / 73.1109 | 57.7766 / 78.9903 | 51.4338 | 74.1015 |
| H3 = `dqu_zvep_cir_reg` | 43.8275 / 69.9058 | 52.6497 / 73.3562 | 57.9296 / 78.9393 | 51.4689 | 74.0671 |

## 阶段结论

- `H2/H3` 都能完整训练、验证并落盘，说明这条结构路线在工程上是通的。
- 但正式结果明显低于当前主线，因此它没有成为后续继续推进的主干。
- 另外，`Z2-Z4` 中真正依赖 MLLM 在线生成/验证的部分当时没有形成完整可运行闭环，因此本阶段被归档为“探索路线”，而不是最终主线。

