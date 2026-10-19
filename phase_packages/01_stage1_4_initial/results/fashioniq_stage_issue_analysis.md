# FashionIQ 阶段结果问题分析与下一步建议

## 1. 结论先说

当前结果可以概括成三句话：

1. `Stage 1 -> Stage 2` 下降，说明当前这版 `DQU-PFP` 没有稳定超过 `DQU baseline`。
2. `Stage 2 -> Stage 4` 有恢复，说明 `filtered pseudo-triplets` 是有用的，但增益还不强。
3. `Stage 4` 只比 `Stage 1` 略高，说明当前真正的瓶颈不在“有没有伪样本”，而在“文本改进是否正确”“伪样本是否足够强”“训练接入是否足够有效”。

换句话说，当前最好不要把重点继续放在“多跑几次现有 Stage 2/4”，而应该先修正结构和实验口径。

---

## 2. 当前实验到底测了什么

### 2.1 Stage 1

- 目标：复现 `DQU-CIR` 在 `FashionIQ VAL-Split` 上的 baseline
- 可信汇总文件：
  - `results/stage1/fashioniq_stage1_summary.json`

### 2.2 Stage 2

- 目标：在 `DQU` 骨架上加入当前实现的 `DQU-PFP`
- 当前实现的本质不是“完整 PFP 复现”，而是：
  - 保留 DQU 的视觉分支
  - 把文本分支改成了一个 `4 个 prompt template + router + residual` 的轻量 prompt mixture
  - 文件位置：
    - `project_stage1_dqu_baseline/src/model.py`

### 2.3 Stage 3

- 目标：基于 `Stage 2 checkpoint` 生成并筛选 `pseudo-triplets`
- 当前筛选后的伪样本是：
  - `dress`: 2604
  - `shirt`: 2694
  - `toptee`: 1631

### 2.4 Stage 4

- 目标：把 `filtered pseudo-triplets` 接入 `DQU-PFP`
- 当前实际跑的是：
  - `human + filtered pseudo + delayed curriculum`
  - 不是完整老师要求的四组对照
  - 还缺：
    - `human + all pseudo`
    - `human + filtered pseudo`（不带 curriculum）

---

## 3. 全部关键数据

## 3.1 DQU 原论文结果 vs 我们当前结果

对比口径：`FashionIQ VAL-Split`

| 方法 | dress R@10 | dress R@50 | shirt R@10 | shirt R@50 | toptee R@10 | toptee R@50 | mean R@10 | mean R@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DQU 原论文 | 57.63 | 78.56 | 62.14 | 80.38 | 66.15 | 85.73 | 61.97 | 81.56 |
| 我们 Stage 1 | 54.3381 | 76.1527 | 61.5800 | 79.6369 | 65.6298 | 85.2116 | 60.5160 | 80.3337 |
| 我们 Stage 4 | 54.6356 | 76.3510 | 61.5800 | 80.1766 | 65.6808 | 85.3646 | 60.6321 | 80.6308 |

差值：

- `Stage 1` 相比论文：
  - mean `R@10`: `-1.4540`
  - mean `R@50`: `-1.2263`
- `Stage 4` 相比论文：
  - mean `R@10`: `-1.3379`
  - mean `R@50`: `-0.9292`

结论：

- 目前我们已经接近 DQU 论文结果，但还没有追平。
- 最大 gap 仍然在 `dress`。

---

## 3.2 我们三个主要阶段的可信汇总结果

可信来源：

- `results/stage1/fashioniq_stage1_summary.json`
- `results/stage2/fashioniq_stage2_summary.json`
- `results/stage4/fashioniq_stage4_summary.json`

| 阶段 | dress R@10 | dress R@50 | shirt R@10 | shirt R@50 | toptee R@10 | toptee R@50 | mean R@10 | mean R@50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1: DQU | 54.3381 | 76.1527 | 61.5800 | 79.6369 | 65.6298 | 85.2116 | 60.5160 | 80.3337 |
| Stage 2: DQU-PFP | 54.1894 | 76.5493 | 61.3837 | 79.0481 | 64.2019 | 84.3957 | 59.9250 | 79.9977 |
| Stage 4: DQU-PFP + filtered pseudo | 54.6356 | 76.3510 | 61.5800 | 80.1766 | 65.6808 | 85.3646 | 60.6321 | 80.6308 |

---

## 3.3 Stage 1 -> Stage 2 的变化

| 类别 | R@10 变化 | R@50 变化 |
| --- | ---: | ---: |
| dress | -0.1487 | +0.3966 |
| shirt | -0.1963 | -0.5888 |
| toptee | -1.4278 | -0.8159 |
| 平均 | -0.5910 | -0.3360 |

观察：

- `dress` 基本持平，略降。
- `shirt` 小幅下降。
- `toptee` 降幅最大，是 Stage 2 整体下降的主要来源。

---

## 3.4 Stage 2 -> Stage 4 的变化

| 类别 | R@10 变化 | R@50 变化 |
| --- | ---: | ---: |
| dress | +0.4462 | -0.1983 |
| shirt | +0.1963 | +1.1286 |
| toptee | +1.4788 | +0.9689 |
| 平均 | +0.7071 | +0.6330 |

观察：

- `filtered pseudo` 对 `shirt` 和 `toptee` 帮助更明显。
- `dress` 的恢复较弱，尤其 `R@50` 还略降。

---

## 3.5 Stage 1 -> Stage 4 的净变化

| 类别 | R@10 变化 | R@50 变化 |
| --- | ---: | ---: |
| dress | +0.2975 | +0.1983 |
| shirt | +0.0000 | +0.5397 |
| toptee | +0.0510 | +0.1530 |
| 平均 | +0.1162 | +0.2970 |

这说明：

- `Stage 4` 的提升是真有，但幅度不大。
- 它更像是“把 Stage 2 的损失补回来并略超 baseline”，而不是“明显拉开新高度”。

---

## 3.6 Stage 3 伪样本统计

### 3.6.1 官方人工训练集规模 vs 当前 pseudo 规模

| 类别 | human train triplets | pseudo triplets | pseudo / human |
| --- | ---: | ---: | ---: |
| dress | 5985 | 2604 | 43.51% |
| shirt | 5988 | 2694 | 44.99% |
| toptee | 6027 | 1631 | 27.06% |

### 3.6.2 当前筛选质量

| 类别 | input images | pairs after text rules | pairs kept | quality threshold | avg quality score | avg direction margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dress | 11452 | 61001 | 2604 | 0.55 | 0.6807 | 0.0446 |
| shirt | 19036 | 59194 | 2694 | 0.55 | 0.6824 | 0.0404 |
| toptee | 16121 | 61018 | 1631 | 0.55 | 0.6755 | 0.0381 |

关键观察：

- `quality score` 看起来不低，但 `direction margin` 很小，只有 `0.038 ~ 0.045`。
- 这说明这些 pseudo 样本“看起来像对”，但模型并没有非常强地把 target 压过 source。
- 这种样本能带来一些增益，但通常不会带来非常大的提升。

---

## 3.7 Stage 3 top examples 透露出的分布问题

从当前 `pseudo_*_summary.json` 的 `top_examples` 看，样本主要集中在：

- `add purple`
- `add green`
- `add blue`
- `add striped`
- `add red`
- `remove brown`
- `add tank`
- `add pocket`

这说明当前 pseudo-triplets 主要覆盖的是：

- 颜色变化
- 简单 pattern 变化
- 单词级加减属性

但覆盖不足的通常是：

- `dress` 中更重要的长度、版型、领口、袖型、剪裁变化
- 多属性组合变化
- 更长、更自然的 modification text

所以当前 pseudo 更像是“高精度简单补样”，而不是“强表达能力增强”。

---

## 3.8 Stage 1 结果文件口径已修正

此前本地 `results/stage1/` 下的分类指标文件与 `fashioniq_stage1_summary.json` 存在不一致。

现在已经完成修正，当前正式口径固定为：

- `results/stage1/dress_stage1_metrics_best.json`
- `results/stage1/shirt_stage1_metrics_best.json`
- `results/stage1/toptee_stage1_metrics_best.json`
- `results/stage1/fashioniq_stage1_summary.json`

四个文件现在已经对齐，且分类指标文件中的元信息也统一为：

- `model_variant = dqu`
- `pseudo_label = none`
- `pseudo_path = ""`

因此，后续汇报和论文写作中：

- **Stage 1 可以同时引用 summary 与分类指标文件**
- **不再把 Stage 1 视为当前的实验管理异常项**

不过，这次不一致本身说明实验产物管理需要更严格，后续建议继续保持：

- 每个阶段单独目录
- 每个阶段单独 run id
- 汇总文件只由对应阶段的指标文件生成

---

## 4. 为什么 Stage 1 -> Stage 2 反而下降

## 4.1 第一原因：当前 Stage 2 不是“真正等价的 PFP 改进”

根据方案文档，Stage 2 原本想做的是：

- 用 `Q-former / image-conditioned prompt generator`
- 输入 `reference image + modification text`
- 生成更灵活的 prompt 表达

但当前真正实现的 `DQU_PFP_CIR` 不是这个结构。

当前代码在 `project_stage1_dqu_baseline/src/model.py` 里做的是：

- 预设 4 个 prompt template
- 对每个 template 编码
- 用 `prompt_router` 做一个 soft mixture
- 再用 residual 回到 base text feature

这更像是“轻量 prompt ensemble”，不是“reference-aware prompt generator”。

这会导致两个问题：

1. 它未必真的学到了更好的条件化文本表达；
2. 它有可能只是在给原本很强的 DQU 文本分支加一个噪声层。

---

## 4.2 第二原因：当前 Stage 2 的文本端实际上依赖的是“写字后的视觉图”

当前 `DQU_PFP_CIR.encode_textual_query()` 里，router 使用的是：

- `base_text`
- `visual_query`

而这个 `visual_query` 不是原始参考图，而是 DQU 的 `written visual query`，也就是：

- 参考图像
- 加上关键词文字

这意味着当前文本分支不是在利用“原始参考图像的视觉语义”做 prompt 条件化，而是在利用“已经被关键词改写过的图像”做 routing。

这会造成两个副作用：

1. 文本分支和视觉分支都在消费同一份“写字图像”信号，分支互补性下降；
2. 文本分支可能被视觉分支里已经注入过的关键词再次干扰，导致重复表达或错误偏置。

对 DQU 这种本来就依赖“文本统一 + 视觉统一 + 线性融合”的模型来说，这种分支间信号重叠很容易伤害效果。

---

## 4.3 第三原因：当前 prompt templates 的信息增量很弱

当前 4 个 template 本质上都在表达同一件事：

- `{reference}`
- `{reference}, but {modification}`
- `Reference item: ... Target change: ...`
- `Find a fashion item keeping ... while changing ...`

这些 template 在语义上很接近，并没有形成真正互补的 prompt space。

结果就是：

- router 需要学习在几个相近模板之间分配权重
- 但这些模板本身带来的信息增量不大
- 所以增加了训练难度，却未必增加有效信息

这类设计最容易出现的结果就是：

- `R@50` 有时轻微浮动
- `R@10` 反而下降

当前结果正符合这个现象。

---

## 4.4 第四原因：DQU baseline 本身已经很强，轻量文本改动很难自然涨点

`DQU` 的优势本来就来自：

- text-oriented unification
- vision-oriented unification
- adaptive fusion

这已经不是一个弱 baseline。

在这种 baseline 上，任何文本分支改动如果不能稳定提供“更强的条件化表达”，就很容易：

- 破坏已有平衡
- 引入额外参数噪声
- 导致小幅退化

所以 `Stage 2` 下降本身并不奇怪。

问题不在于“prompt 方向不能做”，而在于：

- **当前这版 prompt 改动还不够对**

---

## 4.5 第五原因：当前 Stage 2 还是单次结果，不是 5-seed 均值

论文里 DQU 报的是：

- `5 random seeds` 的均值和标准差

我们当前跑的是：

- 单次实验结果

所以 `Stage 1 -> Stage 2` 的小幅下降里，部分波动可能来自随机性。

但这里有一个重要细节：

- `toptee R@10` 从 `65.6298` 降到 `64.2019`
- 降了 `1.4278`

这个幅度已经不是单纯“可以忽略的小抖动”。

所以结论仍然成立：

- 当前这版 `Stage 2` 大概率确实不如 `Stage 1` 稳。

---

## 5. 为什么 Stage 4 只涨了一点点

## 5.1 第一原因：Stage 3 的筛选是建立在一个更弱的 Stage 2 模型上的

当前 `pseudo_*_summary.json` 里记录的 `checkpoint` 都来自：

- `checkpoints/fashioniq_stage2/...`

也就是说：

- pseudo 的打分和筛选依赖的是 `Stage 2 model`
- 但 `Stage 2 model` 本身比 `Stage 1` 弱

这会直接限制 pseudo 的上限：

- 你在用一个稍弱的老师筛样本
- 那它就不太可能给你筛出最强的一批训练增益样本

所以当前 Stage 4 的提升更像：

- “弱老师筛出来的样本仍然有用”

而不是：

- “我们已经找到了最强的 pseudo 筛选策略”

---

## 5.2 第二原因：当前 pseudo 样本虽然不算少，但信息增量仍然有限

规模上看：

- `dress/shirt` pseudo 大约是 human 的 `43% ~ 45%`
- `toptee` pseudo 只有 `27%`

这不算太少，但也不算非常大。

更关键的是内容分布：

- 大量样本只是颜色、条纹、单词级属性增减
- 很多样本与人工数据已经非常接近

这样的 pseudo 能带来：

- 局部补充
- 数据密度增强

但通常很难带来：

- 显著的语义外推能力提升
- 明显的大幅涨点

所以 Stage 4 的增益不大是符合当前 pseudo 分布的。

---

## 5.3 第三原因：当前筛选出来的样本“像是对的”，但还不够“强”

平均 `direction margin`：

- dress: `0.0446`
- shirt: `0.0404`
- toptee: `0.0381`

这个 margin 偏小。

这说明当前 pseudo 样本虽然通过了阈值，但很多样本只是：

- target 比 source 稍微更像一点

而不是：

- target 明显优于 source

这种样本用于训练是会有帮助的，但帮助通常比较有限。

---

## 5.4 第四原因：当前课程式接入可能把 pseudo 作用压后了

当前 `train.py` 里是：

- `pseudo_start_epoch = 10`
- `tolerance_epoch = 5`

这意味着：

- 模型前 10 个 epoch 只看 human
- pseudo 很晚才进入
- 同时 early stopping patience 又比较短

即使这次 Stage 4 最终确实用上了 pseudo，这个设计仍然会削弱 pseudo 的影响力。

因为 pseudo 的有效训练窗口被压缩了。

更直接地说：

- 现在不是 pseudo 完全没用
- 而是 pseudo 进入得太晚、作用得太保守

---

## 5.5 第五原因：当前 Stage 4 更像“补救 Stage 2”，而不是“真正超过 Stage 1 很多”

从数值上看：

- `Stage 2 -> Stage 4` 的恢复很明显
- `Stage 1 -> Stage 4` 的净提升很小

这说明当前 pseudo 的主要作用是：

- 帮助修复 `Stage 2` 带来的退化

而不是：

- 在一个已经很强的模型上继续明显推高指标

这也意味着当前最重要的问题仍然是：

- **Stage 2 本身不够稳**

---

## 6. 当前问题到底在哪里

可以分成五类。

## 6.1 问题一：实验产物管理有问题

体现：

- `results/stage1/*_stage1_metrics_best.json` 与 `stage1 summary` 不一致
- 且 `model_variant` 被写成 `dqu_pfp`

影响：

- 正式汇报时会造成数据口径混乱
- 后续很难确认哪一组文件才是真正 baseline

这不是“小问题”，而是应该先修。

---

## 6.2 问题二：当前 Stage 2 与原计划不一致

计划文档里，Stage 2 想做的是：

- 更接近 PFP 的 image-conditioned flexible prompt

但当前实现是：

- prompt template mixture

也就是说：

- 当前 Stage 2 只是一个“轻量 prompt 试验版”
- 还不是你们计划里的完整模型改进

这意味着：

- 不能因为当前 Stage 2 降了，就直接得出“PFP 思路没用”
- 更准确的说法应该是：
  - **当前这版轻量 prompt 实现没证明有效**

---

## 6.3 问题三：当前 pseudo 的多样性不够

当前 pseudo 的强项是：

- 简单颜色变化
- 简单 pattern 变化
- 简单 add/remove 属性

当前 pseudo 的短板是：

- 结构变化覆盖不足
- 长文本 modification 覆盖不足
- 2 属性变化比例不清楚，且看起来偏低
- dress 这类更依赖版型/长度/袖型的类别，受益不明显

这也是为什么：

- `dress` 仍然是最大 gap 类别

---

## 6.4 问题四：当前质量筛选还不够强

现状：

- `quality score` 不低
- 但 `direction margin` 偏小
- `checkpoint` 还是较弱的 Stage 2

这表明当前筛选是：

- 有一定效果
- 但还不够“尖锐”

它更像是“把明显坏样本去掉了”，还没做到“把真正最有训练价值的样本挑出来”。

---

## 6.5 问题五：当前缺少老师要求的关键对照实验

老师要求的最核心对照是：

1. `human-only`
2. `human + all pseudo`
3. `human + filtered pseudo`
4. `human + filtered pseudo + curriculum`

当前已经有的：

- `human-only`：有
- `human + filtered pseudo + curriculum`：有

当前还没有的：

- `human + all pseudo`
- `human + filtered pseudo`（不带 curriculum）

没有这两组，就不能严谨回答：

- 筛选到底有没有必要
- curriculum 到底有没有必要

---

## 7. 接下来应该怎么做

## 7.1 总策略

接下来最正确的路线不是继续“沿着当前 Stage 2 直接堆更多技巧”，而是先回答两个更关键的问题：

1. **数据侧增强本身，对强 baseline 有没有稳定帮助？**
2. **模型侧 prompt 改动，到底是实现问题，还是方向问题？**

也就是说，下一轮实验要把“数据机制”和“模型机制”拆开验证。

---

## 7.2 优先级最高的动作

### 动作 0：先修实验口径

必须先做：

- 清理 `results/stage1/` 的污染文件
- 明确 `Stage 1` 的可信文件集合
- 重新保存干净的 baseline 指标

否则后续所有对比都可能乱。

### 动作 1：重跑干净的 DQU baseline，做 5-seed

目标：

- 真正判断当前 reproduction 与 DQU 原论文差多少

要做：

- `seed = 42/43/44/45/46`
- 每类都跑 5 次
- 报均值和标准差

这一步是必须的，因为你们现在所有后续判断都建立在 Stage 1 上。

### 动作 2：先做一个“DQU + filtered pseudo”的直接对照

这是当前最重要的新实验。

也就是：

- 不走 Stage 2
- 直接把 `Stage 3/4` 的 pseudo 接到 `DQU baseline`

目标是回答：

- **如果不引入当前这版 prompt 改动，pseudo 本身能不能把强 baseline 拉高？**

这一步非常关键，因为它能直接告诉你：

- 当前瓶颈到底是 pseudo 不行
- 还是 Stage 2 把增益吃掉了

如果 `DQU + filtered pseudo` 明显优于 `Stage 1`，那就说明：

- 数据增强路线是成立的
- 当前该修的是 Stage 2

### 动作 3：不要再把当前 Stage 2 当最终版，应该重做

有两个可选方向：

#### 方向 A：先保守推进

- 暂时把当前 `DQU-PFP` 降级为对照实验
- 主线先走：
  - `DQU`
  - `DQU + pseudo`

优点：

- 更容易先做出稳定提升
- 更符合老师“先强 baseline、先清楚闭环”的要求

#### 方向 B：重做更忠实的 PFP-style text branch

要点：

- 不再用 4 template mixture 充当 flexible prompt
- 改成真正的 `reference-conditioned prompt generator`
- 条件输入应使用：
  - 原始 reference image
  - modification text
- 而不是当前这种：
  - 写字后的 visual query 图像

如果你们要继续保留“PFP 模型改进”这条线，建议走方向 B。

---

## 7.3 伪样本侧应该怎么提升

### 提升 1：改用更强 checkpoint 重新筛 pseudo

建议至少做三组：

1. `Stage 1 DQU` 做 teacher
2. `Stage 2 DQU-PFP` 做 teacher
3. `Stage 1 + Stage 2 ensemble score` 做 teacher

然后看：

- 哪一种 teacher 筛出来的 pseudo 更有用

从当前结果看，`Stage 1 DQU` 很可能比 `Stage 2` 更适合当 teacher。

### 提升 2：提高 pseudo 的结构多样性

建议把 pseudo 分成两桶：

- `Simple`: 1 attribute change
- `Moderate`: 2 attribute changes

并控制每类采样比例。

原因：

- 当前样本过度偏向“单词级简单变化”
- 对 baseline 的信息增量有限

### 提升 3：重新校准质量筛选

建议至少 sweep：

- `score_threshold`: `0.55 / 0.60 / 0.65`
- `pseudo budget`: `1000 / 2000 / 4000 / full filtered`

当前阈值 `0.55` 偏宽松，可能保留了太多“勉强可用但不够强”的样本。

### 提升 4：增强 dress 类 pseudo 的结构属性覆盖

优先补：

- sleeve
- neckline
- length
- fit
- silhouette

因为当前最大 gap 在 `dress`，而现有 top examples 里这类变化覆盖明显不足。

---

## 7.4 训练侧应该怎么提升

### 提升 1：把 pseudo 更早引入

建议把：

- `pseudo_start_epoch`

从当前的 `10` 往前移到：

- `3`
- `5`

做对照。

### 提升 2：把 curriculum 单独做 ablation

必须补这两组：

- `human + filtered pseudo`
- `human + filtered pseudo + curriculum`

否则不能证明 curriculum 真的有用。

### 提升 3：把 unscreened pseudo 单独做 ablation

必须补：

- `human + all pseudo`

这样才能回答：

- 是 pseudo 本身有用
- 还是筛选机制真正有用

---

## 8. 我建议你们下一轮实验按这个顺序做

### 第一批：先把口径和 baseline 立稳

1. 清理 `Stage 1` 污染文件
2. 重跑 `DQU baseline 5-seed`
3. 重新生成干净的 Stage 1 总表

### 第二批：先验证“pseudo 本身是否有用”

4. 跑 `DQU + all pseudo`
5. 跑 `DQU + filtered pseudo`
6. 跑 `DQU + filtered pseudo + curriculum`

如果这三组里有明显提升，就说明：

- 数据侧路线已经成立

### 第三批：再决定要不要继续深挖 PFP

7. 如果你们要保留 PFP 路线，就重做更忠实的 Stage 2
8. 重新比较：
   - `DQU`
   - `DQU-PFP(new)`
   - `DQU + pseudo`
   - `DQU-PFP(new) + pseudo`

---

## 9. 最后判断

当前最重要的判断不是：

- “为什么 Stage 4 只涨了一点”

而是：

- **当前的增益已经说明 pseudo 有潜力**
- **当前的退化已经说明 Stage 2 还没有做对**

因此，最合理的项目判断是：

1. `DQU baseline` 是成立的，虽然仍略低于论文；
2. 当前这版 `DQU-PFP` 不能作为最终模型结论；
3. `filtered pseudo-triplets` 是有效的，但目前只带来小幅增益；
4. 接下来最值得做的是：
   - 先在 `DQU baseline` 上把 pseudo 路线单独做强
   - 再决定是否继续投入更完整的 PFP 文本分支

这条路线最符合老师的要求，因为它会把重点重新放回：

- 强 baseline
- 清楚对照
- 结果导向

而不是继续堆一个还没被证明有效的 prompt 变体。
