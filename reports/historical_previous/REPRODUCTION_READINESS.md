# 复现就绪情况

交付为本地实体工程，非 A100 实验完成结果。正式运行证据为 `runs/delivery-final/`，CPU，6016 条 val 查询。完整 pipeline 退出码2，已正确保存受阻阶段；再次 --resume 验证已完成阶段 input/output hash 后跳过。

| 任务 | 就绪状态 | 已有证据 | 缺失与影响 |
|---|---|---|---|
| 旧 RR 重评估 | 已具备并已执行 | 三类 strict checkpoint 加载；全部查询命中数/分母/预测；macro62.0406/81.0915 | 不证明旧训练可恢复或端到端可重训 |
| true-similarity RR 重训练 | 代码与 fixture 已验证；真实实验受阻 | 真实点积分数、CE、确定性采样、atomic last/best、完整状态恢复 | train三件套、train/val producer证据缺失；见 MISSING_ASSETS |
| CN/旧RR 多K | 已具备并已执行 | 5/10/20/30/50/100；Top100不变；dress差异未抹平 | CN为冻结特征零融合路径，原全网推理未运行 |
| 新 DQU 可视化 | 受阻 | 代码/聚合结果、少量不具备ID映射的旧PNG留档 | DQU真实全查询预测/特征/checkpoint、原始图片和文本缺失 |
| 新 CN/Proposed 可视化 | ID/score全查询导出已完成；完整图片链路受阻 | CN与旧RR完整Top100、rank截断标记、纠正/损害/失败清单 | 原始标注/图片缺；新RR还缺真实训练和新权重 |
| RR成本测量 | 冻结特征CPU推理部分已执行 | 各批原始计时、排除首批后的汇总、显式成本范围 | 没有真实训练成本、GPU性能或端到端DQU/CN成本 |

## 多 K 宏平均（未舍入值计算，展示四位）

| 方法 | R@5 | R@10 | R@20 | R@30 | R@50 | R@100 |
|---|---:|---:|---:|---:|---:|---:|
| CN零融合 |51.8472|61.3659|70.5069|75.3435|80.8952|87.2327|
| 旧RR固定融合 |51.9156|62.0406|70.6839|75.4599|81.0915|87.2327|

旧 RR mR=71.5660。CN零融合 dress 1087/2017；归档 CN coverage1086/2017。这不是按论文数字调整后的输出。各类别统计、命中数、分母、纠正/损害、全查询记录见对应 metrics/predictions/reports。原始归档完整图库 rank 与当前截断 rank 分开保存，详见 archived_rankings_comparison。

## 已完成的本地验收

- 资产schema、归一化/finite、ID/score映射、query-union图库、参考排除和旧权重strict加载。
- 原目标固定1.0输入被真正点积替代；fixture检查候选唯一、标签一致、source排除及loss可回传。
- 独立fixture：连续与step中断恢复的处理顺序/候选/label/loss/optimizer/模型参数一致，当前CPU为逐位相等。
- 实际子进程 SIGTERM 后保存退出，再 --resume；结果与连续运行一致，step行不重复；完整阶段再次启动被跳过。
- 候选/预测块复用、坏块拒绝、错误ID映射拒绝；新配置不能静默续载旧训练。
- 搬迁后的fixture从其他cwd完成短评估；另将真实dress资产/checkpoint复制到另一路径，从 `/tmp` 启动审计并通过。
- 增量整理再次执行：1166项全部校验后跳过，复制0、失败0；源与副本逐项hash核对。
- 主动src/scripts/configs未检出 Windows/历史HPC/workspace绝对路径依赖；旧路径仅存在历史文件/元数据。

验证代码和日志在 tests/、tests/artifacts/、archive/onboarding/，失败的早期测试日志保留；最新通过记录在 LOCAL_VALIDATION.json。不声称 Windows实机、A100/CUDA、多机或断电故障环境测试通过。

## 未完成且未伪造

真实RR重训、新checkpoint/新模型结果、DQU完整排名和真实三方法新图、原始CN特征再生成、A100成本与端到端成本。源码清单5处不符也原样报告，不以修改checksum消除。材料找回后按README先审计和来源校验，使用新run执行真实实验。
