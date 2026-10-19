# 当前交付：真实验证、代码验证和外部材料

本次新增从公开权重出发的真实训练实现，不再把丢失旧上游权重当作停止补齐路线的理由。项目在CIR-FULL分支根目录；此前“同名目录交付”的记录保留于docs/HISTORICAL_LOCAL_HANDOFF和reports/historical_previous。

## A. 已用真实资产执行

- 原min包9个验证feature/TopK和3个旧RR checkpoint真实存在且逐项SHA/schema/ID/norm验证。
- 6016条旧RR评估回归仍为宏R@10=62.040561059683235、R@50=81.09148556455976、mR=71.5660233121215。各类别hits/分母/全查询输出、多K、corrected/harmed和失败清单保存；当前环境完整records可单独导出。
- CN零融合为61.36592095949684 /80.8951757783617；dress较coverage多1个R10命中。原始顺序与当前tie重排差异保留；29处CN rank重排区间分数完全相等，没有改排序凑论文。
- 官方FashionIQ固定commit的12个train/val annotation/split JSON已实际下载验证，哈希随configs/fashioniq_annotations.json交付。它们不是训练图片，不能据此宣称真实全链训练已运行。

## B. 已实现并通过CPU/隔离模拟验证

- DQU/RC真实网络公式移植；实际OpenCLIP包在CPU的随机ViT-B-32接口前向通过，DQU/RC输出维度/有限值/L2/非平凡NCE检查通过。这是库接口测试，不是ViT-H-14预训练性能验证。
- 小型合成图片+显式tiny编码器的完整阶段链：DQU、RC、同源两教师导出、三路RRF训练bank、CN ranking/soft/conflict训练、新CN train/val+TopK+映射导出、真实点积RR、统一预测/案例/报告。
- 全状态保存/恢复，实际SIGTERM，上游与RR连续/恢复的参数/样本顺序/loss一致；scheduler/optimizer/RNG恢复，已完成阶段跳过，指标不重复。配置变化拒绝、损坏chunk/provenance拒绝、换目录与cwd测试。
- K-prefix实验、K超过资产L时拒绝、独立旧bug控制实验、在线编码+检索+RR计时入口、独立训练短基准、模块消融配置均有可执行逻辑；fixture执行验证，而非只打印成功。
- 可恢复下载/导入/校验脚本、免费本地BLIP captions入口、根目录路径解析、无sudo服务器安装脚本和逐阶段中文手册已交付。CPU依赖安装及pip check通过；没有安装GPU CUDA环境。

## C. 尚未在A100执行/仍缺真实外部材料

- 完整FashionIQ授权jpg及作者query缓存未在云端找到；可由用户导入。图片分发来源和导入步骤在A100_RUNBOOK。
- 公开ViT-H-14/BLIP权重下载脚本已实现，但当前云网络对HuggingFace API拒绝访问，没有下载其权重，也没有实测真实caption生成/ViT-H-14完整加载/训练。模型ID、目标文件、权威LFS hash验证与revision记录机制明确，不填假校验值。
- 原作者caption/keyword/correction缓存缺失时，新BLIP+modification写字/身份纠错是显式工程选择，不冒充历史缓存。性能是否接近旧结果需要真实A100实验。
- A100训练、实测GPU吞吐/峰值显存/总训练时间、真实新DQU→CN→Proposed图片案例及新模型数值都等待用户执行。fixture输出不可作论文结果。
- 丢失的旧producer hash/训练资产影响精确旧训练重现，但不影响按新路线重新训练全部模型；旧资产与新实验严格分离。

## 逐项能力状态

| 要求 | 状态 | 实际入口 |
|---|---|---|
| 旧RR重新评估 | 已真实执行 | materialize_old.py + cir.py old-eval |
| true-sim RR真实重训 | 真实代码完整，CPU fixture通过，待真实输入/A100 | full.py rr-train；pipeline自动生成新训练资产 |
| CN/旧RR多K | 旧资产已执行，新资产prefix实验已fixture验证 | cir.py multik / full.py experiments |
| 新DQU可视化 | 真实生成链路已实现并fixture联通，真实结果待训练 | full.py export-dqu + analysis |
| 新CN/Proposed可视化 | 全查询先导出，再真实图片选例 | full.py export-cn/rr-eval/analysis |
| RR及在线成本/离线编码/模型加载 | 分范围计时实现，真实旧RR CPU计时保留；A100待测 | full.py experiments/online-cost/benchmark |
| 完整新路线 | 已实现，不依赖旧上游权重 | full.py pipeline |

最终测试/实际回归记录见BRANCH_VALIDATION.json与archive/branch_validation。原始5个源码hash不符仍在original_integrity.json，源材料未改；无服务器连接、远程GPU占用或正式长训练。
