# 方法证据与阶段契约

## 确认的对应关系

1. StageA 是强 DQU baseline；证据：`phase_packages/02_stageA_E_mainline/README.md` 的表与 `results/local_sync_20260418/artifacts/final_summary_20260418.json`。DQU=`BaseCIRModel/DQU_CIR`：图片/文字编码L2归一化，combiner+dynamic scaler融合，learnable loss_weight初始10的in-batch CE。
2. StageD 的 `DQU_PFP_REFCOND_CIR` 使用 reference caption/modification/visual 三种 conditioning tokens，8个 prompt query、8头 attention、MLP+gate 调制文字。不是把 StageA 另起名字。主线RC独立从公开CLIP训练，原 `run_fashioniq_experiment_hpc.sh` 的 StageD 路径未传DQU warm-start。DQU仍是bank的独立teacher。
3. StageP/CN：`04_stageP_bank_distill/code/src/train.py::_compute_bank_losses` 与 `build_multiteacher_bank.py`。teacherA=StageD，teacherB=DQU，第三路=teacherA text-only；RRF `1/(60+rank)` 三路等权。target rank gate20；bank5项，目标权重.55、其余.45×softmax(50×RRF分数)，训练取前3再归一化。CN从新RC初始化，NCE+.15×soft KL+.08×margin conflict loss，margin.04、temperature.07。参数由v3日志与bankdistill脚本确认；backbone/target/conflict/soft编码与原代码路径对应。
4. StageR 的 `FeatureReranker` 是冻结1024维 query/candidate的第二阶段MLP；hidden768，4种pair表示+基础分数。训练正样本旧固定1.0已修为点积。验证用未融合RR的R10+R50选best；正式固定blend .05/.04/.03。

不能从上述材料证明 CN 有额外 GNN/邻域网络或 RC 必须先加载DQU权重；本实现没有编造。StageB/C/E/N与ZVEP仍保留但不是强制前置。

## 新工程明确的选择/证据不足

- 使用新训练的DQU和RC生成teacher bank，不寻找丢失的历史权重。
- 源图片caption优先导入原缓存；缺失可由免费本地 `Salesforce/blip-image-captioning-base` 生成。新BLIP captions不等同于作者缓存，可能影响性能。模型下载时将实际HF revision固定记录到manifest；当前云网络未验证HF真实权重下载。
- keywords模式auto：有真实对应缓存时用缓存，否则使用当前query的两条人类修改文本（纠错、lowercase、去标点后 ` and ` 拼接）。不是从target图片或ID构造答案。历史缓存不齐、correction_dict缺失时使用身份纠错，doctor明确报告该偏离。可选 `keywords=historical` 严格要求原缓存。
- 修改文本视觉写入复用历史 OpenCV512px布局/白底黑字。DQU/RC都使用相同构造；target/gallery不写入query文本。训练目标ID仅用于取监督图片/标签和teacher训练监督，不作为网络数值特征。历史cache按source_target键寻址只是一种查询身份映射。
- 训练用OpenCLIP train变换；所有新冻结资产train/val统一用确定性val变换，不沿用历史随机train导出，选择已记录在metadata/provenance，防止重复导出漂移。train和val来自同一best模型字节版本；train图库是train query的source/target并集，val图库是val query并集，各自显式ID映射。
- 上游无额外伪样本路线；只使用人类train更新参数。验证只选best。手动单进程batch/no worker预取，以可靠恢复为先。AdamW双LR、BN eval、AMP/GradScaler、衰减规则沿用现有代码。CN epoch24/batch8/lr5e-5/clip_lr5e-7，decay12/max24/patience8；DQU/RC epochs100/batch16/lr1e-4/clip_lr1e-6，decay8/max10/patience5。早停和最优epoch均由实际验证决定，不强制旧指标。

## 阶段契约（根为 runs/new/<run_id>）

| 阶段/CLI参数 | 前置和实际工作 | 输出/消费者 | 完成/恢复 |
|---|---|---|---|
| doctor/preflight | 真实图片、官方split/annotation、source captions、公开CLIP本地文件+完整性记录；检查train/val图像划分、输入SHA、空间 | reports/doctor-*.json、每次preflight报告 | 缺输入退出2；在正式训练前检查，不返回假成功 |
| dqu | publicCLIP+train数据；真实NCE更新 | checkpoints/dqu-<cat>/{latest,best}.pt、step/epoch history | optimizer/scheduler/AMP/RNG/epoch/cursor；step或时间间隔保存 |
| rc | publicCLIP+refcond网络；无需丢失旧模型 | checkpoints/rc-<cat>/；bank/CN初始化使用best | 同上；独立于DQU训练而非错误warm-start |
| export-dqu/export-rc | 对应best、train/val原始数据 | assets/<method>/<cat>/六个PT、provenance、块commit、完整val predictions | 图像/query/TopK按块提交；未提交块重算，已提交hash验证复用 |
| bank | RC/DQU train特征和相同query/gallery ID，provenance验证 | assets/banks/<cat>.jsonl | 每查询块RRF+gate；原始权重分布保留，绝不使用val样本 |
| cn | RC best+bank；真实NCE/soft/conflict训练 | checkpoints/cn-<cat>/、loss分项 | 同上，初始化时严格完整加载RC，重置optimizer符合历史路径 |
| export-cn | 新CN best | assets/cn/<cat>/{train,val}_{query,image,topk}.pt + provenance + predictions/cn-*.jsonl | producer hash完全相同，dim/norm/split/ID/query文本/数据/预处理元数据齐全 |
| rr-train | 新CN同源train/val资产；不用旧RR | checkpoints/<cat>/{last,best}.pt | 全状态事务恢复；epoch验证未提交时可重算该epoch验证 |
| rr-eval | 本run新RR best + 本run新CN val | predictions/new-*.jsonl、hits/count、多K基础指标、raw profile | TopK块原子提交，推理不插入target，source排除 |
| analysis | 同一run DQU/RC/CN/newRR全query输出 | 对齐全查询JSONL、纠正/损害、真实ID案例HTML和图片、分类指标 | 完整全集先导出再选案例，无DQU记录时阻断而不冒充 |
| report | 所选所有类别analysis | reports/final.json、metrics.csv、metrics.svg | 未舍入值等权宏平均；图表从机器指标生成 |
| experiments | 新CN/RR、指定实际候选L>=max(K) | K实验完整JSONL、覆盖率/指标、缓存检索+RR原始计时 | 仅重排prefix K，保持其后L-K顺序；不能伪装K>L |
| online-cost | 新CN/RR+原始val query图/文本 | profiling/online-cost-*.json | 编码reference+text、检索、RR，gallery缓存；明确排除CPU预处理/磁盘和图库离线编码 |
| benchmark | 独立run，默认新DQU；publicCLIP+真实数据 | profiling/benchmark-*.json，实际steps/完整val/峰值显存/耗时估计 | 默认30step，前5预热；只做测量，不作为正式best，不得复用到正式run |
| legacy-control | 同一新CN资产，在独立controls目录复现正目标固定1.0的已知训练bug | controls/legacy-positive-constant/的模型、日志、预测和指标 | 仅显式消融；主Proposed始终真实点积，不能混成历史旧RR或替代新模型 |

## 恢复与完整性

SIGINT/SIGTERM请求在当前step边界保存。latest/last含模型、optimizer、scheduler、AMP（不用时null/空）、Python/NumPy/CPU/CUDA RNG、global step、epoch内cursor、确定性epoch顺序、配置/代码/输入指纹。dropout RNG恢复；无workers/prefetch。原子临时文件+fsync+replace。best写入中断时latest保留pending-best标记，可从该已提交模型重建best；不丢弃上一份有效checkpoint。

训练只重算最近成功提交后未提交的step，默认最多99step或120秒间隔内工作；可将save_every_steps设1降低窗口（会增加I/O）。验证在epoch训练提交后执行，未提交的完整验证可以重算。图像/查询/TopK/teacher bank/正式RR预测按块恢复，最多重算一个未提交块。指标按checkpoint已提交记录原子物化，避免重复行；文本日志追加保留每次重启。

阶段完成以输入/配置/代码和输出SHA共同校验，改变后的完成状态不可静默复用。checkpoint坏或配置不兼容明确失败，保留诊断；不要随意删状态或强行load。benchmark与正式训练目录隔离；legacy control与主RR隔离。CUDA只允许显式单卡可见，防止保存all-visible RNG时触及其他用户GPU。
