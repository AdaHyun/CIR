# A100 从获取分支到正式实验：逐条执行

本次交付只做云端静态/CPU/隔离模拟检查。以下GPU命令由你在自己的服务器执行；没有连接或占用过服务器/A100。本实现不依赖丢失的旧DQU/CN权重，但必须有真实数据和公开CLIP。新结果以实际机器产物为准，不保证达到旧数字。

## 1. 获取正确分支

```bash
git clone --branch CIR-FULL --single-branch https://github.com/AdaHyun/CIR.git
cd CIR
git branch --show-current
git rev-parse HEAD
```

应显示CIR-FULL。已有checkout时先保存自己的改动，再 `git fetch origin CIR-FULL`、`git switch CIR-FULL`（本地没有该分支则 `git switch --track origin/CIR-FULL`），不用reset/force。项目就在仓库根目录，不再进入CIR-FULL子目录。离线搬运也可将整个目录打包后上传；源码中的默认路径相对工程根，与解压路径/cwd无关。

原最小复现包12个核心PT已在原仓库Git中；主动assets/checkpoints副本、运行预测、测试过程、大权重下载等不重复进入Git，逐项清单 `manifests/EXTERNAL_ARTIFACTS.csv`。获取分支之后：

```bash
python3 scripts/materialize_old.py
```

该脚本仅标准库，逐个SHA核对再复制12个原始val/RR文件，遇冲突拒绝覆盖。全新的正式路线无需这些历史PT。当前环境的其它历史运行完整留档可单独导出：

```bash
python3 scripts/artifacts.py export --output ../CIR-artifacts.tar.gz
# 在目标机器于项目根解压（务必核对附带sha256；不要覆盖已有不同实验）
(cd .. && sha256sum -c CIR-artifacts.tar.gz.sha256)
# 首次空目录才直接解压；已有资产时先检查压缩包内容和重名冲突
# tar -tzf ../CIR-artifacts.tar.gz
# tar -xzf ../CIR-artifacts.tar.gz -C .
```

没有自动上传动作。完整全目录包可用scripts/inventory.py + package.py生成；它排除.git、.venv、下载缓存和字节码，保留实际权重/特征/正式输出，禁止输出包递归进入工程。

## 2. 无sudo环境；不触碰其他用户

Ubuntu24.04/Python3.12.3可用于本工程的预编译wheel。驱动570.86.15可运行CUDA12.1构建；`nvidia-smi`的CUDA12.8是驱动支持上限，不是Toolkit安装证明。这里不编译CUDA扩展，不要求nvcc。

```bash
python3.12 --version
nvidia-smi                       # 只查看，不修改服务或占用显卡
bash scripts/setup_server.sh --cu121
source .venv/bin/activate
python -m pip check
python -c 'import torch,torchvision,open_clip,cv2,transformers; print(torch.__version__,torchvision.__version__,torch.version.cuda,torch.cuda.is_available())'
```

脚本仅安装到本项目.venv，不用sudo/conda，不改变系统Python。torch2.3.1/torchvision0.18.1的cu121版本单独从官方索引安装，普通依赖在requirements-full.txt。CPU模式可显式选 `bash scripts/setup_server.sh --cpu`，本次实际接口测试使用的是CPU wheels，而非cu121。

若Python报缺ensurepip/python3.12-venv，不能sudo装系统包；可用已可用的用户级virtualenv，或官方uv用户安装器：

```bash
mkdir -p .tools
curl -LsSf https://astral.sh/uv/install.sh -o .tools/uv-install.sh
UV_INSTALL_DIR="$PWD/.tools/uv" sh .tools/uv-install.sh
.tools/uv/uv venv --python python3.12 .venv
source .venv/bin/activate
.tools/uv/uv pip install --python .venv/bin/python torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
.tools/uv/uv pip install --python .venv/bin/python -r requirements-full.txt
.tools/uv/uv pip check --python .venv/bin/python
```

这同样仅是你在服务器主动执行的安装命令。本云端没有运行GPU安装/公开大模型下载。网络不可达时可事先在有网络的同平台机器下载wheelhouse，再在服务器 `pip install --no-index --find-links ...`，不要跳过TLS/包完整性校验。

## 3. 准备真实FashionIQ

官方标注来源：`XiaoxiaoGuo/fashion-iq`，固定revision `f71b16c3b09ef03b02c0fed160fa072fb3b8aa16`；本次已实际下载并校验train/val各类共12个JSON，hash随configs/fashioniq_annotations.json交付。

```bash
python -u scripts/prepare.py annotations
```

已有文件必须与固定版本hash一致，不能只凭文件名认定正确。JSON下载中断重下该小文件；不会接受半写文件。

真实图片请从官方README指向的授权分发获取：

- https://github.com/XiaoxiaoGuo/fashion-iq
- 该README给出的图片来源：https://github.com/hongwang600/fashion-iq-metadata

本项目不猜旧Amazon链接、不自动抓取或用51张案例图代替图库。将已授权获取的jpg分成dress/shirt/toptee后导入，显式换成你自己的实际路径：

```bash
python -u scripts/prepare.py images --category dress --source /your/authorized/dress_images
python -u scripts/prepare.py images --category shirt --source /your/authorized/shirt_images
python -u scripts/prepare.py images --category toptee --source /your/authorized/toptee_images
```

目标为 `data/fashioniq/resized_image/<category>/<image_id>.jpg`。实际按OpenCLIP变换读图；视觉query先按历史规则512px写字。导入保存原JPEG字节、不猜ID、不移动源文件，逐文件哈希/重复跳过/冲突拒绝。空间建议：原始图片之外，三类别完整模型optimizer/latest/best及原子临时文件至少预留约240–300GiB；单类约80GiB以上，另加公共权重、特征和日志。全权重保存很大，不能只按4GB CLIP文件估算磁盘。

## 4. 源图片captions / keywords / correction缓存

优先使用找回的真实缓存：

```bash
python -u scripts/prepare.py caches --source /your/real/query_caches
```

识别 `image_captions_<cat>_{train,val}.json`（imageID→caption）、`keywords_in_mods_<cat>.json`（历史source_target键→序列，最后一项为文本）、`correction_dict_<cat>.json`。不会覆盖同名不同内容。缓存来源/哈希另存manifest。

没有作者图片caption缓存时，使用本地免费公开BLIP生成；这是**新的、显式记录的输入选择**，不保证等同旧caption质量，不调用付费API：

```bash
python -u scripts/prepare.py model --kind captioner
CUDA_VISIBLE_DEVICES=2 python -u scripts/prepare.py captions --category dress --device cuda:0
CUDA_VISIBLE_DEVICES=2 python -u scripts/prepare.py captions --category shirt --device cuda:0
CUDA_VISIBLE_DEVICES=2 python -u scripts/prepare.py captions --category toptee --device cuda:0
```

模型为Salesforce/blip-image-captioning-base；HF元数据解析出的实际不可变revision、每个模型文件LFS SHA、生成参数、输入图片/标注SHA都会保存。批次完成后原子提交caption字典；重启同命令只处理缺少的ID。生成器不训练模型、不读target图片给query写答案。有其他/不明来源caption文件时拒绝覆盖，优先选择导入。

默认keywords=auto：已存在真实对应缓存时复用，否则从query已有的人类修改文本拼接生成视觉写入字符串。correction_dict缺失则使用身份映射。doctor会报告这些偏离。严格复用旧输入需用custom.json设置 `query.keywords="historical"` 并补全缓存；不能声称自动生成了原作者缓存。

## 5. 公开ViT-H-14预训练权重

```bash
python -u scripts/prepare.py model --kind clip
```

模型标识 laion/CLIP-ViT-H-14-laion2B-s32B-b79K，文件 `weights/clip/open_clip_pytorch_model.bin`；backbone ViT-H-14，输出1024维。HF hub库保留其可恢复下载状态；工具检查权重的权威LFS SHA，保存实际revision和.integrity.json，模型版本变化拒绝覆盖已有版本。训练仅加载本地核验文件，不隐式下载。

Hugging Face API在本次云网络被拒绝，因此不能声称已验证该权重下载、BLIP生成或真实ViT-H-14 GPU训练。若你的服务器也无法访问，可在其他机器获取上述公开文件及权威SHA后导入：

```bash
python -u scripts/prepare.py import --source /your/download/open_clip_pytorch_model.bin --dest weights/clip/open_clip_pytorch_model.bin --sha256 REPLACE_WITH_REAL_AUTHORITATIVE_SHA256
```

不能填假SHA，也不需要API密钥。HF下载需HTTPS访问huggingface.co及其实际下载CDN/Xet目的地；准备工具不使用收费API。

## 6. 检查与CPU冒烟

```bash
python -u scripts/full.py doctor --category dress --run-id doctor-dress --device cpu
python -u -m unittest discover -s tests -p 'test*.py' -v
python -u scripts/materialize_old.py
python -u scripts/cir.py old-eval --run-id old-reproduction --device cpu
```

doctor成功标志为日志中的completed及`reports/doctor-dress.json`；缺图片、caption、官方split、权重或完整性记录会在训练开始前退出2。允许完整fixture在CPU测试所有阶段，但fixture保存在tests/artifacts，不能变成正式训练输入或论文结果。可选真实OpenCLIP CPU接口测试使用随机轻量ViT-B-32，没有用它冒充ViT-H-14预训练结果。

原始min包自带清单有5个源码文件不符，已报告且原样留档；12个模型/feature PT单独验证通过。旧RR宏平均目标不是强制断言新训练输出的数值。

## 7. 先做A100短基准

先自行确认获分配的GPU编号，示例只选择物理2：

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py benchmark --category dress --run-id bench-dress --device cuda:0
```

只可见一张卡，否则程序拒绝运行，避免RNG保存触及其他GPU。逻辑编号为cuda:0。默认30个真实训练step，前5个作预热；随后完整真实验证，记录实际硬件/精度/batch、逐step耗时、峰值显存、验证耗时与基于当前配置的总时间估计。原始记录在 `runs/new/bench-dress/profiling/benchmark-dress-dqu.json`，不是旧论文性能或端到端系统成本。benchmark目录禁止直接续接正式训练。

OOM时停止该任务，换新run，使用配置降低stages.dqu/rc/cn.batch_size或全局batch_size，再测；不要占用另一用户GPU，不自动更换卡。改变batch/精度会改变实验设置，保留到配置和报告中。

## 8. 单类别及三类别正式流程

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category dress --run-id paper-dress --device cuda:0
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category all --run-id paper-all --device cuda:0
```

二者择需要执行，三类模式逐类处理，仍只用一张指定卡。所有完成阶段会留下状态和SHA；完成标志为 `new-workflow-completed`、`reports/final.json` 和对应完整stage_state completed。退出2=缺材料/受阻，退出1=失败，130=已响应中断；不是成功完成。

## 9. 单阶段运行、复用和中断恢复

新建第一次运行后，同一run追加阶段都加--resume：

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py dqu --category dress --run-id staged --device cuda:0
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py rc --category dress --run-id staged --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py export-dqu --category dress --run-id staged --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py export-rc --category dress --run-id staged --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py bank --category dress --run-id staged --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py cn --category dress --run-id staged --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py export-cn --category dress --run-id staged --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py rr-train --category dress --run-id staged --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py rr-eval --category dress --run-id staged --device cuda:0 --resume
python -u scripts/full.py analysis --category dress --run-id staged --resume
python -u scripts/full.py report --category dress --run-id staged --resume
```

Ctrl+C/SIGTERM将尽可能在step边界保存。任意阶段被终止后，可用同配置/设备 `pipeline ... --resume`，已完成且输入/输出没变的阶段校验后跳过，中断阶段恢复。强制结束只能恢复最近成功原子提交点，不能保证最后一个尚未提交的step。数据块恢复、验证重算范围见METHOD_AND_STAGES。

SSH断开不是checkpoint功能的替代；使用服务器已安装的tmux/调度器，不装系统服务：

```bash
tmux new -s cir-dress
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category dress --run-id paper-dress --device cuda:0
# Ctrl+B再按D，脱离；重连后：
tmux attach -t cir-dress
```

没有tmux且无sudo时可用nohup：

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=2 nohup .venv/bin/python -u scripts/full.py pipeline --category dress --run-id paper-dress --device cuda:0 > logs/launcher-paper-dress.log 2>&1 &
tail -f logs/launcher-paper-dress.log
```

nohup避免SIGHUP，不保证调度器不会终止任务。被终止后仍需--resume。实验内部日志同时在 `runs/new/paper-dress/logs/terminal.log`，保留重启时间和异常traceback。

## 10. 补充实验、K与两阶段成本

先完成pipeline，再同run执行：

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py experiments --category dress --run-id paper-dress --device cuda:0 --resume --rerank-ks 5,10,20,30,50,100
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py online-cost --category dress --run-id paper-dress --device cuda:0 --resume
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py legacy-control --category dress --run-id paper-dress --device cuda:0 --resume
```

K实验重排原候选前K项、其后保持顺序，输出完整候选ID、target截断、hit/count、覆盖率和原始计时。K>已导出L将失败，必须以新配置candidate_k>=K的独立run重新导出足够候选并重训/评估兼容RR，不可拿100冒充200。例：`--config configs/candidates-200.json --run-id paper200` 跑完整新流程，再experiments --rerank-ks ...200；不要改已有完成run的配置。

legacy-control只在controls/legacy-positive-constant独立子目录重现正样本固定1.0这一已知bug，用同一新CN资产与相同采样/初始seed/选模规则训练，便于单因素比较。主Proposed永远真实点积。它不是把历史旧RR接到新特征，也不是把验证集用于训练；不要把control结果标为主方法。

实际模块消融独立run（参数overlay叠加默认full.json）：

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category dress --config configs/ablation-no-writing.json --run-id ablate-writing --device cuda:0
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category dress --config configs/ablation-no-rc.json --run-id ablate-rc --device cuda:0
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category dress --config configs/ablation-pfp.json --run-id ablate-pfp --device cuda:0
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category dress --config configs/ablation-no-bank.json --run-id ablate-bank --device cuda:0
```

对应真实代码：关闭视觉写字；RC替换原DQU；替换原PFP类；CN soft/conflict权重归零。没有编造不存在的模块。CN、DQU、Proposed本身也给出两阶段增益；主线report从完整prediction生成机器指标/CSV/SVG。不同run比较必须核对数据/缓存/协议/超参，不拼最优checkpoint。

成本分开记录：`profiling/model-loads.jsonl` 为模型加载；`offline-export-*.json` 为图库/query/text离线编码；`cached-retrieval-RR-*.json` 为缓存后的matmul/sort/RR；`online-cost-*.json` 为reference+text编码、检索和RR（gallery缓存，明确排除CPU预处理/磁盘）。CUDA计时同步，预热标记、精度、batch、硬件、原始batch数据保留。不能把后两者混称整个系统冷启动成本。

## 11. 产物位置与最终图表

`runs/new/<run_id>/{config,manifests,logs,checkpoints,assets,metrics,predictions,profiling,reports}`。完整query输出先保存，再选成功/失败案例；真实DQU→CN→Proposed HTML及图片在 reports/cases-<cat>/，对齐JSONL在reports/aligned-cases-*.jsonl。TopL外只报告outside/lower bound，不编造完整gallery确切排名。原始rank只有被全gallery显式计算的导出阶段才拥有。

report输出机器JSON、CSV、SVG；所有图表来自真实保存的指标。采集多个类别后用category all report做未舍入等权宏平均。源码/配置/环境/随机状态和来源SHA留档，代码变更后复用旧完成状态会拒绝，建议新run。

## 12. 常见错误

- **missing annotations/images/captions/pretrained**：按上述准备命令补齐；doctor不是仅检查文件名，会检查身份/哈希/实际数据。不要把val改名充train。
- **annotation SHA mismatch**：当前文件非固定官方版本或损坏。先保留原文件，核对来源；不要修改期待hash凑通过。
- **HuggingFace403/timeout**：网络域名访问或版本/文件权限问题，未必缺密钥。公开模型不要求付费API；使用可靠网络或人工带权威hash导入，不关TLS。
- **torchvision::nms/operator不存在**：torch/torchvision CPU/CUDA wheel混装。重建隔离venv，按匹配2.3.1/0.18.1同一索引安装，不改系统环境。
- **CUDA driver/runtime不兼容**：记录实际torch.version.cuda/驱动；不是依靠nvidia-smi显示12.8安装任意Toolkit。优先使用文档的cu121 wheel组合。
- **多GPU可见**：只mask你获分配的一张，再--device cuda:0，不能假设0或所有卡可用。
- **OOM/磁盘不足**：先短基准、减batch/增加存储；新run记录参数变化。不自动删除模型/过程文件。
- **completed outputs changed / fingerprint mismatch**：程序防止混实验。检查是否改配置/代码/资产或搬移了工程外依赖；不直接删状态强行跳过。完整项目内相对路径可迁移，显式外部路径需自行修正并新run。
- **恢复会重算部分工作**：未提交训练step最多一个保存窗口；未提交编码/候选块最多一块；epoch验证中断可重算该次验证，已提交指标不重复。
