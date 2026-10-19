# CIR-FULL：可整体搬迁的实验工程

本目录是实体项目，包含两个独立材料包的完整副本、实际可用的验证特征与旧 RR 权重、改造后的统一入口及全部本地验证记录。源材料未移动、删除或修改。当前在仓库的本地 `main` 分支整理；本会话只能访问云工作区，**没有访问 Windows 的 `D:\Study\cjf`**。将整个 `CIR-FULL` 复制到该目录，或未来 Linux 的任意目录即可。运行路径以本文件所在工程根目录解析，与 shell 当前目录无关。

本次没有连接服务器、上传、远程拉取、安装/升级 A100 环境或进行真实训练。CPU 冻结特征评估通过，不代表 A100 验证通过。真实训练和完整新可视化仍缺材料，详见 [资产审计](reports/LOCAL_ASSET_AUDIT.md)、[缺失清单](reports/MISSING_ASSETS.md)、[就绪状态](reports/REPRODUCTION_READINESS.md)。

## 文件布局与材料边界

- `src/cir_full/`、`scripts/cir.py`：新增的可运行入口；不调用 archive 中的历史启动脚本。
- `configs/default.json`：确认过的旧 checkpoint 训练参数、固定融合权重与本次工程选择。
- `assets/<category>/`：最小复现包复制来的 **val** 特征和 Top-100；没有 train 资产。
- `checkpoints/reranker/<category>/`：旧 RR 权重。`dqu/`、`cn/` 中只有缺失说明。
- `archive/CIR-minimal-reproduction-20260919/`、`archive/phase_packages/`：独立、按原相对路径保存的真实来源；禁止默认跨包匹配。
- `archive/onboarding/`：先前验证的副本和执行日志；`archive/requests/`：用户粘贴需求原文件。
- `manifests/source_mapping.csv`：原路径、新路径、大小、SHA-256、来源/类别/阶段/split；`file_manifest.csv`：交付文件全集；`PACKAGE_SHA256SUMS.txt`：可重新校验的交付清单。
- `runs/<run_id>/`：真实资产运行记录；`tests/artifacts/`：显式 synthetic fixture 的检查记录、失败诊断、对照 checkpoint；不进入正式指标。

历史代码和 checkpoint arguments 中的旧服务器路径原样留档，仅作来源证据；当前入口忽略这些路径，不依赖它们。Python/PyTorch 是需自行提供的外部运行时；工程不包含不可迁移的本机虚拟环境或隐藏权重缓存。

## 环境依赖（由你在服务器单独配置）

建议 Python 3.10–3.12。普通依赖为 `requirements.txt`；PyTorch 单独声明在 `requirements-torch.txt`。本地实际验证为 Python 3.12.14、PyTorch 2.3.1+cpu、NumPy 1.26.4。先根据服务器驱动选择官方兼容的 PyTorch/CUDA 构建，再安装普通依赖：

```bash
python -m pip install -r requirements.txt
python -u scripts/cir.py --help
```

本次未执行任何 GPU 安装。历史 raw-image 导出另需 `requirements-export-optional.txt`，包括匹配版本的 torchvision、OpenCLIP、真实 backbone 权重和原始数据；仅准备了原始代码，缺材料时不可启动并宣称成功。当前缓存特征工作流不需要 OpenCLIP、Slurm、数据库或网络服务。

## 完整流水线与 GPU

环境配置完成后，从本目录执行：

```bash
python -u scripts/cir.py pipeline --run-id experiment-001 --device cuda:0 --log-every 10 --heartbeat-seconds 15
```

Linux 可指定物理 GPU：

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/cir.py pipeline --run-id experiment-gpu1 --device cuda:0
```

Windows PowerShell：

```powershell
$env:CUDA_VISIBLE_DEVICES="1"
python -u scripts/cir.py pipeline --run-id experiment-gpu1 --device cuda:0
```

`cuda:0` 是经过 CUDA_VISIBLE_DEVICES 映射后的编号。CUDA 不可用会明确报错，不会静默退回 CPU。只执行短小 CPU 验证可用 `--device cpu`。真实 pipeline 不会使用测试 fixture；本次运行记录位于 `runs/delivery-final/`。

流水线顺序：audit → old-eval → multik → train → new-eval → diagnostics → dqu-export → visualization → cost → summary。缺 train、DQU、原始标注/图片的阶段会写 `blocked`，继续独立阶段。退出码：0=所请求阶段成功，2=有受阻阶段，1=失败，130=中断后已退出；2 不是全流程成功。

## 逐阶段命令

首次建立运行目录，之后同一运行目录都加 `--resume`：

```bash
python -u scripts/cir.py audit --run-id experiment-001 --device cpu
python -u scripts/cir.py old-eval --run-id experiment-001 --resume --device cuda:0
python -u scripts/cir.py multik --run-id experiment-001 --resume --device cuda:0
python -u scripts/cir.py train --run-id experiment-001 --resume --device cuda:0
python -u scripts/cir.py new-eval --run-id experiment-001 --resume --device cuda:0
python -u scripts/cir.py diagnostics --run-id experiment-001 --resume
python -u scripts/cir.py dqu-export --run-id experiment-001 --resume
python -u scripts/cir.py visualization --run-id experiment-001 --resume
python -u scripts/cir.py cost --run-id experiment-001 --resume
python -u scripts/cir.py summary --run-id experiment-001 --resume
```

可加 `--category dress|shirt|toptee`；默认 all。宏平均对所选类别等权，不将单类别称为三类论文结果。只要不是修改既有运行的训练设备/配置/输入，audit 和后处理可用 CPU。恢复训练需使用与原训练匹配的设备类型、PyTorch、模型、配置和资产。

## 配置和路径覆盖

`--config configs/custom.json` 递归覆盖默认 JSON，CLI 优先。支持 `--asset-dir`、`--checkpoint-dir`、`--data-dir`、`--output-dir`、`--eval-batch-size`；相对路径始终相对工程根目录，绝对路径仅在显式覆盖时使用。整体迁移时保留相对路径配置，不指向旧机器。

仅重建缺失的 Top-K（前提是真实 train query/image 特征、source/target 标签和映射齐全）：

```bash
python -u scripts/cir.py build-topk --split train --category dress --run-id rebuild-dress --device cpu
python -u scripts/cir.py build-topk --split train --category dress --run-id rebuild-dress --device cpu --resume
```

输出 `runs/rebuild-dress/reports/dress_train_topk.pt`；按查询块提交，不覆盖原资产。可在 custom.json 中引用：

```json
{
  "asset_files": {
    "dress": {"train": {"topk": "runs/rebuild-dress/reports/dress_train_topk.pt"}}
  }
}
```

仍须给出 `assets/dress/provenance.json` 中的真实来源及该文件 SHA-256。训练特征缺失时不可重建；禁止 val 改名、猜 ID 或生成随机特征补缺。具体 schema 见 MISSING_ASSETS。

## true-similarity RR 与模型选择

所有候选，包括插入的正确目标，都使用冻结的单位特征 `candidate @ query` 作为 base-score；标签只进入交叉熵。保留旧模型结构、候选池和训练超参，无额外 distillation loss。每个 `(seed, epoch, query_id)` 确定采样和标签位置；不足不同负样本会报错，不用目标或无效分数补位。

沿用历史训练的 **未融合 RR 验证 R@10+R@50 最大值**选 best，严格改进才更新。正式对比使用配置中固定融合系数 dress=.05、shirt=.04、toptee=.03，不逐查询/类别择优拼装 checkpoint。训练 AMP/scheduler 未启用，checkpoint 中明确为 null；不声称新增了旧实验没有使用的学习率策略。

当前真实训练被以下检查阻断：缺 train 三件套；缺 train/val 真实同源证据。`provenance.template.json` 只是格式样例，不能伪填模型身份。即便 shape 一致也不足以解除阻断。

## 评估与可视化

使用资产内部显式 gallery ID→行号映射。三类 val 图库已确认分别等于参考/目标图片并集，与历史 VAL-Split 实现一致。推理不插入目标，参考图排除，所有查询计入分母；R@5/10/20/30/50/100 均导出。Top-100 外只记录 `outside Top-100` 和 rank 下界 101，不伪造全图库排名。固定 Top-100 时旧 RR 的 R@100 与 CN 相同。

当前 CN 采用零融合 argsort 路径。dress 得到 1087 个 R@10 命中（53.8919%），归档 coverage 为 1086（53.8423%）；保持两种输出并记录逐查询差异，不修改分母或排序以凑数。

`predictions/cn-*.jsonl`、`old-*.jsonl` 保存全部查询和全部 Top-100 ID/分数、reference/target、截断状态、图库协议、输入指纹和代码指纹。缺原始修改文本时字段为 null，并明确标为缺失，不从指标反推文本。待真实 `data/fashioniq/captions/cap.<category>.val.json` 到位，visualization 仅用唯一 reference+target 对匹配；有重复歧义会阻断而非按行号猜。完整记录先导出，之后才挑失败案例；图片需 `resized_image/<category>/<ID>.jpg`，HTML 使用复制的真实图片。

DQU 缺完整预测、特征和模型。`dqu-export` 只在提供**全查询真实**预测时导入；`custom.json` 可设置 `dqu_predictions: {"dress":"predictions/dqu/dress.jsonl", ...}`。每条需 reference_image_id、target_image_id、retrieved_image_ids、scores、gallery_protocol=`query-union / VAL-Split; reference excluded`，以及 checkpoint_sha256 或 feature_sha256、可用时的原始修改文本。会检查查询全集、图库/参考排除、分数和来源；不将 CSR/StageA 的案例图片或聚合指标冒充 DQU。`new` 专指本次 true-similarity 重训练后的 RR；`old` 是历史 Proposed。

## 实时日志、留档与断点续跑

终端每批/若干 step 输出阶段、类别、run ID、路径、shape、loss/平均 loss/lr、进度、耗时/吞吐/ETA；15 秒心跳覆盖特征加载、校验与长算子；CUDA 使用时输出显存。完整 traceback 同时显示并写 `logs/terminal.log`，日志按行追加，不依赖覆写式进度条。

每次运行的目录：

```text
runs/<run_id>/
  config/       # 生效配置、每次 invocation/environment、代码快照/git 信息
  manifests/    # 每次尝试的输入大小、SHA-256、缺失项
  logs/         # 终端日志，恢复时追加
  checkpoints/  # 各类别 last.pt / best.pt（真实训练当前受阻）
  metrics/      # step/epoch、类别 hits/分母、多 K 指标
  predictions/  # 原子提交的候选块及完整逐查询 JSONL
  profiling/    # 原始批次时间、范围说明、成本汇总
  reports/      # 资产审计、状态/尝试历史、失败查询、纠正/损害、汇总
```

查看日志：

```bash
tail -f runs/experiment-001/logs/terminal.log
```

```powershell
Get-Content runs/experiment-001/logs/terminal.log -Wait
```

Ctrl+C 或 SIGTERM：请求在当前优化 step/数据块安全边界保存退出，退出码 130。**断开 SSH 不保证进程继续存活**；可由你在服务器使用 tmux/调度器启动。如进程结束，重新连接后用同样配置和设备恢复：

```bash
python -u scripts/cir.py pipeline --run-id experiment-001 --device cuda:0 --resume
# 或显式指定运行目录，二者择一
python -u scripts/cir.py train --category dress --run-dir runs/experiment-001 --device cuda:0 --resume
```

训练每 epoch 保存，默认每 100 step 或 120 秒额外保存，可在 training 中配置 `save_every_steps/save_every_seconds`。`last.pt` 是训练恢复入口，包含 model/optimizer、epoch/global step/epoch cursor、最佳记录、完整配置/资产身份、Python/NumPy/CPU/CUDA RNG、确定性采样进度及已提交指标；best 也保留模型与优化器状态，但统一恢复入口使用 last。文件写完并 fsync 后原子替换；指标从成功提交的 last 物化，恢复时不会重复已提交批次或行。未提交 step 在断电后可能重算；SIGKILL/断电不能保证最后一个 step 已保存，也不承诺不同 CUDA/驱动环境逐位一致。测试仅证明当前 CPU 运行的声明范围。

阶段状态为 pending/running/completed/blocked/failed，保留每次尝试。输入/配置/代码及完成输出校验一致才能跳过；已完成阶段发生改变会要求新 run，避免静默混用。blocked 阶段允许补材料后重试；若新增输出改变了已完成下游的输入，使用新 run。预测/Top-K 以块原子提交，重启复用通过校验的块；损坏块报错，不当作成功。残留 `.partial` 从不视作完成文件。不自动清理任何正式输出。

## 本地短测试与整理恢复

```bash
python -u tests/test_workflow.py
python -u tests/test_signal_resume.py
```

所有 synthetic 数据位于 `tests/artifacts/`，fixture 配置被限制只能写这个目录；正式工作流禁止引用 tests 下的资产。本地验证包括真实旧 checkpoint strict 加载、6016 查询评估、恢复后样本/optimizer/参数一致、真实 SIGTERM/恢复、完成阶段跳过、块损坏拒绝、错误 ID 映射拒绝以及搬迁后从其他 cwd 启动短检查。没有进行真实 RR 训练，也没有在 Windows 或 A100 实机执行。

再次增量整理（仅在来源包真实可访问时）：

```bash
python -u scripts/organize.py --source-root /path/containing/both/packages
```

先扫描、计算 SHA/空间，再复制。已校验内容跳过；冲突保存在 archive/conflicts 中并保留原文件；临时副本验证后原子替换。独立日志在 `reports/organization/`，不中断覆盖旧日志。不要把 source-root 指向 CIR-FULL 自身。来源包的位置是显式整理参数，不是实验运行依赖。

## 打包和交付检查

完整携带本目录，尤其不可遗漏 assets、checkpoints、archive、manifests、runs、reports 和真实查询输出；测试 evidence 可保留，不能混入论文结果。本机虚拟环境、依赖下载缓存、Git 内部元数据及 Python 字节码的排除说明见 `reports/EXCLUSIONS.md`。

新增运行/文件后刷新清单，再可选地**本地**打包：

```bash
python -u scripts/inventory.py
python -u scripts/package.py --output ../CIR-FULL.tar.gz
```

压缩包必须写在工程外，避免递归包含；不自动上传，不删文件。SHA-256 检查失败时先诊断，再刷新清单；不能修改原始包的校验清单掩盖差异。仅在已接受的新产物变化后刷新交付 inventory。原始 package 自带清单仍有 5 个源码文件不匹配，保存在审计报告中；本工程自己的交付清单另外记录复制的实际字节。

恢复粒度补充：训练epoch结束后、验证开始前会提交last；若在该次验证中中断，恢复会跳过已完成优化step并重算尚未提交的该epoch验证。正式预测与Top-K导出则按已提交查询块恢复。
