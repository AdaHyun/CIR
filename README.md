# CIR：从公开预训练模型到 CN + true-similarity RR

当前交付为 **`CIR-FULL` 分支的仓库根目录**。无需再进入同名子文件夹。旧源码/历史实验/6016 查询复现证据保留；新训练与旧资产复现独立管理。

**从这里开始：[A100 中文逐步操作手册](docs/A100_RUNBOOK.md)。** 手册覆盖获取分支、无 sudo 环境、真实数据/权重、检查、短基准、训练、恢复、消融和结果导出。

## 第一条命令与执行顺序

```bash
git clone --branch CIR-FULL --single-branch https://github.com/AdaHyun/CIR.git
cd CIR
bash scripts/setup_server.sh --cu121   # 仅由你在服务器执行，本次未执行 GPU 环境安装
source .venv/bin/activate
```

如果已有本地仓库，先保存你的改动，再获取并切换 `CIR-FULL`，不要 reset 或覆盖它们。Python 3.12.3 + 驱动570可使用预编译 PyTorch 2.3.1/cu121；nvidia-smi 的 CUDA12.8 是驱动支持上限，不代表装有 Toolkit。无 venv/ensurepip 时见手册的用户级 uv 替代方式。

准备官方标注、授权真实图片及公开权重/缓存后：

```bash
python -u scripts/full.py doctor --category dress --run-id check-dress --device cpu
# GPU编号由你选择；以下2只是示例，不自动挑选或占用其它卡。
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py benchmark --category dress --run-id bench-dress --device cuda:0
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category dress --run-id paper-dress --device cuda:0
```

新主线 `full.py pipeline`：预检查 → DQU（StageA baseline）→ RC（StageD，独立从公开 CLIP 初始化）→ 两教师 train/val 特征 → 多教师训练候选库 → StageP/CN（初始化自新 RC）→ 同一新 CN 导出 train/val → 修正 RR → 统一评估/案例/报告。**不依赖丢失的旧上游权重。** StageB/C/E/N/ZVEP 是历史探索而非本主线的强制前置，不凭名称虚构额外模块。

## 两条实验路线

| 路线 | 入口 | 输入和结果边界 |
|---|---|---|
| 历史复现 | `scripts/cir.py` | Git 原始最小包的 val 资产 + 旧 RR；实际复现62.0406/81.0915 |
| 全部新训练 | `scripts/full.py` | 官方 train/val、公开 ViT-H-14、真实来源/免费本地生成的源图片 captions；新 DQU/RC/CN/RR 与完整产物 |

```bash
python -u scripts/materialize_old.py
python -u scripts/cir.py old-eval --run-id old-reproduction --device cpu
```

不把旧 RR 默认接到新 CN；也不将新特征与旧 train/val 混合。旧路径中 train 资产缺失不阻断新主线实现，但新主线真实执行仍需原始图片和公开模型。本云端未正式 GPU 训练，不保证新结果与旧论文数值相等。

## 工程与交付

- `src/cir_full/`：DQU/RC 真实网络、FashionIQ处理、上游训练/教师库/同源导出、RR、成本和恢复机制。
- `configs/full.json`：新主线；`configs/default.json`：旧资产 RR；`configs/ablation-*.json`：实际模块消融。
- `scripts/prepare.py`：固定版本官方标注、HF公开模型校验下载、真实图片/缓存导入、免费本地 BLIP captions；不调用付费 API。
- `runs/new/<run_id>/`：新实验；`runs/<old_run>/`：旧复现；`tests/artifacts/`：隔离的模拟证据。日志双写、心跳、原子 latest/best、数据块提交和恢复。
- 原始 `CIR-minimal-reproduction-20260919/`、`phase_packages/` 保留；`archive/` 保留独立来源副本和早期记录，不作为隐藏运行目录。
- `manifests/EXTERNAL_ARTIFACTS.csv` 清楚列出没有放入 Git 的大文件/运行记录；**文件仍在当前环境中**，可用 `python scripts/artifacts.py export --output ../CIR-artifacts.tar.gz` 单独导出。原最小包12个核心资产已在原仓库 Git 中，`materialize_old.py` 可直接重建工作副本。
- `reports/REPRODUCTION_READINESS.md` 区分真实资产验证、模拟代码验证与待补外部材料；`docs/METHOD_AND_STAGES.md` 给出证据和每阶段输入/输出/恢复粒度。

## 常用运行和恢复

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category all --run-id paper-all --device cuda:0
CUDA_VISIBLE_DEVICES=2 python -u scripts/full.py pipeline --category all --run-id paper-all --device cuda:0 --resume
python -u -m unittest discover -s tests -p 'test*.py' -v
```

所有默认路径相对仓库根解析，不依赖当前 cwd；可用 `--config`、`--data-dir`、`--pretrained`、`--output-dir` 覆盖。fixtures 强制只能写 `tests/artifacts/`，不会进入正式结果。原始源包的5处源码清单不符仍保留记录，不改 hash 掩盖问题。
