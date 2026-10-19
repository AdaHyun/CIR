# ZVEP-CIR 进度记录 2026-04-19

## 1. 新路线文档

- 执行版方案：
  [CIR_ZVEP_CIR_new_route_20260418_可执行版.md](C:\Users\qh\Desktop\CIR\next_plan\CIR_ZVEP_CIR_new_route_20260418_可执行版.md)

## 2. 本轮已完成代码改造

### 2.1 新增/修改内容

- 新增 `dqu_zvep_cir`
- 新增 `dqu_zvep_cir_reg`
- `train.py` 现在支持多 loss 求和，而不是只吃 `ranking`
- `datasets.py` 现在会为 `FashionIQ` / `FashionIQPseudo` 显式返回：
  - `reference_caption`
  - `keep_instruction`
  - `change_instruction`
- 新增 ZVEP 运行入口：
  - [zvep_base.env](C:\Users\qh\Desktop\CIR\project_stage1_dqu_baseline\configs\zvep\zvep_base.env)
  - [run_fashioniq_zvep_hpc.sh](C:\Users\qh\Desktop\CIR\project_stage1_dqu_baseline\scripts\run_fashioniq_zvep_hpc.sh)
  - [train_fashioniq_zvep_generic.slurm](C:\Users\qh\Desktop\CIR\project_stage1_dqu_baseline\slurm\train_fashioniq_zvep_generic.slurm)

### 2.2 涉及文件

- [model.py](C:\Users\qh\Desktop\CIR\project_stage1_dqu_baseline\src\model.py)
- [train.py](C:\Users\qh\Desktop\CIR\project_stage1_dqu_baseline\src\train.py)
- [evaluate.py](C:\Users\qh\Desktop\CIR\project_stage1_dqu_baseline\src\evaluate.py)
- [datasets.py](C:\Users\qh\Desktop\CIR\project_stage1_dqu_baseline\src\datasets.py)

## 3. 阻塞与修复

### 3.1 已定位并修复

- `ZVEP` 入口第一次 smoke 失败，原因不是模型，而是 `FashionIQ` 路径错误。
- 错误表现：
  - 缺少 `correction_dict_dress.json`
- 修复：
  - 将 `run_fashioniq_zvep_hpc.sh` 的 `FASHIONIQ_ROOT` 从外部 dataset 根目录改回项目内完整数据根目录：
    - `/data/user/hqi881/CIR/project_stage1_dqu_baseline/data/FashionIQ`

### 3.2 当前仍存在的外部阻塞

- 远端环境变量中没有现成的 `OPENAI/AZURE/GEMINI/ANTHROPIC/HF_TOKEN/OLLAMA/VLLM` 入口。
- 这意味着：
  - `H` 系列 human-only 可以继续推进
  - `Z2-Z4` 里真正依赖 MLLM 生成/验证的部分，目前还不能宣称已经可执行

## 4. smoke 结果

### 4.1 smoke job

- `269713`: `H2 smoke`
- `269714`: `H3 smoke`

### 4.2 配置

- dataset: `dress`
- epochs: `2`
- H2: `dqu_zvep_cir`
- H3: `dqu_zvep_cir_reg`

### 4.3 smoke 指标

#### H2

- `dress_r1 = 15.7660`
- `dress_r10 = 42.7863`
- `dress_r50 = 68.3193`

#### H3

- `dress_r1 = 16.1626`
- `dress_r10 = 42.3401`
- `dress_r50 = 68.3193`

### 4.4 观察

- 两条线都已经证明：
  - 能正常加载模型
  - 能正常进入训练
  - 能正常验证
  - 能正常落 `metrics_best.json` 与 `history.jsonl`
- 2 epoch 下：
  - `H2` 的 `R@10` 略高
  - `H3` 的 `R@1` 略高
- 这不足以决定最终优劣，所以已经把 `H2/H3` 两条 full run 都提上去了

## 5. 正式 full run

### 5.1 job id

- `269736`: `H2 dress`
- `269740`: `H2 shirt`
- `269739`: `H2 toptee`
- `269738`: `H3 dress`
- `269741`: `H3 shirt`
- `269737`: `H3 toptee`

### 5.2 当前目标

- 先完成 `H2/H3` 的 human-only 正式训练
- 比较它们是否能超过当前冻结基线：
  - `Stage D = 61.1446 / 80.8781`

## 6. 当前结论

1. 新路线已经从“只有文档”进入“真实可训练”的状态。
2. 当前最合理的推进顺序仍然是：
   - 先看 `H2/H3` 能否站住 human-only
   - 再决定是否进入 `Z2-Z4`
3. 目前不能诚实地说 ZVEP 全路线已经可跑完，因为 `MLLM` 生成/验证通道尚未落地。

