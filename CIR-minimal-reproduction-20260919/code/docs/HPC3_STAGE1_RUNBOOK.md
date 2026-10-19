# HPC3 Stage 1 Runbook

This runbook assumes you upload the folder as:

```text
/data/user/<your_id>/project_stage1_dqu_baseline
```

## 1. Upload

Upload the whole local folder to HPC3, then put the FashionIQ resized images under:

```text
data/FashionIQ/resized_image/dress/
data/FashionIQ/resized_image/shirt/
data/FashionIQ/resized_image/toptee/
```

## 2. Create Environment

```bash
module load miniconda3
conda env create -f environment.yml -p /data/user/$USER/envs/dqu-stage1
conda activate /data/user/$USER/envs/dqu-stage1
```

## 3. Validate Benchmark Files

```bash
python scripts/check_dataset.py --dataset all
```

If this fails, do not start training. Fix the missing files first.

## 4. Smoke Test One FashionIQ Category

```bash
bash scripts/run_fashioniq_stage1.sh dress
```

Expected outputs:

```text
checkpoints/fashioniq/dress/dress_stage1_best_model.pt
checkpoints/fashioniq/dress/dress_stage1_metrics_best.json
checkpoints/fashioniq/dress/dress_stage1_train.log
```

## 5. Run the Full Stage 1 Benchmark

Option A, sequential inside one shell:

```bash
bash scripts/run_all_fashioniq_stage1.sh
```

Option B, one Slurm job per category:

```bash
sbatch --export=ALL,CATEGORY=dress slurm/train_fashioniq_stage1.slurm
sbatch --export=ALL,CATEGORY=shirt slurm/train_fashioniq_stage1.slurm
sbatch --export=ALL,CATEGORY=toptee slurm/train_fashioniq_stage1.slurm
```

Option C, Slurm array:

```bash
sbatch slurm/train_fashioniq_stage1_array.slurm
```

## 6. Aggregate Final Metrics

```bash
python scripts/aggregate_fashioniq_metrics.py \
  --dress checkpoints/fashioniq/dress/dress_stage1_metrics_best.json \
  --shirt checkpoints/fashioniq/shirt/shirt_stage1_metrics_best.json \
  --toptee checkpoints/fashioniq/toptee/toptee_stage1_metrics_best.json
```

Expected output file:

```text
checkpoints/fashioniq/fashioniq_stage1_summary.json
```

## 7. Stage 1 Completion Checklist

- `python scripts/check_dataset.py --dataset all` passes
- `dress`, `shirt`, and `toptee` each produce a best checkpoint
- `dress`, `shirt`, and `toptee` each produce a `*_metrics_best.json`
- `checkpoints/fashioniq/fashioniq_stage1_summary.json` is created

After that, the folder is ready to serve as the frozen baseline for Stage 2.
