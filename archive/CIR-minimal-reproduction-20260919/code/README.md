# Stage 1 DQU Baseline

This folder is a self-contained local project for **Stage 1: DQU-CIR baseline reproduction**.

The goal of Stage 1 is narrow:

- reproduce the official **DQU-CIR** baseline
- fix the project layout so it can be uploaded to `HPC3` and run from the repo root
- standardize the **FashionIQ benchmark** workflow for `dress`, `shirt`, and `toptee`
- leave a clean base for Stage 2 (`DQU + PFP`) later

This project is based on the official implementation of:

> Haokun Wen, Xuemeng Song, Xiaolin Chen, Yinwei Wei, Liqiang Nie, Tat-Seng Chua.  
> *Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval*.

## Stage 1 Scope

Stage 1 only targets the **human-only DQU baseline**.

Primary benchmark:

- `FashionIQ`

Secondary benchmarks prepared but not required for the first milestone:

- `Shoes`
- `CIRR`

Main metric for Stage 1 reporting:

- `R@10`
- `R@50`

## Project Layout

```text
project_stage1_dqu_baseline/
├── configs/
├── data/
├── docs/
├── scripts/
├── slurm/
├── src/
├── tools/
├── environment.yml
├── requirements.txt
└── Makefile
```

## What Was Fixed Relative to the Upstream Repo

- data paths now work when you run `python src/train.py` from the repo root
- dataset path joins were normalized to avoid slash-related failures
- the model now exposes `backbone` and `pretrained` as actual runtime arguments
- training now saves a standard best checkpoint for FashionIQ and other datasets
- deterministic seeding is enabled by default
- Stage 1 benchmark scripts, data checks, Slurm template, and metric aggregation were added

## Environment

Recommended:

```bash
conda env create -f environment.yml
conda activate dqu-stage1
```

Or:

```bash
pip install -r requirements.txt
```

For `HPC3`, prefer the `conda env create -f environment.yml` path so `PyTorch` is installed with GPU support.

## Data Preparation

This repo already includes the **extra metadata files** released by the DQU authors, such as:

- corrected modification text dictionaries
- generated image captions
- keyword files for visual query writing

It now also includes the official FashionIQ annotation files:

- `cap.*.{train,val,test}.json`
- `split.*.{train,val,test}.json`

It does **not** include the full raw benchmark images.

Before training on FashionIQ, place the official data under:

```text
data/FashionIQ/
├── captions/
├── image_splits/
└── resized_image/
```

What you still need to provide manually:

- `data/FashionIQ/resized_image/dress/*.jpg`
- `data/FashionIQ/resized_image/shirt/*.jpg`
- `data/FashionIQ/resized_image/toptee/*.jpg`

If you only have the original images, use [tools/resize_images.py](tools/resize_images.py) to create the resized folders.

Then validate:

```bash
python scripts/check_dataset.py --dataset all
```

More details are in [data/README.md](data/README.md).

## Stage 1 Training

Train FashionIQ categories separately:

```bash
bash scripts/run_fashioniq_stage1.sh dress
bash scripts/run_fashioniq_stage1.sh shirt
bash scripts/run_fashioniq_stage1.sh toptee
```

Or run the full Stage 1 benchmark sequentially:

```bash
bash scripts/run_all_fashioniq_stage1.sh
```

Evaluate a saved checkpoint:

```bash
python src/evaluate.py --dataset dress --checkpoint checkpoints/fashioniq/dress/dress_stage1_best_model.pt
```

Aggregate the three FashionIQ category results:

```bash
python scripts/aggregate_fashioniq_metrics.py \
  --dress checkpoints/fashioniq/dress/dress_stage1_metrics_best.json \
  --shirt checkpoints/fashioniq/shirt/shirt_stage1_metrics_best.json \
  --toptee checkpoints/fashioniq/toptee/toptee_stage1_metrics_best.json
```

## HPC3 Usage

The folder is designed to be uploaded directly to `HPC3`.

Suggested order on `HPC3`:

1. create / activate environment
2. run dataset checks
3. run one category as a smoke test
4. submit the full Stage 1 benchmark by Slurm

See:

- [docs/STAGE1_DQU_BASELINE.md](docs/STAGE1_DQU_BASELINE.md)
- [docs/HPC3_STAGE1_RUNBOOK.md](docs/HPC3_STAGE1_RUNBOOK.md)
- [slurm/train_fashioniq_stage1.slurm](slurm/train_fashioniq_stage1.slurm)
- [slurm/train_fashioniq_stage1_array.slurm](slurm/train_fashioniq_stage1_array.slurm)

## Citation

If you use the original DQU baseline, cite:

```bibtex
@inproceedings{dqu_cir,
    author = {Wen, Haokun and Song, Xuemeng and Chen, Xiaolin and Wei, Yinwei and Nie, Liqiang and Chua, Tat-Seng},
    title = {Simple but Effective Raw-Data Level Multimodal Fusion for Composed Image Retrieval},
    booktitle = {Proceedings of the International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {229-239},
    publisher = {{ACM}},
    year = {2024}
}
```
