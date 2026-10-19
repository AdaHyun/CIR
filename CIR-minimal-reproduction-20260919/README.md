# CIR minimal paper-results reproduction package

This package is the smallest self-contained release that recomputes the paper's
reported Candidate-Neighbourhood (CN) and final residual-reranking results from
the genuine archived query-level rankings. It verifies every validation query,
not only the aggregate numbers shown in the paper.

## Quick verification

The ranking verifier uses only Python's standard library:

```bash
python3 scripts/verify_rankings.py
```

The expected final macro results are:

- Recall@10: `62.0406`
- Recall@50: `81.0915`
- mR: `71.5660`

The full packaged check verifies file integrity, recomputes the ranking-level
metrics, and reruns the selected residual rerankers on the included frozen
validation features:

```bash
PYTHON_BIN=/path/to/python-with-pytorch bash scripts/run_verify.sh
```

PyTorch checkpoint files must only be loaded after their SHA-256 values have
been checked against `manifests/SHA256SUMS.txt`.

## Included

- The executed project source, configurations, utilities, launch scripts, and
  environment files copied from the HPC experiment tree.
- Genuine per-query CN and final R-mean rankings for dress, shirt, and toptee.
- Reported main-comparison and progressive-ablation values.
- Frozen validation query/image features and Top-100 candidate lists.
- The three selected residual-reranker checkpoints and fixed blend weights.
- Reproduction and integrity-check scripts.

## Intentionally excluded

- OpenCLIP ViT-H/14 `laion2B-s32B-b79K` (3.945 GB); it is a separately downloadable public dependency.
- Raw FashionIQ images; they are not needed for cached-feature evaluation and should be obtained through the dataset's authorized distribution channel.
- Virtual environments, package caches, duplicate data archives, failed jobs, sweep checkpoints, and raw Slurm logs.

## Evidence boundary

The original final Stage-P/Candidate-Neighbourhood network checkpoints for dress,
shirt, and toptee are absent from the local archive and all accessible HPC CIR
directories. Consequently, this package reproduces the reported evaluation from
the exact frozen features and genuine query-level rankings, but it cannot regenerate
those first-stage features from raw images. It must not be described as a complete
end-to-end retraining release.

The reranker training implementation represented in the historical experiment
inserted the target with a base-score input of 1.0, whereas inference used actual
candidate similarities. The included results characterize that executed procedure.

## Layout

```text
code/                    cleaned source and configurations
assets/<category>/       frozen validation features and Top-100 candidates
checkpoints/reranker/    selected dress, shirt, and toptee rerankers
results/rankings/        genuine CN and R-mean query-level rankings
results/                 reported and recomputed metrics
scripts/                 verification and cached-feature evaluation
manifests/               file hashes and inventory
```
