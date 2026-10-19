# Stage 1 DQU Baseline

## Goal

Finish the **human-only DQU baseline** before any pseudo-triplet or flexible-prompt work starts.

Stage 1 answers only one question:

> Can we reproduce a stable DQU-CIR baseline on FashionIQ with a clean and reusable project structure?

## Benchmark

Primary benchmark:

- `FashionIQ`

Stage 1 run units:

- `dress`
- `shirt`
- `toptee`

Report:

- per-category `R@10`
- per-category `R@50`
- macro average over the three categories

## Deliverables

After Stage 1, the folder should contain:

- runnable training code
- runnable evaluation code
- benchmark configs
- dataset checking script
- full FashionIQ annotation json files
- Slurm script for HPC3
- best checkpoint for each FashionIQ category
- metrics json for each category
- aggregated FashionIQ summary

## Recommended Execution Order

1. Create the environment.
2. Put the FashionIQ resized images under `data/FashionIQ/resized_image/{dress,shirt,toptee}`.
3. Run dataset checks for `dress`, `shirt`, and `toptee`.
4. Smoke test one category, usually `dress`.
5. Train all three FashionIQ categories.
6. Aggregate metrics.
7. Freeze Stage 1 outputs for later Stage 2 comparison.

## HPC3 Notes

Use:

- `sbatch` for real training
- `tmux + srun` only for debugging

Suggested checkpoint layout:

```text
checkpoints/
└── fashioniq/
    ├── dress/
    ├── shirt/
    └── toptee/
```

## Exit Criteria

Stage 1 is complete when:

1. `dress`, `shirt`, and `toptee` all run end-to-end
2. each category has a best checkpoint and metrics json
3. FashionIQ aggregate summary is produced
4. the project can be uploaded to HPC3 without structural edits
