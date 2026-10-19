# StageP BankDistill Final Summary

Date: 2026-04-23

## Goal

Improve on the current internal best FashionIQ result from `Stage D`:

- `mean R@10 = 61.1446`
- `mean R@50 = 80.8781`

## What Was Run

### Bank Construction

- `banksmoke_dress_ue`
  - `accepted_count = 163 / 200`
  - `accepted_ratio = 0.8150`
- `banksmoke_dress_v2`
  - target-anchored soft-weight fix applied
  - `accepted_count = 37 / 50`
  - `accepted_ratio = 0.7400`
- `bankfull_dress`
  - `accepted_count = 5089 / 5985`
  - `accepted_ratio = 0.8503`
- `bankfull_shirt`
  - `accepted_count = 4675 / 5988`
  - `accepted_ratio = 0.7807`
- `bankfull_toptee`
  - `accepted_count = 5339 / 6027`
  - `accepted_ratio = 0.8858`

### Training Iterations

#### v1 / v2

- `trainbd_dress_v1`, `trainbd_toptee_v1`, `trainbd_dress_v2`
  - failed
  - root cause: CUDA OOM in bank-loss target feature extraction

#### v3_safe

Configuration:

- `batch_size = 8`
- `bank_soft_k = 3`
- `soft_loss_weight = 0.15`
- `conflict_loss_weight = 0.08`
- `conflict_margin = 0.04`
- resumed from `Stage D`

Best per-category results:

- `dress`
  - best epoch: `2`
  - `R@10 = 53.8423`
  - `R@50 = 77.1443`
- `shirt`
  - best epoch: `3`
  - `R@10 = 63.1992`
  - `R@50 = 80.1766`
- `toptee`
  - best epoch: `2`
  - `R@10 = 67.0066`
  - `R@50 = 85.3646`

Observed behavior:

- `dress` never exceeded `Stage D`
- `shirt` exceeded `Stage D` on `R@10`
- `toptee` exceeded `Stage D` on `R@10`
- after the best epoch, `dress/toptee` started to fall back

#### v4b_delay

Configuration:

- lower `lr` / `clip_lr`
- lower soft/conflict weights
- smaller `bank_soft_k = 2`
- delayed bank start

Results:

- `dress`
  - `R@10 = 53.8919`
  - `R@50 = 76.2023`
- `shirt`
  - `R@10 = 61.7272`
  - `R@50 = 80.0785`
- `toptee`
  - `R@10 = 65.7828`
  - `R@50 = 84.7527`

Conclusion:

- `v4b` was clearly worse than `v3_safe`
- jobs were stopped after confirmation

#### v5_dress_rank5soft

Configuration:

- `dress` only
- strict bank: `target_rank <= 5`
- `soft-only`
- lower `lr = 2e-5`
- lower `clip_lr = 2e-7`
- delayed bank start

First result:

- `dress`
  - `R@10 = 53.0491`
  - `R@50 = 76.2519`

Conclusion:

- worse than both `Stage D dress` and `v3 dress`
- job was stopped

## Best Result Found In This Round

The best result found in this round is `StageP v3_safe`.

Best category-wise results inside `v3_safe`:

- `dress`
  - `R@10 = 53.8423`
  - `R@50 = 77.1443`
- `shirt`
  - `R@10 = 63.1992`
  - `R@50 = 80.1766`
- `toptee`
  - `R@10 = 67.0066`
  - `R@50 = 85.3646`

Combined mean:

- `mean R@10 = 61.3494`
- `mean R@50 = 80.8952`
- `sum score = 142.2446`

Compared with `Stage D`:

- `mean R@10`: `+0.2048`
- `mean R@50`: `+0.0171`
- `sum score`: `+0.2218`

## Final Judgment

- `bank-distill` did not make every category stronger than `Stage D`
- `dress` remained the bottleneck
- but `v3_safe` still became the best overall result of this round because `shirt` and `toptee` improvements were enough to offset the weaker `dress`
- therefore, the best practical result from this round is simply:
  - `StageP v3_safe`

## Local Sync

All lightweight artifacts have been pulled back to:

- `C:\\Users\\qh\\Desktop\\CIR\\results\\stageP_bankdistill_20260423\\sync_light_20260423`

This local sync contains:

- bank summaries
- `v3` metrics / histories / train logs
- `v4b` metrics / histories / train logs
- `v5` metrics / histories / train logs
- related Slurm `out/err` logs
