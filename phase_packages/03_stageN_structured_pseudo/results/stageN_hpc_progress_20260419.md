# Stage N HPC Progress 2026-04-19

## Completed

- `N3 structured pseudo` generation completed for all three FashionIQ categories.
- Fixed a `DataLoader` crash in mixed training caused by human samples missing the `sample_weight` field while pseudo samples had it.
- Synced the fix to `HPC3`.

## N3 Outputs

### Raw structured pseudo

| Category | Count | Mean quality | Mean direction margin |
| --- | ---: | ---: | ---: |
| dress | 2412 | 0.7054 | 0.0428 |
| shirt | 2757 | 0.7066 | 0.0317 |
| toptee | 2245 | 0.7040 | 0.0354 |

### High-quality subset

Filter:
- `quality_score >= 0.72`
- `direction_margin >= 0.03`

| Category | Count | Mean quality | Mean direction margin |
| --- | ---: | ---: | ---: |
| dress | 652 | 0.7498 | 0.0781 |
| shirt | 599 | 0.7522 | 0.0660 |
| toptee | 510 | 0.7460 | 0.0694 |

Remote paths:
- `/data/user/hqi881/CIR/project_stage1_dqu_baseline/artifacts/stageN3_structured/raw`
- `/data/user/hqi881/CIR/project_stage1_dqu_baseline/artifacts/stageN3_structured/highq_q72_m03`

## Human-only reference

Current `Stage D human-only` best:

| Category | R@10 | R@50 |
| --- | ---: | ---: |
| dress | 54.7348 | 76.7476 |
| shirt | 62.6104 | 80.4711 |
| toptee | 66.0887 | 85.4156 |

## Running N4 Jobs

### v2

Settings:
- `mix_mode in {static, curriculum, dynamic_weighted}`
- `pseudo_start_epoch = 3`
- `pseudo_max_ratio = 0.25`

Dress:
- `269429` `n4s2_dress`
- `269427` `n4c2_dress`
- `269428` `n4d2_dress`

Toptee:
- `269433` `n4s2_toptee`
- `269431` `n4c2_toptee`
- `269432` `n4d2_toptee`

Shirt:
- `269435` `n4s2_shirt`
- `269437` `n4c2_shirt`
- `269436` `n4d2_shirt`

### v3

Conservative curriculum sweep:
- `pseudo_start_epoch = 5`
- `pseudo_max_ratio = 0.15`

Jobs:
- `269439` `n4c3_dress`
- `269440` `n4c3_toptee`
- `269441` `n4c3_shirt`

### v4

High-quality curriculum sweep:
- pseudo source: `highq_q72_m03`
- `pseudo_start_epoch = 5`
- `pseudo_max_ratio = 0.10`
- `pseudo_quality_power = 1.2`

Jobs:
- `269450` `n4c4_dress`
- `269452` `n4c4_toptee`
- `269451` `n4c4_shirt`

## Early observations

### v2 curriculum

Best early validation values observed so far:

| Category | R@10 | R@50 |
| --- | ---: | ---: |
| dress | 54.4869 | 76.4998 |
| shirt | 62.4141 | 80.0294 |
| toptee | 65.7318 | 84.8547 |

These are still below the current `Stage D human-only` reference on all three categories.

### v2 dynamic_weighted

Observed on dress after pseudo injection:
- fell to around `52.9995 / 75.8552`

This was the main signal that triggered the conservative `v3` and high-quality `v4` sweeps.

### v3 and v4

At the time of this snapshot:
- `v3` is still in early epochs and not yet at pseudo-injection evaluation
- `v4` is also still in early epochs

## Code fix applied

File:
- [datasets.py](C:\Users\qh\Desktop\CIR\project_stage1_dqu_baseline\src\datasets.py)

Change:
- add `sample_weight = 1.0` to human samples in `FashionIQ.__getitem__`
- add `sample_weight = 1.0` to human samples in `Shoes.__getitem__`

Reason:
- mixed batches contained both human and pseudo samples
- pseudo samples had `sample_weight`
- human samples did not
- default PyTorch `collate` then raised `KeyError: 'sample_weight'`

## Next monitoring focus

1. Watch `v4 high-quality curriculum` first.
2. Compare `v4` against `v2 curriculum` once pseudo is injected.
3. If `v4` still does not beat `Stage D human-only`, stop low-hope lines and move to stricter pseudo filtering or per-category strategy.
