# Release inventory and provenance

Inventory date: 2026-09-19

## Evaluation evidence

The six files in `results/rankings/` are the genuine query-level exports used for
the dissertation and ICASSP paper. They contain 2,017 dress queries, 2,038 shirt
queries, and 1,961 toptee queries for both the Candidate-Neighbourhood order and
the selected final R-mean order.

`scripts/verify_rankings.py` verifies query identities, exact hit counts, macro
metrics, and corrected/harmed transitions. It requires no third-party package.

## Binary evaluation assets

`assets/<category>/` contains the frozen validation query
features, validation image features, and Top-100 candidate assets produced from
the selected Stage-P retriever. `checkpoints/reranker/<category>/` then contains
the selected residual scorer. These files allow
`scripts/evaluate_cached_reranker.py` to recompute the final ordering without
OpenCLIP or raw images. All of these files are included in this HPC release.

The selected operating points are:

| Category | Reranker run | Blend weight |
|---|---|---:|
| dress | `human_wide_v3` | 0.05 |
| shirt | `shirt_toprank_v2` | 0.04 |
| toptee | `toptee_toprank_v2` | 0.03 |

## Source provenance

- Executed source, configurations, utilities, and job launchers: the HPC3 project
  tree at packaging time.
- Frozen features and reranker checkpoints: HPC3
  `/data/user/hqi881/CIR/project_stage1_dqu_baseline/artifacts/stageR_feature_rerank_20260423`.
- Query-level rankings: HPC3 CPU export dated 2026-09-11.
- Reported comparison and ablation values: final dissertation/ICASSP evidence set.

## Known unavailable files

The following original first-stage checkpoints were not present locally, on HPC3,
or in the accessible HPC2/HPC4 project locations at inventory time:

- `stageA_dqu_baseline_v3/{dress,shirt,toptee}/*_stageA3_best_model.pt`
- `stageD_refcond/{dress,shirt,toptee}/*_stageD_refcond_best_model.pt`
- `stageP_bankdistill_20260423/v3_safe/{dress,shirt,toptee}/*_stagePbd3_best_model.pt`

Their absence prevents raw-image regeneration of the frozen first-stage features.
It does not affect verification of the packaged query-level results or evaluation
of the packaged residual rerankers on those features.
