# StageR Final Report

## Local Paths
- Summary JSON: [STAGER_FINAL_SUMMARY.json](C:\Users\qh\Desktop\CIR\results\stageR_feature_rerank_20260423_final\STAGER_FINAL_SUMMARY.json)
- Summary Markdown: [STAGER_FINAL_SUMMARY.md](C:\Users\qh\Desktop\CIR\results\stageR_feature_rerank_20260423_final\STAGER_FINAL_SUMMARY.md)
- Synced artifacts root: [C:\Users\qh\Desktop\CIR\results\stageR_feature_rerank_20260423_final](C:\Users\qh\Desktop\CIR\results\stageR_feature_rerank_20260423_final)

## Previous Baseline
- Previous internal best: mean R@10 = 61.3494, mean R@50 = 80.8952
- DQU paper reported: mean R@10 = 61.97, mean R@50 = 81.56

## Best Mean R@10 Combo
- dress: dress_toprank_v6 blend=0.07
- shirt: shirt_toprank_v2 blend=0.04
- toptee: toptee_toprank_v2 blend=0.03
- mean R@10 = 62.1067
- mean R@50 = 80.9262
- delta vs previous internal best: R@10 +0.7573, R@50 +0.0310
- delta vs DQU paper: R@10 +0.1367, R@50 -0.6338

## Best Sum Combo
- dress: human_wide_v3 blend=0.05
- shirt: shirt_toprank_v2 blend=0.04
- toptee: toptee_toprank_v2 blend=0.03
- mean R@10 = 62.0406
- mean R@50 = 81.0915
- delta vs previous internal best: R@10 +0.6912, R@50 +0.1963
- delta vs DQU paper: R@10 +0.0706, R@50 -0.4685

## Conclusion
- This round produced a new internal best.
- We exceeded the DQU paper on mean R@10 with the best-R@10 combination.
- We did not exceed the DQU paper on mean R@50 in this round.
- All lightweight logs, histories, blend sweeps, asset summaries, and slurm logs are synced locally in this folder.
