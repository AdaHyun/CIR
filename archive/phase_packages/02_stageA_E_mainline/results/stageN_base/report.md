# Next-Stage Frozen Base

这一组结果冻结为下一阶段唯一基线，来源于当前最优的 `DQU-PFP(new) human-only`。

## Summary

```json
{
  "dress_r10": 54.73475458601884,
  "dress_r50": 76.7476450173525,
  "shirt_r10": 62.610402355250244,
  "shirt_r50": 80.47105004906771,
  "toptee_r10": 66.08873023967364,
  "toptee_r50": 85.41560428352881,
  "fashioniq_mean_r10": 61.14462906031424,
  "fashioniq_mean_r50": 80.87809978331633
}
```

## Per-Category

| category | R@10 | R@50 | model_variant | pseudo_label |
| --- | ---: | ---: | --- | --- |
| dress | 54.7348 | 76.7476 | dqu_pfp_refcond | none |
| shirt | 62.6104 | 80.4711 | dqu_pfp_refcond | none |
| toptee | 66.0887 | 85.4156 | dqu_pfp_refcond | none |

## History Tail

```json
{
  "dress_last_epoch": {
    "epoch": 9,
    "train_loss": 0.10051564487814903,
    "score": 129.1026276648488,
    "best_score": 131.48239960337133,
    "used_pseudo": false,
    "dress_r1": 21.715418939018345,
    "dress_r10": 53.04908279623203,
    "dress_r50": 76.05354486861675
  },
  "shirt_last_epoch": {
    "epoch": 6,
    "train_loss": 0.11284641600896915,
    "score": 139.05789990186457,
    "best_score": 143.08145240431796,
    "used_pseudo": false,
    "shirt_r1": 28.851815505397447,
    "shirt_r10": 60.10794896957802,
    "shirt_r50": 78.94995093228655
  },
  "toptee_last_epoch": {
    "epoch": 8,
    "train_loss": 0.08411497703755882,
    "score": 147.7817440081591,
    "best_score": 151.50433452320243,
    "used_pseudo": false,
    "toptee_r1": 30.90260071392147,
    "toptee_r10": 63.488016318205,
    "toptee_r50": 84.29372768995411
  }
}
```
