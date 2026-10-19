# Stage N Summary 2026-04-19

## 本地已拉回

目录：
- [stageN_sync_20260419](C:\Users\qh\Desktop\CIR\results\stageN_sync_20260419)

已同步内容：
- `N3 structured pseudo` 产物
- `N4` 各轮的 `metrics_best.json`
- `history.jsonl`
- `train.log`
- `slurm/stageN` 日志

主汇总文件：
- [stageN_summary_20260419.json](C:\Users\qh\Desktop\CIR\results\stageN_sync_20260419\stageN_summary_20260419.json)

## 结果总表

| Run | dress R@10 / R@50 | shirt R@10 / R@50 | toptee R@10 / R@50 | mean R@10 | mean R@50 |
| --- | --- | --- | --- | ---: | ---: |
| Stage D human-only | 54.7348 / 76.7476 | 62.6104 / 80.4711 | 66.0887 / 85.4156 | 61.1446 | 80.8781 |
| N4 v2 static | 54.0902 / 76.4502 | 62.6595 / 79.8332 | 65.8848 / 84.8037 | 60.8782 | 80.3623 |
| N4 v2 curriculum | 54.4869 / 76.4998 | 62.4141 / 80.0294 | 65.7318 / 84.8547 | 60.8776 | 80.4613 |
| N4 v2 dynamic | 54.1398 / 76.5989 | 62.2179 / 80.2748 | 65.7318 / 84.9057 | 60.6965 | 80.5931 |
| N4 v3 curriculum | 53.9415 / 76.9460 | 62.1688 / 80.1766 | 66.5987 / 85.3646 | 60.9030 | 80.8291 |
| N4 v4 highq curriculum | 54.0902 / 76.2023 | 62.6104 / 80.2748 | 66.3947 / 85.5176 | 61.0318 | 80.6649 |

## 当前结论

- 当前这轮 `N4` 最好的是 `v4 highq curriculum`。
- 但它仍然没有超过 `Stage D human-only`。

差值相对 `Stage D human-only`：

| Run | mean R@10 delta | mean R@50 delta |
| --- | ---: | ---: |
| N4 v2 static | -0.2664 | -0.5158 |
| N4 v2 curriculum | -0.2670 | -0.4168 |
| N4 v2 dynamic | -0.4481 | -0.2850 |
| N4 v3 curriculum | -0.2416 | -0.0490 |
| N4 v4 highq curriculum | -0.1128 | -0.2132 |

## 解释

- `v2` 三条线都没有形成稳定提升。
- `v3` 的保守 curriculum 把 `mean R@50` 拉到了接近 `Stage D`，但 `mean R@10` 还是低。
- `v4` 的高质量 pseudo 子集是这一轮最接近 `Stage D` 的方案，尤其：
  - `shirt R@10` 已经追平 `Stage D`
  - `toptee R@50` 还略高于 `Stage D`
- 主要拖累仍然在 `dress`。

## 远端状态

这些 job 在 `sacct` 里显示为 `FAILED 127`，但对应的 `metrics_best.json` 已经完整落下来了。
从日志看，训练过程本身是正常跑完若干 epoch 并完成验证后才退出，因此目前应以落下来的 metrics 文件为准。
