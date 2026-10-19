# Baseline Config Snapshot

冻结后的 Stage A baseline 使用原始 DQU 单次训练口径，后续所有对照都以这一份配置为准。

| category | DATASET | FASHIONIQ_SPLIT | LR | CLIP_LR | BATCH_SIZE | NUM_EPOCHS | NUM_WORKERS | BACKBONE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dress | dress | val-split | 1e-4 | 1e-6 | 16 | 100 | 8 | ViT-H-14 |
| shirt | shirt | val-split | 1e-4 | 1e-6 | 16 | 100 | 8 | ViT-H-14 |
| toptee | toptee | val-split | 1e-4 | 1e-6 | 16 | 100 | 8 | ViT-H-14 |
