# DQU Pseudo Ablation Table

| experiment | mean R@10 | mean R@50 |
| --- | ---: | ---: |
| E1 DQU human-only | 61.0292 | 80.1985 |
| E2 DQU + all pseudo | 59.9681 | 80.3474 |
| E3 DQU + filtered pseudo | 58.9209 | 79.8498 |
| E4 DQU + filtered pseudo + curriculum | 60.9817 | 80.4854 |

## Delta

```json
{
  "E2_minus_E1_r10": -1.0611258874110518,
  "E3_minus_E2_r10": -1.0472366484232936,
  "E4_minus_E3_r10": 2.060841988070436,
  "E4_minus_E1_r10": -0.04752054776390935,
  "E2_minus_E1_r50": 0.14886710327381536,
  "E3_minus_E2_r50": -0.49755676801865434,
  "E4_minus_E3_r50": 0.6355609115328633,
  "E4_minus_E1_r50": 0.28687124678802434
}
```
