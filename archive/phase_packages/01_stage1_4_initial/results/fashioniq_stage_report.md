# FashionIQ Stage Report

## Stage 1 DQU Baseline

- mean R@10: 60.5160
- mean R@50: 80.3337

## Stage 2 DQU-PFP

- mean R@10: 59.9250
- mean R@50: 79.9977
- delta vs Stage 1 R@10: -0.5910
- delta vs Stage 1 R@50: -0.3360

## Stage 3 Pseudo-Triplets

- dress: kept 2604 / 61001 pairs, avg_quality_score=0.6807, avg_direction_margin=0.0446
- shirt: kept 2694 / 59194 pairs, avg_quality_score=0.6824, avg_direction_margin=0.0404
- toptee: kept 1631 / 61018 pairs, avg_quality_score=0.6755, avg_direction_margin=0.0381

## Stage 4 DQU-PFP + Filtered Pseudo

- mean R@10: 60.6321
- mean R@50: 80.6308
- delta vs Stage 2 R@10: +0.7071
- delta vs Stage 2 R@50: +0.6330
- delta vs Stage 1 R@10: +0.1162
- delta vs Stage 1 R@50: +0.2970

## Per-category

- dress: S1 R@10=54.3381, S2 R@10=54.1894, S4 R@10=54.6356; S1 R@50=76.1527, S2 R@50=76.5493, S4 R@50=76.3510
- shirt: S1 R@10=61.5800, S2 R@10=61.3837, S4 R@10=61.5800; S1 R@50=79.6369, S2 R@50=79.0481, S4 R@50=80.1766
- toptee: S1 R@10=65.6298, S2 R@10=64.2019, S4 R@10=65.6808; S1 R@50=85.2116, S2 R@50=84.3957, S4 R@50=85.3646
