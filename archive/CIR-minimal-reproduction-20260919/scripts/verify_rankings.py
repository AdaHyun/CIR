#!/usr/bin/env python3
"""Recompute the paper's CN and final reranking metrics from saved rankings."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RANKING_DIR = ROOT / "results" / "rankings"
OUTPUT = ROOT / "results" / "reproduced_metrics.json"
CATEGORIES = ("dress", "shirt", "toptee")
CUTOFFS = (10, 50)

EXPECTED_COUNTS = {
    "dress": {"n": 2017, "cn": {10: 1086, 50: 1556}, "rmean": {10: 1107, 50: 1560}},
    "shirt": {"n": 2038, "cn": {10: 1288, 50: 1634}, "rmean": {10: 1308, 50: 1643}},
    "toptee": {"n": 1961, "cn": {10: 1314, 50: 1674}, "rmean": {10: 1315, 50: 1673}},
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"query_index", "source_id", "target_id", "target_rank", "top_ids"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number}: missing {sorted(missing)}")
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    hits = {cutoff: sum(int(row["target_rank"]) <= cutoff for row in rows) for cutoff in CUTOFFS}
    return {
        "queries": n,
        "hits": {f"R@{k}": hits[k] for k in CUTOFFS},
        "recall_percent": {f"R@{k}": hits[k] * 100.0 / n for k in CUTOFFS},
    }


def main() -> None:
    result: dict[str, object] = {"categories": {}}
    macro = {"cn": {10: [], 50: []}, "rmean": {10: [], 50: []}}

    for category in CATEGORIES:
        cn_rows = load_jsonl(RANKING_DIR / f"{category}_cn_rankings.jsonl")
        rr_rows = load_jsonl(RANKING_DIR / f"{category}_rmean_rankings.jsonl")
        expected = EXPECTED_COUNTS[category]
        if len(cn_rows) != expected["n"] or len(rr_rows) != expected["n"]:
            raise AssertionError(f"Unexpected query count for {category}")

        for left, right in zip(cn_rows, rr_rows):
            identity_left = (left["query_index"], left["source_id"], left["target_id"])
            identity_right = (right["query_index"], right["source_id"], right["target_id"])
            if identity_left != identity_right:
                raise AssertionError(f"Ranking identity mismatch for {category}: {identity_left}")

        cn_summary = summarize(cn_rows)
        rr_summary = summarize(rr_rows)
        transitions = {}
        for cutoff in CUTOFFS:
            corrected = sum(
                int(cn["target_rank"]) > cutoff and int(rr["target_rank"]) <= cutoff
                for cn, rr in zip(cn_rows, rr_rows)
            )
            harmed = sum(
                int(cn["target_rank"]) <= cutoff and int(rr["target_rank"]) > cutoff
                for cn, rr in zip(cn_rows, rr_rows)
            )
            transitions[f"R@{cutoff}"] = {"corrected": corrected, "harmed": harmed}

        for stage, summary in (("cn", cn_summary), ("rmean", rr_summary)):
            for cutoff in CUTOFFS:
                actual = summary["hits"][f"R@{cutoff}"]
                if actual != expected[stage][cutoff]:
                    raise AssertionError(
                        f"{category} {stage} R@{cutoff}: expected {expected[stage][cutoff]}, got {actual}"
                    )
                macro[stage][cutoff].append(summary["recall_percent"][f"R@{cutoff}"])

        result["categories"][category] = {
            "candidate_neighbourhood": cn_summary,
            "proposed_rmean": rr_summary,
            "transitions": transitions,
        }

    result["macro_percent"] = {}
    for stage, label in (("cn", "candidate_neighbourhood"), ("rmean", "proposed_rmean")):
        r10 = sum(macro[stage][10]) / len(CATEGORIES)
        r50 = sum(macro[stage][50]) / len(CATEGORIES)
        result["macro_percent"][label] = {"R@10": r10, "R@50": r50, "mR": (r10 + r50) / 2.0}

    final = result["macro_percent"]["proposed_rmean"]
    expected_final = {"R@10": 62.0406, "R@50": 81.0915, "mR": 71.5660}
    for metric, expected_value in expected_final.items():
        if abs(final[metric] - expected_value) > 5e-5:
            raise AssertionError(f"Macro {metric}: expected {expected_value}, got {final[metric]}")

    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nPASS: reproduced macro R@10={final['R@10']:.4f}, R@50={final['R@50']:.4f}, mR={final['mR']:.4f}")


if __name__ == "__main__":
    main()
