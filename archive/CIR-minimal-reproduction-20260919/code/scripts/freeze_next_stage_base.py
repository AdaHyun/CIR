from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_history_tail(path: Path) -> dict:
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last or {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dress_metrics", type=Path, required=True)
    parser.add_argument("--shirt_metrics", type=Path, required=True)
    parser.add_argument("--toptee_metrics", type=Path, required=True)
    parser.add_argument("--dress_history", type=Path, required=True)
    parser.add_argument("--shirt_history", type=Path, required=True)
    parser.add_argument("--toptee_history", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "dress": load_json(args.dress_metrics),
        "shirt": load_json(args.shirt_metrics),
        "toptee": load_json(args.toptee_metrics),
    }
    summary = {
        "dress_r10": metrics["dress"]["dress_r10"],
        "dress_r50": metrics["dress"]["dress_r50"],
        "shirt_r10": metrics["shirt"]["shirt_r10"],
        "shirt_r50": metrics["shirt"]["shirt_r50"],
        "toptee_r10": metrics["toptee"]["toptee_r10"],
        "toptee_r50": metrics["toptee"]["toptee_r50"],
    }
    summary["fashioniq_mean_r10"] = (
        summary["dress_r10"] + summary["shirt_r10"] + summary["toptee_r10"]
    ) / 3.0
    summary["fashioniq_mean_r50"] = (
        summary["dress_r50"] + summary["shirt_r50"] + summary["toptee_r50"]
    ) / 3.0
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    history_snapshot = {
        "dress_last_epoch": load_history_tail(args.dress_history),
        "shirt_last_epoch": load_history_tail(args.shirt_history),
        "toptee_last_epoch": load_history_tail(args.toptee_history),
    }
    (args.output_dir / "train_history.json").write_text(
        json.dumps(history_snapshot, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = [
        "# Next-Stage Frozen Base",
        "",
        "这一组结果冻结为下一阶段唯一基线，来源于当前最优的 `DQU-PFP(new) human-only`。",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Per-Category",
        "",
        "| category | R@10 | R@50 | model_variant | pseudo_label |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for category in ["dress", "shirt", "toptee"]:
        report.append(
            "| {} | {:.4f} | {:.4f} | {} | {} |".format(
                category,
                metrics[category][f"{category}_r10"],
                metrics[category][f"{category}_r50"],
                metrics[category].get("model_variant", ""),
                metrics[category].get("pseudo_label", ""),
            )
        )
    report.extend(
        [
            "",
            "## History Tail",
            "",
            "```json",
            json.dumps(history_snapshot, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
