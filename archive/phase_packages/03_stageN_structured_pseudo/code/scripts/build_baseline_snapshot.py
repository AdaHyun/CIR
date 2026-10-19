from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_env(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def render_config_table(configs: dict[str, dict[str, str]]) -> str:
    headers = ["category", "DATASET", "FASHIONIQ_SPLIT", "LR", "CLIP_LR", "BATCH_SIZE", "NUM_EPOCHS", "NUM_WORKERS", "BACKBONE"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for category, config in configs.items():
        row = [category] + [config.get(header, "") for header in headers[1:]]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_result_table(metrics: dict[str, dict[str, float]]) -> str:
    rows = [
        "| category | R@10 | R@50 |",
        "| --- | ---: | ---: |",
    ]
    for category in ["dress", "shirt", "toptee"]:
        rows.append(
            "| {} | {:.4f} | {:.4f} |".format(
                category,
                metrics[category][f"{category}_r10"],
                metrics[category][f"{category}_r50"],
            )
        )
    mean_r10 = sum(metrics[category][f"{category}_r10"] for category in metrics) / 3.0
    mean_r50 = sum(metrics[category][f"{category}_r50"] for category in metrics) / 3.0
    rows.append("| mean | {:.4f} | {:.4f} |".format(mean_r10, mean_r50))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_dir", type=Path, required=True)
    parser.add_argument("--dress", type=Path, required=True)
    parser.add_argument("--shirt", type=Path, required=True)
    parser.add_argument("--toptee", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    configs = {
        category: load_env(args.config_dir / f"fashioniq_{category}.env")
        for category in ["dress", "shirt", "toptee"]
    }
    metrics = {
        "dress": load_json(args.dress),
        "shirt": load_json(args.shirt),
        "toptee": load_json(args.toptee),
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    config_md = [
        "# Baseline Config Snapshot",
        "",
        "冻结后的 Stage A baseline 使用原始 DQU 单次训练口径，后续所有对照都以这一份配置为准。",
        "",
        render_config_table(configs),
        "",
    ]
    (args.output_dir / "baseline_config_snapshot.md").write_text("\n".join(config_md), encoding="utf-8")

    result_md = [
        "# Baseline Result Snapshot",
        "",
        "这一份结果是当前项目后续实验的唯一基准线。",
        "",
        render_result_table(metrics),
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    (args.output_dir / "baseline_result_snapshot.md").write_text("\n".join(result_md), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
