from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dress", type=Path, required=True)
    parser.add_argument("--shirt", type=Path, required=True)
    parser.add_argument("--toptee", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/fashioniq/fashioniq_stage1_summary.json"),
    )
    args = parser.parse_args()

    metrics = {
        "dress": load_metrics(args.dress),
        "shirt": load_metrics(args.shirt),
        "toptee": load_metrics(args.toptee),
    }

    summary = {
        "dress_r10": metrics["dress"]["dress_r10"],
        "dress_r50": metrics["dress"]["dress_r50"],
        "shirt_r10": metrics["shirt"]["shirt_r10"],
        "shirt_r50": metrics["shirt"]["shirt_r50"],
        "toptee_r10": metrics["toptee"]["toptee_r10"],
        "toptee_r50": metrics["toptee"]["toptee_r50"],
        "fashioniq_mean_r10": (
            metrics["dress"]["dress_r10"]
            + metrics["shirt"]["shirt_r10"]
            + metrics["toptee"]["toptee_r10"]
        )
        / 3.0,
        "fashioniq_mean_r50": (
            metrics["dress"]["dress_r50"]
            + metrics["shirt"]["shirt_r50"]
            + metrics["toptee"]["toptee_r50"]
        )
        / 3.0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
