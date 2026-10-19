from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_summary_row(label: str, summary: dict) -> str:
    return "| {} | {:.4f} | {:.4f} |".format(
        label,
        summary["fashioniq_mean_r10"],
        summary["fashioniq_mean_r50"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e1", type=Path, required=True)
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--e3", type=Path, required=True)
    parser.add_argument("--e4", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    summaries = {
        "E1 DQU human-only": load_json(args.e1),
        "E2 DQU + all pseudo": load_json(args.e2),
        "E3 DQU + filtered pseudo": load_json(args.e3),
        "E4 DQU + filtered pseudo + curriculum": load_json(args.e4),
    }

    report = {
        label: {
            "fashioniq_mean_r10": summary["fashioniq_mean_r10"],
            "fashioniq_mean_r50": summary["fashioniq_mean_r50"],
        }
        for label, summary in summaries.items()
    }
    report["delta"] = {
        "E2_minus_E1_r10": summaries["E2 DQU + all pseudo"]["fashioniq_mean_r10"] - summaries["E1 DQU human-only"]["fashioniq_mean_r10"],
        "E3_minus_E2_r10": summaries["E3 DQU + filtered pseudo"]["fashioniq_mean_r10"] - summaries["E2 DQU + all pseudo"]["fashioniq_mean_r10"],
        "E4_minus_E3_r10": summaries["E4 DQU + filtered pseudo + curriculum"]["fashioniq_mean_r10"] - summaries["E3 DQU + filtered pseudo"]["fashioniq_mean_r10"],
        "E4_minus_E1_r10": summaries["E4 DQU + filtered pseudo + curriculum"]["fashioniq_mean_r10"] - summaries["E1 DQU human-only"]["fashioniq_mean_r10"],
        "E2_minus_E1_r50": summaries["E2 DQU + all pseudo"]["fashioniq_mean_r50"] - summaries["E1 DQU human-only"]["fashioniq_mean_r50"],
        "E3_minus_E2_r50": summaries["E3 DQU + filtered pseudo"]["fashioniq_mean_r50"] - summaries["E2 DQU + all pseudo"]["fashioniq_mean_r50"],
        "E4_minus_E3_r50": summaries["E4 DQU + filtered pseudo + curriculum"]["fashioniq_mean_r50"] - summaries["E3 DQU + filtered pseudo"]["fashioniq_mean_r50"],
        "E4_minus_E1_r50": summaries["E4 DQU + filtered pseudo + curriculum"]["fashioniq_mean_r50"] - summaries["E1 DQU human-only"]["fashioniq_mean_r50"],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# DQU Pseudo Ablation Table",
        "",
        "| experiment | mean R@10 | mean R@50 |",
        "| --- | ---: | ---: |",
    ]
    for label in summaries:
        lines.append(render_summary_row(label, summaries[label]))
    lines.extend(
        [
            "",
            "## Delta",
            "",
            "```json",
            json.dumps(report["delta"], indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
