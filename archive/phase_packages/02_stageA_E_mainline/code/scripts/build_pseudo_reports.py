from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected label=path")
    label, raw_path = value.split("=", 1)
    return label, Path(raw_path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", action="append", type=parse_labeled_path, default=[])
    parser.add_argument("--sweep", action="append", type=parse_labeled_path, default=[])
    parser.add_argument("--distribution", action="append", type=parse_labeled_path, default=[])
    parser.add_argument("--teacher_md", type=Path, required=True)
    parser.add_argument("--sweep_md", type=Path, required=True)
    parser.add_argument("--distribution_md", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    output_payload = {}

    teacher_rows = []
    teacher_payload = {}
    for label, path in args.teacher:
        data = load_json(path)
        teacher_payload[label] = data
        teacher_rows.append(
            [
                label,
                data.get("category", ""),
                str(data.get("pairs_kept", "")),
                "{:.4f}".format(float(data.get("avg_quality_score", 0.0))),
                "{:.4f}".format(float(data.get("avg_direction_margin", 0.0))),
            ]
        )
    output_payload["teacher"] = teacher_payload

    sweep_rows = []
    sweep_payload = {}
    for label, path in args.sweep:
        data = load_json(path)
        sweep_payload[label] = data
        sweep_rows.append(
            [
                label,
                data.get("category", ""),
                str(data.get("pairs_kept", "")),
                str(data.get("keep_top_n", "")),
                "{:.4f}".format(float(data.get("score_threshold", 0.0))),
                "{:.4f}".format(float(data.get("avg_quality_score", 0.0))),
            ]
        )
    output_payload["sweep"] = sweep_payload

    distribution_sections = ["# Pseudo Distribution Report", ""]
    distribution_payload = {}
    for label, path in args.distribution:
        data = load_json(path)
        distribution_payload[label] = data
        distribution_sections.extend(
            [
                f"## {label}",
                "",
                "```json",
                json.dumps(
                    {
                        "category": data.get("category"),
                        "pairs_kept": data.get("pairs_kept"),
                        "num_changes_distribution": data.get("num_changes_distribution"),
                        "unique_modification_count": data.get("unique_modification_count"),
                        "top_modifications": data.get("top_modifications"),
                        "top_added_tokens": data.get("top_added_tokens"),
                        "top_removed_tokens": data.get("top_removed_tokens"),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                "```",
                "",
            ]
        )
    output_payload["distribution"] = distribution_payload

    args.teacher_md.parent.mkdir(parents=True, exist_ok=True)
    args.teacher_md.write_text(
        "\n".join(
            [
                "# Pseudo Teacher Comparison",
                "",
                render_table(
                    ["label", "category", "pairs_kept", "avg_quality_score", "avg_direction_margin"],
                    teacher_rows,
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    args.sweep_md.parent.mkdir(parents=True, exist_ok=True)
    args.sweep_md.write_text(
        "\n".join(
            [
                "# Pseudo Filter Sweep Report",
                "",
                render_table(
                    ["label", "category", "pairs_kept", "keep_top_n", "score_threshold", "avg_quality_score"],
                    sweep_rows,
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )

    args.distribution_md.parent.mkdir(parents=True, exist_ok=True)
    args.distribution_md.write_text("\n".join(distribution_sections), encoding="utf-8")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
