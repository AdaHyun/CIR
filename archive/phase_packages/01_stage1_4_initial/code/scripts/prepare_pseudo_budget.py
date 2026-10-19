from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_path", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=0)
    parser.add_argument("--min_score", type=float, default=0.0)
    parser.add_argument("--min_num_changes", type=int, default=0)
    parser.add_argument("--max_num_changes", type=int, default=0)
    args = parser.parse_args()

    records = [
        record for record in load_records(args.input_jsonl)
        if record.get("quality_score", 0.0) >= args.min_score
    ]
    if args.min_num_changes > 0:
        records = [
            record for record in records
            if (record.get("num_removed_tokens", 0) + record.get("num_added_tokens", 0)) >= args.min_num_changes
        ]
    if args.max_num_changes > 0:
        records = [
            record for record in records
            if (record.get("num_removed_tokens", 0) + record.get("num_added_tokens", 0)) <= args.max_num_changes
        ]
    records.sort(key=lambda item: item.get("quality_score", 0.0), reverse=True)
    selected = records[: args.topk] if args.topk > 0 else records

    write_records(args.output_jsonl, selected)

    summary = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "input_records": len(records),
        "selected_records": len(selected),
        "topk": args.topk,
        "min_score": args.min_score,
        "min_num_changes": args.min_num_changes,
        "max_num_changes": args.max_num_changes,
        "avg_quality_score": (
            sum(record.get("quality_score", 0.0) for record in selected) / max(len(selected), 1)
        ),
    }
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
