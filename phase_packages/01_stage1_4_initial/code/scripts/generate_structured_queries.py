from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import string

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from query_decomposition import decompose_query


def normalize_text(text: str, correction_dict: dict[str, str]) -> str:
    trans = str.maketrans({key: " " for key in string.punctuation})
    tokens = str(text).lower().translate(trans).strip().split()
    return " ".join(correction_dict.get(word, word) for word in tokens)


def concat_modification(captions: list[str], correction_dict: dict[str, str]) -> str:
    return "{} and {}".format(
        normalize_text(captions[0], correction_dict),
        normalize_text(captions[1], correction_dict),
    )


def build_records(fashioniq_path: Path, category: str, split: str) -> list[dict]:
    caption_dir = fashioniq_path / "captions"
    correction_dict = json.loads((caption_dir / f"correction_dict_{category}.json").read_text(encoding="utf-8"))
    image_captions = json.loads(
        (caption_dir / f"image_captions_{category}_{split}.json").read_text(encoding="utf-8")
    )
    caption_pairs = json.loads((caption_dir / f"cap.{category}.{split}.json").read_text(encoding="utf-8"))

    records = []
    for idx, triplet in enumerate(caption_pairs):
        candidate = triplet["candidate"]
        target = triplet["target"]
        modification_text = concat_modification(triplet["captions"], correction_dict)
        raw_query = image_captions[candidate] + ", but " + modification_text
        record = decompose_query(raw_query)
        records.append(
            {
                "id": f"{category}_{split}_{idx}",
                "category": category,
                "split": split,
                "candidate": candidate,
                "target": target,
                **record.to_json_dict(),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fashioniq_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["dress", "shirt", "toptee"])
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for category in args.categories:
        for split in args.splits:
            records = build_records(args.fashioniq_path, category, split)
            output_path = args.output_dir / f"structured_{category}_{split}.json"
            output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
            index[f"{category}_{split}"] = {
                "path": str(output_path),
                "num_records": len(records),
            }

    (args.output_dir / "structured_query_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
