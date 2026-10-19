from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
FASHIONIQ_CATEGORIES = ("dress", "shirt", "toptee")


def collect_missing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def count_images(path: Path) -> int:
    return len(list(path.glob("*.jpg"))) + len(list(path.glob("*.png")))


def check_fashioniq(category: str) -> list[str]:
    base = DATA_ROOT / "FashionIQ"
    image_dir = base / "resized_image" / category
    required = [
        base / "captions" / f"cap.{category}.train.json",
        base / "captions" / f"cap.{category}.val.json",
        base / "captions" / f"cap.{category}.test.json",
        base / "captions" / f"correction_dict_{category}.json",
        base / "captions" / f"image_captions_{category}_train.json",
        base / "captions" / f"image_captions_{category}_val.json",
        base / "captions" / f"keywords_in_mods_{category}.json",
        base / "image_splits" / f"split.{category}.train.json",
        base / "image_splits" / f"split.{category}.val.json",
        base / "image_splits" / f"split.{category}.test.json",
        image_dir,
    ]
    missing = [str(path) for path in collect_missing(required)]
    if not missing and count_images(image_dir) == 0:
        missing.append(f"{image_dir} (directory exists but contains no .jpg/.png files)")
    return missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=[*FASHIONIQ_CATEGORIES, "all"],
        default="all",
    )
    args = parser.parse_args()
    categories = FASHIONIQ_CATEGORIES if args.dataset == "all" else (args.dataset,)

    missing_by_category: dict[str, list[str]] = {}
    for category in categories:
        missing = check_fashioniq(category)
        if missing:
            missing_by_category[category] = missing
            continue
        print(f"[OK] FashionIQ {category} dataset check passed.")

    if missing_by_category:
        for category, missing in missing_by_category.items():
            print(f"[MISSING] FashionIQ {category}:")
            for path in missing:
                print(f"  - {path}")
        raise SystemExit(1)

    print("All requested FashionIQ dataset checks passed.")


if __name__ == "__main__":
    main()
