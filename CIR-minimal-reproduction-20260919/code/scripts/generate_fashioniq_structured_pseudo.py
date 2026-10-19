from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys

import cv2
import open_clip
import PIL.Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model as cir_model  # noqa: E402
from query_decomposition import extract_attributes  # noqa: E402


ATTRIBUTE_ORDER = [
    "color",
    "pattern",
    "neckline",
    "sleeve",
    "length",
    "fit",
    "silhouette",
    "material",
    "decoration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["dress", "shirt", "toptee"], required=True)
    parser.add_argument("--fashioniq_path", type=Path, default=PROJECT_ROOT / "data" / "FashionIQ")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--output_summary", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--neighbor_topk", type=int, default=12)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--score_threshold", type=float, default=0.58)
    parser.add_argument("--min_direction_margin", type=float, default=0.0)
    parser.add_argument("--min_num_changes", type=int, default=1)
    parser.add_argument("--max_num_changes", type=int, default=2)
    parser.add_argument("--keep_top_n", type=int, default=0)
    parser.add_argument("--max_records_per_change", type=int, default=150)
    parser.add_argument("--max_examples", type=int, default=10)
    parser.add_argument("--query_batch_size", type=int, default=128)
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def summarize_slots(slots: dict[str, list[str]], empty_text: str) -> str:
    if not slots:
        return empty_text
    parts = []
    for attribute_name in ATTRIBUTE_ORDER:
        values = slots.get(attribute_name)
        if values:
            parts.append(f"{attribute_name}={', '.join(values)}")
    if not parts:
        return empty_text
    return "; ".join(parts)


def build_structured_change(source_caption: str, target_caption: str):
    source_slots = extract_attributes(source_caption)
    target_slots = extract_attributes(target_caption)

    retain_slots: dict[str, list[str]] = {}
    modify_slots: dict[str, list[str]] = {}
    for attribute_name in ATTRIBUTE_ORDER:
        source_values = source_slots.get(attribute_name, [])
        target_values = target_slots.get(attribute_name, [])
        if source_values and target_values and source_values == target_values:
            retain_slots[attribute_name] = target_values
        elif target_values and source_values != target_values:
            modify_slots[attribute_name] = target_values

    num_changes = len(modify_slots)
    if num_changes == 0:
        return None, {}

    change_text = "keep {}. change {}.".format(
        summarize_slots(retain_slots, "reference core attributes"),
        summarize_slots(modify_slots, "raw modification request"),
    )
    change_key = summarize_slots(modify_slots, "no-change")
    return change_text, {
        "retain_slots": retain_slots,
        "modify_slots": modify_slots,
        "source_slots": source_slots,
        "target_slots": target_slots,
        "num_changes": num_changes,
        "change_key": change_key,
    }


def overlay_text_on_image(image_path: Path, text: str, transform):
    image = cv2.imread(str(image_path))
    image = cv2.resize(image, (512, 512))
    draw_y = 25
    for line in text.split(". "):
        if not line:
            continue
        cv2.rectangle(image, (10, draw_y - 18), (500, draw_y + 8), (255, 255, 255), -1)
        cv2.putText(
            image,
            line[:120],
            (15, draw_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
        draw_y += 22
    image = PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return transform(image)


def load_model(args: argparse.Namespace):
    payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = payload.get("args", {})
    model_variant = checkpoint_args.get("model_variant", "dqu_pfp_refcond")
    net = cir_model.build_model(
        model_variant=model_variant,
        hidden_dim=checkpoint_args.get("hidden_dim", 1024),
        dropout=checkpoint_args.get("dropout_rate", 0.5),
        backbone=checkpoint_args.get("backbone", "ViT-H-14"),
        pretrained=checkpoint_args.get("pretrained", "laion2B-s32B-b79K"),
    )
    net.load_state_dict(payload["model_state_dict"])
    net = net.cuda().eval()
    _, _, preprocess_val = open_clip.create_model_and_transforms(
        checkpoint_args.get("backbone", "ViT-H-14"),
        pretrained=checkpoint_args.get("pretrained", "laion2B-s32B-b79K"),
    )
    return net, preprocess_val


def encode_images(net, image_paths, preprocess, batch_size):
    all_features = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="encode_images"):
        batch_paths = image_paths[start : start + batch_size]
        batch_tensors = []
        for image_path in batch_paths:
            with image_path.open("rb") as f:
                image = PIL.Image.open(f).convert("RGB")
            batch_tensors.append(preprocess(image))
        batch = torch.stack(batch_tensors).float().cuda()
        with torch.no_grad():
            features = net.extract_target(batch).detach().cpu()
        all_features.append(features)
    return torch.cat(all_features, dim=0)


def encode_texts(net, texts, batch_size):
    all_features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="encode_texts"):
        batch_text = texts[start : start + batch_size]
        with torch.no_grad():
            features = F.normalize(net.extract_text_fea(batch_text), p=2, dim=-1).detach().cpu()
        all_features.append(features)
    return torch.cat(all_features, dim=0)


def find_candidate_neighbors(image_features, text_features, topk, chunk_size):
    image_features = image_features.cuda()
    text_features = text_features.cuda()
    combined = 0.7 * image_features + 0.3 * text_features
    combined = F.normalize(combined, p=2, dim=-1)

    neighbors = []
    total = combined.size(0)
    for start in tqdm(range(0, total, chunk_size), desc="search_neighbors"):
        chunk = combined[start : start + chunk_size]
        sims = chunk @ combined.T
        row_ids = torch.arange(start, min(start + chunk_size, total), device=sims.device)
        sims[torch.arange(sims.size(0), device=sims.device), row_ids] = -1e9
        top_scores, top_indices = torch.topk(sims, k=topk, dim=1)
        neighbors.append((top_indices.cpu(), top_scores.cpu()))
    return neighbors


def compute_quality_score(pair_score, direction_margin, num_changes, num_retained):
    pair_term = (pair_score + 1.0) / 2.0
    direction_term = 1.0 / (1.0 + math.exp(-8.0 * direction_margin))
    sparsity_term = 1.0 / float(max(num_changes, 1))
    retain_term = min(float(num_retained) / 3.0, 1.0)
    return 0.35 * pair_term + 0.35 * direction_term + 0.20 * sparsity_term + 0.10 * retain_term


def score_candidate_records(
    net,
    preprocess,
    image_paths,
    image_features,
    candidate_records,
    batch_size,
    score_threshold,
    min_direction_margin,
):
    kept_records = []
    image_feature_gpu = image_features.cuda()
    for start in tqdm(range(0, len(candidate_records), batch_size), desc="score_pairs"):
        batch_records = candidate_records[start : start + batch_size]
        visual_queries = []
        textual_queries = []
        for record in batch_records:
            visual_queries.append(
                overlay_text_on_image(
                    image_paths[record["source_idx"]],
                    record["modification_text"],
                    preprocess,
                )
            )
            textual_queries.append(record["source_caption"] + ", but " + record["modification_text"])

        visual_batch = torch.stack(visual_queries).float().cuda()
        with torch.no_grad():
            query_features = net.extract_query(textual_queries, visual_batch)

        target_indices = torch.tensor([record["target_idx"] for record in batch_records], device=query_features.device)
        source_indices = torch.tensor([record["source_idx"] for record in batch_records], device=query_features.device)
        target_scores = torch.sum(query_features * image_feature_gpu[target_indices], dim=-1)
        source_scores = torch.sum(query_features * image_feature_gpu[source_indices], dim=-1)
        direction_margins = target_scores - source_scores

        for record, target_score, source_score, direction_margin in zip(
            batch_records,
            target_scores.detach().cpu().tolist(),
            source_scores.detach().cpu().tolist(),
            direction_margins.detach().cpu().tolist(),
        ):
            if direction_margin <= min_direction_margin:
                continue
            quality_score = compute_quality_score(
                pair_score=record["pair_score"],
                direction_margin=direction_margin,
                num_changes=max(record["num_changes"], 1),
                num_retained=record["num_retained"],
            )
            if quality_score < score_threshold:
                continue
            kept_records.append(
                {
                    "category": record["category"],
                    "candidate": record["candidate"],
                    "target": record["target"],
                    "source_caption": record["source_caption"],
                    "target_caption": record["target_caption"],
                    "modification_text": record["modification_text"],
                    "pair_score": record["pair_score"],
                    "target_score": target_score,
                    "source_score": source_score,
                    "direction_margin": direction_margin,
                    "quality_score": quality_score,
                    "num_changes": record["num_changes"],
                    "num_retained": record["num_retained"],
                    "change_key": record["change_key"],
                    "retain_slots": record["retain_slots"],
                    "modify_slots": record["modify_slots"],
                }
            )
    return kept_records


def rebalance_records(records, keep_top_n, max_records_per_change):
    if keep_top_n <= 0 and max_records_per_change <= 0:
        return records

    selected = []
    change_counter = Counter()
    for record in records:
        change_key = record["change_key"]
        if max_records_per_change > 0 and change_counter[change_key] >= max_records_per_change:
            continue
        change_counter[change_key] += 1
        selected.append(record)
        if keep_top_n > 0 and len(selected) >= keep_top_n:
            break
    return selected


def main() -> None:
    args = parse_args()
    net, preprocess = load_model(args)

    image_split = load_json(args.fashioniq_path / "image_splits" / f"split.{args.category}.train.json")
    image_captions = load_json(args.fashioniq_path / "captions" / f"image_captions_{args.category}_train.json")
    human_triplets = load_json(args.fashioniq_path / "captions" / f"cap.{args.category}.train.json")
    human_pairs = {(item["candidate"], item["target"]) for item in human_triplets}

    image_paths = [
        args.fashioniq_path / "resized_image" / args.category / f"{image_id}.jpg"
        for image_id in image_split
    ]
    captions = [image_captions[image_id] for image_id in image_split]

    image_features = encode_images(net, image_paths, preprocess, args.batch_size)
    text_features = encode_texts(net, captions, args.batch_size)
    neighbor_chunks = find_candidate_neighbors(
        image_features=image_features,
        text_features=text_features,
        topk=args.neighbor_topk,
        chunk_size=args.chunk_size,
    )

    candidate_records = []
    total_pairs_considered = 0
    hardest_coverage = defaultdict(int)
    for chunk_id, (neighbor_indices, neighbor_scores) in enumerate(tqdm(neighbor_chunks, desc="build_candidates")):
        base_index = chunk_id * args.chunk_size
        for row_offset in range(neighbor_indices.size(0)):
            source_idx = base_index + row_offset
            source_id = image_split[source_idx]
            source_caption = image_captions[source_id]
            for col_idx in range(neighbor_indices.size(1)):
                target_idx = int(neighbor_indices[row_offset, col_idx].item())
                pair_score = float(neighbor_scores[row_offset, col_idx].item())
                target_id = image_split[target_idx]
                if (source_id, target_id) in human_pairs:
                    continue

                target_caption = image_captions[target_id]
                modification_text, change_meta = build_structured_change(source_caption, target_caption)
                if modification_text is None:
                    continue
                if change_meta["num_changes"] < args.min_num_changes or change_meta["num_changes"] > args.max_num_changes:
                    continue

                total_pairs_considered += 1
                hardest_coverage[source_id] += 1
                candidate_records.append(
                    {
                        "category": args.category,
                        "source_idx": source_idx,
                        "target_idx": target_idx,
                        "candidate": source_id,
                        "target": target_id,
                        "source_caption": source_caption,
                        "target_caption": target_caption,
                        "modification_text": modification_text,
                        "pair_score": pair_score,
                        "num_changes": change_meta["num_changes"],
                        "num_retained": len(change_meta["retain_slots"]),
                        "change_key": change_meta["change_key"],
                        "retain_slots": change_meta["retain_slots"],
                        "modify_slots": change_meta["modify_slots"],
                    }
                )

    kept_records = score_candidate_records(
        net=net,
        preprocess=preprocess,
        image_paths=image_paths,
        image_features=image_features,
        candidate_records=candidate_records,
        batch_size=args.query_batch_size,
        score_threshold=args.score_threshold,
        min_direction_margin=args.min_direction_margin,
    )

    kept_records.sort(key=lambda item: item["quality_score"], reverse=True)
    kept_before_rebalance = len(kept_records)
    kept_records = rebalance_records(
        kept_records,
        keep_top_n=args.keep_top_n,
        max_records_per_change=args.max_records_per_change,
    )

    change_counter = Counter(record["change_key"] for record in kept_records)
    change_cardinality = Counter(record["num_changes"] for record in kept_records)
    attribute_counter = Counter()
    for record in kept_records:
        for attribute_name, values in record["modify_slots"].items():
            attribute_counter.update(f"{attribute_name}:{value}" for value in values)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for record in kept_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "category",
                    "candidate",
                    "target",
                    "modification_text",
                    "quality_score",
                    "pair_score",
                    "direction_margin",
                    "num_changes",
                    "num_retained",
                    "change_key",
                ],
            )
            writer.writeheader()
            for record in kept_records:
                writer.writerow({key: record.get(key) for key in writer.fieldnames})

    summary = {
        "category": args.category,
        "checkpoint": str(args.checkpoint),
        "input_images": len(image_split),
        "neighbor_topk": args.neighbor_topk,
        "pairs_considered_after_structured_diff": total_pairs_considered,
        "pairs_kept_before_rebalance": kept_before_rebalance,
        "pairs_kept": len(kept_records),
        "score_threshold": args.score_threshold,
        "min_direction_margin": args.min_direction_margin,
        "min_num_changes": args.min_num_changes,
        "max_num_changes": args.max_num_changes,
        "keep_top_n": args.keep_top_n,
        "max_records_per_change": args.max_records_per_change,
        "avg_quality_score": float(sum(item["quality_score"] for item in kept_records) / max(len(kept_records), 1)),
        "avg_direction_margin": float(sum(item["direction_margin"] for item in kept_records) / max(len(kept_records), 1)),
        "num_changes_distribution": dict(sorted(change_cardinality.items())),
        "unique_change_count": len(change_counter),
        "top_changes": change_counter.most_common(args.max_examples),
        "top_attribute_values": attribute_counter.most_common(args.max_examples),
        "hardest_coverage_count": sum(1 for _, count in hardest_coverage.items() if count > 0),
        "top_examples": kept_records[: args.max_examples],
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
