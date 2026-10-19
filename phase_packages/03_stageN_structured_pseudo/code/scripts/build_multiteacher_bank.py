#!/usr/bin/env python
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "FashionIQ"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import datasets
import model
import open_clip


def build_transforms(backbone: str, pretrained: str):
    _, preprocess_train, _ = open_clip.create_model_and_transforms(
        backbone,
        pretrained=pretrained,
    )
    return [preprocess_train, preprocess_train]


def load_teacher(variant: str, checkpoint_path: str, backbone: str, pretrained: str):
    net = model.build_model(
        model_variant=variant,
        hidden_dim=1024,
        dropout=0.5,
        backbone=backbone,
        pretrained=pretrained,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    net.load_state_dict(state_dict, strict=False)
    net.cuda()
    net.eval()
    return net


def reciprocal_rank_fusion(rank_maps, size: int, k: float = 60.0):
    fused = np.zeros(size, dtype=np.float32)
    for rank_map in rank_maps:
        fused += 1.0 / (k + rank_map.astype(np.float32))
    return fused


def compute_gallery_features(net, dataset, gallery_ids, batch_size: int):
    features = []
    with torch.no_grad():
        for start in tqdm(range(0, len(gallery_ids), batch_size), desc="gallery", leave=False):
            batch_ids = gallery_ids[start : start + batch_size]
            imgs = [
                dataset.get_img(f"{dataset.category}_{image_id}", stage=0)[0]
                for image_id in batch_ids
            ]
            tensor = torch.stack(imgs).float().cuda()
            feat = net.extract_target(tensor).cpu().numpy()
            features.append(feat)
    return np.concatenate(features, axis=0)


def compute_rank_map(scores: np.ndarray):
    order = np.argsort(-scores)
    rank_map = np.empty_like(order)
    rank_map[order] = np.arange(1, len(order) + 1)
    return order, rank_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=["dress", "shirt", "toptee"])
    parser.add_argument("--fashioniq_path", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--teacher_a_checkpoint", required=True)
    parser.add_argument("--teacher_a_variant", default="dqu_pfp_refcond")
    parser.add_argument("--teacher_b_checkpoint", required=True)
    parser.add_argument("--teacher_b_variant", default="dqu")
    parser.add_argument("--backbone", default="ViT-H-14")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--candidate_topk", type=int, default=20)
    parser.add_argument("--soft_bank_size", type=int, default=5)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--text_teacher_weight", type=float, default=1.0)
    parser.add_argument("--rank_gate", type=int, default=20)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--query_batch_size", type=int, default=64)
    args = parser.parse_args()

    transforms = build_transforms(args.backbone, args.pretrained)
    train_dataset = datasets.FashionIQ(
        path=args.fashioniq_path,
        category=args.category,
        transform=transforms,
        split="val-split",
    )

    teacher_a = load_teacher(args.teacher_a_variant, args.teacher_a_checkpoint, args.backbone, args.pretrained)
    teacher_b = load_teacher(args.teacher_b_variant, args.teacher_b_checkpoint, args.backbone, args.pretrained)

    gallery_ids = sorted(
        {
            item["candidate"].split("_", 1)[1]
            for item in train_dataset.train_data
        }
        | {
            item["target"].split("_", 1)[1]
            for item in train_dataset.train_data
        }
    )
    gallery_index = {image_id: idx for idx, image_id in enumerate(gallery_ids)}

    gallery_a = compute_gallery_features(teacher_a, train_dataset, gallery_ids, args.batch_size)
    gallery_b = compute_gallery_features(teacher_b, train_dataset, gallery_ids, args.batch_size)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    accepted = 0
    query_limit = int(args.max_queries) if int(args.max_queries) > 0 else len(train_dataset)
    with open(args.output_path, "w", encoding="utf-8") as f:
        with torch.no_grad():
            for start in tqdm(range(0, query_limit, int(args.query_batch_size)), desc=f"bank-{args.category}"):
                batch_indices = list(range(start, min(start + int(args.query_batch_size), query_limit)))
                batch_items = [train_dataset[idx] for idx in batch_indices]
                textual_queries = [item["textual_query"] for item in batch_items]
                visual_queries = torch.stack([item["visual_query"] for item in batch_items]).float().cuda()

                query_a_batch = teacher_a.extract_query(textual_queries, visual_queries).cpu().numpy()
                query_b_batch = teacher_b.extract_query(textual_queries, visual_queries).cpu().numpy()
                query_text_batch = teacher_a.encode_textual_query(textual_queries).cpu().numpy()

                scores_a_batch = query_a_batch @ gallery_a.T
                scores_b_batch = query_b_batch @ gallery_b.T
                scores_t_batch = query_text_batch @ gallery_a.T

                for offset, query_idx in enumerate(batch_indices):
                    scores_a = scores_a_batch[offset].copy()
                    scores_b = scores_b_batch[offset].copy()
                    scores_t = scores_t_batch[offset].copy()

                    caption = train_dataset.train_data[query_idx]
                    source_id = caption["candidate"].split("_", 1)[1]
                    target_id = caption["target"].split("_", 1)[1]
                    source_gallery_idx = gallery_index[source_id]
                    target_gallery_idx = gallery_index[target_id]

                    scores_a[source_gallery_idx] = -1e9
                    scores_b[source_gallery_idx] = -1e9
                    scores_t[source_gallery_idx] = -1e9

                    _, rank_a = compute_rank_map(scores_a)
                    _, rank_b = compute_rank_map(scores_b)
                    _, rank_t = compute_rank_map(scores_t)

                    fused = reciprocal_rank_fusion([rank_a, rank_b, rank_t], len(gallery_ids))
                    fused[source_gallery_idx] = -1e9
                    fused_order = np.argsort(-fused)

                    target_rank = int(np.where(fused_order == target_gallery_idx)[0][0]) + 1
                    use_bank = target_rank <= int(args.rank_gate)

                    selected_ids = [target_id]
                    selected_scores = [float(fused[target_gallery_idx])]
                    conflict_id = target_id
                    conflict_score = float(fused[target_gallery_idx])

                    for gallery_pos in fused_order:
                        image_id = gallery_ids[int(gallery_pos)]
                        if image_id == target_id:
                            continue
                        if conflict_id == target_id:
                            conflict_id = image_id
                            conflict_score = float(fused[int(gallery_pos)])
                        if len(selected_ids) < int(args.soft_bank_size):
                            selected_ids.append(image_id)
                            selected_scores.append(float(fused[int(gallery_pos)]))
                        if len(selected_ids) >= int(args.soft_bank_size):
                            break

                    while len(selected_ids) < int(args.soft_bank_size):
                        selected_ids.append(target_id)
                        selected_scores.append(float(fused[target_gallery_idx]))

                    if len(selected_scores) == 1:
                        soft_weights = [1.0]
                    else:
                        negative_scores = np.array(selected_scores[1:], dtype=np.float32)
                        negative_scores = negative_scores - negative_scores.max()
                        negative_exp = np.exp(negative_scores * 50.0)
                        negative_exp = negative_exp / np.clip(negative_exp.sum(), a_min=1e-6, a_max=None)
                        soft_weights = [0.55] + (0.45 * negative_exp).tolist()

                    teacher_gap = float(fused[target_gallery_idx] - conflict_score)
                    record = {
                        "category": args.category,
                        "query_idx": query_idx,
                        "source_id": source_id,
                        "target_id": target_id,
                        "target_rank": target_rank,
                        "use_bank": bool(use_bank),
                        "teacher_gap": teacher_gap,
                        "soft_ids": selected_ids,
                        "soft_weights": soft_weights,
                        "conflict_id": conflict_id,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if use_bank:
                        accepted += 1

    summary = {
        "category": args.category,
        "query_count": query_limit,
        "accepted_count": accepted,
        "accepted_ratio": accepted / max(query_limit, 1),
        "soft_bank_size": args.soft_bank_size,
        "rank_gate": args.rank_gate,
        "teacher_a_variant": args.teacher_a_variant,
        "teacher_b_variant": args.teacher_b_variant,
    }
    summary_path = os.path.splitext(args.output_path)[0] + "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
