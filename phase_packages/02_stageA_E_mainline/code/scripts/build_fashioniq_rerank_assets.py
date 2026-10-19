from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import open_clip
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import datasets
import model as cir_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["dress", "shirt", "toptee"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--fashioniq_path", type=str, required=True)
    parser.add_argument("--backbone", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2B-s32B-b79K")
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--model_variant", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_topk", type=int, default=64)
    parser.add_argument("--val_topk", type=int, default=100)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    payload = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_args = payload.get("args", {}) if isinstance(payload, dict) else {}
    model_variant = args.model_variant or checkpoint_args.get("model_variant", "dqu")
    net = cir_model.build_model(
        model_variant=model_variant,
        hidden_dim=checkpoint_args.get("hidden_dim", args.hidden_dim),
        dropout=checkpoint_args.get("dropout_rate", args.dropout_rate),
        backbone=checkpoint_args.get("backbone", args.backbone),
        pretrained=checkpoint_args.get("pretrained", args.pretrained),
    )
    if isinstance(payload, dict) and "model_state_dict" in payload:
        net.load_state_dict(payload["model_state_dict"], strict=False)
    else:
        net = payload
    net = net.cuda().eval()
    args.model_variant = model_variant
    args.backbone = checkpoint_args.get("backbone", args.backbone)
    args.pretrained = checkpoint_args.get("pretrained", args.pretrained)
    return net


def load_dataset(args: argparse.Namespace):
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.backbone,
        pretrained=args.pretrained,
    )
    return datasets.FashionIQ(
        path=args.fashioniq_path,
        category=args.category,
        transform=[preprocess_train, preprocess_val],
        split="val-split",
    )


def _batch_extract_target_features(net: torch.nn.Module, tensors: list[torch.Tensor]) -> np.ndarray:
    batch = torch.stack(tensors).float().cuda()
    with torch.no_grad():
        feats = net.extract_target(batch)
    return feats.detach().cpu().numpy()


def _batch_extract_query_features(net: torch.nn.Module, texts: list[str], visuals: list[torch.Tensor]) -> np.ndarray:
    batch = torch.stack(visuals).float().cuda()
    with torch.no_grad():
        feats = net.extract_query(texts, batch)
    return feats.detach().cpu().numpy()


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    return x / norms


def collect_train_image_ids(dataset: datasets.FashionIQ) -> list[str]:
    ids = set()
    for row in dataset.train_data:
        ids.add(row["candidate"].split("_", 1)[1])
        ids.add(row["target"].split("_", 1)[1])
    return sorted(ids)


def extract_image_features(
    net: torch.nn.Module,
    dataset: datasets.FashionIQ,
    image_ids: Iterable[str],
    stage: int,
    batch_size: int,
) -> dict:
    image_ids = list(image_ids)
    features = []
    batch_tensors: list[torch.Tensor] = []
    batch_ids: list[str] = []
    for image_id in tqdm(image_ids, desc=f"image-feats-stage{stage}", ncols=80):
        img, _ = dataset.get_img(f"{dataset.category}_{image_id}", stage=stage)
        batch_tensors.append(img)
        batch_ids.append(image_id)
        if len(batch_tensors) >= batch_size:
            features.append(_batch_extract_target_features(net, batch_tensors))
            batch_tensors = []
            batch_ids = []
    if batch_tensors:
        features.append(_batch_extract_target_features(net, batch_tensors))
    feature_array = _normalize(np.concatenate(features, axis=0))
    return {
        "image_ids": image_ids,
        "features": torch.from_numpy(feature_array).float(),
    }


def extract_train_query_features(
    net: torch.nn.Module,
    dataset: datasets.FashionIQ,
    batch_size: int,
) -> dict:
    query_features = []
    textual_batch: list[str] = []
    visual_batch: list[torch.Tensor] = []
    source_ids: list[str] = []
    target_ids: list[str] = []
    for idx in tqdm(range(len(dataset)), desc="train-query-feats", ncols=80):
        row = dataset[idx]
        caption = dataset.train_data[idx]
        source_ids.append(caption["candidate"].split("_", 1)[1])
        target_ids.append(caption["target"].split("_", 1)[1])
        textual_batch.append(row["textual_query"])
        visual_batch.append(row["visual_query"])
        if len(visual_batch) >= batch_size:
            query_features.append(_batch_extract_query_features(net, textual_batch, visual_batch))
            textual_batch = []
            visual_batch = []
    if visual_batch:
        query_features.append(_batch_extract_query_features(net, textual_batch, visual_batch))
    feature_array = _normalize(np.concatenate(query_features, axis=0))
    return {
        "query_features": torch.from_numpy(feature_array).float(),
        "source_ids": source_ids,
        "target_ids": target_ids,
    }


def extract_val_query_features(
    net: torch.nn.Module,
    dataset: datasets.FashionIQ,
    batch_size: int,
) -> dict:
    query_features = []
    textual_batch: list[str] = []
    visual_batch: list[torch.Tensor] = []
    for row in tqdm(dataset.test_queries, desc="val-query-feats", ncols=80):
        textual_batch.append(row["textual_query"])
        visual_batch.append(row["visual_query"])
        if len(visual_batch) >= batch_size:
            query_features.append(_batch_extract_query_features(net, textual_batch, visual_batch))
            textual_batch = []
            visual_batch = []
    if visual_batch:
        query_features.append(_batch_extract_query_features(net, textual_batch, visual_batch))

    source_image_ids = [Path(row["source_img_path"]).stem for row in dataset.test_queries]
    target_image_ids = [Path(row["target_img_path"]).stem for row in dataset.test_queries]

    feature_array = _normalize(np.concatenate(query_features, axis=0))
    return {
        "query_features": torch.from_numpy(feature_array).float(),
        "source_ids": source_image_ids,
        "target_ids": target_image_ids,
    }


def build_topk_assets(
    query_features: torch.Tensor,
    source_ids: list[str],
    target_ids: list[str],
    image_features: torch.Tensor,
    image_ids: list[str],
    topk: int,
) -> dict:
    q = _normalize(query_features.numpy())
    i = _normalize(image_features.numpy())
    sims = q @ i.T
    image_index = {image_id: idx for idx, image_id in enumerate(image_ids)}

    sorted_ids = []
    sorted_scores = []
    target_ranks = []
    target_in_topk = []

    for row_idx in tqdm(range(len(source_ids)), desc=f"topk-{topk}", ncols=80):
        source_id = source_ids[row_idx]
        target_id = target_ids[row_idx]
        source_idx = image_index.get(source_id)
        if source_idx is not None:
            sims[row_idx, source_idx] = -1e10
        order = np.argsort(-sims[row_idx])
        target_idx = image_index[target_id]
        target_rank = int(np.where(order == target_idx)[0][0]) + 1
        top_order = order[:topk]
        sorted_ids.append([image_ids[idx] for idx in top_order])
        sorted_scores.append([float(sims[row_idx, idx]) for idx in top_order])
        target_ranks.append(target_rank)
        target_in_topk.append(target_rank <= topk)

    coverage = {
        "num_queries": len(source_ids),
        "top10_coverage": float(sum(rank <= 10 for rank in target_ranks) * 100.0 / len(target_ranks)),
        "top50_coverage": float(sum(rank <= 50 for rank in target_ranks) * 100.0 / len(target_ranks)),
        f"top{topk}_coverage": float(sum(target_in_topk) * 100.0 / len(target_ranks)),
        "mean_target_rank": float(np.mean(target_ranks)),
        "median_target_rank": float(np.median(target_ranks)),
    }
    return {
        "sorted_ids": sorted_ids,
        "sorted_scores": sorted_scores,
        "source_ids": source_ids,
        "target_ids": target_ids,
        "target_ranks": target_ranks,
        "target_in_topk": target_in_topk,
        "coverage": coverage,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    net = load_model(args)
    dataset = load_dataset(args)

    train_image_ids = collect_train_image_ids(dataset)
    train_image_assets = extract_image_features(
        net=net,
        dataset=dataset,
        image_ids=train_image_ids,
        stage=0,
        batch_size=args.batch_size,
    )
    torch.save(train_image_assets, args.output_dir / f"{args.category}_train_image_features.pt")

    val_image_ids = [Path(target["target_img_path"]).stem for target in dataset.test_targets]
    val_image_assets = extract_image_features(
        net=net,
        dataset=dataset,
        image_ids=val_image_ids,
        stage=1,
        batch_size=args.batch_size,
    )
    torch.save(val_image_assets, args.output_dir / f"{args.category}_val_image_features.pt")

    train_query_assets = extract_train_query_features(net, dataset, args.batch_size)
    torch.save(train_query_assets, args.output_dir / f"{args.category}_train_query_features.pt")

    val_query_assets = extract_val_query_features(net, dataset, args.batch_size)
    torch.save(val_query_assets, args.output_dir / f"{args.category}_val_query_features.pt")

    train_topk = build_topk_assets(
        query_features=train_query_assets["query_features"],
        source_ids=train_query_assets["source_ids"],
        target_ids=train_query_assets["target_ids"],
        image_features=train_image_assets["features"],
        image_ids=train_image_assets["image_ids"],
        topk=args.train_topk,
    )
    torch.save(train_topk, args.output_dir / f"{args.category}_train_topk.pt")
    (args.output_dir / f"{args.category}_train_topk_summary.json").write_text(
        json.dumps(train_topk["coverage"], indent=2),
        encoding="utf-8",
    )

    val_topk = build_topk_assets(
        query_features=val_query_assets["query_features"],
        source_ids=val_query_assets["source_ids"],
        target_ids=val_query_assets["target_ids"],
        image_features=val_image_assets["features"],
        image_ids=val_image_assets["image_ids"],
        topk=args.val_topk,
    )
    torch.save(val_topk, args.output_dir / f"{args.category}_val_topk.pt")
    (args.output_dir / f"{args.category}_val_topk_summary.json").write_text(
        json.dumps(val_topk["coverage"], indent=2),
        encoding="utf-8",
    )

    run_summary = {
        "category": args.category,
        "checkpoint": str(args.checkpoint),
        "model_variant": args.model_variant,
        "backbone": args.backbone,
        "pretrained": args.pretrained,
        "train_topk_summary": train_topk["coverage"],
        "val_topk_summary": val_topk["coverage"],
    }
    (args.output_dir / f"{args.category}_asset_summary.json").write_text(
        json.dumps(run_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
