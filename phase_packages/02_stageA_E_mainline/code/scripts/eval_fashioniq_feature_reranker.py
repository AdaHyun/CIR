from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_fashioniq_feature_reranker import FeatureRerankValDataset, FeatureReranker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["dress", "shirt", "toptee"], required=True)
    parser.add_argument("--asset_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--blend_lambdas", type=str, default="0.0,0.25,0.5,0.75,1.0,1.25")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def load_assets(args: argparse.Namespace):
    val_query = torch.load(args.asset_dir / f"{args.category}_val_query_features.pt", map_location="cpu")
    val_image = torch.load(args.asset_dir / f"{args.category}_val_image_features.pt", map_location="cpu")
    val_topk = torch.load(args.asset_dir / f"{args.category}_val_topk.pt", map_location="cpu")
    return val_query, val_image, val_topk


def evaluate_blends(
    model: FeatureReranker,
    loader: DataLoader,
    category: str,
    blend_lambdas: list[float],
    device: torch.device,
) -> dict:
    model.eval()
    stats = {
        lam: {"hit10": 0, "hit50": 0, "hit100": 0, "count": 0}
        for lam in blend_lambdas
    }

    with torch.no_grad():
        for batch in loader:
            query = batch["query"].to(device).float()
            candidates = batch["candidates"].to(device).float()
            base_scores = batch["base_scores"].to(device).float()
            model_scores = model(query, candidates, base_scores)
            candidate_ids = batch["candidate_ids"]
            target_ids = batch["target_id"]

            for lam in blend_lambdas:
                blended = base_scores + lam * (model_scores - base_scores)
                order = torch.argsort(blended, dim=-1, descending=True).cpu().numpy()
                bucket = stats[lam]
                for row_idx in range(len(target_ids)):
                    ranked_ids = [candidate_ids[col_idx][row_idx] for col_idx in range(len(candidate_ids))]
                    sorted_ids = [ranked_ids[col] for col in order[row_idx]]
                    target_id = target_ids[row_idx]
                    bucket["count"] += 1
                    if target_id in sorted_ids[:10]:
                        bucket["hit10"] += 1
                    if target_id in sorted_ids[:50]:
                        bucket["hit50"] += 1
                    if target_id in sorted_ids[:100]:
                        bucket["hit100"] += 1

    results = {}
    best = None
    for lam, bucket in stats.items():
        count = max(bucket["count"], 1)
        metrics = {
            f"{category}_r10": bucket["hit10"] * 100.0 / count,
            f"{category}_r50": bucket["hit50"] * 100.0 / count,
            f"{category}_r100": bucket["hit100"] * 100.0 / count,
            "sum_score": (bucket["hit10"] + bucket["hit50"]) * 100.0 / count,
        }
        results[str(lam)] = metrics
        if best is None or metrics["sum_score"] > best["metrics"]["sum_score"]:
            best = {"lambda": lam, "metrics": metrics}
    return {"best": best, "all": results}


def main() -> None:
    args = parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = checkpoint["args"]

    val_query, val_image, val_topk = load_assets(args)
    val_dataset = FeatureRerankValDataset(
        query_features=val_query["query_features"],
        image_features=val_image["features"],
        image_ids=val_image["image_ids"],
        topk_assets=val_topk,
        val_k=ckpt_args["val_k"],
    )
    loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FeatureReranker(
        input_dim=int(val_query["query_features"].shape[1]),
        hidden_dim=int(ckpt_args["hidden_dim"]),
        dropout=float(ckpt_args["dropout"]),
        base_score_scale=float(ckpt_args["base_score_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    blend_lambdas = [float(x) for x in args.blend_lambdas.split(",") if x.strip()]
    summary = {
        "category": args.category,
        "checkpoint": str(args.checkpoint),
        "blend_lambdas": blend_lambdas,
        "device": str(device),
        **evaluate_blends(model, loader, args.category, blend_lambdas, device),
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
