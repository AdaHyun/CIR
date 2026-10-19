#!/usr/bin/env python3
"""Evaluate the packaged reranker checkpoints on frozen validation features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "dress": {"lambda": 0.05, "checkpoint": "dress/dress_best.pt", "hits": (1107, 1560)},
    "shirt": {"lambda": 0.04, "checkpoint": "shirt/shirt_best.pt", "hits": (1308, 1643)},
    "toptee": {"lambda": 0.03, "checkpoint": "toptee/toptee_best.pt", "hits": (1315, 1673)},
}


def trusted_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class FeatureReranker(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, base_score_scale: float):
        super().__init__()
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.cand_proj = nn.Linear(input_dim, hidden_dim)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.base_scale = nn.Parameter(torch.tensor(float(base_score_scale)))

    def forward(self, query: torch.Tensor, candidates: torch.Tensor, base_scores: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query)
        c = self.cand_proj(candidates)
        q_expand = q.unsqueeze(1).expand_as(c)
        pair = torch.cat(
            (q_expand, c, q_expand * c, torch.abs(q_expand - c), base_scores.unsqueeze(-1)), dim=-1
        )
        return self.score_head(pair).squeeze(-1) + self.base_scale * base_scores


def evaluate_category(category: str, batch_size: int, device: torch.device) -> dict:
    setting = SETTINGS[category]
    asset_dir = ROOT / "assets" / category
    checkpoint_path = ROOT / "checkpoints" / "reranker" / setting["checkpoint"]
    query_data = trusted_torch_load(asset_dir / f"{category}_val_query_features.pt")
    image_data = trusted_torch_load(asset_dir / f"{category}_val_image_features.pt")
    topk = trusted_torch_load(asset_dir / f"{category}_val_topk.pt")
    checkpoint = trusted_torch_load(checkpoint_path)

    query_features = query_data["query_features"].float()
    image_features = image_data["features"].float()
    image_index = {image_id: idx for idx, image_id in enumerate(image_data["image_ids"])}
    sorted_ids = topk["sorted_ids"]
    sorted_scores = topk["sorted_scores"]
    target_ids = topk["target_ids"]
    args = checkpoint["args"]
    val_k = int(args["val_k"])

    model = FeatureReranker(
        input_dim=int(query_features.shape[1]),
        hidden_dim=int(args["hidden_dim"]),
        dropout=float(args["dropout"]),
        base_score_scale=float(args["base_score_scale"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    hit10 = hit50 = 0
    blend_lambda = float(setting["lambda"])
    with torch.no_grad():
        for start in range(0, len(target_ids), batch_size):
            stop = min(start + batch_size, len(target_ids))
            ids_batch = [list(row[:val_k]) for row in sorted_ids[start:stop]]
            index_batch = torch.tensor(
                [[image_index[image_id] for image_id in row] for row in ids_batch], dtype=torch.long
            )
            candidates = image_features[index_batch].to(device)
            queries = query_features[start:stop].to(device)
            base_scores = torch.as_tensor(sorted_scores[start:stop], dtype=torch.float32)[:, :val_k].to(device)
            learned_scores = model(queries, candidates, base_scores)
            final_scores = base_scores + blend_lambda * (learned_scores - base_scores)
            order = torch.argsort(final_scores, dim=-1, descending=True).cpu().tolist()
            for row_idx, permutation in enumerate(order):
                ranked = [ids_batch[row_idx][position] for position in permutation]
                target = target_ids[start + row_idx]
                hit10 += target in ranked[:10]
                hit50 += target in ranked[:50]

    expected_hit10, expected_hit50 = setting["hits"]
    if (hit10, hit50) != (expected_hit10, expected_hit50):
        raise AssertionError(
            f"{category}: expected hits {(expected_hit10, expected_hit50)}, got {(hit10, hit50)}"
        )
    count = len(target_ids)
    return {
        "queries": count,
        "lambda": blend_lambda,
        "hit10": hit10,
        "hit50": hit50,
        "R@10": hit10 * 100.0 / count,
        "R@50": hit50 * 100.0 / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    results = {category: evaluate_category(category, args.batch_size, device) for category in SETTINGS}
    macro_r10 = sum(item["R@10"] for item in results.values()) / len(results)
    macro_r50 = sum(item["R@50"] for item in results.values()) / len(results)
    payload = {
        "device": str(device),
        "categories": results,
        "macro": {"R@10": macro_r10, "R@50": macro_r50, "mR": (macro_r10 + macro_r50) / 2.0},
    }
    output = ROOT / "results" / "cached_reranker_metrics.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("\nPASS: cached-feature reranker evaluation matches the archived query-level results.")


if __name__ == "__main__":
    main()
