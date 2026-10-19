from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=["dress", "shirt", "toptee"], required=True)
    parser.add_argument("--asset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--train_k", type=int, default=16)
    parser.add_argument("--train_pool", type=int, default=32)
    parser.add_argument("--val_k", type=int, default=100)
    parser.add_argument("--distill_weight", type=float, default=0.0)
    parser.add_argument("--distill_temperature", type=float, default=0.07)
    parser.add_argument("--base_score_scale", type=float, default=1.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureRerankTrainDataset(Dataset):
    def __init__(
        self,
        query_features: torch.Tensor,
        image_features: torch.Tensor,
        image_ids: list[str],
        topk_assets: dict,
        train_k: int,
        train_pool: int,
    ):
        self.query_features = query_features.float()
        self.image_features = image_features.float()
        self.image_index = {image_id: idx for idx, image_id in enumerate(image_ids)}
        self.sorted_ids = topk_assets["sorted_ids"]
        self.sorted_scores = topk_assets["sorted_scores"]
        self.target_ids = topk_assets["target_ids"]
        self.train_k = train_k
        self.train_pool = max(train_pool, train_k)

    def __len__(self) -> int:
        return len(self.target_ids)

    def __getitem__(self, idx: int):
        query = self.query_features[idx]
        target_id = self.target_ids[idx]
        candidate_ids = list(self.sorted_ids[idx][: self.train_pool])
        candidate_scores = list(self.sorted_scores[idx][: self.train_pool])

        neg_ids = [cid for cid in candidate_ids if cid != target_id]
        neg_scores = [score for cid, score in zip(candidate_ids, candidate_scores) if cid != target_id]

        if len(neg_ids) >= self.train_k - 1:
            sampled = random.sample(list(range(len(neg_ids))), self.train_k - 1)
            selected_neg_ids = [neg_ids[i] for i in sampled]
            selected_neg_scores = [neg_scores[i] for i in sampled]
        else:
            selected_neg_ids = neg_ids
            selected_neg_scores = neg_scores
            while len(selected_neg_ids) < self.train_k - 1:
                selected_neg_ids.append(selected_neg_ids[-1] if selected_neg_ids else target_id)
                selected_neg_scores.append(selected_neg_scores[-1] if selected_neg_scores else -1e10)

        label = random.randint(0, self.train_k - 1)
        candidate_ids_final = []
        candidate_scores_final = []
        neg_ptr = 0
        for pos in range(self.train_k):
            if pos == label:
                candidate_ids_final.append(target_id)
                candidate_scores_final.append(1.0)
            else:
                candidate_ids_final.append(selected_neg_ids[neg_ptr])
                candidate_scores_final.append(selected_neg_scores[neg_ptr])
                neg_ptr += 1

        candidate_idx = [self.image_index[cid] for cid in candidate_ids_final]
        candidate_features = self.image_features[candidate_idx]
        return {
            "query": query,
            "candidates": candidate_features,
            "base_scores": torch.tensor(candidate_scores_final, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


class FeatureRerankValDataset(Dataset):
    def __init__(
        self,
        query_features: torch.Tensor,
        image_features: torch.Tensor,
        image_ids: list[str],
        topk_assets: dict,
        val_k: int,
    ):
        self.query_features = query_features.float()
        self.image_features = image_features.float()
        self.image_index = {image_id: idx for idx, image_id in enumerate(image_ids)}
        self.sorted_ids = [row[:val_k] for row in topk_assets["sorted_ids"]]
        self.sorted_scores = [row[:val_k] for row in topk_assets["sorted_scores"]]
        self.target_ids = topk_assets["target_ids"]
        self.target_in_topk = topk_assets["target_in_topk"]

    def __len__(self) -> int:
        return len(self.target_ids)

    def __getitem__(self, idx: int):
        candidate_ids = self.sorted_ids[idx]
        candidate_idx = [self.image_index[cid] for cid in candidate_ids]
        return {
            "query": self.query_features[idx],
            "candidates": self.image_features[candidate_idx],
            "base_scores": torch.tensor(self.sorted_scores[idx], dtype=torch.float32),
            "candidate_ids": candidate_ids,
            "target_id": self.target_ids[idx],
            "target_in_topk": bool(self.target_in_topk[idx]),
        }


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
            [
                q_expand,
                c,
                q_expand * c,
                torch.abs(q_expand - c),
                base_scores.unsqueeze(-1),
            ],
            dim=-1,
        )
        residual = self.score_head(pair).squeeze(-1)
        return residual + self.base_scale * base_scores


def load_assets(args: argparse.Namespace):
    train_query = torch.load(args.asset_dir / f"{args.category}_train_query_features.pt", map_location="cpu")
    train_image = torch.load(args.asset_dir / f"{args.category}_train_image_features.pt", map_location="cpu")
    train_topk = torch.load(args.asset_dir / f"{args.category}_train_topk.pt", map_location="cpu")

    val_query = torch.load(args.asset_dir / f"{args.category}_val_query_features.pt", map_location="cpu")
    val_image = torch.load(args.asset_dir / f"{args.category}_val_image_features.pt", map_location="cpu")
    val_topk = torch.load(args.asset_dir / f"{args.category}_val_topk.pt", map_location="cpu")
    return train_query, train_image, train_topk, val_query, val_image, val_topk


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    hit10 = 0
    hit50 = 0
    hit100 = 0
    count = 0
    with torch.no_grad():
        for batch in loader:
            query = batch["query"].cuda().float()
            candidates = batch["candidates"].cuda().float()
            base_scores = batch["base_scores"].cuda().float()
            logits = model(query, candidates, base_scores)
            order = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
            candidate_ids = batch["candidate_ids"]
            target_ids = batch["target_id"]

            for row_idx in range(len(target_ids)):
                ranked_ids = [candidate_ids[col_idx][row_idx] for col_idx in range(len(candidate_ids))]
                sorted_ids = [ranked_ids[col] for col in order[row_idx]]
                target_id = target_ids[row_idx]
                count += 1
                if target_id in sorted_ids[:10]:
                    hit10 += 1
                if target_id in sorted_ids[:50]:
                    hit50 += 1
                if target_id in sorted_ids[:100]:
                    hit100 += 1

    return {
        f"{args.category}_r10": hit10 * 100.0 / count,
        f"{args.category}_r50": hit50 * 100.0 / count,
        f"{args.category}_r100": hit100 * 100.0 / count,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in tqdm(loader, desc="train", ncols=80):
        query = batch["query"].cuda().float()
        candidates = batch["candidates"].cuda().float()
        base_scores = batch["base_scores"].cuda().float()
        labels = batch["label"].cuda()

        optimizer.zero_grad()
        logits = model(query, candidates, base_scores)
        ce_loss = F.cross_entropy(logits, labels)

        if args.distill_weight > 0:
            student = F.log_softmax(logits / args.distill_temperature, dim=-1)
            teacher = F.softmax(base_scores / args.distill_temperature, dim=-1)
            distill_loss = F.kl_div(student, teacher, reduction="batchmean") * (args.distill_temperature ** 2)
        else:
            distill_loss = logits.new_zeros(())

        loss = ce_loss + args.distill_weight * distill_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * query.size(0)
        total_items += query.size(0)
    return total_loss / max(total_items, 1)


def main() -> None:
    global args
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_query, train_image, train_topk, val_query, val_image, val_topk = load_assets(args)

    train_dataset = FeatureRerankTrainDataset(
        query_features=train_query["query_features"],
        image_features=train_image["features"],
        image_ids=train_image["image_ids"],
        topk_assets=train_topk,
        train_k=args.train_k,
        train_pool=args.train_pool,
    )
    val_dataset = FeatureRerankValDataset(
        query_features=val_query["query_features"],
        image_features=val_image["features"],
        image_ids=val_image["image_ids"],
        topk_assets=val_topk,
        val_k=args.val_k,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=0)

    input_dim = int(train_query["query_features"].shape[1])
    model = FeatureReranker(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        base_score_scale=args.base_score_scale,
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_path = args.output_dir / f"{args.category}_history.jsonl"
    best_metrics_path = args.output_dir / f"{args.category}_metrics_best.json"
    best_checkpoint_path = args.output_dir / f"{args.category}_best.pt"
    best_score = -math.inf

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, args)
        metrics = evaluate(model, val_loader)
        score = metrics[f"{args.category}_r10"] + metrics[f"{args.category}_r50"]
        history_record = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "score": float(score),
            **{k: float(v) for k, v in metrics.items()},
        }
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(history_record) + "\n")

        if score > best_score:
            best_score = score
            best_payload = {
                **history_record,
                "seed": args.seed,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "train_k": args.train_k,
                "train_pool": args.train_pool,
                "val_k": args.val_k,
                "distill_weight": args.distill_weight,
                "distill_temperature": args.distill_temperature,
                "base_score_scale": float(model.base_scale.detach().cpu().item()),
            }
            best_metrics_path.write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "best_metrics": best_payload,
                },
                best_checkpoint_path,
            )
        print(json.dumps(history_record))


if __name__ == "__main__":
    main()
