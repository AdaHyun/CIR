from __future__ import annotations

import argparse
import json
from pathlib import Path

import open_clip
import torch

import datasets
import model as cir_model
import test


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dress", "shirt", "toptee", "shoes", "cirr"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fashioniq_split", default="val-split")
    parser.add_argument("--fashioniq_path", default=str(DATA_ROOT / "FashionIQ"))
    parser.add_argument("--shoes_path", default=str(DATA_ROOT / "Shoes"))
    parser.add_argument("--cirr_path", default=str(DATA_ROOT / "CIRR"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--backbone", default="ViT-H-14")
    parser.add_argument("--pretrained", default="laion2B-s32B-b79K")
    parser.add_argument(
        "--model_variant",
        choices=[
            "dqu",
            "dqu_pfp",
            "dqu_pfp_refcond",
            "dqu_structured_text",
            "dqu_structured_text_conf",
            "dqu_structured_context",
            "dqu_structured_rm",
            "dqu_structured_rm_attr",
            "dqu_zvep_cir",
            "dqu_zvep_cir_reg",
        ],
        default=None,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser.parse_args()


def load_dataset(args: argparse.Namespace):
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.backbone,
        pretrained=args.pretrained,
    )
    transforms = [preprocess_train, preprocess_val]

    if args.dataset in ["dress", "shirt", "toptee"]:
        return datasets.FashionIQ(
            path=args.fashioniq_path,
            category=args.dataset,
            transform=transforms,
            split=args.fashioniq_split,
        )
    if args.dataset == "shoes":
        return datasets.Shoes(path=args.shoes_path, transform=transforms)
    if args.dataset == "cirr":
        return datasets.CIRR(path=args.cirr_path, transform=transforms)
    raise ValueError(args.dataset)


def load_model(args: argparse.Namespace) -> tuple[torch.nn.Module, dict]:
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
        net.load_state_dict(payload["model_state_dict"])
    else:
        net = payload
    return net.cuda().eval(), checkpoint_args


def main() -> None:
    args = parse_args()
    net, checkpoint_args = load_model(args)
    if checkpoint_args:
        args.backbone = checkpoint_args.get("backbone", args.backbone)
        args.pretrained = checkpoint_args.get("pretrained", args.pretrained)
        args.model_variant = checkpoint_args.get("model_variant", args.model_variant)

    dataset = load_dataset(args)

    with torch.no_grad():
        if args.dataset == "cirr":
            metrics = test.test_cirr_valset(args, net, dataset)
        else:
            metrics = test.test(args, net, dataset, args.dataset)

    result = {name: float(value) for name, value in metrics}
    result["model_variant"] = args.model_variant
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
