import argparse
import json
import logging
import os
import random
import warnings
from pathlib import Path

import numpy as np
import open_clip
import torch
import utils
import datasets
import data_mixer
import model
import test

from torch.cuda.amp import autocast as autocast, GradScaler
from torch.utils.data import ConcatDataset
from torch.utils.data import dataloader
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.set_num_threads(2)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=os.getenv("LOCAL_RANK", -1), type=int)
parser.add_argument("--dataset", default="dress", help="data set type")
parser.add_argument("--fashioniq_split", default="val-split")
parser.add_argument("--fashioniq_path", default=str(DEFAULT_DATA_ROOT / "FashionIQ"))
parser.add_argument("--shoes_path", default=str(DEFAULT_DATA_ROOT / "Shoes"))
parser.add_argument("--fashion200k_path", default=str(DEFAULT_DATA_ROOT / "Fashion200K"))
parser.add_argument("--cirr_path", default=str(DEFAULT_DATA_ROOT / "CIRR"))

parser.add_argument("--optimizer", default="adamw")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--eps", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--dropout_rate", type=float, default=0.5)
parser.add_argument("--hidden_dim", type=int, default=1024)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--clip_lr", type=float, default=1e-6)

parser.add_argument("--backbone", type=str, default="ViT-H-14")
parser.add_argument("--pretrained", type=str, default="laion2B-s32B-b79K")
parser.add_argument("--debug_cuda_launch_blocking", action="store_true")
parser.add_argument(
    "--model_variant",
    choices=[
        "dqu",
        "dqu_pfp",
        "dqu_pfp_refcond",
        "dqu_structured_text",
        "dqu_structured_rm",
        "dqu_structured_rm_attr",
    ],
    default="dqu",
)

parser.add_argument("--lr_decay", type=int, default=8)
parser.add_argument("--lr_div", type=float, default=0.1)
parser.add_argument("--max_decay_epoch", type=int, default=10)
parser.add_argument("--tolerance_epoch", type=int, default=5)

parser.add_argument("--pseudo_path", type=str, default="")
parser.add_argument("--pseudo_start_epoch", type=int, default=0)
parser.add_argument("--pseudo_min_score", type=float, default=0.0)
parser.add_argument("--pseudo_label", type=str, default="none")
parser.add_argument("--pseudo_repeat", type=int, default=1)
parser.add_argument("--mix_mode", choices=["concat", "static", "curriculum", "dynamic_weighted"], default="concat")
parser.add_argument("--pseudo_max_ratio", type=float, default=0.3)
parser.add_argument("--pseudo_ramp_epochs", type=int, default=5)
parser.add_argument("--pseudo_quality_power", type=float, default=1.0)
parser.add_argument("--pseudo_samples_per_epoch", type=int, default=0)

parser.add_argument("--model_dir", default="./checkpoints", help="Directory containing params.json")
parser.add_argument("--save_summary_steps", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--i", type=str, default="0")

args = parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_transforms():
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        args.backbone,
        pretrained=args.pretrained,
    )
    return preprocess_train, preprocess_val


def load_dataset():
    preprocess_train, preprocess_val = build_transforms()
    if args.dataset in ["dress", "shirt", "toptee"]:
        fashioniq_dataset = datasets.FashionIQ(
            path=args.fashioniq_path,
            category=args.dataset,
            transform=[preprocess_train, preprocess_val],
            split=args.fashioniq_split,
        )
        return [fashioniq_dataset]
    if args.dataset == "shoes":
        shoes_dataset = datasets.Shoes(path=args.shoes_path, transform=[preprocess_train, preprocess_val])
        return [shoes_dataset]
    if args.dataset == "cirr":
        cirr_dataset = datasets.CIRR(path=args.cirr_path, transform=[preprocess_train, preprocess_val])
        return [cirr_dataset]
    if args.dataset == "fashion200k":
        fashion200k_dataset = datasets.Fashion200k(
            path=args.fashion200k_path,
            split="train",
            transform=[preprocess_train, preprocess_val],
        )
        fashion200k_testset = datasets.Fashion200k(
            path=args.fashion200k_path,
            split="test",
            transform=[preprocess_train, preprocess_val],
        )
        return [fashion200k_dataset, fashion200k_testset]
    raise ValueError(args.dataset)


def build_train_dataset(base_dataset):
    if args.dataset not in ["dress", "shirt", "toptee"]:
        return base_dataset
    if not args.pseudo_path:
        return base_dataset
    pseudo_dataset = datasets.FashionIQPseudo(
        path=args.fashioniq_path,
        category=args.dataset,
        pseudo_path=args.pseudo_path,
        transform=base_dataset.transform,
        min_score=args.pseudo_min_score,
    )
    logging.info("Loaded %d pseudo samples for %s", len(pseudo_dataset), args.dataset)
    if len(pseudo_dataset) == 0:
        return base_dataset
    repeat_count = max(int(args.pseudo_repeat), 1)
    concat_parts = [base_dataset] + [pseudo_dataset for _ in range(repeat_count)]
    logging.info(
        "Using pseudo dataset repeat=%d base_size=%d pseudo_size=%d mixed_size=%d",
        repeat_count,
        len(base_dataset),
        len(pseudo_dataset),
        sum(len(part) for part in concat_parts),
    )
    return ConcatDataset(concat_parts)


def load_pseudo_dataset(base_dataset):
    if args.dataset not in ["dress", "shirt", "toptee"]:
        return None
    if not args.pseudo_path:
        return None
    pseudo_dataset = datasets.FashionIQPseudo(
        path=args.fashioniq_path,
        category=args.dataset,
        pseudo_path=args.pseudo_path,
        transform=base_dataset.transform,
        min_score=args.pseudo_min_score,
    )
    logging.info("Loaded %d pseudo samples for %s", len(pseudo_dataset), args.dataset)
    return pseudo_dataset


def set_bn_eval(module):
    classname = module.__class__.__name__
    if classname.find("BatchNorm2d") != -1:
        module.eval()


def create_model_and_optimizer():
    net = model.build_model(
        model_variant=args.model_variant,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout_rate,
        backbone=args.backbone,
        pretrained=args.pretrained,
    )
    net.cuda()

    params = list(net.named_parameters())
    param_group = [
        {"params": [p for n, p in params if "clip" in n], "lr": args.clip_lr},
        {"params": [p for n, p in params if "clip" not in n], "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_group, lr=args.lr, weight_decay=args.weight_decay)
    return net, optimizer


def train_epoch(net, optimizer, train_loader, scaler):
    net.train()
    net.apply(set_bn_eval)
    loss_avg = utils.RunningAverage()
    with tqdm(total=len(train_loader)) as progress:
        for step, batch in enumerate(train_loader):
            target_img = batch["target_img_data"].cuda()
            textual_query = batch["textual_query"]
            visual_query = batch["visual_query"].cuda()
            optimizer.zero_grad()
            with autocast():
                loss = net.compute_loss(textual_query, visual_query, target_img)
                total_loss = loss["ranking"]

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_avg.update(total_loss.item())
            progress.set_postfix(loss="{:05.3f}".format(loss_avg()))
            progress.update()

            if step % args.save_summary_steps == 0:
                logging.info("step=%d total_loss=%.6f", step, total_loss.item())
    return loss_avg()


def evaluate_once(net, dataset_list):
    if args.dataset in ["dress", "shirt", "toptee", "shoes"]:
        with torch.no_grad():
            return test.test(args, net, dataset_list[0], args.dataset)
    if args.dataset == "fashion200k":
        return test.test_fashion200k_dataset(args, net, dataset_list[1])
    if args.dataset == "cirr":
        return test.test_cirr_valset(args, net, dataset_list[0])
    raise ValueError(args.dataset)


def score_metrics(metrics):
    if args.dataset in ["dress", "shirt", "toptee", "shoes"]:
        return metrics[1][1] + metrics[2][1]
    if args.dataset == "fashion200k":
        return metrics[0][1] + metrics[1][1] + metrics[2][1]
    if args.dataset == "cirr":
        return sum(metric[1] for metric in metrics)
    raise ValueError(args.dataset)


def save_best(net, optimizer, epoch, best_score, metrics):
    best_json_path = os.path.join(args.model_dir, "{}_{}_metrics_best.json".format(args.dataset, args.i))
    test_metrics = {metric_name: metric_value for metric_name, metric_value in metrics}
    test_metrics["model_variant"] = args.model_variant
    test_metrics["pseudo_label"] = args.pseudo_label
    test_metrics["pseudo_path"] = args.pseudo_path
    utils.save_dict_to_json(test_metrics, best_json_path)

    checkpoint_path = os.path.join(args.model_dir, "{}_{}_best_model.pt".format(args.dataset, args.i))
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "epoch": epoch + 1,
            "best_score": best_score,
            "metrics": test_metrics,
        },
        checkpoint_path,
    )


def append_history(epoch, train_loss, metrics, current_score, best_score, used_pseudo):
    history_path = os.path.join(args.model_dir, "{}_{}_history.jsonl".format(args.dataset, args.i))
    history_record = {
        "epoch": epoch + 1,
        "train_loss": float(train_loss),
        "score": float(current_score),
        "best_score": float(best_score),
        "used_pseudo": bool(used_pseudo),
    }
    for metric_name, metric_value in metrics:
        history_record[metric_name] = float(metric_value)
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(history_record, ensure_ascii=False) + "\n")


def build_train_loader(dataset):
    return dataloader.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )


def train_and_evaluate(net, optimizer, dataset_list):
    scaler = GradScaler()
    best_score = float("-inf")
    tolerance = 0

    if args.dataset == "fashion200k":
        base_train_dataset = dataset_list[0]
        mixed_train_dataset = base_train_dataset
        pseudo_dataset = None
    else:
        base_train_dataset = dataset_list[0]
        pseudo_dataset = load_pseudo_dataset(base_train_dataset)
        mixed_train_dataset = build_train_dataset(base_train_dataset) if pseudo_dataset is not None else base_train_dataset

    logging.info(
        "Dataset sizes: base=%d mixed=%d pseudo_path=%s pseudo_start_epoch=%d pseudo_repeat=%d pseudo_min_score=%.4f",
        len(base_train_dataset),
        len(mixed_train_dataset),
        args.pseudo_path,
        args.pseudo_start_epoch,
        args.pseudo_repeat,
        args.pseudo_min_score,
    )

    for epoch in range(args.num_epochs):
        tolerance += 1
        if epoch != 0 and (epoch + 1) % args.lr_decay == 0 and epoch < args.max_decay_epoch:
            for group in optimizer.param_groups:
                group["lr"] *= args.lr_div

        logging.info("Epoch %d/%d", epoch + 1, args.num_epochs)
        use_pseudo = bool(args.pseudo_path) and epoch >= args.pseudo_start_epoch
        if args.mix_mode == "concat":
            if args.pseudo_path and epoch < args.pseudo_start_epoch:
                train_loader = build_train_loader(base_train_dataset)
            else:
                train_loader = build_train_loader(mixed_train_dataset)
        elif pseudo_dataset is None:
            train_loader = build_train_loader(base_train_dataset)
        else:
            train_loader, mix_meta = data_mixer.build_epoch_loader(
                base_dataset=base_train_dataset,
                pseudo_dataset=pseudo_dataset,
                args=args,
                epoch=epoch,
            )
            logging.info(
                "Epoch %d mixer: mode=%s pseudo_ratio=%.4f base_size=%d pseudo_size=%d num_samples=%d",
                epoch + 1,
                mix_meta.mix_mode,
                mix_meta.pseudo_ratio,
                mix_meta.base_size,
                mix_meta.pseudo_size,
                mix_meta.num_samples,
            )
        train_loss = train_epoch(net, optimizer, train_loader, scaler)

        if tolerance >= args.tolerance_epoch:
            break

        metrics = evaluate_once(net, dataset_list)
        logging.info(metrics)
        current_score = score_metrics(metrics)

        if current_score > best_score:
            best_score = current_score
            tolerance = 0
            save_best(net, optimizer, epoch, best_score, metrics)

        try:
            append_history(
                epoch=epoch,
                train_loss=train_loss,
                metrics=metrics,
                current_score=current_score,
                best_score=best_score,
                used_pseudo=use_pseudo,
            )
        except Exception as exc:  # pragma: no cover
            logging.warning("Failed to append history: %s", exc)


if __name__ == "__main__":
    print("Arguments:")
    for key in args.__dict__.keys():
        print("   ", key, ":", str(args.__dict__[key]))

    if args.debug_cuda_launch_blocking:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    seed_everything(args.seed)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    utils.set_logger(os.path.join(args.model_dir, "{}_{}_train.log".format(args.dataset, args.i)))
    logging.info("Loading the datasets and model...")

    dataset_list = load_dataset()
    net, optimizer = create_model_and_optimizer()
    logging.info("Starting train for %d epoch(s)", args.num_epochs)
    train_and_evaluate(net, optimizer, dataset_list)
