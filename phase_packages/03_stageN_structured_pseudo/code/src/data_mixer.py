from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import ConcatDataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import dataloader


@dataclass
class MixerMeta:
    mix_mode: str
    pseudo_ratio: float
    num_samples: int
    pseudo_size: int
    base_size: int


def _normalize_quality_weights(pseudo_dataset, power: float) -> list[float]:
    raw_weights = list(getattr(pseudo_dataset, "sample_weights", []))
    if not raw_weights:
        return [1.0] * len(pseudo_dataset)
    min_weight = min(raw_weights)
    max_weight = max(raw_weights)
    if max_weight <= min_weight:
        return [1.0] * len(raw_weights)
    normalized = []
    for weight in raw_weights:
        scaled = (weight - min_weight) / (max_weight - min_weight)
        normalized.append(max(scaled, 1e-3) ** power)
    return normalized


def _compute_pseudo_ratio(args, epoch: int) -> float:
    if epoch < args.pseudo_start_epoch:
        return 0.0
    if args.mix_mode == "static":
        return float(args.pseudo_max_ratio)

    ramp_epochs = max(int(args.pseudo_ramp_epochs), 1)
    progress = min(max((epoch - args.pseudo_start_epoch + 1) / ramp_epochs, 0.0), 1.0)
    if args.mix_mode == "curriculum":
        return float(args.pseudo_max_ratio) * progress
    if args.mix_mode == "dynamic_weighted":
        return float(args.pseudo_max_ratio) * (0.5 + 0.5 * progress)
    return float(args.pseudo_max_ratio)


def build_epoch_loader(base_dataset, pseudo_dataset, args, epoch: int):
    pseudo_ratio = _compute_pseudo_ratio(args, epoch)
    if pseudo_dataset is None or pseudo_ratio <= 0.0:
        loader = dataloader.DataLoader(
            base_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        return loader, MixerMeta(
            mix_mode=args.mix_mode,
            pseudo_ratio=0.0,
            num_samples=len(base_dataset),
            pseudo_size=0,
            base_size=len(base_dataset),
        )

    concat_dataset = ConcatDataset([base_dataset, pseudo_dataset])
    base_weights = [1.0] * len(base_dataset)
    pseudo_quality_weights = _normalize_quality_weights(pseudo_dataset, float(args.pseudo_quality_power))
    pseudo_scale = (pseudo_ratio / max(1.0 - pseudo_ratio, 1e-6)) * (
        len(base_dataset) / max(len(pseudo_dataset), 1)
    )
    sample_weights = base_weights + [pseudo_scale * weight for weight in pseudo_quality_weights]
    num_samples = int(args.pseudo_samples_per_epoch) if int(args.pseudo_samples_per_epoch) > 0 else len(base_dataset)
    sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
    loader = dataloader.DataLoader(
        concat_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    return loader, MixerMeta(
        mix_mode=args.mix_mode,
        pseudo_ratio=float(pseudo_ratio),
        num_samples=num_samples,
        pseudo_size=len(pseudo_dataset),
        base_size=len(base_dataset),
    )
