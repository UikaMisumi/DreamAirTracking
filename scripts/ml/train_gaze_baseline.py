#!/usr/bin/env python3
"""Train a small binocular gaze regressor from a Dream Air manifest."""

from __future__ import annotations

import argparse
import csv
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small


@dataclass(frozen=True)
class Sample:
    sample_id: str
    left_file: Path
    right_file: Path
    target: tuple[float, float]
    centers: tuple[float, float, float, float]
    weight: float
    split: str
    source: str


def read_manifest(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            samples.append(
                Sample(
                    sample_id=row["sample_id"],
                    left_file=Path(row["left_file"]),
                    right_file=Path(row["right_file"]),
                    target=(float(row["target_x"]), float(row["target_y"])),
                    centers=(
                        float(row["left_center_x"]),
                        float(row["left_center_y"]),
                        float(row["right_center_x"]),
                        float(row["right_center_y"]),
                    ),
                    weight=float(row.get("sample_weight", "1.0") or "1.0"),
                    split=row.get("split", "train"),
                    source=row.get("source", "real"),
                )
            )
    return samples


def augment_image(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.8:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.8:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.35:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.8)))
    return image


def image_to_tensor(path: Path, size: int, train: bool, rng: random.Random) -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    if train:
        image = augment_image(image, rng)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if train and rng.random() < 0.45:
        noise = rng.normalvariate(0.0, 0.025)
        array = np.clip(array + np.random.normal(noise, 0.018, array.shape), 0.0, 1.0)
    return torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0)


def geometry_to_tensor(sample: Sample) -> torch.Tensor:
    with Image.open(sample.left_file) as left_image:
        left_w, left_h = left_image.size
    with Image.open(sample.right_file) as right_image:
        right_w, right_h = right_image.size
    lx, ly, rx, ry = sample.centers
    left_x = lx / max(left_w, 1)
    left_y = ly / max(left_h, 1)
    right_x = rx / max(right_w, 1)
    right_y = ry / max(right_h, 1)
    return torch.tensor(
        [
            left_x,
            left_y,
            right_x,
            right_y,
            right_x - left_x,
            right_y - left_y,
        ],
        dtype=torch.float32,
    )


class GazeDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, train: bool, seed: int) -> None:
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        rng = random.Random(self.seed + index * 1009)
        left = image_to_tensor(sample.left_file, self.image_size, self.train, rng)
        right = image_to_tensor(sample.right_file, self.image_size, self.train, rng)
        target = torch.tensor(sample.target, dtype=torch.float32)
        weight = torch.tensor(sample.weight, dtype=torch.float32)
        return {
            "image": torch.cat([left, right], dim=0),
            "geometry": geometry_to_tensor(sample),
            "target": target,
            "weight": weight,
            "sample_id": sample.sample_id,
        }


class BinocularGazeNet(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(64, 2),
            nn.Tanh(),
        )
        self.image_size = image_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))


class MediumBinocularGazeNet(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 24, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 160, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(160),
            nn.SiLU(inplace=True),
            nn.Conv2d(160, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
            nn.Tanh(),
        )
        self.image_size = image_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(image))


class HybridGazeNet(nn.Module):
    def __init__(self, image_size: int, geometry_dim: int = 6) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, 24),
            nn.ReLU(inplace=True),
            nn.Linear(24, 24),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(96 + 24, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.12),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 2),
            nn.Tanh(),
        )
        self.image_size = image_size
        self.geometry_dim = geometry_dim

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.encoder(image), self.geometry(geometry)], dim=1))


class MobileNetV3SmallGazeNet(nn.Module):
    def __init__(self, image_size: int) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        first = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            2,
            first.out_channels,
            kernel_size=first.kernel_size,
            stride=first.stride,
            padding=first.padding,
            bias=False,
        )
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = backbone.classifier[0].in_features
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 160),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(160, 64),
            nn.Hardswish(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )
        self.image_size = image_size

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(image)))


def create_model(model_name: str, image_size: int) -> nn.Module:
    if model_name == "cnn":
        return BinocularGazeNet(image_size)
    if model_name == "medium_cnn":
        return MediumBinocularGazeNet(image_size)
    if model_name == "hybrid":
        return HybridGazeNet(image_size)
    if model_name == "mobilenetv3_small":
        return MobileNetV3SmallGazeNet(image_size)
    raise ValueError(f"Unknown model: {model_name}")


def forward_model(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    image = batch["image"].to(device)
    if isinstance(model, HybridGazeNet):
        return model(image, batch["geometry"].to(device))
    return model(image)


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=1)
    weighted = loss * weight
    return weighted.sum() / weight.sum().clamp_min(1e-6)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_l2 = 0.0
    total = 0
    for batch in loader:
        target = batch["target"].to(device)
        weight = batch["weight"].to(device)
        pred = forward_model(model, batch, device)
        loss = weighted_huber(pred, target, weight)
        l2 = torch.linalg.vector_norm(pred - target, dim=1).mean()
        count = target.shape[0]
        total_loss += float(loss) * count
        total_l2 += float(l2) * count
        total += count
    return {
        "loss": total_loss / max(total, 1),
        "l2": total_l2 / max(total, 1),
    }


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = read_manifest(args.manifest.resolve())
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    if not train_samples or not val_samples:
        raise SystemExit("Manifest must contain both train and val samples.")

    train_loader = DataLoader(
        GazeDataset(train_samples, args.image_size, train=True, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        GazeDataset(val_samples, args.image_size, train=False, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = create_model(args.model, args.image_size).to(device)
    if args.init_checkpoint:
        init = torch.load(args.init_checkpoint.resolve(), map_location=device, weights_only=False)
        init_model_type = str(init.get("model_type", args.model))
        init_image_size = int(init.get("image_size", args.image_size))
        if init_model_type != args.model:
            raise SystemExit(f"--init-checkpoint model_type={init_model_type} does not match --model {args.model}.")
        if init_image_size != args.image_size:
            raise SystemExit(f"--init-checkpoint image_size={init_image_size} does not match --image-size {args.image_size}.")
        model.load_state_dict(init["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history: list[dict[str, float]] = []
    best_val = math.inf
    best_state = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in train_loader:
            target = batch["target"].to(device)
            weight = batch["weight"].to(device)
            pred = forward_model(model, batch, device)
            loss = weighted_huber(pred, target, weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            count = target.shape[0]
            total_loss += float(loss.detach()) * count
            total += count

        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": total_loss / max(total, 1),
            "val_loss": val_metrics["loss"],
            "val_l2": val_metrics["l2"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_loss={row['val_loss']:.5f} val_l2={row['val_l2']:.5f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = args.output_dir / "gaze_baseline.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_type": args.model,
            "image_size": args.image_size,
            "manifest": str(args.manifest.resolve()),
            "stage_target_range": [-1.0, 1.0],
        },
        checkpoint_path,
    )

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        "device": str(device),
        "model_type": args.model,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "history": history,
    }
    report_path = args.output_dir / "training_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--model", choices=("cnn", "medium_cnn", "hybrid", "mobilenetv3_small"), default="cnn")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--init-checkpoint", type=Path, help="Optional checkpoint to warm-start fine-tuning.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    report = train(args)
    print(f"Saved checkpoint: {report['checkpoint']}")
    print(f"Best validation loss: {report['best_val_loss']:.6f}")
    print(f"Best epoch: {report['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
