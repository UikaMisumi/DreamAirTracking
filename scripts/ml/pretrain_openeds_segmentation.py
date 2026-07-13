#!/usr/bin/env python3
"""Pretrain the MobileNetV3-small eye encoder on OpenEDS 2020 segmentation (E3 step 2).

Intermediate pretraining on PUBLIC multi-person IR near-eye data (OpenEDS 2020:
87 subjects, VR HMD IR cameras, per-pixel eye-region/iris/pupil masks) so the
encoder learns cross-person eye features instead of overfitting a single subject.
The trained encoder ('features' stack) is saved for
`train_eye_multitask.py --pretrained-encoder <path>`.

Dataset: phorosyne/OpenEDS_2020_Shards (MDS columns: image bytes, mask ndarray,
subject int, has_mask int). License: OpenEDS is research / non-commercial — cite
the OpenEDS/OpenEDS2020 papers; compatible with this project's non-commercial use.

  python pretrain_openeds_segmentation.py --data datasets/openeds2020 --output runs/openeds_encoder/encoder.pt
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_eye_multitask import build_pretrainable_backbone  # noqa: E402


def _mask_to_labels(mask: np.ndarray) -> np.ndarray:
    """Coerce an OpenEDS mask ndarray to a 2D int label map."""
    arr = np.asarray(mask)
    if arr.ndim == 3:
        # channel-last one-hot / RGB -> collapse to a label map
        arr = arr[..., 0] if arr.shape[-1] <= 4 else arr.argmax(-1)
    return arr.astype(np.int64)


class OpenEdsSegDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: Path, split: str, image_size: int, limit: int = 0):
        from streaming import LocalDataset

        self.ds = LocalDataset(local=str(Path(data_dir) / split))
        self.image_size = image_size
        n = len(self.ds)
        # This OpenEDS 2020 mirror ships dense per-frame masks, so use all frames
        # directly (a per-sample has_mask scan would decode the whole set).
        self.indices = list(range(n if limit <= 0 else min(limit, n)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, k: int):
        sample = self.ds[self.indices[k]]
        image = Image.open(io.BytesIO(sample["image"])).convert("L").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).unsqueeze(0)
        labels = _mask_to_labels(sample["mask"])
        mask_img = Image.fromarray(labels.astype(np.uint8)).resize(
            (self.image_size, self.image_size), Image.NEAREST
        )
        y = torch.from_numpy(np.asarray(mask_img, dtype=np.int64))
        return x, y


class SegModel(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone: bool = True):
        super().__init__()
        backbone = build_pretrainable_backbone(1, pretrained_backbone)
        self.features = backbone.features
        feat_channels = backbone.classifier[0].in_features  # 576 for mobilenet_v3_small
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Hardswish(inplace=True),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        logits = self.decoder(feat)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def infer_num_classes(dataset: OpenEdsSegDataset, probe: int = 32) -> int:
    hi = 0
    for k in range(min(probe, len(dataset))):
        _, y = dataset[k]
        hi = max(hi, int(y.max().item()))
    return hi + 1


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_px = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += F.cross_entropy(logits, y).item() * x.size(0)
            pred = logits.argmax(1)
            total_correct += (pred == y).sum().item()
            total_px += y.numel()
    n = len(loader.dataset)
    return total_loss / max(n, 1), total_correct / max(total_px, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="OpenEDS 2020 local dir (with train/ val/).")
    ap.add_argument("--output", required=True, type=Path, help="Destination encoder .pt (features state_dict).")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Cap masked samples per split (debug).")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    train_ds = OpenEdsSegDataset(args.data, "train", args.image_size, args.limit)
    val_ds = OpenEdsSegDataset(args.data, "val", args.image_size, args.limit)
    print(f"OpenEDS seg: train masked={len(train_ds)} val masked={len(val_ds)}")
    num_classes = infer_num_classes(train_ds)
    print(f"inferred num_classes={num_classes} | device={device}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    model = SegModel(num_classes, pretrained_backbone=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    best_acc = -1.0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        started = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        train_loss = running / max(len(train_ds), 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_px_acc={val_acc:.4f} ({time.time()-started:.0f}s)")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.features.state_dict(), args.output)
    print(f"Saved best encoder (val_px_acc={best_acc:.4f}) -> {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
