#!/usr/bin/env python3
"""Train a MobileNetV3 binocular multitask eye model."""

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
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


WEAK_FIELDS = (
    "weak_left_openness",
    "weak_left_pupil_x",
    "weak_left_pupil_y",
    "weak_left_pupil_radius",
    "weak_left_quality",
    "weak_right_openness",
    "weak_right_pupil_x",
    "weak_right_pupil_y",
    "weak_right_pupil_radius",
    "weak_right_quality",
    "weak_pair_quality",
)
METADATA_FEATURE_NAMES = (
    "left_pupil_x",
    "left_pupil_y",
    "left_pupil_radius",
    "left_quality",
    "right_pupil_x",
    "right_pupil_y",
    "right_pupil_radius",
    "right_quality",
    "left_delta_x_from_session_center",
    "left_delta_y_from_session_center",
    "left_delta_radius_from_session_center",
    "right_delta_x_from_session_center",
    "right_delta_y_from_session_center",
    "right_delta_radius_from_session_center",
    "pair_quality",
    "visible_pupil_fraction",
)
METADATA_FEATURE_COUNT = len(METADATA_FEATURE_NAMES)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    left_file: Path
    right_file: Path
    gaze: tuple[float, float]
    openness: tuple[float, float]
    wide: tuple[float, float]
    squint: tuple[float, float]
    pupil: tuple[float, float, float, float, float, float]
    confidence: tuple[float, float, float]
    expression_mask: float
    openness_mask: tuple[float, float]
    wide_mask: tuple[float, float]
    squint_mask: tuple[float, float]
    pupil_mask: tuple[float, float]
    confidence_mask: tuple[float, float, float]
    has_weak: bool
    weight: float
    split: str
    session: str
    stage: str


def parse_float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    value = row.get(name, "")
    return default if value == "" else float(value)


def parse_optional_float(row: dict[str, str], name: str) -> float | None:
    value = row.get(name, "")
    return None if value == "" else float(value)


def parse_mask(row: dict[str, str], name: str, fallback: float = -1.0) -> float:
    value = parse_optional_float(row, name)
    return fallback if value is None else max(0.0, float(value))


def expression_targets_for_stage(stage: str) -> tuple[tuple[float, float], tuple[float, float], float]:
    normalized = stage.strip().lower()
    wide = (0.0, 0.0)
    squint = (0.0, 0.0)
    mask = 1.0
    if normalized in {"open_wide", "both_open_wide"}:
        wide = (1.0, 1.0)
    elif normalized == "left_wide_right_relaxed":
        wide = (1.0, 0.0)
    elif normalized == "right_wide_left_relaxed":
        wide = (0.0, 1.0)
    elif normalized in {"squint", "both_squint"}:
        squint = (1.0, 1.0)
    elif normalized in {"left_squint_right_open", "left_squint_right_relaxed"}:
        squint = (1.0, 0.0)
    elif normalized in {"right_squint_left_open", "right_squint_left_relaxed", "left_open_right_squint"}:
        squint = (0.0, 1.0)
    elif normalized == "":
        mask = 0.0
    return wide, squint, mask


def read_manifest(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            has_weak = all(field in row and row[field] != "" for field in WEAK_FIELDS)
            left_quality = parse_float(row, "weak_left_quality") if has_weak else 0.0
            right_quality = parse_float(row, "weak_right_quality") if has_weak else 0.0
            pair_quality = parse_float(row, "weak_pair_quality") if has_weak else 0.0
            stage = row.get("stage", "")
            stage_wide, stage_squint, stage_expression_mask = expression_targets_for_stage(stage)
            expression_mask = parse_float(row, "weak_expression_mask", stage_expression_mask)
            gaze_weight = parse_float(row, "gaze_weight", float(row.get("sample_weight", "1.0") or "1.0"))
            wide = (
                parse_float(row, "weak_left_wide", stage_wide[0]),
                parse_float(row, "weak_right_wide", stage_wide[1]),
            )
            squint = (
                parse_float(row, "weak_left_squint", stage_squint[0]),
                parse_float(row, "weak_right_squint", stage_squint[1]),
            )
            samples.append(
                Sample(
                    sample_id=row["sample_id"],
                    left_file=Path(row["left_file"]),
                    right_file=Path(row["right_file"]),
                    gaze=(float(row["target_x"]), float(row["target_y"])),
                    openness=(
                        parse_float(row, "weak_left_openness", 1.0),
                        parse_float(row, "weak_right_openness", 1.0),
                    ),
                    wide=wide,
                    squint=squint,
                    pupil=(
                        parse_float(row, "weak_left_pupil_x", 0.5),
                        parse_float(row, "weak_left_pupil_y", 0.54),
                        parse_float(row, "weak_left_pupil_radius", 0.08),
                        parse_float(row, "weak_right_pupil_x", 0.5),
                        parse_float(row, "weak_right_pupil_y", 0.54),
                        parse_float(row, "weak_right_pupil_radius", 0.08),
                    ),
                    confidence=(left_quality, right_quality, pair_quality),
                    expression_mask=expression_mask,
                    openness_mask=(
                        parse_mask(row, "openness_valid_left"),
                        parse_mask(row, "openness_valid_right"),
                    ),
                    wide_mask=(
                        parse_mask(row, "wide_valid_left"),
                        parse_mask(row, "wide_valid_right"),
                    ),
                    squint_mask=(
                        parse_mask(row, "squint_valid_left"),
                        parse_mask(row, "squint_valid_right"),
                    ),
                    pupil_mask=(
                        parse_mask(row, "pupil_valid_left"),
                        parse_mask(row, "pupil_valid_right"),
                    ),
                    confidence_mask=(
                        parse_mask(row, "confidence_valid_left"),
                        parse_mask(row, "confidence_valid_right"),
                        parse_mask(row, "confidence_valid_pair"),
                    ),
                    has_weak=has_weak,
                    weight=gaze_weight,
                    split=row.get("split", "train"),
                    session=row.get("session", ""),
                    stage=stage,
                )
            )
    return samples


def augment_image(image: Image.Image, rng: random.Random,
                  ty_frac: float = 0.10, tx_frac: float = 0.12, shadow: bool = False,
                  motion_blur: bool = False) -> Image.Image:
    if motion_blur and rng.random() < 0.25:
        # P0-1b: vertical motion blur — headset shake smears the eyelid vertically; without this
        # a blurred lid reads as half-closed. Box blur along y via padded cumsum (fast, exact).
        arr = np.asarray(image).astype(np.float32)
        k = rng.randint(3, 9)
        pad = np.pad(arr, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
        csum = np.cumsum(np.vstack([np.zeros((1, arr.shape[1]), dtype=np.float32), pad]), axis=0)
        blurred = (csum[k:] - csum[:-k]) / float(k)
        image = Image.fromarray(np.clip(blurred, 0.0, 255.0).astype(np.uint8))
    if rng.random() < 0.80:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.65, 1.35))
    if rng.random() < 0.80:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.75, 1.25))
    if rng.random() < 0.12:
        image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.0, 0.9)))
    if rng.random() < 0.85:
        width, height = image.size
        scale = rng.uniform(0.88, 1.15)
        # P0-1: vertical-translation range is row-type aware (eyelid aggressive / open gaze medium /
        # gaze moderate) so the openness head learns invariance to headset slide.
        tx = rng.uniform(-tx_frac, tx_frac) * width
        ty = rng.uniform(-ty_frac, ty_frac) * height
        cx = width * 0.5
        cy = height * 0.5
        fill = int(float(np.asarray(image).mean()))  # skin-ish fill so exposed edge isn't a black "closed" bar
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (
                1.0 / scale,
                0.0,
                cx - (cx + tx) / scale,
                0.0,
                1.0 / scale,
                cy - (cy + ty) / scale,
            ),
            resample=Image.Resampling.BILINEAR,
            fillcolor=fill,
        )
    if shadow and rng.random() < 0.35:
        # simulate a lens-rim / eyelid shadow band intruding from top or bottom on a fit shift
        arr = np.asarray(image).astype(np.float32)
        band_h = int(rng.uniform(0.12, 0.30) * arr.shape[0])
        factor = rng.uniform(0.3, 0.7)
        if rng.random() < 0.5:
            arr[:band_h] *= factor
        else:
            arr[arr.shape[0] - band_h:] *= factor
        image = Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8))
    return image


def image_to_tensor(path: Path, size: int, train: bool, rng: random.Random,
                    ty_frac: float = 0.10, tx_frac: float = 0.12, shadow: bool = False,
                    motion_blur: bool = False,
                    recenter_canonical: "tuple[float, float] | None" = None,
                    recenter_fallback: "tuple[float, float] | None" = None,
                    recenter_is_left: bool = True) -> torch.Tensor:
    image = Image.open(path).convert("L")
    if recenter_canonical is not None:
        # P0-2: normalize geometry BEFORE resize. Training uses the SESSION-median anchor
        # (one constant shift per session): it kills the wear-level geometry variance —
        # the actual problem — without per-frame jitter, and matches the runtime's slow
        # EMA behaviour. Per-frame estimation is only the fallback for unknown sessions.
        from pupil_recenter import estimate_dark_centroid, recenter_gray
        gray = np.asarray(image, dtype=np.uint8)
        center = recenter_fallback
        if center is None:
            cx, cy, conf = estimate_dark_centroid(gray, recenter_is_left)
            center = (cx, cy) if conf > 0.5 else None
        if center is not None:
            gray = recenter_gray(gray, center, recenter_canonical)
            image = Image.fromarray(gray)
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    if train:
        image = augment_image(image, rng, ty_frac, tx_frac, shadow, motion_blur)
    array = np.asarray(image, dtype=np.float32) / 255.0
    if train and rng.random() < 0.45:
        array = np.clip(array + np.random.normal(0.0, 0.018, array.shape), 0.0, 1.0)
    return torch.from_numpy(array.astype(np.float32, copy=False)).unsqueeze(0)


def build_session_recenter_centers(samples: "list[Sample]") -> dict:
    """Per-(session, side) median dark-centroid over a few open-ish frames — the fallback
    center for frames whose own centroid is untrustworthy (closed eyes). P0-2."""
    from pupil_recenter import estimate_dark_centroid
    by_session: dict[str, list[Sample]] = {}
    for sample in samples:
        by_session.setdefault(sample.session, []).append(sample)
    centers: dict = {}
    for session, rows in by_session.items():
        candidates = [s for s in rows if "closed" not in s.stage][:4] or rows[:4]
        for side, attr in (("left", "left_file"), ("right", "right_file")):
            pts = []
            for s in candidates:
                try:
                    gray = np.asarray(Image.open(getattr(s, attr)).convert("L"), dtype=np.uint8)
                except OSError:
                    continue
                cx, cy, conf = estimate_dark_centroid(gray, side == "left")
                if conf > 0.5:
                    pts.append((cx, cy))
            if pts:
                centers[(session, side)] = (
                    float(np.median([p[0] for p in pts])),
                    float(np.median([p[1] for p in pts])),
                )
    return centers


class EyeMultitaskDataset(Dataset):
    def __init__(self, samples: list[Sample], image_size: int, train: bool, seed: int, min_weak_quality: float,
                 gaze_openness_open_label: bool = False, pupil_recenter_input: bool = False) -> None:
        self.samples = samples
        self.image_size = image_size
        self.train = train
        self.seed = seed
        self.min_weak_quality = min_weak_quality
        self.gaze_openness_open_label = gaze_openness_open_label
        self.pupil_recenter_input = pupil_recenter_input
        self.center_anchors = build_session_center_anchors(samples, min_weak_quality)
        self.recenter_centers = build_session_recenter_centers(samples) if pupil_recenter_input else {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        rng = random.Random(self.seed + index * 1009)
        is_gaze = sample.weight > 0.0
        # P0-1/P1-7/P0-1b: row-type-aware vertical-translation + shake-blur for openness slide/shake-robustness
        if not is_gaze:
            ty_frac, tx_frac, shadow, mblur = 0.32, 0.15, True, True   # eyelid/expression: aggressive
        elif self.gaze_openness_open_label:
            ty_frac, tx_frac, shadow, mblur = 0.20, 0.13, True, True   # open gaze frames (now openness-supervised): medium
        else:
            ty_frac, tx_frac, shadow, mblur = 0.10, 0.12, False, False  # gaze without open-label: moderate (protect gaze)
        if self.pupil_recenter_input:
            # P0-2: geometry is normalized at the input, so big synthetic translations are no
            # longer needed (and would fight the normalization). Keep a small residual range
            # to cover centroid estimation error.
            from pupil_recenter import CANONICAL_LEFT, CANONICAL_RIGHT
            ty_frac, tx_frac = min(ty_frac, 0.10), min(tx_frac, 0.08)
            left = image_to_tensor(sample.left_file, self.image_size, self.train, rng, ty_frac, tx_frac, shadow, mblur,
                                   recenter_canonical=CANONICAL_LEFT,
                                   recenter_fallback=self.recenter_centers.get((sample.session, "left")),
                                   recenter_is_left=True)
            right = image_to_tensor(sample.right_file, self.image_size, self.train, rng, ty_frac, tx_frac, shadow, mblur,
                                    recenter_canonical=CANONICAL_RIGHT,
                                    recenter_fallback=self.recenter_centers.get((sample.session, "right")),
                                    recenter_is_left=False)
        else:
            left = image_to_tensor(sample.left_file, self.image_size, self.train, rng, ty_frac, tx_frac, shadow, mblur)
            right = image_to_tensor(sample.right_file, self.image_size, self.train, rng, ty_frac, tx_frac, shadow, mblur)
        left_quality, right_quality, pair_quality = sample.confidence
        weak_pair = 1.0 if sample.has_weak and pair_quality >= self.min_weak_quality else 0.0
        left_pupil = 1.0 if sample.has_weak and left_quality >= self.min_weak_quality else 0.0
        right_pupil = 1.0 if sample.has_weak and right_quality >= self.min_weak_quality else 0.0
        old_expression_mask = sample.expression_mask if sample.has_weak else 0.0
        openness_mask = sample.openness_mask if sample.openness_mask[0] >= 0.0 else (weak_pair, weak_pair)
        openness = sample.openness
        if self.gaze_openness_open_label and is_gaze:
            # P1-7: open-eye gaze frames get a hard open=1 label (protocol-known open), supervised at many
            # positions; a constant label means the openness head can't read gaze -> stays de-entangled.
            openness = (1.0, 1.0)
            openness_mask = (1.0, 1.0)
        wide_mask = sample.wide_mask if sample.wide_mask[0] >= 0.0 else (old_expression_mask, old_expression_mask)
        squint_mask = sample.squint_mask if sample.squint_mask[0] >= 0.0 else (old_expression_mask, old_expression_mask)
        pupil_side_mask = sample.pupil_mask if sample.pupil_mask[0] >= 0.0 else (left_pupil, right_pupil)
        confidence_mask = sample.confidence_mask if sample.confidence_mask[0] >= 0.0 else (
            1.0 if sample.has_weak else 0.0,
            1.0 if sample.has_weak else 0.0,
            1.0 if sample.has_weak else 0.0,
        )
        return {
            "image": torch.cat([left, right], dim=0),
            "metadata": torch.tensor(metadata_features(sample, self.center_anchors, self.min_weak_quality), dtype=torch.float32),
            "gaze": torch.tensor(sample.gaze, dtype=torch.float32),
            "openness": torch.tensor(openness, dtype=torch.float32),
            "wide": torch.tensor(sample.wide, dtype=torch.float32),
            "squint": torch.tensor(sample.squint, dtype=torch.float32),
            "pupil": torch.tensor(sample.pupil, dtype=torch.float32),
            "confidence": torch.tensor(sample.confidence, dtype=torch.float32),
            "gaze_weight": torch.tensor(sample.weight, dtype=torch.float32),
            "openness_mask": torch.tensor(openness_mask, dtype=torch.float32),
            "expression_mask": torch.tensor(max(wide_mask + squint_mask), dtype=torch.float32),
            "wide_mask": torch.tensor(wide_mask, dtype=torch.float32),
            "squint_mask": torch.tensor(squint_mask, dtype=torch.float32),
            "pupil_mask": torch.tensor(
                [pupil_side_mask[0], pupil_side_mask[0], pupil_side_mask[0], pupil_side_mask[1], pupil_side_mask[1], pupil_side_mask[1]],
                dtype=torch.float32,
            ),
            "confidence_mask": torch.tensor(confidence_mask, dtype=torch.float32),
            "sample_id": sample.sample_id,
        }


def build_session_center_anchors(samples: list[Sample], min_weak_quality: float) -> dict[str, tuple[float, float, float, float, float, float]]:
    grouped: dict[str, list[tuple[float, float, float, float, float, float]]] = {}
    for sample in samples:
        left_quality, right_quality, _ = sample.confidence
        if sample.stage != "center" or left_quality < min_weak_quality or right_quality < min_weak_quality:
            continue
        grouped.setdefault(sample.session, []).append(sample.pupil)
    anchors: dict[str, tuple[float, float, float, float, float, float]] = {}
    for session, values in grouped.items():
        array = np.asarray(values, dtype=np.float32)
        anchors[session] = tuple(float(x) for x in np.median(array, axis=0))
    return anchors


def metadata_features(
    sample: Sample,
    center_anchors: dict[str, tuple[float, float, float, float, float, float]],
    min_weak_quality: float,
) -> tuple[float, ...]:
    neutral = (0.5, 0.54, 0.08, 0.5, 0.54, 0.08)
    anchor = center_anchors.get(sample.session, neutral)
    left_quality, right_quality, pair_quality = sample.confidence
    left_visible = 1.0 if left_quality >= min_weak_quality else 0.0
    right_visible = 1.0 if right_quality >= min_weak_quality else 0.0
    left_x, left_y, left_radius, right_x, right_y, right_radius = sample.pupil
    center_left_x, center_left_y, center_left_radius, center_right_x, center_right_y, center_right_radius = anchor
    if not left_visible:
        left_x, left_y, left_radius = center_left_x, center_left_y, center_left_radius
    if not right_visible:
        right_x, right_y, right_radius = center_right_x, center_right_y, center_right_radius
    return (
        left_x,
        left_y,
        left_radius,
        left_quality,
        right_x,
        right_y,
        right_radius,
        right_quality,
        left_x - center_left_x,
        left_y - center_left_y,
        left_radius - center_left_radius,
        right_x - center_right_x,
        right_y - center_right_y,
        right_radius - center_right_radius,
        pair_quality,
        0.5 * (left_visible + right_visible),
    )


def build_pretrainable_backbone(in_channels: int, pretrained: bool) -> nn.Module:
    """MobileNetV3-small backbone with conv1 re-shaped for grayscale eye input.

    pretrained=True loads ImageNet weights and folds the RGB conv1 filters to the
    requested input-channel count (sum over RGB, normalized), so low-level
    edge/texture priors survive the domain change. E3 step 1: stop the weights=None
    from-scratch training that let the encoder overfit a single subject.
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    backbone = mobilenet_v3_small(weights=weights)
    first = backbone.features[0][0]
    new_conv = nn.Conv2d(
        in_channels,
        first.out_channels,
        kernel_size=first.kernel_size,
        stride=first.stride,
        padding=first.padding,
        bias=False,
    )
    if pretrained:
        with torch.no_grad():
            gray = first.weight.sum(dim=1, keepdim=True)  # [out,3,k,k] -> [out,1,k,k]
            new_conv.weight.copy_(gray.repeat(1, in_channels, 1, 1) / float(in_channels))
    backbone.features[0][0] = new_conv
    return backbone


class MobileNetV3SmallEyeMultitask(nn.Module):
    def __init__(self, image_size: int, pretrained: bool = False) -> None:
        super().__init__()
        backbone = build_pretrainable_backbone(2, pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = backbone.classifier[0].in_features
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 192),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(192, 128),
            nn.Hardswish(inplace=True),
        )
        self.gaze_head = nn.Sequential(nn.Linear(128, 2), nn.Tanh())
        self.openness_head = nn.Sequential(nn.Linear(128, 2), nn.Sigmoid())
        self.wide_head = nn.Sequential(nn.Linear(128, 2), nn.Sigmoid())
        self.squint_head = nn.Sequential(nn.Linear(128, 2), nn.Sigmoid())
        self.pupil_head = nn.Sequential(nn.Linear(128, 6), nn.Sigmoid())
        self.confidence_head = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())
        self.image_size = image_size

    def forward(self, image: torch.Tensor, metadata: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = self.shared(self.pool(self.features(image)))
        return (
            self.gaze_head(feature),
            self.openness_head(feature),
            self.wide_head(feature),
            self.squint_head(feature),
            self.pupil_head(feature),
            self.confidence_head(feature),
        )


class SiameseMobileNetV3SmallEyeMultitask(nn.Module):
    def __init__(self, image_size: int, metadata_size: int = 0, pretrained: bool = False) -> None:
        super().__init__()
        backbone = build_pretrainable_backbone(1, pretrained)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = backbone.classifier[0].in_features
        self.eye_shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 192),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(192, 128),
            nn.Hardswish(inplace=True),
        )
        self.metadata_size = metadata_size
        self.fusion = nn.Sequential(
            nn.Linear(128 * 4 + metadata_size, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.Hardswish(inplace=True),
        )
        self.gaze_head = nn.Sequential(nn.Linear(128, 2), nn.Tanh())
        self.left_openness_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.right_openness_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.left_wide_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.right_wide_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.left_squint_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.right_squint_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.left_pupil_head = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())
        self.right_pupil_head = nn.Sequential(nn.Linear(128, 3), nn.Sigmoid())
        self.left_confidence_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.right_confidence_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.pair_confidence_head = nn.Sequential(nn.Linear(128, 1), nn.Sigmoid())
        self.image_size = image_size

    def encode_eye(self, eye: torch.Tensor) -> torch.Tensor:
        return self.eye_shared(self.pool(self.features(eye)))

    def forward(self, image: torch.Tensor, metadata: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left = self.encode_eye(image[:, 0:1])
        right = self.encode_eye(image[:, 1:2])
        fused_input = torch.cat([left, right, right - left, left * right], dim=1)
        if self.metadata_size > 0:
            if metadata is None:
                metadata = torch.zeros(image.shape[0], self.metadata_size, dtype=image.dtype, device=image.device)
            fused_input = torch.cat([fused_input, metadata], dim=1)
        fused = self.fusion(fused_input)
        openness = torch.cat([self.left_openness_head(left), self.right_openness_head(right)], dim=1)
        wide = torch.cat([self.left_wide_head(left), self.right_wide_head(right)], dim=1)
        squint = torch.cat([self.left_squint_head(left), self.right_squint_head(right)], dim=1)
        pupil = torch.cat([self.left_pupil_head(left), self.right_pupil_head(right)], dim=1)
        confidence = torch.cat(
            [
                self.left_confidence_head(left),
                self.right_confidence_head(right),
                self.pair_confidence_head(fused),
            ],
            dim=1,
        )
        return (
            self.gaze_head(fused),
            openness,
            wide,
            squint,
            pupil,
            confidence,
        )


class SplitSiameseMobileNetV3SmallEyeMultitask(nn.Module):
    """Two isolated Siamese branches: geometry heads from one branch, expression heads from another."""

    def __init__(self, image_size: int, pretrained: bool = False) -> None:
        super().__init__()
        self.geometry = SiameseMobileNetV3SmallEyeMultitask(image_size, pretrained=pretrained)
        self.expression = SiameseMobileNetV3SmallEyeMultitask(image_size, pretrained=pretrained)
        self.image_size = image_size

    def forward(self, image: torch.Tensor, metadata: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        geometry_gaze, geometry_openness, _, _, geometry_pupil, geometry_confidence = self.geometry(image, metadata)
        _, _, expression_wide, expression_squint, _, _ = self.expression(image, metadata)
        return (
            geometry_gaze,
            geometry_openness,
            expression_wide,
            expression_squint,
            geometry_pupil,
            geometry_confidence,
        )


def model_type_from_architecture(architecture: str) -> str:
    if architecture == "two_channel":
        return "mobilenetv3_small_multitask"
    if architecture == "siamese":
        return "siamese_mobilenetv3_small_multitask"
    if architecture == "siamese_metadata":
        return "siamese_metadata_mobilenetv3_small_multitask"
    if architecture == "split_siamese":
        return "split_siamese_mobilenetv3_small_multitask"
    raise ValueError(f"Unsupported architecture: {architecture}")


def model_uses_metadata(model_type: str) -> bool:
    return model_type == "siamese_metadata_mobilenetv3_small_multitask"


def build_model(model_type: str, image_size: int, pretrained: bool = False) -> nn.Module:
    if model_type == "mobilenetv3_small_multitask":
        return MobileNetV3SmallEyeMultitask(image_size, pretrained=pretrained)
    if model_type == "siamese_mobilenetv3_small_multitask":
        return SiameseMobileNetV3SmallEyeMultitask(image_size, pretrained=pretrained)
    if model_type == "siamese_metadata_mobilenetv3_small_multitask":
        return SiameseMobileNetV3SmallEyeMultitask(image_size, metadata_size=METADATA_FEATURE_COUNT, pretrained=pretrained)
    if model_type == "split_siamese_mobilenetv3_small_multitask":
        return SplitSiameseMobileNetV3SmallEyeMultitask(image_size, pretrained=pretrained)
    raise ValueError(f"Unsupported multitask model_type={model_type}")


def load_pretrained_encoder(model: nn.Module, path: Path, device: str) -> None:
    """Load an OpenEDS-pretrained encoder ('features' state_dict) into a DreamAir model.

    Applies to the shared MobileNetV3 feature stack of the two_channel / siamese /
    siamese_metadata models, and to both branches of split_siamese. strict=False so a
    channel-mismatched conv1 (e.g. 2-ch two_channel vs 1-ch pretrain) is skipped.
    """
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "features" in state and isinstance(state["features"], dict):
        state = state["features"]
    stacks: list[tuple[str, nn.Module]] = []
    if hasattr(model, "features"):
        stacks.append(("features", model.features))
    if hasattr(model, "geometry") and hasattr(model.geometry, "features"):
        stacks.append(("geometry.features", model.geometry.features))
        stacks.append(("expression.features", model.expression.features))
    for name, feats in stacks:
        missing, unexpected = feats.load_state_dict(state, strict=False)
        print(f"Loaded pretrained encoder into {name}: matched={len(state) - len(unexpected)} "
              f"missing={len(missing)} unexpected={len(unexpected)}")


def weighted_mean(loss: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def compute_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    args: argparse.Namespace,
    teacher_outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    gaze_pred, openness_pred, wide_pred, squint_pred, pupil_pred, confidence_pred = outputs
    gaze_target = batch["gaze"]
    gaze_weight = batch["gaze_weight"]
    gaze_loss_each = nn.functional.smooth_l1_loss(gaze_pred, gaze_target, reduction="none").mean(dim=1)
    gaze_loss = weighted_mean(gaze_loss_each, gaze_weight)

    openness_mask = batch["openness_mask"]
    openness_loss_raw = nn.functional.l1_loss(openness_pred, batch["openness"], reduction="none")
    openness_loss = weighted_mean(openness_loss_raw, openness_mask)

    wide_target = batch["wide"].clamp(0, 1)
    wide_loss_raw = nn.functional.binary_cross_entropy(wide_pred, wide_target, reduction="none")
    wide_loss_raw = wide_loss_raw * (1.0 + (args.wide_positive_weight - 1.0) * wide_target)
    wide_loss = weighted_mean(wide_loss_raw, batch["wide_mask"])
    squint_target = batch["squint"].clamp(0, 1)
    squint_loss_raw = nn.functional.binary_cross_entropy(squint_pred, squint_target, reduction="none")
    squint_loss_raw = squint_loss_raw * (1.0 + (args.squint_positive_weight - 1.0) * squint_target)
    squint_loss = weighted_mean(squint_loss_raw, batch["squint_mask"])

    pupil_mask = batch["pupil_mask"]
    pupil_loss_raw = nn.functional.l1_loss(pupil_pred, batch["pupil"], reduction="none")
    pupil_loss = weighted_mean(pupil_loss_raw, pupil_mask)

    confidence_mask = batch["confidence_mask"]
    confidence_loss_raw = nn.functional.binary_cross_entropy(confidence_pred, batch["confidence"].clamp(0, 1), reduction="none")
    confidence_loss = weighted_mean(confidence_loss_raw, confidence_mask)

    center_consistency = torch.linalg.vector_norm(gaze_pred, dim=1)
    gaze_active = (gaze_weight > 0.0).float()
    center_targets = (gaze_target.abs().sum(dim=1) < 1e-6).float() * gaze_active
    center_loss = weighted_mean(center_consistency, center_targets)

    total = (
        args.gaze_weight * gaze_loss
        + args.openness_weight * openness_loss
        + args.wide_weight * wide_loss
        + args.squint_weight * squint_loss
        + args.pupil_weight * pupil_loss
        + args.confidence_weight * confidence_loss
        + args.center_consistency_weight * center_loss
    )
    parts = {
        "gaze_loss": float(gaze_loss.detach()),
        "openness_loss": float(openness_loss.detach()),
        "wide_loss": float(wide_loss.detach()),
        "squint_loss": float(squint_loss.detach()),
        "pupil_loss": float(pupil_loss.detach()),
        "confidence_loss": float(confidence_loss.detach()),
        "center_loss": float(center_loss.detach()),
    }
    if teacher_outputs is not None:
        teacher_gaze, teacher_openness, _, _, teacher_pupil, teacher_confidence = teacher_outputs
        teacher_gaze_loss_each = nn.functional.smooth_l1_loss(
            gaze_pred,
            teacher_gaze.detach(),
            reduction="none",
        ).mean(dim=1)
        teacher_gaze_loss = weighted_mean(teacher_gaze_loss_each, gaze_weight)
        teacher_openness_loss = weighted_mean(
            nn.functional.l1_loss(openness_pred, teacher_openness.detach(), reduction="none"),
            openness_mask,
        )
        teacher_pupil_loss = weighted_mean(
            nn.functional.l1_loss(pupil_pred, teacher_pupil.detach(), reduction="none"),
            pupil_mask,
        )
        teacher_confidence_loss = weighted_mean(
            nn.functional.l1_loss(confidence_pred, teacher_confidence.detach(), reduction="none"),
            confidence_mask,
        )
        total = (
            total
            + args.teacher_gaze_weight * teacher_gaze_loss
            + args.teacher_openness_weight * teacher_openness_loss
            + args.teacher_pupil_weight * teacher_pupil_loss
            + args.teacher_confidence_weight * teacher_confidence_loss
        )
        parts.update(
            {
                "teacher_gaze_loss": float(teacher_gaze_loss.detach()),
                "teacher_openness_loss": float(teacher_openness_loss.detach()),
                "teacher_pupil_loss": float(teacher_pupil_loss.detach()),
                "teacher_confidence_loss": float(teacher_confidence_loss.detach()),
            }
        )
    return total, parts


def forward_model(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return model(batch["image"], batch.get("metadata"))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    teacher_model: nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    if teacher_model is not None:
        teacher_model.eval()
    totals: dict[str, float] = {}
    weighted_gaze_sum = 0.0
    weighted_gaze_count = 0.0
    edge_gaze_sum = 0.0
    edge_gaze_count = 0.0
    micro_gaze_sum = 0.0
    micro_gaze_count = 0.0
    openness_mae_sum = 0.0
    openness_mae_count = 0.0
    wide_mae_sum = 0.0
    wide_mae_count = 0.0
    squint_mae_sum = 0.0
    squint_mae_count = 0.0
    pupil_mae_sum = 0.0
    pupil_mae_count = 0.0
    confidence_mae_sum = 0.0
    confidence_mae_count = 0.0
    count = 0
    for raw_batch in loader:
        batch = tensor_batch(raw_batch, device)
        outputs = forward_model(model, batch)
        teacher_outputs = None
        if teacher_model is not None:
            with torch.no_grad():
                teacher_outputs = forward_model(teacher_model, batch)
        loss, parts = compute_loss(outputs, batch, args, teacher_outputs)
        gaze_pred, openness_pred, wide_pred, squint_pred, pupil_pred, confidence_pred = outputs
        batch_size = batch["gaze"].shape[0]
        gaze_l2_each = torch.linalg.vector_norm(gaze_pred - batch["gaze"], dim=1)
        gaze_weight = batch["gaze_weight"].clamp_min(0.0)
        target_abs_max = batch["gaze"].abs().amax(dim=1)
        edge_mask = ((target_abs_max >= 0.45) & (gaze_weight > 0.0)).float()
        micro_mask = ((target_abs_max >= 0.20) & (target_abs_max < 0.45) & (gaze_weight > 0.0)).float()
        count += batch_size
        totals["loss"] = totals.get("loss", 0.0) + float(loss) * batch_size
        for key, value in parts.items():
            totals[key] = totals.get(key, 0.0) + value * batch_size
        totals["gaze_l2"] = totals.get("gaze_l2", 0.0) + float(gaze_l2_each.mean()) * batch_size
        weighted_gaze_sum += float((gaze_l2_each * gaze_weight).sum())
        weighted_gaze_count += float(gaze_weight.sum())
        edge_gaze_sum += float((gaze_l2_each * edge_mask).sum())
        edge_gaze_count += float(edge_mask.sum())
        micro_gaze_sum += float((gaze_l2_each * micro_mask).sum())
        micro_gaze_count += float(micro_mask.sum())
        openness_abs = torch.abs(openness_pred - batch["openness"])
        openness_mask = batch["openness_mask"]
        openness_mae_sum += float((openness_abs * openness_mask).sum())
        openness_mae_count += float(openness_mask.sum())
        wide_abs = torch.abs(wide_pred - batch["wide"])
        wide_mask = batch["wide_mask"]
        wide_mae_sum += float((wide_abs * wide_mask).sum())
        wide_mae_count += float(wide_mask.sum())
        squint_abs = torch.abs(squint_pred - batch["squint"])
        squint_mask = batch["squint_mask"]
        squint_mae_sum += float((squint_abs * squint_mask).sum())
        squint_mae_count += float(squint_mask.sum())
        pupil_abs = torch.abs(pupil_pred - batch["pupil"])
        pupil_mask = batch["pupil_mask"]
        pupil_mae_sum += float((pupil_abs * pupil_mask).sum())
        pupil_mae_count += float(pupil_mask.sum())
        confidence_abs = torch.abs(confidence_pred - batch["confidence"])
        confidence_mask = batch["confidence_mask"]
        confidence_mae_sum += float((confidence_abs * confidence_mask).sum())
        confidence_mae_count += float(confidence_mask.sum())
    result = {key: value / max(count, 1) for key, value in totals.items()}
    result["gaze_l2_weighted"] = weighted_gaze_sum / max(weighted_gaze_count, 1e-6)
    result["gaze_l2_edge"] = edge_gaze_sum / max(edge_gaze_count, 1e-6)
    result["gaze_l2_micro"] = micro_gaze_sum / max(micro_gaze_count, 1e-6)
    result["gaze_edge_count"] = edge_gaze_count
    result["gaze_micro_count"] = micro_gaze_count
    result["gaze_weighted_count"] = weighted_gaze_count
    result["openness_mae"] = openness_mae_sum / max(openness_mae_count, 1e-6)
    result["wide_mae"] = wide_mae_sum / max(wide_mae_count, 1e-6)
    result["squint_mae"] = squint_mae_sum / max(squint_mae_count, 1e-6)
    result["pupil_mae"] = pupil_mae_sum / max(pupil_mae_count, 1e-6)
    result["confidence_mae"] = confidence_mae_sum / max(confidence_mae_count, 1e-6)
    result["openness_valid_count"] = openness_mae_count
    result["wide_valid_count"] = wide_mae_count
    result["squint_valid_count"] = squint_mae_count
    result["pupil_valid_count"] = pupil_mae_count
    result["confidence_valid_count"] = confidence_mae_count
    return result


def selection_metric_value(val: dict[str, float], metric: str) -> float:
    if metric == "expression_gaze_combo":
        return (
            val["wide_mae"]
            + 0.5 * val["squint_mae"]
            + 0.25 * val.get("gaze_l2_edge", val["gaze_l2"])
            + 0.2 * val["openness_mae"]
        )
    if metric == "eye_control_combo":
        return (
            val["openness_mae"]
            + val["wide_mae"]
            + 0.5 * val["squint_mae"]
            + 0.45 * val.get("gaze_l2_edge", val["gaze_l2"])
            + 0.2 * val["pupil_mae"]
        )
    if metric == "eye_control_edge_combo":
        return (
            val["openness_mae"]
            + val["wide_mae"]
            + 0.5 * val["squint_mae"]
            + 0.65 * val.get("gaze_l2_edge", val.get("gaze_l2_weighted", val["gaze_l2"]))
            + 0.15 * val["pupil_mae"]
        )
    if metric == "eye_control_edge_center_combo":
        return (
            val["openness_mae"]
            + val["wide_mae"]
            + 0.5 * val["squint_mae"]
            + 0.65 * val.get("gaze_l2_edge", val.get("gaze_l2_weighted", val["gaze_l2"]))
            + 0.25 * val.get("center_loss", 0.0)
            + 0.15 * val["pupil_mae"]
        )
    return {
        "gaze_l2": val["gaze_l2"],
        "gaze_l2_weighted": val.get("gaze_l2_weighted", val["gaze_l2"]),
        "gaze_l2_edge": val.get("gaze_l2_edge", val.get("gaze_l2_weighted", val["gaze_l2"])),
        "openness_mae": val["openness_mae"],
        "wide_mae": val["wide_mae"],
        "squint_mae": val["squint_mae"],
        "loss": val["loss"],
    }[metric]


def keep_frozen_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        params = list(module.parameters(recurse=False))
        if params and not any(parameter.requires_grad for parameter in params):
            module.eval()


def keep_all_batchnorm_eval(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def tensor_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def train(args: argparse.Namespace) -> dict[str, object]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    from manifest_head_mask_gate import enforce as enforce_manifest_gate

    enforce_manifest_gate(
        args.manifest.resolve(),
        allow_weak_fallback=getattr(args, "allow_weak_label_fallback", False),
        allow_gaze_head_cosupervision=getattr(args, "allow_gaze_head_cosupervision", False),
        skip=getattr(args, "skip_manifest_audit", False),
    )
    samples = read_manifest(args.manifest.resolve())
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    if not train_samples or not val_samples:
        raise SystemExit("Manifest must contain both train and val samples.")

    train_loader = DataLoader(
        EyeMultitaskDataset(train_samples, args.image_size, train=True, seed=args.seed, min_weak_quality=args.min_weak_quality, gaze_openness_open_label=args.gaze_openness_open_label, pupil_recenter_input=args.pupil_recenter_input),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        EyeMultitaskDataset(val_samples, args.image_size, train=False, seed=args.seed, min_weak_quality=args.min_weak_quality, pupil_recenter_input=args.pupil_recenter_input),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.cpu,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_type = model_type_from_architecture(args.architecture)
    model = build_model(model_type, args.image_size, pretrained=getattr(args, "pretrained_backbone", False)).to(device)
    if getattr(args, "pretrained_encoder", None):
        load_pretrained_encoder(model, args.pretrained_encoder, device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        checkpoint_model_type = checkpoint.get("model_type")
        if checkpoint_model_type and checkpoint_model_type != model_type:
            raise SystemExit(f"Checkpoint model_type={checkpoint_model_type} does not match requested model_type={model_type}.")
        missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
        if missing:
            print(f"Init checkpoint missing new parameters: {', '.join(missing)}")
        if unexpected:
            print(f"Init checkpoint ignored unexpected parameters: {', '.join(unexpected)}")
    teacher_model = None
    if args.teacher_checkpoint:
        teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
        teacher_model_type = teacher_checkpoint.get("model_type")
        if teacher_model_type and teacher_model_type != model_type:
            raise SystemExit(f"Teacher checkpoint model_type={teacher_model_type} does not match requested model_type={model_type}.")
        teacher_model = build_model(model_type, args.image_size).to(device)
        missing, unexpected = teacher_model.load_state_dict(teacher_checkpoint["model_state"], strict=False)
        if missing:
            print(f"Teacher checkpoint missing new parameters: {', '.join(missing)}")
        if unexpected:
            print(f"Teacher checkpoint ignored unexpected parameters: {', '.join(unexpected)}")
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False
    if args.train_expression_heads_only and args.train_expression_tail_only:
        raise ValueError("--train-expression-heads-only and --train-expression-tail-only are mutually exclusive")
    if args.train_expression_heads_only or args.train_expression_tail_only:
        trainable_keywords = ["wide_head", "squint_head"]
        if args.train_expression_tail_only:
            trainable_keywords += ["features.10.", "features.11.", "features.12.", "eye_shared.4."]
        trainable_count = 0
        frozen_count = 0
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(keyword in name for keyword in trainable_keywords)
            if parameter.requires_grad:
                trainable_count += parameter.numel()
            else:
                frozen_count += parameter.numel()
        mode = "expression tail only" if args.train_expression_tail_only else "expression heads only"
        print(f"Training {mode}: trainable={trainable_count} frozen={frozen_count}")
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_metric = math.inf
    best_state = None
    best_epoch = 0
    train_steps = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_batchnorm:
            keep_all_batchnorm_eval(model)
        if args.train_expression_heads_only or args.train_expression_tail_only:
            keep_frozen_batchnorm_eval(model)
        total_loss = 0.0
        total = 0
        stop_training = False
        for raw_batch in train_loader:
            batch = tensor_batch(raw_batch, device)
            teacher_outputs = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = forward_model(teacher_model, batch)
            loss, _ = compute_loss(forward_model(model, batch), batch, args, teacher_outputs)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch["gaze"].shape[0]
            total += batch["gaze"].shape[0]
            train_steps += 1
            if args.max_train_steps and train_steps >= args.max_train_steps:
                stop_training = True
                break

        val = evaluate(model, val_loader, device, args, teacher_model)
        row = {"epoch": float(epoch), "train_steps": float(train_steps), "train_loss": total_loss / max(total, 1), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(
            f"epoch={epoch:03d} train_loss={row['train_loss']:.5f} "
            f"val_loss={row['val_loss']:.5f} val_gaze_l2={row['val_gaze_l2']:.5f} "
            f"val_gaze_edge={row['val_gaze_l2_edge']:.5f} "
            f"val_open={row['val_openness_mae']:.5f} val_wide={row['val_wide_mae']:.5f} "
            f"val_squint={row['val_squint_mae']:.5f} val_pupil={row['val_pupil_mae']:.5f}",
            flush=True,
        )
        selection_metric = selection_metric_value(val, args.selection_metric)
        if selection_metric < best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        if stop_training:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    last_state = copy.deepcopy(model.state_dict())

    def checkpoint_payload(state_dict: dict[str, torch.Tensor]) -> dict[str, object]:
        return {
            "model_state": state_dict,
            "model_type": model_type,
            "architecture": args.architecture,
            "image_size": args.image_size,
            "manifest": str(args.manifest.resolve()),
            "heads": ["gaze_xy", "openness_lr", "wide_lr", "squint_lr", "pupil_lr", "confidence"],
            "metadata_features": list(METADATA_FEATURE_NAMES) if model_uses_metadata(model_type) else [],
            "loss_weights": {
                "gaze": args.gaze_weight,
                "openness": args.openness_weight,
                "wide": args.wide_weight,
                "squint": args.squint_weight,
                "pupil": args.pupil_weight,
                "confidence": args.confidence_weight,
                "center_consistency": args.center_consistency_weight,
                "teacher_gaze": args.teacher_gaze_weight,
                "teacher_openness": args.teacher_openness_weight,
                "teacher_pupil": args.teacher_pupil_weight,
                "teacher_confidence": args.teacher_confidence_weight,
            },
            "teacher_checkpoint": str(args.teacher_checkpoint.resolve()) if args.teacher_checkpoint else None,
        }

    last_checkpoint_path = args.output_dir / "eye_multitask.last.pt"
    torch.save(checkpoint_payload(last_state), last_checkpoint_path)
    if best_state is not None:
        model.load_state_dict(best_state)
    checkpoint_path = args.output_dir / "eye_multitask.pt"
    torch.save(checkpoint_payload(model.state_dict()), checkpoint_path)

    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "last_checkpoint": str(last_checkpoint_path.resolve()),
        "device": str(device),
        "model_type": model_type,
        "architecture": args.architecture,
        "image_size": args.image_size,
        "init_checkpoint": str(args.init_checkpoint.resolve()) if args.init_checkpoint else None,
        "teacher_checkpoint": str(args.teacher_checkpoint.resolve()) if args.teacher_checkpoint else None,
        "teacher_weights": {
            "gaze": args.teacher_gaze_weight,
            "openness": args.teacher_openness_weight,
            "pupil": args.teacher_pupil_weight,
            "confidence": args.teacher_confidence_weight,
        },
        "train_expression_heads_only": args.train_expression_heads_only,
        "train_expression_tail_only": args.train_expression_tail_only,
        "freeze_batchnorm": args.freeze_batchnorm,
        "max_train_steps": args.max_train_steps,
        "train_steps": train_steps,
        "manifest": str(args.manifest.resolve()),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "weak_train_samples": sum(1 for sample in train_samples if sample.has_weak),
        "weak_val_samples": sum(1 for sample in val_samples if sample.has_weak),
        "best_epoch": best_epoch,
        "selection_metric": args.selection_metric,
        "best_selection_metric": best_metric,
        "history": history,
    }
    report_path = args.output_dir / "train_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--allow-weak-label-fallback",
        action="store_true",
        help="Opt into the legacy silent fallback: when a manifest lacks openness/pupil valid "
        "columns, supervise those heads from weak labels instead of failing the head-mask gate.",
    )
    parser.add_argument(
        "--allow-gaze-head-cosupervision",
        action="store_true",
        help="Allow gaze rows to also supervise openness/pupil (demote that head-mask check to a warning). "
        "Note: the current v5-style manifests trip this; the clean fix is to set openness/pupil valid=0 on gaze rows.",
    )
    parser.add_argument(
        "--skip-manifest-audit",
        action="store_true",
        help="Bypass the head-mask training gate entirely (not recommended).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pupil-recenter-input", action="store_true",
                        help="P0-2: train on pupil-recentered crops (geometric normalization); export/stamp metadata recentered_input accordingly.")
    parser.add_argument("--gaze-openness-open-label", action="store_true",
                        help="P1-7: supervise openness=1.0 (hard, protocol-known open) on open-eye gaze rows so the "
                             "openness head sees open eyes at many positions; constant label keeps gaze de-entangled.")
    parser.add_argument("--architecture", choices=("two_channel", "siamese", "siamese_metadata", "split_siamese"), default="two_channel")
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Initialize the MobileNetV3 backbone from ImageNet (conv1 folded to grayscale) instead of "
        "weights=None. E3 step 1: for training a base model; backbone weights are overridden by --init-checkpoint.",
    )
    parser.add_argument(
        "--pretrained-encoder",
        type=Path,
        default=None,
        help="Load an OpenEDS-pretrained encoder ('features' state_dict from "
        "pretrain_openeds_segmentation.py) into the backbone. E3 step 2.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-train-steps", type=int, default=0, help="Stop after this many optimizer steps; 0 means use all epochs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--min-weak-quality", type=float, default=0.12)
    parser.add_argument("--gaze-weight", type=float, default=1.0)
    parser.add_argument("--openness-weight", type=float, default=0.25)
    parser.add_argument("--wide-weight", type=float, default=0.25)
    parser.add_argument("--squint-weight", type=float, default=0.20)
    parser.add_argument("--wide-positive-weight", type=float, default=1.0)
    parser.add_argument("--squint-positive-weight", type=float, default=1.0)
    parser.add_argument("--pupil-weight", type=float, default=0.20)
    parser.add_argument("--confidence-weight", type=float, default=0.10)
    parser.add_argument("--center-consistency-weight", type=float, default=0.05)
    parser.add_argument("--teacher-gaze-weight", type=float, default=0.0)
    parser.add_argument("--teacher-openness-weight", type=float, default=0.0)
    parser.add_argument("--teacher-pupil-weight", type=float, default=0.0)
    parser.add_argument("--teacher-confidence-weight", type=float, default=0.0)
    parser.add_argument(
        "--selection-metric",
        choices=(
            "gaze_l2",
            "gaze_l2_weighted",
            "gaze_l2_edge",
            "openness_mae",
            "wide_mae",
            "squint_mae",
            "loss",
            "expression_gaze_combo",
            "eye_control_combo",
            "eye_control_edge_combo",
            "eye_control_edge_center_combo",
        ),
        default="gaze_l2",
    )
    parser.add_argument("--train-expression-heads-only", action="store_true")
    parser.add_argument("--train-expression-tail-only", action="store_true")
    parser.add_argument("--freeze-batchnorm", action="store_true", help="Keep BatchNorm modules in eval mode during training.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    report = train(args)
    print(f"Saved checkpoint: {report['checkpoint']}")
    print(f"Best {report['selection_metric']}: {report['best_selection_metric']:.6f}")
    print(f"Best epoch: {report['best_epoch']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
