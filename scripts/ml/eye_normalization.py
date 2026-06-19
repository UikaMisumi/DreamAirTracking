"""Eye crop normalization helpers for BrokenEye JPEG streams."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EyeNormalizationOptions:
    mode: str = "off"
    target_x: float = 0.50
    target_y: float = 0.54
    max_shift_x: float = 0.18
    max_shift_y: float = 0.16
    scale_min: float = 0.86
    scale_max: float = 1.18
    hold_frames: int = 5
    min_confidence: float = 0.10
    min_openness: float = 0.18
    min_aperture_height: int = 8
    target_radius: float = 0.085


@dataclass(frozen=True)
class EyeNormalizationMetadata:
    found: bool
    confidence: float
    shift_x: float
    shift_y: float
    scale: float
    pupil_x: float
    pupil_y: float
    drop_reason: str | None

    @staticmethod
    def identity(options: EyeNormalizationOptions, reason: str | None) -> "EyeNormalizationMetadata":
        return EyeNormalizationMetadata(
            found=False,
            confidence=0.0,
            shift_x=0.0,
            shift_y=0.0,
            scale=1.0,
            pupil_x=float(options.target_x),
            pupil_y=float(options.target_y),
            drop_reason=reason,
        )


@dataclass(frozen=True)
class EyeNormalizationResult:
    image: np.ndarray
    metadata: EyeNormalizationMetadata


@dataclass(frozen=True)
class _Transform:
    pupil_x: float
    pupil_y: float
    shift_x: float
    shift_y: float
    scale: float
    confidence: float


class EyeNormalizationState:
    def __init__(self) -> None:
        self.previous: _Transform | None = None
        self.missing_frames = 0

    def update(
        self,
        jpeg: bytes,
        image_size: int,
        options: EyeNormalizationOptions,
        openness: float | None = None,
        aperture_height: int | None = None,
    ) -> EyeNormalizationResult:
        if options.mode == "off":
            return EyeNormalizationResult(_resize_jpeg(jpeg, image_size), EyeNormalizationMetadata.identity(options, "off"))
        if options.mode not in ("probe_only", "pupil_center"):
            raise ValueError(f"Unsupported normalization mode: {options.mode}")

        try:
            image = Image.open(BytesIO(jpeg)).convert("L")
        except Exception as exc:
            return self._fallback(jpeg, image_size, options, f"decode failed: {exc}")

        gray = np.asarray(image, dtype=np.uint8)
        candidate = _detect_pupil_candidate(gray)
        drop_reason = _quality_drop_reason(candidate, options, openness, aperture_height)
        if drop_reason is None and candidate is not None:
            transform = _build_transform(candidate, options)
            self.previous = transform
            self.missing_frames = 0
            metadata = EyeNormalizationMetadata(
                found=True,
                confidence=transform.confidence,
                shift_x=transform.shift_x,
                shift_y=transform.shift_y,
                scale=transform.scale,
                pupil_x=transform.pupil_x,
                pupil_y=transform.pupil_y,
                drop_reason=None,
            )
            if options.mode == "probe_only":
                return EyeNormalizationResult(_resize_image(image, image_size), metadata)
            return EyeNormalizationResult(_apply_transform(image, image_size, transform, options), metadata)

        return self._fallback(jpeg, image_size, options, drop_reason or "pupil not found")

    def _fallback(
        self,
        jpeg: bytes,
        image_size: int,
        options: EyeNormalizationOptions,
        reason: str,
    ) -> EyeNormalizationResult:
        self.missing_frames += 1
        if self.previous is not None and self.missing_frames <= max(0, options.hold_frames):
            metadata = EyeNormalizationMetadata(
                found=False,
                confidence=max(0.0, self.previous.confidence * 0.5),
                shift_x=self.previous.shift_x,
                shift_y=self.previous.shift_y,
                scale=self.previous.scale,
                pupil_x=self.previous.pupil_x,
                pupil_y=self.previous.pupil_y,
                drop_reason=f"held_previous: {reason}",
            )
            try:
                image = Image.open(BytesIO(jpeg)).convert("L")
            except Exception:
                return EyeNormalizationResult(_neutral_image(image_size), metadata)
            if options.mode == "probe_only":
                return EyeNormalizationResult(_resize_image(image, image_size), metadata)
            return EyeNormalizationResult(_apply_transform(image, image_size, self.previous, options), metadata)

        return EyeNormalizationResult(_resize_jpeg(jpeg, image_size), EyeNormalizationMetadata.identity(options, reason))


@dataclass(frozen=True)
class _PupilCandidate:
    x: float
    y: float
    radius: float
    confidence: float
    area: int


def _resize_jpeg(jpeg: bytes, image_size: int) -> np.ndarray:
    try:
        image = Image.open(BytesIO(jpeg)).convert("L")
    except Exception:
        return _neutral_image(image_size)
    return _resize_image(image, image_size)


def _resize_image(image: Image.Image, image_size: int) -> np.ndarray:
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def _neutral_image(image_size: int) -> np.ndarray:
    return np.full((image_size, image_size), 0.5, dtype=np.float32)


def _detect_pupil_candidate(gray: np.ndarray) -> _PupilCandidate | None:
    height, width = gray.shape
    if height < 12 or width < 12:
        return None

    x0 = int(width * 0.12)
    x1 = int(width * 0.88)
    y0 = int(height * 0.10)
    y1 = int(height * 0.90)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    threshold = int(min(np.percentile(roi, 12.0) + 10.0, 105.0))
    mask = roi <= threshold
    if int(mask.sum()) < 8:
        return None

    component = _largest_component(mask)
    if component is None:
        return None

    ys, xs = component
    area = int(xs.size)
    if area < 8:
        return None

    intensities = roi[ys, xs].astype(np.float32)
    weights = np.maximum(1.0, float(threshold) + 1.0 - intensities)
    center_x = float(np.average(xs + x0, weights=weights))
    center_y = float(np.average(ys + y0, weights=weights))
    radius = float(np.sqrt(area / np.pi) / max(width, height))
    local_darkness = max(0.0, (float(np.median(roi)) - float(np.mean(intensities))) / 80.0)
    area_score = min(1.0, area / max(1.0, width * height * 0.018))
    confidence = float(np.clip((area_score * 0.65) + (local_darkness * 0.35), 0.0, 1.0))
    return _PupilCandidate(center_x / width, center_y / height, radius, confidence, area)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    starts = np.argwhere(mask)
    for start_y, start_x in starts:
        if seen[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        seen[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if len(points) > len(best):
            best = points

    if not best:
        return None
    ys = np.fromiter((point[0] for point in best), dtype=np.int32)
    xs = np.fromiter((point[1] for point in best), dtype=np.int32)
    return ys, xs


def _quality_drop_reason(
    candidate: _PupilCandidate | None,
    options: EyeNormalizationOptions,
    openness: float | None,
    aperture_height: int | None,
) -> str | None:
    if candidate is None:
        return "pupil not found"
    if candidate.confidence < options.min_confidence:
        return f"low pupil confidence {candidate.confidence:.3f}"
    if openness is not None and openness < options.min_openness:
        return f"low openness {openness:.3f}"
    if aperture_height is not None and aperture_height < options.min_aperture_height:
        return f"low aperture {aperture_height}"
    return None


def _build_transform(candidate: _PupilCandidate, options: EyeNormalizationOptions) -> _Transform:
    shift_x = float(np.clip(candidate.x - options.target_x, -options.max_shift_x, options.max_shift_x))
    shift_y = float(np.clip(candidate.y - options.target_y, -options.max_shift_y, options.max_shift_y))
    if candidate.radius > 1e-6:
        scale = options.target_radius / candidate.radius
    else:
        scale = 1.0
    scale = float(np.clip(scale, options.scale_min, options.scale_max))
    return _Transform(candidate.x, candidate.y, shift_x, shift_y, scale, candidate.confidence)


def _apply_transform(
    image: Image.Image,
    image_size: int,
    transform: _Transform,
    options: EyeNormalizationOptions,
) -> np.ndarray:
    source = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    fill = int(np.median(np.asarray(source, dtype=np.uint8)))
    scale = max(transform.scale, 1e-6)
    effective_x = options.target_x + transform.shift_x
    effective_y = options.target_y + transform.shift_y
    a = 1.0 / scale
    e = 1.0 / scale
    c = image_size * (effective_x - (options.target_x / scale))
    f = image_size * (effective_y - (options.target_y / scale))
    normalized = source.transform(
        (image_size, image_size),
        Image.Transform.AFFINE,
        (a, 0.0, c, 0.0, e, f),
        resample=Image.Resampling.BILINEAR,
        fillcolor=fill,
    )
    return np.asarray(normalized, dtype=np.float32) / 255.0
