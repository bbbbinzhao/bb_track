"""Causal search-region state shared by diagnostics and policy training."""

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F


OBSERVATION_DIM = 32
FACTORS = (4.0, 6.0, 8.0)
EXTRAPOLATIONS = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class SearchRegion:
    center_xy: tuple
    factor: float

    def as_dict(self):
        return {"center_xy": list(self.center_xy), "factor": float(self.factor)}


@dataclass(frozen=True)
class CropGeometry:
    center_xy: tuple
    origin_xy: tuple
    side: int
    factor: float
    output_size: int

    @classmethod
    def from_box(cls, box, factor, output_size):
        x, y, w, h = map(float, box)
        side = math.ceil(math.sqrt(w * h) * factor)
        if side < 1:
            raise ValueError("A search crop must have positive area")
        center = (x + w / 2, y + h / 2)
        origin = tuple(round(c - side / 2) for c in center)
        return cls(center, origin, side, float(factor), output_size)

    def crop_to_image(self, box):
        """Map a pixel xywh box through the actual integer crop origin."""
        x, y, w, h = map(float, box)
        scale = self.side / self.output_size
        return [self.origin_xy[0] + x * scale, self.origin_xy[1] + y * scale,
                w * scale, h * scale]

    def image_to_crop(self, box):
        x, y, w, h = map(float, box)
        scale = self.output_size / self.side
        return [(x - self.origin_xy[0]) * scale, (y - self.origin_xy[1]) * scale,
                w * scale, h * scale]

    def as_dict(self):
        return {"center_xy": list(self.center_xy), "origin_xy": list(self.origin_xy),
                "side": self.side, "factor": self.factor, "output_size": self.output_size}


def box_center(box):
    return np.asarray(box[:2], dtype=np.float64) + np.asarray(box[2:], dtype=np.float64) / 2


def allowed_actions(mode="joint"):
    if mode == "joint":
        return tuple(range(9))
    if mode == "center":
        return (1, 4, 7)  # Keep factor 6.
    if mode == "scale":
        return (0, 1, 2)  # Keep the predicted target center.
    raise ValueError("Unknown action mode: " + mode)


def candidate_regions(box, velocity, geometry, image_shape, expanded=False):
    height, width = image_shape[:2]
    center = box_center(box)
    displacement = np.asarray(velocity, dtype=np.float64).copy()
    limit = geometry.side * 0.25
    norm = np.linalg.norm(displacement)
    if norm > limit:
        displacement *= limit / norm
    if expanded:
        centers = [center + limit * np.asarray([dx, dy])
                   for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    else:
        centers = [center + k * displacement for k in EXTRAPOLATIONS]
    return [SearchRegion(tuple(np.clip(c, [0, 0], [width - 1, height - 1])), factor)
            for c in centers for factor in FACTORS]


class MotionState:
    def __init__(self):
        self.velocity = np.zeros(2, dtype=np.float64)
        self.previous_box = None
        self.previous_reliable = False

    def update(self, box, reliable):
        previous = self.previous_box
        if reliable and self.previous_reliable and previous is not None:
            self.velocity = 0.8 * self.velocity + 0.2 * (box_center(box) - box_center(previous))
        else:
            self.velocity *= 0.8
        scale_change = (np.log(np.maximum(box[2:], 1e-6) / np.maximum(previous[2:], 1e-6))
                        if previous is not None else np.zeros(2))
        self.previous_box = np.asarray(box, dtype=np.float64).copy()
        self.previous_reliable = bool(reliable)
        return scale_change


def make_observation(score_map, presence, box, velocity, scale_change, geometry, image_shape):
    scores = score_map.detach().float().reshape(1, 1, *score_map.shape[-2:])
    pooled = F.adaptive_avg_pool2d(scores, (4, 4)).flatten()
    flat = scores.flatten().clamp_min(1e-8)
    distribution = flat / flat.sum()
    entropy = -(distribution * distribution.log()).sum() / math.log(flat.numel())
    index = int(flat.argmax())
    peak_xy = [index % scores.shape[-1] / scores.shape[-1],
               index // scores.shape[-1] / scores.shape[-2]]
    height, width = image_shape[:2]
    normalized_box = np.asarray(box) / [width, height, width, height]
    geometry_values = np.concatenate([
        [presence, float(flat.max()), float(entropy)], peak_xy, normalized_box,
        np.asarray(velocity) / geometry.side, np.clip(scale_change, -5, 5),
        [geometry.factor / 8], np.asarray(geometry.center_xy) / [width, height]])
    observation = torch.cat([pooled.cpu(), torch.as_tensor(geometry_values, dtype=torch.float32)])
    if observation.numel() != OBSERVATION_DIM or not torch.isfinite(observation).all():
        raise ValueError("Invalid search-policy observation")
    return observation
