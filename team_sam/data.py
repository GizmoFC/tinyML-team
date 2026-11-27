"""COCO data access and the frozen train/val/test splits (proposal Sec. 3.3)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO

REPO_ROOT = Path(__file__).resolve().parent.parent
COCO_ROOT = REPO_ROOT / "data" / "coco"
SPLITS_PATH = REPO_ROOT / "splits" / "splits.json"

MIN_MASK_PIXELS = 100  # exclude masks covering fewer than 100 pixels in the original image


@dataclass
class Instance:
    image_id: int
    ann_id: int
    category_id: int
    mask: np.ndarray  # HxW bool, original image resolution
    box_xyxy: np.ndarray  # (4,) float, original image coordinates
    area_bucket: str  # "small" | "medium" | "large" (COCO area thresholds)


def area_bucket(area: float) -> str:
    if area < 32**2:
        return "small"
    if area < 96**2:
        return "medium"
    return "large"


def load_coco(split: str = "val2017") -> COCO:
    return COCO(str(COCO_ROOT / "annotations" / f"instances_{split}.json"))


def eligible_instances(coco: COCO, image_id: int) -> list[Instance]:
    """All non-crowd instances of an image whose decoded mask has >= MIN_MASK_PIXELS pixels."""
    out = []
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
    for ann in coco.loadAnns(ann_ids):
        if ann.get("iscrowd", 0):
            continue
        m = coco.annToMask(ann).astype(bool)
        npix = int(m.sum())
        if npix < MIN_MASK_PIXELS:
            continue
        x, y, w, h = ann["bbox"]
        out.append(
            Instance(
                image_id=image_id,
                ann_id=ann["id"],
                category_id=ann["category_id"],
                mask=m,
                box_xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
                area_bucket=area_bucket(npix),
            )
        )
    return out


def image_path(coco: COCO, image_id: int, split: str = "val2017") -> Path:
    return COCO_ROOT / split / coco.loadImgs([image_id])[0]["file_name"]


def load_splits(path: Path = SPLITS_PATH) -> dict:
    with open(path) as f:
        return json.load(f)
