"""Load EfficientViT-SAM models from the pinned third_party/efficientvit checkout."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
EVIT_ROOT = REPO_ROOT / "third_party" / "efficientvit"
if str(EVIT_ROOT) not in sys.path:
    sys.path.insert(0, str(EVIT_ROOT))

from efficientvit.models.efficientvit.sam import EfficientViTSam, EfficientViTSamPredictor  # noqa: E402
from efficientvit.sam_model_zoo import create_efficientvit_sam_model  # noqa: E402

DEFAULT_WEIGHTS = REPO_ROOT / "assets" / "checkpoints" / "efficientvit_sam_l0.pt"


def load_sam(
    name: str = "efficientvit-sam-l0",
    weights: str | os.PathLike | None = None,
    device: str = "cpu",
) -> EfficientViTSam:
    weights = Path(weights) if weights else DEFAULT_WEIGHTS
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found. Run scripts/setup_assets.sh first.")
    model = create_efficientvit_sam_model(name, pretrained=True, weight_url=str(weights))
    return model.to(device).eval()


def load_predictor(**kwargs) -> EfficientViTSamPredictor:
    return EfficientViTSamPredictor(load_sam(**kwargs))


__all__ = ["load_sam", "load_predictor", "EfficientViTSam", "EfficientViTSamPredictor", "torch"]
