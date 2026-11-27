"""Core ML backend for team_sam.evaluate (macOS only: Core ML runtime is not on Linux).

Mirrors what the iPhone app must do, so its IoU is the "desktop Core ML" number:
  1. resize longest side to 512 (PIL bilinear, same as the PyTorch pipeline)
  2. pad bottom/right to 512x512 with the mean color RGB(124, 116, 104)
  3. encoder -> (1,256,64,64) embedding, cached per image
  4. prompts scaled to the 1024 frame; with no box, append padding point (0,0,-1)
  5. decoder -> 4 low-res masks; index 0 = single-mask, 1..3 = multimask
  6. upsample 256 -> 1024 (bilinear), crop the valid region, resize to original
"""

from __future__ import annotations

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PAD_RGB = (124, 116, 104)
ENC_SIZE = 512
FRAME = 1024


def _preprocess_shape(h: int, w: int, long_side: int) -> tuple[int, int]:
    s = long_side / max(h, w)
    return int(h * s + 0.5), int(w * s + 0.5)


class CoreMLSegmenter:
    def __init__(self, encoder_path: str, decoder_path: str, compute_units: str = "ALL"):
        cu = getattr(ct.ComputeUnit, compute_units)
        self.enc = ct.models.MLModel(encoder_path, compute_units=cu)
        self.dec = ct.models.MLModel(decoder_path, compute_units=cu)

    def set_image(self, rgb: np.ndarray) -> None:
        h, w = rgb.shape[:2]
        self.original_size = (h, w)
        nh, nw = _preprocess_shape(h, w, ENC_SIZE)
        canvas = Image.new("RGB", (ENC_SIZE, ENC_SIZE), PAD_RGB)
        canvas.paste(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR), (0, 0))
        self.emb = self.enc.predict({"image": canvas})["image_embeddings"].astype(np.float32)
        self.input_size = _preprocess_shape(h, w, FRAME)

    def _to_frame(self, xy: np.ndarray) -> np.ndarray:
        h, w = self.original_size
        nh, nw = self.input_size
        xy = np.asarray(xy, dtype=np.float32).reshape(-1, 2).copy()
        xy[:, 0] *= nw / w
        xy[:, 1] *= nh / h
        return xy

    def predict(self, points=None, labels=None, box=None, multimask=True):
        coords, labs = [], []
        if points is not None:
            coords.append(self._to_frame(points))
            labs.append(np.asarray(labels, dtype=np.float32))
        if box is not None:
            coords.append(self._to_frame(box))
            labs.append(np.array([2, 3], dtype=np.float32))
        else:
            coords.append(np.zeros((1, 2), np.float32))
            labs.append(np.array([-1], np.float32))
        out = self.dec.predict(
            {
                "image_embeddings": self.emb,
                "point_coords": np.concatenate(coords)[None],
                "point_labels": np.concatenate(labs)[None],
            }
        )
        low = torch.from_numpy(out["low_res_masks"].astype(np.float32))
        iou = out["iou_predictions"].astype(np.float32)[0]
        sel = slice(1, 4) if multimask else slice(0, 1)
        low, iou = low[:, sel], iou[sel]
        m = F.interpolate(low, (FRAME, FRAME), mode="bilinear", align_corners=False)
        m = m[..., : self.input_size[0], : self.input_size[1]]
        m = F.interpolate(m, self.original_size, mode="bilinear", align_corners=False)
        return (m[0] > 0).numpy(), iou
