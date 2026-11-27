"""Interactive-segmentation evaluation on the frozen COCO splits (proposal Sec. 3.3).

Per instance it records:
  - iou@1, iou@3, iou@5: click-based IoU (first click = deepest point of GT;
    corrective clicks = deepest point of the largest error region; all clicks so far
    are passed to the decoder; if no errors remain, the current mask is kept).
    With one click the decoder returns 3 masks and we keep the one with the highest
    predicted IoU. With 2+ clicks we use the single-mask output (SAM convention).
  - iou_box: GT-box prompt IoU (single-mask output).
Results are written as JSONL (one row per instance) so paired bootstrap CIs can be
computed later by resampling images.

Usage:
  python -m team_sam.evaluate --split val --out results/fp32_torch_val.jsonl
  # macOS only: Core ML FP16 export
  python -m team_sam.evaluate --split val --backend coreml \
      --encoder coreml/L0_encoder_fp16.mlpackage --decoder coreml/L0_decoder_fp16.mlpackage \
      --out results/fp16_coreml_val.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from team_sam.clicks import corrective_click, first_click, mask_iou
from team_sam.data import eligible_instances, image_path, load_coco, load_splits


class TorchSegmenter:
    """Wraps EfficientViTSamPredictor. Other backends (Core ML, pruned models) implement
    the same two methods so the evaluation code is shared."""

    def __init__(self, predictor):
        self.p = predictor

    def set_image(self, rgb: np.ndarray) -> None:
        self.p.set_image(rgb)

    def predict(self, points=None, labels=None, box=None, multimask=True):
        masks, ious, _ = self.p.predict(
            point_coords=None if points is None else np.asarray(points, dtype=np.float32),
            point_labels=None if labels is None else np.asarray(labels, dtype=np.int32),
            box=None if box is None else np.asarray(box, dtype=np.float32),
            multimask_output=multimask,
        )
        return masks, ious


def click_ious(seg, gt: np.ndarray, n_clicks: int = 5) -> list[float]:
    points, labels, ious = [], [], []
    pred = None
    for k in range(n_clicks):
        if k == 0:
            x, y, lab = first_click(gt)
        else:
            c = corrective_click(pred, gt)
            if c is None:  # no errors remain: keep current mask
                ious.extend([ious[-1]] * (n_clicks - k))
                break
            x, y, lab = c
        points.append([x, y])
        labels.append(lab)
        masks, scores = seg.predict(points, labels, multimask=(k == 0))
        pred = masks[int(np.argmax(scores))]
        ious.append(mask_iou(pred, gt))
    return ious


def box_iou(seg, gt: np.ndarray, box: np.ndarray) -> float:
    masks, _ = seg.predict(box=box, multimask=False)
    return mask_iou(masks[0], gt)


def summarize(rows: list[dict]) -> dict:
    s = {k: float(np.mean([r[k] for r in rows])) for k in ("iou@1", "iou@3", "iou@5", "iou_box")}
    by_size = defaultdict(list)
    for r in rows:
        by_size[r["size"]].append(r["iou_box"])
    s["iou_box_by_size"] = {k: float(np.mean(v)) for k, v in sorted(by_size.items())}
    s["num_instances"] = len(rows)
    s["num_images"] = len({r["image_id"] for r in rows})
    return s


def run(seg, split: str, out: Path, limit: int | None = None) -> dict:
    coco = load_coco("val2017")
    ids = load_splits()[split]
    if limit:
        ids = ids[:limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    with open(out, "w") as f:
        for n, img_id in enumerate(ids, 1):
            bgr = cv2.imread(str(image_path(coco, img_id)))
            seg.set_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for inst in eligible_instances(coco, img_id):
                c = click_ious(seg, inst.mask, 5)
                row = {
                    "image_id": img_id,
                    "ann_id": inst.ann_id,
                    "size": inst.area_bucket,
                    "iou@1": c[0],
                    "iou@3": c[2],
                    "iou@5": c[4],
                    "iou_box": box_iou(seg, inst.mask, inst.box_xyxy),
                }
                rows.append(row)
                f.write(json.dumps(row) + "\n")
            if n % 50 == 0:
                print(f"[{n}/{len(ids)}] {time.time() - t0:.0f}s  mIoU@1={np.mean([r['iou@1'] for r in rows]):.4f}", flush=True)
    summary = summarize(rows)
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N images (smoke test)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--backend", choices=["torch", "coreml"], default="torch")
    ap.add_argument("--encoder", help="coreml backend: encoder .mlpackage")
    ap.add_argument("--decoder", help="coreml backend: decoder .mlpackage")
    ap.add_argument("--compute-units", default="ALL", help="coreml: ALL | CPU_ONLY | CPU_AND_GPU | CPU_AND_NE")
    args = ap.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.backend == "coreml":
        from team_sam.coreml_backend import CoreMLSegmenter

        seg = CoreMLSegmenter(args.encoder, args.decoder, args.compute_units)
    else:
        from team_sam.model import load_predictor

        seg = TorchSegmenter(load_predictor(weights=args.weights))
    print(json.dumps(run(seg, args.split, args.out, args.limit), indent=1))


if __name__ == "__main__":
    main()
