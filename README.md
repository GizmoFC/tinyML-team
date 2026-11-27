# TEAM: Tap and Edit Anything Model

Compressing the EfficientViT-SAM-L0 image encoder (structured pruning + 8-bit weights) for
interactive photo editing on iPhone. This repo holds the evaluation, export and on-device
benchmark code. The upstream model code is pinned as a submodule in `third_party/efficientvit`.

```
team_sam/          Python library: model loading, COCO splits, click simulation, evaluation
  evaluate.py      1/3/5-click + box-prompt mIoU on the frozen splits (torch or Core ML backend)
  coreml_backend.py  Core ML backend (macOS only), mirrors the iPhone preprocessing
export/
  export_coreml.py stock L0 -> Core ML (encoder, decoder, optional per-stage sub-models)
ios/TEAMBench/     Swift sources for the on-device latency benchmark app
scripts/           setup_assets.sh, make_splits.py, make_bench_images.py
splits/            frozen train/val/test image ids (DO NOT regenerate after results exist)
results/           per-instance JSONL + summaries
```

## Setup (Mac or Linux)

```bash
git clone --recurse-submodules https://github.com/GizmoFC/tinyML-team && cd tinyML-team
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
scripts/setup_assets.sh            # L0 weights + COCO val2017 (~1 GB). Add --train for the 2,000 train images
```

Use **torch 2.7** for export: coremltools 9 fails to convert the encoder with newer torch.

## Phase 0 status

| Step | Where | Status |
|---|---|---|
| Frozen splits: 2,000 train2017 / 500 val / 1,000 test (val2017, disjoint) | any | done: `splits/splits.json` |
| Eval script: 1/3/5-click + box IoU by object size | any | done: `team_sam/evaluate.py` |
| FP32 PyTorch reference on val | any | done: 1-click **57.9**, 3-click 75.9, 5-click 79.6, box 80.0 mIoU (3,241 instances) |
| FP16 Core ML export (encoder, decoder, per-stage) | any | done: `export/export_coreml.py` |
| FP16 Core ML IoU on val (desktop check) | **Mac** | todo |
| Compute-unit sweep + op placement | **Mac + iPhone** | todo |
| Per-block-group timing to choose 2 groups to prune | **Mac + iPhone** | todo |
| First-mask / repeat-prompt p50/p95 on iPhone | **Mac + iPhone** | todo |

### 1. Export (about 1 min on CPU)

```bash
python export/export_coreml.py --out coreml --per-stage
```

This writes `coreml/L0_encoder_fp16.mlpackage` (59 MB), `L0_decoder_fp16.mlpackage` (12 MB) and
`L0_stage{0..4}_fp16.mlpackage` + `L0_neck_fp16.mlpackage`.

Export notes:
- Core ML has no bicubic resize, so the neck's bicubic upsamples (32->64, 16->64) are
  replaced by matmuls with interpolation matrices taken from PyTorch itself. The max
  difference from torch bicubic is printed at export (8e-7 on random input, float rounding only).
- ImageNet normalization is inside the encoder. The app pads with RGB(124,116,104), which
  normalizes to about 0. Checked against the PyTorch pipeline: input max diff 0.008, embedding
  relative diff 0.2–0.4%.
- The decoder is rewritten for batch size 1 without `repeat_interleave`, which Core ML cannot
  convert. It matches the PyTorch decoder exactly (max diff 0.0). It accepts 1–16 points; the
  box is points with labels 2/3; append (0,0,-1) when there is no box.

### 2. Desktop Core ML IoU check (Mac)

```bash
python -m team_sam.evaluate --split val --backend coreml \
  --encoder coreml/L0_encoder_fp16.mlpackage --decoder coreml/L0_decoder_fp16.mlpackage \
  --compute-units ALL --out results/fp16_coreml_val.jsonl
```

Compare with `results/fp32_torch_val.summary.json`. Report the mean IoU change.

### 3. Compute units + per-block timing (Mac + iPhone, no code needed)

Open each `.mlpackage` in Xcode, go to **Performance**, click **+**, choose the connected
iPhone, and run it once per compute unit (CPU only / CPU+GPU / All). Record the median
prediction time and load time, and note which ops fall back to CPU in the op-placement view.
Do this for `L0_encoder_fp16`, `L0_decoder_fp16` and each `L0_stage*` / `L0_neck` sub-model.
The two slowest stage groups are the pruning targets. Size hint: stage4 alone holds 45 MB of
the 59 MB encoder.

### 4. First-mask latency on iPhone (Xcode app)

1. In Xcode, create a new iOS App project named `TEAMBench` (SwiftUI, iOS 17+).
2. Delete the generated `ContentView.swift` and `TEAMBenchApp.swift`, then drag in everything in
   `ios/TEAMBench/`, including the `BenchImages` folder ("Copy items" + add to target).
3. Drag in `coreml/L0_encoder_fp16.mlpackage` and `coreml/L0_decoder_fp16.mlpackage`. Xcode
   compiles them to `.mlmodelc`.
4. Build in **Release** on the device, tap **Run benchmark**, then **Share CSV**. The default is
   5 warm-ups and 30 trials over 20 images per compute-unit setting. The CSV header records
   the device, iOS version, thermal state and settings.
5. For peak memory, use Instruments (Allocations / VM Tracker) on a fresh launch through the
   first mask.

Pick the compute-unit setting with the lowest **p95 first_mask** for all later comparisons.

## Results so far

Stock L0, FP32 PyTorch (CPU), val split (500 images, 3,241 instances), mIoU %:

| 1 click | 3 clicks | 5 clicks | box | box small | box medium | box large |
|---|---|---|---|---|---|---|
| 57.9 | 75.9 | 79.6 | 80.0 | 76.6 | 81.4 | 83.1 |

Per-instance rows: `results/fp32_torch_val.jsonl`. The full run takes about 25 minutes on 4 CPU cores.

## Evaluation protocol (frozen)

- Instances: non-crowd, mask ≥ 100 px in the original image.
- Click 1: the point inside the GT farthest from its boundary (image border counts as
  boundary); multimask output, keep the mask with the highest predicted IoU.
- Clicks 2+: the deepest point of the largest error region (positive if missed foreground,
  negative if false positive); all clicks so far are passed; single-mask output; if no
  errors remain, the mask is kept.
- Box: the GT box, single-mask output; reported overall and by COCO size (small < 32², medium < 96²).
- Per-instance rows are saved so image-level paired bootstrap CIs can be computed later.
