"""Export EfficientViT-SAM-L0 to Core ML (ML Program, FP16 by default).

Produces:
  <out>/L0_encoder_<prec>.mlpackage
      input  image: 512x512 RGB image (CVPixelBuffer in the app)
      output image_embeddings: (1, 256, 64, 64)
      Normalization (ImageNet mean/std) is done INSIDE the model, so the app only
      resizes the longest side to 512 and pads bottom/right to 512x512 with the mean
      color RGB(124, 116, 104). That pad value normalizes to ~0, matching the
      PyTorch pipeline, which pads with 0 after normalization.
  <out>/L0_decoder_<prec>.mlpackage
      inputs image_embeddings (1,256,64,64), point_coords (1,N,2), point_labels (1,N)
        with 1 <= N <= 16. Coordinates are in the 1024-long-side frame (that is
        original_xy * 1024 / max(H, W)). Labels: 1 = positive, 0 = negative,
        2/3 = box top-left/bottom-right, -1 = padding. With no box, append one
        padding point (0, 0, -1), as SAM's ONNX export does.
      outputs low_res_masks (1,4,256,256) logits, iou_predictions (1,4)
        index 0 = single-mask output, indices 1-3 = multimask outputs.
  With --per-stage, it also writes one encoder sub-model per backbone stage plus
  the neck (L0_stage*_<prec>.mlpackage) for per-block timing in Xcode's
  Performance report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from team_sam.model import load_sam  # noqa: E402

MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
MAX_POINTS = 16


class EncoderWithNorm(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.register_buffer("mean", MEAN)
        self.register_buffer("std", STD)

    def forward(self, image: torch.Tensor) -> torch.Tensor:  # image in [0, 255]
        return self.encoder((image - self.mean) / self.std)


class Decoder(nn.Module):
    """Prompt encoder (points/box) + mask decoder; adapted from SAM's ONNX export."""

    def __init__(self, sam):
        super().__init__()
        self.sam = sam
        self.img_size = sam.image_size[0]  # 1024
        w = sam.prompt_encoder.no_mask_embed.weight.detach()
        self.register_buffer("dense", w.reshape(1, -1, 1, 1).repeat(1, 1, 64, 64))
        self.register_buffer("image_pe", sam.prompt_encoder.get_dense_pe().detach())

    def _embed_points(self, coords: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        pe = self.sam.prompt_encoder
        coords = (coords + 0.5) / self.img_size
        emb = pe.pe_layer._pe_encoding(coords)
        labels = labels.unsqueeze(-1).expand_as(emb)
        emb = emb * (labels != -1)
        emb = emb + pe.not_a_point_embed.weight * (labels == -1)
        for i in range(pe.num_point_embeddings):
            emb = emb + pe.point_embeddings[i].weight * (labels == i)
        return emb

    def forward(self, image_embeddings, point_coords, point_labels):
        sparse = self._embed_points(point_coords, point_labels)
        dense = self.dense  # batch is fixed to 1, so the "no mask" embedding is a constant
        return self._predict_masks(image_embeddings, self.image_pe, sparse, dense)

    def _predict_masks(self, image_embeddings, image_pe, sparse, dense):
        """SAM's MaskDecoder.predict_masks for batch size 1, without repeat_interleave
        (which Core ML cannot convert with a variable number of points)."""
        md = self.sam.mask_decoder
        output_tokens = torch.cat([md.iou_token.weight, md.mask_tokens.weight], dim=0).unsqueeze(0)
        tokens = torch.cat((output_tokens, sparse), dim=1)
        src = image_embeddings + dense
        b, c, h, w = src.shape
        hs, src = md.transformer(src, image_pe, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + md.num_mask_tokens), :]
        src = src.transpose(1, 2).reshape(b, c, h, w)
        up = md.output_upscaling(src)
        hyper_in = torch.stack(
            [md.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(md.num_mask_tokens)], dim=1
        )
        b, c, h, w = up.shape
        masks = (hyper_in @ up.reshape(b, c, h * w)).reshape(b, -1, h, w)
        return masks, md.iou_prediction_head(iou_token_out)


def _interp_matrix(n_in: int, n_out: int, mode: str, align_corners: bool) -> torch.Tensor:
    """(n_out, n_in) matrix A with interpolate_1d(x) == A @ x, taken from PyTorch itself.

    PyTorch's bicubic resize is separable, so a 2D resize is A_h @ X @ A_w^T.
    """
    eye = torch.eye(n_in).view(n_in, 1, n_in, 1)
    out = torch.nn.functional.interpolate(eye, size=(n_out, 1), mode=mode, align_corners=align_corners)
    return out.view(n_in, n_out).T.contiguous()


def patch_bicubic_upsample(module: nn.Module) -> int:
    """Core ML has no bicubic resize. Replace each fixed-size bicubic UpSampleLayer with
    two matmuls against precomputed interpolation matrices (numerically the same as
    torch's bicubic up to float rounding). Returns the number of patched layers."""
    from efficientvit.models.nn.ops import UpSampleLayer

    count = 0
    for m in module.modules():
        if isinstance(m, UpSampleLayer) and m.mode == "bicubic" and m.size is not None:
            m._mats = {}

            def fwd(x, m=m):
                h, w = x.shape[-2:]
                if (h, w) == tuple(m.size):
                    return x
                key = (int(h), int(w))
                if key not in m._mats:
                    m._mats[key] = (
                        _interp_matrix(key[0], m.size[0], "bicubic", m.align_corners),
                        _interp_matrix(key[1], m.size[1], "bicubic", m.align_corners),
                    )
                ah, aw = m._mats[key]
                return torch.matmul(torch.matmul(ah, x), aw.T)

            m.forward = fwd
            count += 1
    return count


def _precision(p: str):
    return ct.precision.FLOAT16 if p == "fp16" else ct.precision.FLOAT32


def export_encoder(sam, out: Path, prec: str, target) -> Path:
    model = EncoderWithNorm(sam.image_encoder).eval()
    ex = torch.rand(1, 3, 512, 512) * 255
    traced = torch.jit.trace(model, ex)
    ml = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, 512, 512), color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="image_embeddings")],
        convert_to="mlprogram",
        compute_precision=_precision(prec),
        minimum_deployment_target=target,
    )
    ml.short_description = "EfficientViT-SAM-L0 image encoder (stock). Input: 512x512 RGB, mean-color padded."
    path = out / f"L0_encoder_{prec}.mlpackage"
    ml.save(str(path))
    return path


def export_decoder(sam, out: Path, prec: str, target) -> Path:
    model = Decoder(sam).eval()
    ex = (
        torch.randn(1, 256, 64, 64),
        torch.randint(0, 1024, (1, 2, 2)).float(),
        torch.tensor([[1.0, -1.0]]),
    )
    traced = torch.jit.trace(model, ex)
    n = ct.RangeDim(lower_bound=1, upper_bound=MAX_POINTS, default=2)
    ml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="image_embeddings", shape=(1, 256, 64, 64)),
            ct.TensorType(name="point_coords", shape=(1, n, 2)),
            ct.TensorType(name="point_labels", shape=(1, n)),
        ],
        outputs=[ct.TensorType(name="low_res_masks"), ct.TensorType(name="iou_predictions")],
        convert_to="mlprogram",
        compute_precision=_precision(prec),
        minimum_deployment_target=target,
    )
    ml.short_description = "EfficientViT-SAM prompt encoder + mask decoder. Coordinates in 1024 frame."
    path = out / f"L0_decoder_{prec}.mlpackage"
    ml.save(str(path))
    return path


class _Stage(nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, x):
        return self.mod(x)


class _Neck(nn.Module):
    def __init__(self, neck, norm):
        super().__init__()
        self.neck, self.norm = neck, norm

    def forward(self, s2, s3, s4):
        return self.norm(self.neck({"stage2": s2, "stage3": s3, "stage4": s4})["sam_encoder"])


def export_stages(sam, out: Path, prec: str, target) -> list[Path]:
    """One sub-model per backbone stage (stem = input_stem/stage0) plus the neck."""
    bb = sam.image_encoder.backbone
    x = (torch.rand(1, 3, 512, 512) * 255 - MEAN) / STD
    parts, feats = [], {}
    with torch.no_grad():
        for sid, stage in enumerate(bb.stages):
            parts.append((f"stage{sid}", _Stage(stage).eval(), (x,)))
            x = stage(x)
            feats[f"stage{sid}"] = x
    neck_in = (feats["stage2"], feats["stage3"], feats["stage4"])
    parts.append(("neck", _Neck(sam.image_encoder.neck, sam.image_encoder.norm).eval(), neck_in))
    paths = []
    for name, mod, ex in parts:
        traced = torch.jit.trace(mod, ex)
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name=f"in{i}", shape=t.shape) for i, t in enumerate(ex)],
            convert_to="mlprogram",
            compute_precision=_precision(prec),
            minimum_deployment_target=target,
        )
        p = out / f"L0_{name}_{prec}.mlpackage"
        ml.save(str(p))
        shapes = ", ".join(str(tuple(t.shape)) for t in ex)
        print(f"  {name}: inputs {shapes} -> {p.name}")
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", type=Path, default=Path("coreml"))
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--ios", type=int, default=17, choices=[16, 17, 18])
    ap.add_argument("--per-stage", action="store_true")
    ap.add_argument("--skip-encoder", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()

    target = {16: ct.target.iOS16, 17: ct.target.iOS17, 18: ct.target.iOS18}[args.ios]
    args.out.mkdir(parents=True, exist_ok=True)
    sam = load_sam(weights=args.weights)
    x = torch.rand(1, 3, 512, 512)
    with torch.no_grad():
        ref = sam.image_encoder(x)
        n = patch_bicubic_upsample(sam.image_encoder)
        err = (sam.image_encoder(x) - ref).abs().max().item()
    print(f"patched {n} bicubic upsample layers; max |diff| vs torch bicubic = {err:.2e}")
    if not args.skip_encoder:
        print("encoder ->", export_encoder(sam, args.out, args.precision, target))
    if not args.skip_decoder:
        print("decoder ->", export_decoder(sam, args.out, args.precision, target))
    if args.per_stage:
        export_stages(sam, args.out, args.precision, target)


if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    main()
