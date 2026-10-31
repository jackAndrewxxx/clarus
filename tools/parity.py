"""Numerical parity: engine model (any backend mix) vs the original network, called like its
official test script (padding / cropping, see _common.build_reference).

  python tools/parity.py [--config CFG] [--backends torch --dtype fp32] [--images N] [--odd]
                         [--crop H,W ...]
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from _common import GT, LQ, add_config_args, build_engine_model, build_reference, load_rgb
from clarus.metrics import psnr  # the worker's PSNR (basicsr peak rule: 1 if pred max <= 1, else 255)


def sample_names(names, k):
    """`k` names spread evenly over the sorted list `names` (all of them must exist)."""
    if k < 1:
        raise SystemExit("--images must be >= 1")
    if k > len(names):
        raise SystemExit(f"--images {k} requested but only {len(names)} images available")
    return sorted(names)[:: len(names) // k][:k]


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--odd", action="store_true", help="also test a non-multiple-of-16 crop (251x333)")
    ap.add_argument("--crop", action="append", default=[], metavar="H,W",
                    help="also test the top-left HxW crop of each image (repeatable)")
    args = ap.parse_args()
    if not LQ or not GT:
        raise SystemExit("set CLARUS_LQ and CLARUS_GT")
    model, ops, dev = build_engine_model(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = build_reference(args.config, dev.torch_device)
    print(ref.describe(), flush=True)
    d = dev.torch_device
    windows = [None] + ([(37, 101, 251, 333)] if args.odd else [])
    windows += [(0, 0, *map(int, c.split(","))) for c in args.crop]
    dps = []
    for name in sample_names(os.listdir(LQ), args.images):
        if not os.path.exists(os.path.join(GT, name)):
            raise SystemExit(f"{name}: no ground truth in {GT}")
        x = load_rgb(os.path.join(LQ, name), d)
        gt = load_rgb(os.path.join(GT, name), d)
        for win in windows:
            inp, g = x, gt
            if win is not None:
                y0, x0, h, w = win
                if y0 + h > x.shape[2] or x0 + w > x.shape[3]:
                    raise SystemExit(f"crop {h}x{w} at ({y0}, {x0}) exceeds {name} {tuple(x.shape[2:])}")
                inp = x[:, :, y0:y0 + h, x0:x0 + w].contiguous()
                g = gt[:, :, y0:y0 + h, x0:x0 + w]
            with torch.no_grad():
                y_ref = ref(inp)
                dev.synchronize()
                t = time.perf_counter()
                y = model.forward(inp)
                dev.synchronize()
                dt = time.perf_counter() - t
            diff = (y.float() - y_ref).abs()
            u8 = lambda t: (t.clamp(0, 1) * 255).round()  # noqa: E731  (tensor2img quantization)
            mism = (u8(y) != u8(y_ref)).float().mean()
            pr, pt = psnr(u8(y_ref), u8(g)), psnr(u8(y), u8(g))
            dps.append(pt - pr)
            print(f"{name} {tuple(inp.shape[2:])}: max|d|={diff.max():.3e} mean|d|={diff.mean():.3e} "
                  f"u8-mismatch={mism:.2e} t={dt * 1000:.1f}ms psnr_ref={pr:.4f} psnr_rt={pt:.4f} "
                  f"dPSNR={pt - pr:+.5f}", flush=True)
    print(f"mean dPSNR {sum(dps) / len(dps):+.5f} dB, max |dPSNR| {max(abs(v) for v in dps):.5f} dB "
          f"over {len(dps)} inputs ({len(dps) // len(windows)} images x {len(windows)} windows)")
    print("op dispatch:", ops.report()["ops"])


if __name__ == "__main__":
    main()
