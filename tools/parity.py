"""Numerical parity: engine model (any backend mix) vs the original basicsr network.

  python tools/parity.py [--config CFG] [--backends torch --dtype fp32] [--images N] [--odd]
"""
from __future__ import annotations

import argparse
import os
import time

import torch

from _common import GT, LQ, add_config_args, build_engine_model, build_reference, load_rgb


def psnr_u8(a, b):
    a = (a.clamp(0, 1) * 255).round()
    b = (b.clamp(0, 1) * 255).round()
    mse = (a.double() - b.double()).square().mean()
    return float(20 * torch.log10(255.0 / mse.sqrt()))


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--odd", action="store_true", help="also test a non-multiple-of-16 crop")
    args = ap.parse_args()
    model, ops, dev = build_engine_model(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ref = build_reference(args.config, dev.torch_device)
    d = dev.torch_device
    dps = []
    names = sorted(os.listdir(LQ))[:: max(1, 1111 // max(args.images, 1))][: args.images]
    for name in names:
        x = load_rgb(os.path.join(LQ, name), d)
        gt = load_rgb(os.path.join(GT, name), d)
        crops = [x]
        if args.odd:
            crops.append(x[:, :, 37:37 + 251, 101:101 + 333].contiguous())
        for inp in crops:
            with torch.no_grad():
                y_ref = ref(inp)
                dev.synchronize()
                t = time.perf_counter()
                y = model.forward(inp)
                dev.synchronize()
                dt = time.perf_counter() - t
            diff = (y.float() - y_ref).abs()
            mism = ((y.clamp(0, 1) * 255).round() != (y_ref.clamp(0, 1) * 255).round()).float().mean()
            msg = (f"{name} {tuple(inp.shape[2:])}: max|d|={diff.max():.3e} mean|d|={diff.mean():.3e} "
                   f"u8-mismatch={mism:.2e} t={dt * 1000:.1f}ms")
            if inp is x:
                pr, pt = psnr_u8(y_ref, gt), psnr_u8(y, gt)
                dps.append(pt - pr)
                msg += f" psnr_ref={pr:.4f} psnr_rt={pt:.4f}"
            print(msg, flush=True)
    print(f"mean dPSNR {sum(dps) / len(dps):+.5f} dB, max |dPSNR| {max(abs(v) for v in dps):.5f} dB "
          f"over {len(dps)} images")
    print("op dispatch:", ops.report()["ops"])


if __name__ == "__main__":
    main()
