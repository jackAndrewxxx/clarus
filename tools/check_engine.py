"""End-to-end check of the Rust engine against the original basicsr model + metrics.

Builds a small mixed dataset (several shapes incl. non-multiple-of-16, a grayscale and an
RGBA PNG), runs `clarus eval --save`, then recomputes every image with the
reference fp32 network and basicsr's calculate_psnr / calculate_ssim.

  python tools/check_engine.py [--device 0]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEFAULT_CONFIG, GT, LQ, ROOT, build_reference  # noqa: E402
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402

BIN = os.path.join(ROOT, "engine/target/release/clarus")


def make_dataset(dst):
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(f"{dst}/lq")
    os.makedirs(f"{dst}/gt")
    names = sorted(os.listdir(LQ))
    crops = [None, None, (37, 101, 251, 333), (300, 500, 251, 333), (0, 0, 480, 640), (100, 200, 480, 640),
             (5, 7, 129, 1000), "gray", "rgba"]
    for i, c in enumerate(crops):
        name = names[i * 97]
        a, b = cv2.imread(os.path.join(LQ, name)), cv2.imread(os.path.join(GT, name))
        out = f"img{i:02d}.png"
        if isinstance(c, tuple):
            y, x, h, w = c
            a, b = a[y:y + h, x:x + w], b[y:y + h, x:x + w]
        if c == "gray":
            a = cv2.cvtColor(a[:256, :384], cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(b[:256, :384], cv2.COLOR_BGR2GRAY)
        if c == "rgba":
            a = np.concatenate([a[:256, :384], np.full((256, 384, 1), 77, np.uint8)], 2)
            b = b[:256, :384]
        cv2.imwrite(f"{dst}/lq/{out}", a)
        cv2.imwrite(f"{dst}/gt/{out}", b)
    return sorted(os.listdir(f"{dst}/lq"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="0")
    args = ap.parse_args()
    dst = "/tmp/clarus_check"
    names = make_dataset(dst)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device)
    cmd = [BIN, "eval", "--lq", f"{dst}/lq", "--gt", f"{dst}/gt", "--save", f"{dst}/out",
           "--report", f"{dst}/report.json", "--max-batch", "2"]
    subprocess.run(cmd, check=True, env=env)
    rep = {r["name"]: r for r in json.load(open(f"{dst}/report.json"))["images"]}

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    ref = build_reference(DEFAULT_CONFIG)
    worst = {"psnr": 0.0, "ssim": 0.0, "pix": 0.0}
    for name in names:
        lq = cv2.imread(f"{dst}/lq/{name}", cv2.IMREAD_COLOR)
        gt = cv2.imread(f"{dst}/gt/{name}", cv2.IMREAD_COLOR)
        x = torch.from_numpy(lq[:, :, ::-1].astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
        with torch.no_grad():
            y = ref(x)[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
        y_bgr = np.ascontiguousarray(y[:, :, ::-1])
        p_ref, s_ref = calculate_psnr(y_bgr, gt, 0), calculate_ssim(y_bgr, gt, 0)
        saved = cv2.imread(f"{dst}/out/{name}", cv2.IMREAD_COLOR)
        pix = float((saved.astype(int) != y_bgr.astype(int)).mean())
        r = rep[name]
        dp, ds = abs(r["psnr"] - p_ref), abs(r["ssim"] - s_ref)
        worst = {"psnr": max(worst["psnr"], dp), "ssim": max(worst["ssim"], ds), "pix": max(worst["pix"], pix)}
        print(f"{name} {lq.shape[1]}x{lq.shape[0]}: engine psnr {r['psnr']:.4f} ssim {r['ssim']:.5f} | "
              f"reference {p_ref:.4f} {s_ref:.5f} | saved-vs-ref u8 mismatch {pix:.2e}")
    print(f"max |dPSNR| {worst['psnr']:.4f} dB, max |dSSIM| {worst['ssim']:.2e}, max pixel mismatch {worst['pix']:.2e}")
    ok = worst["psnr"] < 0.01 and worst["ssim"] < 2e-4 and worst["pix"] < 0.05
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
