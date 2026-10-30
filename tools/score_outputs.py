"""Score a directory of restored PNGs against GT with basicsr's calculate_psnr / calculate_ssim,
and optionally compare it pixel by pixel with another output directory.

  python tools/score_outputs.py OUT_DIR GT_DIR [--other OTHER_DIR] [--report r.json] [--suffix _out]

Used to put official inference scripts (which only write images) on the same metric as
`clarus eval`. Metrics run on the GPU through basicsr's 3-D SSIM, like the training code.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402,F401  (puts the basicsr checkout on sys.path)
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("gt_dir")
    ap.add_argument("--other", default=None, help="second output dir to compare pixels with")
    ap.add_argument("--suffix", default="", help="suffix of output file names, e.g. _out")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    rows = []
    for name in sorted(os.listdir(args.gt_dir)):
        stem, ext = os.path.splitext(name)
        out_path = os.path.join(args.out_dir, stem + args.suffix + ".png")
        if not os.path.exists(out_path):
            continue
        out = cv2.imread(out_path)
        gt = cv2.imread(os.path.join(args.gt_dir, name))
        row = {"name": name, "psnr": float(calculate_psnr(out, gt, 0)), "ssim": float(calculate_ssim(out, gt, 0))}
        if args.other:
            o2 = cv2.imread(os.path.join(args.other, name))
            if o2 is not None and o2.shape == out.shape:
                d = np.abs(out.astype(int) - o2.astype(int))
                row["pix_mismatch"] = float((d > 0).mean())
                row["max_abs_diff"] = int(d.max())
        rows.append(row)
    res = {"images": len(rows), "psnr": float(np.mean([r["psnr"] for r in rows])),
           "ssim": float(np.mean([r["ssim"] for r in rows]))}
    if args.other:
        mm = [r["pix_mismatch"] for r in rows if "pix_mismatch" in r]
        res["mean_pix_mismatch"] = float(np.mean(mm)) if mm else None
        res["max_abs_diff"] = max((r["max_abs_diff"] for r in rows if "max_abs_diff" in r), default=None)
    print(json.dumps(res, indent=1))
    if args.report:
        json.dump({**res, "rows": rows}, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
