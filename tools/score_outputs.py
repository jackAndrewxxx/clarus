"""Score a directory of restored PNGs against GT with basicsr's calculate_psnr / calculate_ssim,
and optionally compare it pixel by pixel with another output directory.

  python tools/score_outputs.py OUT_DIR GT_DIR [--other OTHER_DIR] [--report r.json] [--suffix _out] [--subset]

Used to put official inference scripts (which only write images) on the same metric as
`clarus eval`. Metrics run on the GPU through basicsr's 3-D SSIM, like the training code.
Every GT image must have exactly one output (and one image in OTHER_DIR) and vice versa;
otherwise the missing / extra / duplicate names are listed and the exit code is 1. --subset
scores the images present everywhere instead and lists the excluded ones.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common  # noqa: E402,F401  (puts the basicsr checkout on sys.path)

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def match_files(out_dir, gt_dir, suffix="", other=None, subset=False):
    """-> ([(gt_name, out_name)], problems, excluded). OUT_DIR holds <stem><suffix>.png per GT
    image, OTHER_DIR the GT file names."""
    problems, excluded = [], []
    gt_names = sorted(n for n in os.listdir(gt_dir) if n.lower().endswith(IMAGE_EXTS))
    stems = collections.Counter(os.path.splitext(n)[0] for n in gt_names)
    dup = sorted(s for s, c in stems.items() if c > 1)
    if dup:
        problems.append(f"GT names that map to the same output file: {dup}")
        return [], problems, excluded
    expected = {os.path.splitext(n)[0] + suffix + ".png": n for n in gt_names}
    outs = {n for n in os.listdir(out_dir) if n.lower().endswith(".png")}
    keep = set(gt_names)
    checks = [(f"GT images without output in {out_dir}", sorted(expected[o] for o in set(expected) - outs)),
              (f"outputs in {out_dir} without GT", sorted(outs - set(expected)))]
    keep -= {expected[o] for o in set(expected) - outs}
    if other:
        others = {n for n in os.listdir(other) if n.lower().endswith(IMAGE_EXTS)}
        checks += [(f"GT images missing from {other}", sorted(set(gt_names) - others)),
                   (f"images in {other} without GT", sorted(others - set(gt_names)))]
        keep &= others
    for what, names in checks:
        if names:
            (excluded if subset else problems).append(f"{len(names)} {what}: {names}")
    if not keep:
        problems.append("no images to score")
    inv = {g: o for o, g in expected.items()}
    return [(g, inv[g]) for g in sorted(keep)], problems, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("gt_dir")
    ap.add_argument("--other", default=None, help="second output dir to compare pixels with")
    ap.add_argument("--suffix", default="", help="suffix of output file names, e.g. _out")
    ap.add_argument("--report", default=None)
    ap.add_argument("--subset", action="store_true", help="score only the images present in every directory")
    args = ap.parse_args()
    pairs, problems, excluded = match_files(args.out_dir, args.gt_dir, args.suffix, args.other, args.subset)
    for line in excluded:
        print(f"excluded (--subset): {line}", file=sys.stderr)
    if problems:
        for line in problems:
            print(f"ERROR: {line}", file=sys.stderr)
        sys.exit(1)
    from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim

    rows = []
    for name, out_name in pairs:
        out = cv2.imread(os.path.join(args.out_dir, out_name))
        gt = cv2.imread(os.path.join(args.gt_dir, name))
        o2 = cv2.imread(os.path.join(args.other, name)) if args.other else None
        for path, img in ((out_name, out), (name, gt)) + (((name + " (other)", o2),) if args.other else ()):
            if img is None:
                sys.exit(f"ERROR: cannot read {path}")
        if out.shape != gt.shape or (o2 is not None and o2.shape != out.shape):
            sys.exit(f"ERROR: {name}: shape mismatch out {out.shape} gt {gt.shape}"
                     + (f" other {o2.shape}" if o2 is not None else ""))
        row = {"name": name, "psnr": float(calculate_psnr(out, gt, 0)), "ssim": float(calculate_ssim(out, gt, 0))}
        if o2 is not None:
            d = np.abs(out.astype(int) - o2.astype(int))
            row["pix_mismatch"] = float((d > 0).mean())
            row["max_abs_diff"] = int(d.max())
        rows.append(row)
    res = {"images": len(rows), "psnr": float(np.mean([r["psnr"] for r in rows])),
           "ssim": float(np.mean([r["ssim"] for r in rows]))}
    if excluded:
        res["excluded"] = excluded
    if args.other:
        res["mean_pix_mismatch"] = float(np.mean([r["pix_mismatch"] for r in rows]))
        res["max_abs_diff"] = max(r["max_abs_diff"] for r in rows)
    print(json.dumps(res, indent=1))
    if args.report:
        json.dump({**res, "rows": rows}, open(args.report, "w"), indent=1)


if __name__ == "__main__":
    main()
