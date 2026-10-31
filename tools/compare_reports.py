"""Per-image comparison of an engine report against tools/reference_eval.py output.

  python tools/compare_reports.py engine_report.json reference_per_image.json [--subset]

Both reports must contain the same image names, each once; otherwise the missing / extra /
duplicate names are listed and the exit code is 1. --subset compares the intersection instead
(duplicates are still an error) and lists the excluded images.
"""
import argparse
import collections
import json
import sys

import numpy as np


def match_names(eng_rows, ref_rows, subset=False):
    """-> (sorted common names, list of problem lines, list of excluded-name lines)."""
    problems, excluded = [], []
    for label, rows in (("engine", eng_rows), ("reference", ref_rows)):
        dup = sorted(n for n, c in collections.Counter(r["name"] for r in rows).items() if c > 1)
        if dup:
            problems.append(f"duplicate names in {label} report: {dup}")
    eng, ref = {r["name"] for r in eng_rows}, {r["name"] for r in ref_rows}
    missing, extra = sorted(ref - eng), sorted(eng - ref)
    for what, names in (("missing from the engine report", missing), ("not in the reference report", extra)):
        if names:
            (excluded if subset else problems).append(f"{len(names)} images {what}: {names}")
    common = sorted(eng & ref)
    if not common:
        problems.append("no images in common")
    return common, problems, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine_report")
    ap.add_argument("reference_report")
    ap.add_argument("--subset", action="store_true", help="compare the intersection of the two name sets")
    args = ap.parse_args()
    eng_rows = json.load(open(args.engine_report))["images"]
    ref_rows = json.load(open(args.reference_report))["images"]
    names, problems, excluded = match_names(eng_rows, ref_rows, args.subset)
    for line in excluded:
        print(f"excluded (--subset): {line}")
    if problems:
        for line in problems:
            print(f"ERROR: {line}", file=sys.stderr)
        sys.exit(1)
    eng = {r["name"]: r for r in eng_rows}
    ref = {r["name"]: r for r in ref_rows}
    dp = np.array([eng[n]["psnr"] - ref[n]["psnr"] for n in names])
    ds = np.array([eng[n]["ssim"] - ref[n]["ssim"] for n in names])
    print(f"images compared      {len(names)} (engine {len(eng)}, reference {len(ref)})")
    print(f"mean PSNR  engine {np.mean([eng[n]['psnr'] for n in names]):.6f}  reference {np.mean([ref[n]['psnr'] for n in names]):.6f}")
    print(f"mean SSIM  engine {np.mean([eng[n]['ssim'] for n in names]):.6f}  reference {np.mean([ref[n]['ssim'] for n in names]):.6f}")
    print(f"per-image dPSNR      mean {dp.mean():+.5f}  |max| {np.abs(dp).max():.5f}  p99 {np.percentile(np.abs(dp), 99):.5f} dB")
    print(f"per-image dSSIM      mean {ds.mean():+.2e}  |max| {np.abs(ds).max():.2e}  p99 {np.percentile(np.abs(ds), 99):.2e}")


if __name__ == "__main__":
    main()
