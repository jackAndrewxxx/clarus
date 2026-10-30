"""Per-image comparison of an engine report against tools/reference_eval.py output.

  python tools/compare_reports.py engine_report.json reference_per_image.json
"""
import json
import sys

import numpy as np

eng = {r["name"]: r for r in json.load(open(sys.argv[1]))["images"]}
ref = {r["name"]: r for r in json.load(open(sys.argv[2]))["images"]}
names = sorted(set(eng) & set(ref))
dp = np.array([eng[n]["psnr"] - ref[n]["psnr"] for n in names])
ds = np.array([eng[n]["ssim"] - ref[n]["ssim"] for n in names])
print(f"images compared      {len(names)} (engine {len(eng)}, reference {len(ref)})")
print(f"mean PSNR  engine {np.mean([eng[n]['psnr'] for n in names]):.6f}  reference {np.mean([ref[n]['psnr'] for n in names]):.6f}")
print(f"mean SSIM  engine {np.mean([eng[n]['ssim'] for n in names]):.6f}  reference {np.mean([ref[n]['ssim'] for n in names]):.6f}")
print(f"per-image dPSNR      mean {dp.mean():+.5f}  |max| {np.abs(dp).max():.5f}  p99 {np.percentile(np.abs(dp), 99):.5f} dB")
print(f"per-image dSSIM      mean {ds.mean():+.2e}  |max| {np.abs(ds).max():.2e}  p99 {np.percentile(np.abs(ds), 99):.2e}")
