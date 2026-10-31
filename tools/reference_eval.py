"""Per-image reference numbers from the original project code path (dev / audit tool).

Mirrors basicsr/test.py + ImageRestorationModel.dist_validation for one GPU:
cv2 decode -> float32/255 -> the config's original network called like its official test script
(padding / cropping, see _common.build_reference; fp32, PyTorch default TF32 settings,
cudnn.benchmark like the baseline) -> tensor2img -> calculate_psnr / calculate_ssim, but keeps
every per-image value so the engine can be compared image by image. The reference repo and its
git revision are recorded in OUT.json ("reference").

  python tools/reference_eval.py OUT.json [--limit N] [--config CFG]
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEFAULT_CONFIG, GT, LQ, ROOT, build_reference  # noqa: E402
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402
from basicsr.utils import img2tensor, tensor2img  # noqa: E402


def main():
    out_path = sys.argv[1]
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    torch.backends.cudnn.benchmark = True
    cfg = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else DEFAULT_CONFIG
    net = build_reference(cfg)
    print(net.describe(), flush=True)
    names = sorted(os.listdir(LQ))[:limit]
    rows = []
    t0 = time.perf_counter()
    for i, name in enumerate(names):
        lq = cv2.imread(os.path.join(LQ, name)).astype(np.float32) / 255.0
        gt = cv2.imread(os.path.join(GT, name)).astype(np.float32) / 255.0
        lq_t, gt_t = img2tensor([lq, gt], bgr2rgb=True, float32=True)
        with torch.no_grad():
            pred = net(lq_t[None].cuda()).detach().cpu()
        sr = tensor2img([pred], rgb2bgr=True)
        gt_img = tensor2img([gt_t[None]], rgb2bgr=True)
        rows.append({"name": name, "psnr": float(calculate_psnr(sr, gt_img, 0)),
                     "ssim": float(calculate_ssim(sr, gt_img, 0))})
        if i % 100 == 0:
            print(f"{i}/{len(names)} {time.perf_counter() - t0:.0f}s", flush=True)
    res = {"reference": net.info, "psnr": float(np.mean([r["psnr"] for r in rows])), "ssim": float(np.mean([r["ssim"] for r in rows])),
           "seconds": time.perf_counter() - t0, "images": rows}
    json.dump(res, open(out_path, "w"), indent=1)
    print(f"reference PSNR {res['psnr']:.6f} SSIM {res['ssim']:.6f} ({len(rows)} images)")


if __name__ == "__main__":
    main()
