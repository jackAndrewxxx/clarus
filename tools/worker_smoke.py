"""Smoke test of ModelWorker without the Rust engine (dev tool).

Fills pinned slots from Python (cv2), runs batches, prints mean PSNR/SSIM and throughput.
  python tools/worker_smoke.py [N_IMAGES] [MAX_BATCH]
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import torch

from _common import DEFAULT_CONFIG, GT, LQ, load_config  # noqa: E402
from clarus.worker import create  # noqa: E402


def main():
    n_img = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    mb = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    cfg = load_config(DEFAULT_CONFIG)
    w = create(json.dumps({"model": cfg["model"], "runtime": cfg["runtime"],
                           "worker": {"slots": 2 * mb, "max_pixels": 720 * 1280, "batch_sizes": sorted({1, mb})}}))
    print(w.info())
    print(w.warmup(json.dumps({"shapes": [[720, 1280]], "metrics": True})))
    names = sorted(os.listdir(LQ))[:n_img]
    t0 = time.perf_counter()
    psnr, ssim = [], []
    bid = 0
    for i in range(0, len(names), mb):
        chunk = names[i:i + mb]
        slots = []
        for j, name in enumerate(chunk):
            s = (bid % 2) * mb + j
            for plane, d in (("lq", LQ), ("gt", GT)):
                img = cv2.imread(os.path.join(d, name))[:, :, ::-1]
                w._plane(s, plane, img.size).copy_(torch.from_numpy(np.ascontiguousarray(img)).view(-1))
            slots.append(s)
        w.submit(json.dumps({"id": bid, "h": 720, "w": 1280, "slots": slots, "metrics": True}))
        bid += 1
        if int(w.inflight()) >= 2:
            r = json.loads(w.wait())
            psnr += [x["psnr"] for x in r["results"]]
            ssim += [x["ssim"] for x in r["results"]]
    while True:
        r = json.loads(w.wait())
        if r is None:
            break
        psnr += [x["psnr"] for x in r["results"]]
        ssim += [x["ssim"] for x in r["results"]]
    dt = time.perf_counter() - t0
    print(f"{len(psnr)} images: PSNR {np.mean(psnr):.4f} SSIM {np.mean(ssim):.5f} | {dt:.2f}s "
          f"{len(psnr) / dt:.2f} img/s (python-side IO, cv2)")
    print(w.shutdown())


if __name__ == "__main__":
    main()
