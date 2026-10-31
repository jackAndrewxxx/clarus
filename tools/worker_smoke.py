"""Smoke test of ModelWorker without the Rust engine (dev tool).

Fills pinned slots from Python (cv2), runs batches, prints mean PSNR/SSIM and throughput.
  python tools/worker_smoke.py [N_IMAGES] [MAX_BATCH]

Self-test (CUDA, synthetic PNG pairs in /tmp/clarus_smoke, no dataset needed):
  python tools/worker_smoke.py --selftest
  * lazily created entries (no warmup) in graphs-off + torch-backend mode and in graph mode,
    with the current stream stalled and its allocator pool poisoned with NaN right before each
    new entry: outputs must match an eager reference bit for bit, PSNR must match metrics.psnr
  * metrics scratch / entry memory is released on LRU eviction (max_graphs=2, 6 shapes)
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
from clarus.metrics import psnr as host_psnr  # noqa: E402
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


# ------------------------------------------------------------------ self-test
SMOKE_DIR = "/tmp/clarus_smoke"


def _write_pairs(shapes, n_per):
    """Synthetic (blurred, sharp) PNG pairs; returns [(lq_rgb, gt_rgb), ...] as read back by cv2."""
    os.makedirs(SMOKE_DIR, exist_ok=True)
    rng = np.random.default_rng(0)
    pairs = []
    for h, w in shapes:
        for k in range(n_per):
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            # (smooth low-frequency content overflows NAFNet's fp16 activations -> keep it busy)
            base = np.stack([127 + 100 * np.sin(xx / (5 + 3 * c + k) + yy / (7 + c)) for c in range(3)], -1)
            gt = np.clip(base + rng.normal(0, 12, base.shape), 0, 255).astype(np.uint8)
            lq = cv2.GaussianBlur(gt, (7, 7), 2.0)
            out = []
            for tag, img in (("lq", lq), ("gt", gt)):
                p = os.path.join(SMOKE_DIR, f"{h}x{w}_{k}_{tag}.png")
                cv2.imwrite(p, img)
                out.append(np.ascontiguousarray(cv2.imread(p)[:, :, ::-1]))
            pairs.append((h, w, *out))
    return pairs


def _poison_and_stall(dev):
    """Leave NaN-filled free blocks in the current stream's allocator pool and queue ~0.3 s of
    work on the current stream. Entries initialized on this stream (instead of the worker
    stream) would then be read uninitialized / overwritten late by the worker stream."""
    junk = [torch.full((1 << 17,), float("nan"), device=dev) for _ in range(64)]  # 64 x 512 KiB
    junk += [torch.full((1 << 24,), float("nan"), device=dev) for _ in range(4)]  # 4 x 64 MiB
    del junk
    a = torch.randn(4096, 4096, device=dev)
    for _ in range(20):
        a = torch.mm(a, a).mul_(1e-3)


def _worker(model_cfg, rt, max_pixels, batch_sizes=(1,), slots=2):
    return create(json.dumps({"model": model_cfg, "runtime": rt,
                              "worker": {"slots": slots, "max_pixels": max_pixels,
                                         "batch_sizes": list(batch_sizes)}}))


def _run_one(w, h, wd, lq, gt, output=True):
    w._plane(0, "lq", lq.size).copy_(torch.from_numpy(lq).view(-1))
    w._plane(0, "gt", gt.size).copy_(torch.from_numpy(gt).view(-1))
    w.submit(json.dumps({"id": 0, "h": h, "w": wd, "slots": [0], "metrics": True, "output": output}))
    r = json.loads(w.wait())["results"][0]
    out = w._plane(0, "out", gt.size).numpy().reshape(gt.shape).copy() if output else None
    return r, out


def _reference(w, lq):
    """The worker pipeline's math, eagerly on the default stream after a full sync."""
    m = max(1, w.io.pad_multiple)
    h, wd, _ = lq.shape
    torch.cuda.synchronize()
    x = torch.zeros((1, 3, -(-h // m) * m, -(-wd // m) * m), device=w.dev)
    x[:, :, :h, :wd].copy_(torch.from_numpy(lq).to(w.dev).permute(2, 0, 1)).div_(255.0)
    y = w.model.forward(x)[:, :, :h, :wd]
    u8 = y.clamp(0.0, 1.0).mul(255.0).round_().permute(0, 2, 3, 1)[0].to(torch.uint8).cpu().numpy()
    return u8


def test_lazy_entries(cfg, graphs, backends):
    rt = dict(cfg["runtime"], graphs=graphs, max_graphs=8)
    if backends:
        rt["backends"] = backends
    shapes = [(64, 96), (60, 90)]  # 60x90 pads to the model multiple -> padding must be zero
    pairs = _write_pairs(shapes, 2)
    w = _worker(cfg["model"], rt, max(h * wd for h, wd in shapes))
    tag = f"graphs={graphs} backends={rt['backends']} (use_graph={w.use_graph})"
    seen, worst = set(), 0.0
    for h, wd, lq, gt in pairs:
        if (h, wd) not in seen:  # this submit creates the entry lazily
            seen.add((h, wd))
            _poison_and_stall(w.dev)
        r, out = _run_one(w, h, wd, lq, gt)
        ref = _reference(w, lq)
        mism = int((out != ref).sum())
        p_host = host_psnr(out, gt)
        worst = max(worst, abs(r["psnr"] - p_host))
        print(f"  {tag} {h}x{wd}: psnr {r['psnr']:.10f} host {p_host:.10f} ssim {r['ssim']:.8f} "
              f"nonfinite {r['nonfinite']} u8-mismatch {mism}")
        assert r["nonfinite"] == 0, "non-finite model output (uninitialized input padding?)"
        assert mism == 0, "worker output differs from the eager reference"
        assert abs(r["psnr"] - host_psnr(ref, gt)) < 1e-9, "worker PSNR != metrics.psnr of the reference"
    w.shutdown()
    print(f"  OK {tag}: max |psnr - metrics.psnr| = {worst:.2e}")


def test_eviction(graphs):
    cfg = {"name": "example_unsharp", "plugins": ["clarus.models.template"]}
    rt = {"device": "cuda:0", "graphs": graphs, "max_graphs": 2, "backends": ["triton", "torch"]}
    shapes = [(240, 320), (320, 240), (160, 480), (480, 160), (192, 400), (400, 192)]  # 76800 px each
    w = _worker(cfg, rt, 76800)
    rng = np.random.default_rng(1)
    alloc = []
    for h, wd in shapes:
        lq = rng.integers(0, 256, (h, wd, 3), dtype=np.uint8)
        gt = rng.integers(0, 256, (h, wd, 3), dtype=np.uint8)
        _run_one(w, h, wd, lq, gt, output=False)
        torch.cuda.synchronize()
        alloc.append(torch.cuda.memory_allocated())
        live = len(getattr(w.metrics, "_live", {}))
        print(f"  graphs={graphs} {h}x{wd}: entries {len(w.graphs)} live-scratch {live} "
              f"allocated {alloc[-1] / 2 ** 20:.2f} MiB")
    scratch_mib = 60 * 76800 / 2 ** 20
    growth = (max(alloc[2:]) - alloc[1]) / 2 ** 20
    w.shutdown()
    print(f"  graphs={graphs}: growth after the LRU filled = {growth:.2f} MiB "
          f"(one leaked scratch = {scratch_mib:.2f} MiB)")
    assert growth < 0.5 * scratch_mib, "device memory grows with every new shape (scratch not released)"
    print(f"  OK graphs={graphs} eviction")


def selftest():
    cfg = load_config(DEFAULT_CONFIG)
    cfg["runtime"] = dict(cfg["runtime"], device="cuda:0")
    test_lazy_entries(cfg, graphs=False, backends=["torch"])
    test_lazy_entries(cfg, graphs=True, backends=None)
    test_eviction(graphs=False)
    test_eviction(graphs=True)
    print("selftest passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
