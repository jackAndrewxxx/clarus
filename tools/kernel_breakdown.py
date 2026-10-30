"""Per-launch timing of the runtime's kernels at a given resolution (dev tool).

Wraps the kernel entry points, records every call made by one forward pass,
then re-times each call in isolation with triton.testing.do_bench.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import torch
from triton.testing import do_bench

from _common import add_config_args, build_engine_model  # noqa: E402
from clarus.kernels import conv3x3 as kc, gemm as kg, spatial as ks  # noqa: E402
from clarus.ops import triton_cuda  # noqa: E402

calls = []   # (label, ms, args-descriptor) -- timed inline, so no tensor outlives its call


def wrap(mod, name, label):
    fn = getattr(mod, name)

    def inner(*a, **k):
        t = do_bench(lambda: fn(*a, **k), warmup=3, rep=20)
        meta = {f: k.get(f) for f in ("m", "n", "k", "pro", "a_mode", "c_mode", "gate", "act", "epi")}
        meta["shape"] = tuple(a[0].shape)
        meta["dtype"] = a[0].dtype
        calls.append((label, t, meta, sum(nbytes(v) for v in a) + sum(nbytes(v) for v in k.values())))
        return fn(*a, **k)
    setattr(mod, name, inner)


def nbytes(t):
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--hw", default="720,1280")
    args = ap.parse_args()
    hw = tuple(map(int, args.hw.split(",")))
    net, _, dev = build_engine_model(args)
    x = torch.rand(1, 3, *hw, device=dev.torch_device)
    net.forward(x)
    from clarus.kernels import attention as ka, freq as kf, ssm as kss
    for mod, names in ((kg, ["fused_gemm", "row_stats"]), (ks, ["dwconv", "boxmean", "ln_rows", "mul_small",
                                                                 "resize_bilinear"]),
                       (kc, ["conv3x3", "intro3x3"]), (ka, ["gram"]),
                       (kf, ["ln_gemm_patch_filter", "patch_corr"]),
                       (kss, ["selective_scan", "seq_dwconv1d", "crops_gather", "crops_merge"])):
        for n in names:
            wrap(mod, n, n)
    triton_cuda.kg, triton_cuda.ks, triton_cuda.kc = kg, ks, kc
    triton_cuda.ka, triton_cuda.kf, triton_cuda.kss = ka, kf, kss
    net.forward(x)
    torch.cuda.synchronize()
    agg = defaultdict(float)
    by_m = defaultdict(lambda: [0.0, 0, 0.0])
    rows = []
    total = 0.0
    for label, t, k, io in calls:
        total += t
        desc = label
        if label == "fused_gemm":
            m, n, kk = k["m"], k["n"], k["k"]
            flags = ",".join(f"{f}={k[f]}" for f in ("pro", "a_mode", "c_mode", "gate", "act", "epi") if k.get(f))
            flops = 2 * m * n * kk * (2 if k.get("gate") else 1)
            desc = f"gemm M={m} N={n} K={kk} {flags} {k['dtype']}"
            rows.append((t, desc, flops / t / 1e9, io / t / 1e6))
            g = by_m[(m, n, kk, flags, str(k['dtype']))]
            g[0] += t
            g[1] += 1
            g[2] = flops / t / 1e9
        else:
            desc = f"{label} {k['shape']}"
            rows.append((t, desc, 0, io / t / 1e6))
        agg[label] += t
    rows.sort(reverse=True)
    for t, d, tf, gbs in rows[:40]:
        print(f"{t * 1000:8.1f}us  {tf:6.1f}TF  {gbs:6.0f}GB/s  {d}")
    print("--- gemm groups (total ms, count, TF)")
    for key, (t, c, tf) in sorted(by_m.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"{t:7.2f} x{c:3d} {tf:5.1f}TF {key}")
    print("--- per kernel type (ms)")
    for k_, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"{k_:16s} {v:7.2f}")
    print(f"total {total:.2f} ms over {len(calls)} launches")


if __name__ == "__main__":
    main()
