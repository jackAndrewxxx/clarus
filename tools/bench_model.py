"""Latency benchmark of the runtime model forward (dev tool).

  python tools/bench_model.py [--config CFG] [--backends triton,torch] [--graph] [--batch 1] [--prof]
"""
from __future__ import annotations

import argparse
import time

import torch

from _common import add_config_args, build_engine_model


def main():
    ap = add_config_args(argparse.ArgumentParser())
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--hw", default="720,1280")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--prof", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    h, w = map(int, args.hw.split(","))
    net, ops, dev = build_engine_model(args)
    x = torch.rand(args.batch, 3, h, w, device=dev.torch_device)
    fn = net.forward
    for _ in range(3):
        fn(x)
    torch.cuda.synchronize()
    if args.graph:
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn(x)
        torch.cuda.current_stream().wait_stream(s)
        with torch.cuda.graph(g):
            y = fn(x)
        run = g.replay
    else:
        def run():
            fn(x)
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(args.iters):
        run()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / args.iters
    print(f"{ops.report()['backends']}/{ops.dtype} B={args.batch} {h}x{w} graph={args.graph}: "
          f"{dt * 1000:.2f} ms/batch, {dt * 1000 / args.batch:.2f} ms/img, "
          f"peak mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    if args.prof:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(x)
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40, max_name_column_width=70))


if __name__ == "__main__":
    main()
