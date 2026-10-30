"""Profile the original basicsr network forward at GoPro resolution (dev tool).

  python tools/profile_reference.py [--config CFG] [--prof]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DEFAULT_CONFIG, build_reference as _build  # noqa: E402


def build_reference():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--prof", action="store_true")
    return _build(ap.parse_args().config)


def main():
    torch.backends.cudnn.benchmark = True
    net = build_reference()
    x = torch.rand(1, 3, 720, 1280, device="cuda")
    with torch.no_grad():
        for _ in range(3):
            net(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        n = 10
        for _ in range(n):
            net(x)
        torch.cuda.synchronize()
        print(f"fp32 forward: {(time.perf_counter() - t) / n * 1000:.1f} ms")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                net(x)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                net(x)
            torch.cuda.synchronize()
            print(f"bf16 autocast forward: {(time.perf_counter() - t) / n * 1000:.1f} ms")
        if "--prof" in sys.argv:
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
                net(x)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))


if __name__ == "__main__":
    main()
