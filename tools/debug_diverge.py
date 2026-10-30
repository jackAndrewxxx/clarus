"""Debug helper: run two backend configurations side by side, report per-op divergence.

  python tools/debug_diverge.py [--config CFG] [--hw 256,384] [--a torch] [--b triton,torch]
"""
from __future__ import annotations

import argparse

import torch

from _common import DEFAULT_CONFIG, LQ, build_from_config, load_rgb  # noqa: E402
from clarus.ops.torch_ref import Small  # noqa: E402


def record(net, log):
    ops = net.ops
    depth = [0]
    cls_attrs = ("report", "dtype", "device", "backends", "skipped", "by_name", "overrides", "served")
    names = sorted({n for b in ops.backends for n in dir(b) if not n.startswith("_") and n not in
                    ("cached", "expand", "name", "dtype", "device") and callable(getattr(b, n))})
    for name in names:
        if name in cls_attrs:
            continue
        fn = getattr(ops, name)

        def wrap(fn=fn, name=name):
            def inner(*a, **k):
                depth[0] += 1
                try:
                    out = fn(*a, **k)
                finally:
                    depth[0] -= 1
                if depth[0] == 0:
                    log.append((name, out.data if isinstance(out, Small) else out))
                return out
            return inner
        setattr(ops, name, wrap())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--hw", default="256,384")
    ap.add_argument("--a", default="torch")
    ap.add_argument("--b", default="triton,torch")
    ap.add_argument("--noise", action="store_true",
                    help="uniform noise input instead of a crop of the first CLARUS_LQ image (out of distribution: expect fp16 overflow "
                         "unless the residual stream is fp32, i.e. model.options.residual_fp32_from_level=0)")
    args = ap.parse_args()
    hw = tuple(map(int, args.hw.split(",")))
    torch.manual_seed(0)
    a, _, dev = build_from_config(args.config, backends=args.a.split(","))
    b, _, _ = build_from_config(args.config, backends=args.b.split(","))
    if args.noise:
        x = torch.rand(1, 3, *hw, device=dev.torch_device)
    else:
        import os
        x = load_rgb(os.path.join(LQ, sorted(os.listdir(LQ))[0]), dev.torch_device)[:, :, :hw[0], :hw[1]].contiguous()
    la, lb = [], []
    record(a, la)
    record(b, lb)
    a.forward(x)
    b.forward(x)
    print(len(la), len(lb))
    for i, ((na, ta), (nb, tb)) in enumerate(zip(la, lb)):
        ta, tb = ta.float(), tb.float()
        if ta.shape != tb.shape:
            tb = tb.reshape(ta.shape) if tb.numel() == ta.numel() else tb
        d = (ta - tb).abs().max().item()
        s = ta.abs().max().item() + 1e-6
        flag = "  <<<" if d / s > 2e-2 else ""
        print(f"{i:4d} {na:20s} {tuple(ta.shape)} rel={d / s:.2e} max_t={s:.3e} "
              f"max_tr={tb.abs().max().item():.3e} nan_t={ta.isnan().any().item()} nan_tr={tb.isnan().any().item()}{flag}")


if __name__ == "__main__":
    main()
