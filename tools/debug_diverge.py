"""Debug helper: run two backend configurations side by side, report per-op divergence.

  python tools/debug_diverge.py [--config CFG] [--hw 256,384] [--a torch] [--b triton,torch]

Every top-level op output is snapshotted (outputs may live in buffers the model reuses). Both
runs must call the same op sequence; the first mismatch is reported and the exit code is 1.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

from _common import DEFAULT_CONFIG, LQ, build_from_config, load_rgb  # noqa: E402
from clarus.ops.torch_ref import Small  # noqa: E402


def snapshot(out):
    """A copy of an op result that later in-place writes to its buffer cannot change."""
    if isinstance(out, Small):
        out = out.data
    if isinstance(out, torch.Tensor):
        return out.detach().clone()
    if isinstance(out, (tuple, list)):
        return tuple(snapshot(o) for o in out)
    return out


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
                    log.append((name, snapshot(out)))
                return out
            return inner
        setattr(ops, name, wrap())


def _tensors(v):
    if isinstance(v, torch.Tensor):
        return [v]
    if isinstance(v, tuple):
        return [t for o in v for t in _tensors(o)]
    return []


def compare(la, lb, out=print) -> int:
    """Print per-op divergence of two recorded logs; -> 0, or 1 if the op sequences differ."""
    out(f"ops recorded: a={len(la)} b={len(lb)}")
    n = min(len(la), len(lb))
    first = next((i for i in range(n) if la[i][0] != lb[i][0]), n if len(la) != len(lb) else None)
    for i in range(n if first is None else first):
        (na, va), (_, vb) = la[i], lb[i]
        ta_list, tb_list = _tensors(va), _tensors(vb)
        if len(ta_list) != len(tb_list):
            out(f"{i:4d} {na:20s} output arity differs: {len(ta_list)} vs {len(tb_list)} tensors")
            continue
        for ta, tb in zip(ta_list, tb_list):
            ta, tb = ta.float(), tb.float()
            if ta.shape != tb.shape:
                if tb.numel() != ta.numel():
                    out(f"{i:4d} {na:20s} shape mismatch {tuple(ta.shape)} vs {tuple(tb.shape)}")
                    continue
                tb = tb.reshape(ta.shape)
            d = (ta - tb).abs().max().item()
            s = ta.abs().max().item() + 1e-6
            flag = "  <<<" if d / s > 2e-2 else ""
            out(f"{i:4d} {na:20s} {tuple(ta.shape)} rel={d / s:.2e} max_t={s:.3e} "
                f"max_tr={tb.abs().max().item():.3e} nan_t={ta.isnan().any().item()} "
                f"nan_tr={tb.isnan().any().item()}{flag}")
    if first is None:
        return 0
    a = la[first][0] if first < len(la) else "<end>"
    b = lb[first][0] if first < len(lb) else "<end>"
    out(f"OP SEQUENCE MISMATCH at #{first}: a={a} b={b} (compared the {first} ops before it)")
    return 1


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
        x = load_rgb(os.path.join(LQ, sorted(os.listdir(LQ))[0]), dev.torch_device)[:, :, :hw[0], :hw[1]].contiguous()
    la, lb = [], []
    record(a, la)
    record(b, lb)
    a.forward(x)
    b.forward(x)
    sys.exit(compare(la, lb))


if __name__ == "__main__":
    main()
