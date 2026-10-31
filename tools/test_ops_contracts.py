"""Backend contract tests: the Triton adapters either match the PyTorch reference or decline
(Unsupported -> the dispatcher falls back to `torch`); they never return a different answer
or crash on inputs the reference accepts.

  CUDA_VISIBLE_DEVICES=1 python tools/test_ops_contracts.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from clarus.ops import Ops  # noqa: E402
from clarus.ops.torch_ref import Small, TorchOps, box_geometry  # noqa: E402
from clarus.ops.triton_cuda import TritonOps  # noqa: E402

dev = torch.device("cuda")
dt = torch.float16
torch.manual_seed(0)
FAIL = []
REF = TorchOps(dt, dev)


def report(name, ok, msg=""):
    print(f"{'OK ' if ok else 'BAD'} {name:52s} {msg}")
    if not ok:
        FAIL.append(name)


def run(op, *args, **kw):
    """(output, backend that served it) through a fresh triton -> torch dispatcher."""
    ops = Ops(["triton", "torch"], dt, dev, log=lambda *a: None)
    out = getattr(ops, op)(*args, **kw)
    served = [b for (o, b) in ops.served if o == op]
    return out, served[0]


def close(got, ref, tol):
    got, ref = got.float(), ref.float()
    if not (torch.isfinite(got).all() and torch.isfinite(ref).all()):
        return False, "non-finite"
    err = (got - ref).abs().max().item()
    rel = err / (ref.abs().max().item() + 1e-6)
    return rel < tol, f"rel={rel:.2e}"


def check(name, op, args, kw=None, expect=None, tol=2e-2, ref=None):
    """Dispatch `op`, compare with the reference, and check which backend served it."""
    kw = kw or {}
    try:
        got, served = run(op, *args, **kw)
    except Exception as e:  # noqa: BLE001  (only Unsupported may leave an adapter, and it is caught)
        report(name, False, f"raised {type(e).__name__}: {e}")
        return
    want = ref if ref is not None else getattr(REF, op)(*args, **kw)
    if isinstance(got, Small):
        got, want = got.data, want.data
    ok, msg = close(got, want, tol)
    if got.dtype != want.dtype or tuple(got.shape) != tuple(want.shape):
        ok, msg = False, f"dtype/shape {got.dtype}{tuple(got.shape)} vs {want.dtype}{tuple(want.shape)}"
    if expect is not None and served != expect:
        ok, msg = False, msg + f" served by {served}, expected {expect}"
    report(name, ok, f"[{served}] {msg}")


def rnd(*s, dtype=dt, scale=1.0):
    return (torch.randn(*s, device=dev) * scale).to(dtype)


# --------------------------------------------- 1. fp32 GEMM accumulation / epilogue in the reference
def test_gemm_range():
    """|x| ~ 400, |w| ~ 200: x @ w (~5e6) overflows fp16, the scaled / fp32 result does not.
    Both backends must give the same finite answer (the reference used to round the product to fp16)."""
    n, h, w = 2, 8, 12
    for k, nout, gate in [(64, 32, False), (64, 96, False), (256, 48, False), (64, 32, True)]:
        x = (400 + 40 * torch.randn(n, h, w, k, device=dev)).to(dt)
        wt = (200 + 20 * torch.randn(k, 2 * nout if gate else nout, device=dev)).to(dt)
        b = torch.randn(2 * nout if gate else nout, device=dev)
        sc = torch.full((nout,), 1e-4, device=dev)
        check(f"linear |x|=400 |w|=200 scale=1e-4 k{k} n{nout}{' gate' if gate else ''}", "linear", (x, wt, b),
              dict(scale=sc, gate=gate, out_dtype=torch.float32), expect="triton", tol=1e-3)
    # the mul prologue: (20 * 20) @ 200
    for k in (64, 256):
        x = (20 + torch.randn(n, h, w, k, device=dev)).to(dt)
        k1, k2, oh, ow, pt, pl = box_geometry(h, w, (4, 4))
        s = Small((20 + torch.randn(n, oh, ow, k, device=dev)).to(dt), pt, pl, h, w)
        wt = (200 + 20 * torch.randn(k, 32, device=dev)).to(dt)
        check(f"linear mul prologue large k{k}", "linear", (x, wt, None),
              dict(mul=s, scale=torch.full((32,), 1e-4, device=dev), out_dtype=torch.float32), expect="triton",
              tol=1e-3)
    x = rnd(n, h, w, 64)
    lw, lb = torch.ones(64, device=dev), torch.zeros(64, device=dev)
    m = (400 + 40 * torch.randn(n, h, w, 64, device=dev)).to(dt)
    wt = (200 + 20 * torch.randn(64, 48, device=dev)).to(dt)
    check("ln_mul_linear |m|=400 |w|=200 -> fp32", "ln_mul_linear", (x, lw, lb, m, wt),
          dict(out_dtype=torch.float32), expect="triton", tol=2e-2)
    x = (400 + 40 * torch.randn(1, 16, 24, 64, device=dev)).to(dt)
    wt = (200 + 20 * torch.randn(64, 32, device=dev)).to(dt)
    f = torch.full((32, 8, 5), 1e-4, device=dev)
    check("linear_patch_filter |x @ w|=5e6, filter 1e-4", "linear_patch_filter",
          (x, wt, torch.zeros(32, device=dev), f), expect="triton", tol=2e-2)
    # the reference is also the CPU path
    cpu = TorchOps(dt, torch.device("cpu"))
    y = cpu.linear(torch.full((1, 1, 1, 64), 400.0).half(), torch.full((64, 4), 200.0).half(),
                   scale=torch.full((4,), 1e-4), out_dtype=torch.float32)
    report("cpu reference linear 400 * 200 * 64 * 1e-4 = 512", bool(torch.all(y == 512.0)), f"got {y.flatten()[0].item()}")


# ------------------------------------------------------------- 2. ln + mul falls back
def test_ln_mul():
    n, h, w, k = 2, 16, 16, 64
    x = rnd(n, h, w, k)
    k1, k2, oh, ow, pt, pl = box_geometry(h, w, (8, 8))
    s = Small(rnd(n, oh, ow, k), pt, pl, h, w)
    wt = rnd(k, 48, scale=0.1)
    for ln in (True, "rms"):
        check(f"linear ln={ln} + mul -> reference", "linear", (x, wt, torch.randn(48, device=dev)),
              dict(ln=ln, mul=s), expect="torch", tol=1e-6)
    check("linear mul only (k < 256) stays on triton", "linear", (x, wt, None), dict(mul=s), expect="triton")
    check("linear ln only stays on triton", "linear", (x, wt, None), dict(ln=True), expect="triton")


# ------------------------------------------------------- 3. patch_corr with arbitrary strides
def test_patch_corr():
    c = 32
    base = rnd(4, 16, 24, 2 * c + 16)
    cases = [
        ("base[::2] (batch stride 2HWC)", base[::2], "triton"),
        ("base[:1].expand(3) (batch stride 0)", base[:1].expand(3, -1, -1, -1), "triton"),
        ("channel slice base[..., 8:]", base[..., 8:8 + 2 * c], "triton"),
        ("H/W transposed view", base.transpose(1, 2), "triton"),
        ("contiguous", base[:, :, :, :2 * c].contiguous(), "triton"),
        ("channel stride != 1 -> reference", rnd(2, 2 * c, 16, 16).permute(0, 2, 3, 1), "torch"),
    ]
    for name, x, expect in cases:
        check(f"patch_corr {name}", "patch_corr", (x, c), expect=expect)


# ------------------------------------------------------------------- 4. selective scan
def _scan_inputs(t, hh, ww, d, r, ns):
    u = rnd(t, hh, ww, d)
    dbl = torch.randn(t, hh, ww, r + 2 * ns, device=dev)
    dtw = torch.randn(d, r, device=dev) * 0.5
    dtb = torch.randn(d, device=dev) - 2
    A = -torch.rand(d, ns, device=dev) * 8 - 0.1
    return u, dbl, dtw, dtb, A, torch.randn(d, device=dev)


def _serial_scan(u, dbl, dt_w, dt_b, A, D):
    """Step-by-step fp64 recurrence (ground truth; no closed-form decay products)."""
    t, hh, ww, d = u.shape
    L, r, ns = hh * ww, dt_w.shape[1], A.shape[1]
    u, dbl = u.double().reshape(t, L, d), dbl.double().reshape(t, L, -1)
    delta = torch.nn.functional.softplus(dbl[..., :r] @ dt_w.double().t() + dt_b.double())
    h = torch.zeros(t, d, ns, dtype=torch.float64, device=u.device)
    ys = []
    for i in range(L):
        h = torch.exp(delta[:, i, :, None] * A.double()) * h + (delta[:, i] * u[:, i])[..., None] * dbl[:, i, None, r:r + ns]
        ys.append((h * dbl[:, i, None, r + ns:r + 2 * ns]).sum(-1) + D.double() * u[:, i])
    return torch.stack(ys, 1).view(t, hh, ww, d)


def test_selective_scan():
    for ns in (4, 16):
        check(f"selective_scan d_state {ns} -> reference", "selective_scan", _scan_inputs(2, 8, 8, 32, 3, ns),
              expect="torch", tol=1e-6)
    args = _scan_inputs(2, 16, 16, 32, 3, 8)
    check("selective_scan d_state 8 on triton", "selective_scan", args, expect="triton", tol=1e-3)
    # segmented scan: segment 1 decays by exp(-110) (fp32 underflow) while the incoming state is
    # 2.5e31, so decay * state ~ 4e-17 is representable; C = 1e18 makes it visible in y
    t, L, d, r, ns = 1, 64, 16, 1, 8
    u = torch.zeros(t, 1, L, d, device=dev)
    dbl = torch.zeros(t, 1, L, r + 2 * ns, device=dev)
    dbl[..., 0] = -30.0
    dbl[0, 0, 15, 0], dbl[0, 0, 15, r:r + ns], u[0, 0, 15] = 25.0, 1e30, 1.0
    dbl[0, 0, 16:32, 0] = 110.0 / 16
    dbl[0, 0, 32:, r + ns:] = 1e18
    args = (u.to(dt), dbl, torch.ones(d, r, device=dev), torch.zeros(d, device=dev),
            -torch.ones(d, ns, device=dev), torch.zeros(d, device=dev))
    want = _serial_scan(*args)
    check("selective_scan segment decay below fp32 range", "selective_scan", args, expect="triton", tol=1e-3,
          ref=want.float())


# ------------------------------------------------- 3b. stride / dtype validation of the other adapters
def test_adapter_validation():
    n, h, w, k, c = 2, 16, 24, 64, 48
    wide = rnd(n, h, w, 2 * k)
    x = wide[..., :k].contiguous()
    wt = rnd(k, c, scale=0.1)
    b = torch.randn(c, device=dev)
    res = rnd(n, h, w, c)
    check("linear contiguous", "linear", (x, wt, b), dict(residual=res), expect="triton")
    check("linear x channel slice -> reference", "linear", (wide[..., :k], wt, b), expect="torch")
    check("linear x batch slice -> reference", "linear", (rnd(2 * n, h, w, k)[::2], wt, b), expect="torch")
    check("linear residual transposed -> reference", "linear", (x, wt, b),
          dict(residual=rnd(n, w, h, c).transpose(1, 2)), expect="torch")
    check("linear residual broadcast -> reference", "linear", (x, wt, b), dict(residual=rnd(1, 1, 1, c)),
          expect="torch")

    w4 = rnd(c, k, 2, 2, scale=0.1)
    check("down contiguous", "down", (x, w4, b), expect="triton")
    check("down x channel slice -> reference", "down", (wide[..., :k], w4, b), expect="torch")
    wu = rnd(k, 4 * 16, scale=0.1)
    check("up residual contiguous", "up", (x, wu), dict(residual=rnd(n, 2 * h, 2 * w, 16)), expect="triton")
    check("up residual channel slice -> reference", "up", (x, wu),
          dict(residual=rnd(n, 2 * h, 2 * w, 32)[..., :16]), expect="torch")

    check("boxmean x channel slice -> reference", "boxmean", (wide[..., :k], (8, 8)), expect="torch")
    check("boxmean fp32 x -> reference (output dtype)", "boxmean", (x.float(), (8, 8)), expect="torch")
    wd, bd = rnd(k, 1, 3, 3, scale=0.3), torch.randn(k, device=dev)
    check("dwconv", "dwconv", (x, wd, bd), expect="triton")
    check("dwconv bias None -> reference", "dwconv", (x, wd, None), expect="torch")
    check("dwconv fp32 x -> reference (output dtype)", "dwconv", (x.float(), wd, bd), expect="torch")

    w3 = rnd(c, k, 3, 3, scale=0.05)
    check("conv3x3 x channel slice (uniform pixel stride)", "conv3x3", (wide[..., :k], w3, b), expect="triton")
    check("conv3x3 x H/W transposed -> reference (was ValueError)", "conv3x3",
          (rnd(n, w, h, k).transpose(1, 2), w3, b), expect="torch")
    check("conv3x3 residual broadcast -> reference", "conv3x3", (x, w3, b), dict(residual=rnd(1, 1, 1, c)),
          expect="torch")
    we = rnd(3, k, 3, 3, scale=0.05)
    inp = torch.rand(n, 3, h, w, device=dev)
    check("ending", "ending", (x, we, torch.randn(3, device=dev), inp), expect="triton")
    check("ending x H/W transposed -> reference", "ending",
          (rnd(n, w, h, k).transpose(1, 2), we, torch.randn(3, device=dev), inp), expect="torch")

    lw, lb = torch.randn(k, device=dev), torch.randn(k, device=dev)
    xf = torch.randn(n, h, w, k, device=dev) * 100
    check("ln_mul_linear m channel slice", "ln_mul_linear", (xf, lw, lb, wide[..., k:], wt, b), expect="triton")
    check("ln_mul_linear m batch slice -> reference", "ln_mul_linear",
          (xf, lw, lb, rnd(2 * n, h, w, k)[::2], wt, b), expect="torch")
    check("ln_mul_linear m expanded batch -> reference", "ln_mul_linear",
          (xf, lw, lb, rnd(1, h, w, k).expand(n, -1, -1, -1), wt, b), expect="torch")
    check("ln_mul_linear residual broadcast -> reference", "ln_mul_linear", (xf, lw, lb, x, wt, b),
          dict(residual=rnd(1, 1, 1, c)), expect="torch")

    mix = torch.randn(n, k, c, device=dev) * 0.1
    check("linear_per_image x channel slice", "linear_per_image", (wide[..., k:], mix), dict(residual=res),
          expect="triton")
    check("linear_per_image residual transposed -> reference", "linear_per_image", (x, mix),
          dict(residual=rnd(n, w, h, c).transpose(1, 2)), expect="torch")

    seq = torch.randn(n, h, w, k, device=dev)
    check("seq_dwconv1d", "seq_dwconv1d", (seq, torch.randn(k, 5, device=dev), torch.randn(k, device=dev)),
          expect="triton", tol=1e-4)
    check("seq_dwconv1d bias None -> reference", "seq_dwconv1d", (seq, torch.randn(k, 5, device=dev), None),
          expect="torch")

    f = torch.randn(c, 8, 5, device=dev)
    check("linear_patch_filter", "linear_patch_filter", (x, wt, b, f), expect="triton")
    check("linear_patch_filter fp32 x -> reference (output dtype)", "linear_patch_filter", (x.float(), wt, b, f),
          expect="torch")
    check("linear_patch_filter residual broadcast -> reference", "linear_patch_filter", (x, wt, b, f),
          dict(residual=rnd(1, 1, 1, c)), expect="torch")

    idx = [(0, 0), (0, 8), (8, 0), (8, 8)]
    parts = rnd(n * len(idx), 8, 16, k)
    check("tiles_merge", "tiles_merge", (x, parts, 8, 16, idx), expect="triton")
    check("tiles_merge fp32 base -> reference (output dtype)", "tiles_merge", (x.float(), parts, 8, 16, idx),
          expect="torch")


# ------------------------------------------------------------ 6. device capability
def test_capability():
    ops = Ops(["triton", "torch"], torch.bfloat16, dev, log=lambda *a: None)
    cap = torch.cuda.get_device_capability(dev)
    report(f"bf16 on sm_{cap[0]}{cap[1]}: triton {'active' if cap >= (8, 0) else 'skipped'}",
           ("triton" in ops.by_name) == (cap >= (8, 0)), str(ops.skipped))
    with mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        ops = Ops(["triton", "torch"], torch.bfloat16, dev, log=lambda *a: None)
        report("bf16 on (mocked) sm_75: triton skipped, torch active",
               "triton" in ops.skipped and [b.name for b in ops.backends] == ["torch"], str(ops.skipped))
        ok = True
        try:
            TritonOps(torch.float16, dev)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(e)
        report("fp16 on (mocked) sm_75: triton active", ok)


if __name__ == "__main__":
    test_gemm_range()
    test_ln_mul()
    test_patch_corr()
    test_selective_scan()
    test_adapter_validation()
    test_capability()
    print("FAILED:" if FAIL else "ALL OK", FAIL)
    sys.exit(1 if FAIL else 0)
