"""Kernel contract tests: scale handling, numeric range, box-mean precision, in/out aliasing
under cold tuning, and the tuning cache (paths, per-device state, concurrent saves).

  CUDA_VISIBLE_DEVICES=1 CLARUS_TUNE_CACHE=/tmp/clarus_tune_kernels.json python tools/test_kernel_contracts.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from clarus.kernels import conv3x3 as kc  # noqa: E402
from clarus.kernels import gemm as kg  # noqa: E402
from clarus.kernels import spatial as ks  # noqa: E402
from clarus.kernels import tuning as kt  # noqa: E402

dev = "cuda"
f16, f32, f64 = torch.float16, torch.float32, torch.float64
torch.manual_seed(0)
FAIL = []


def check(name, ok, msg=""):
    print(f"{'OK ' if ok else 'BAD'} {name:52s} {msg}")
    if not ok:
        FAIL.append(name)


def check_rows(name, got, ref, tol=3e-3):
    """Max over rows (last dim = channels) of |got - ref| / max|ref row|: per-row relative error,
    so a tiny-magnitude row cannot hide behind a large one."""
    got, ref = got.double().reshape(-1, got.shape[-1]), ref.double().reshape(-1, ref.shape[-1])
    den = ref.abs().amax(1).clamp_min(1e-300)
    err = ((got - ref).abs().amax(1) / den).max().item()
    check(name, err < tol and torch.isfinite(got).all().item(), f"row-rel={err:.2e}")


class fresh_tuner:
    """Swap every kernel module's TUNER for an empty one (cold tuning), file in a temp dir."""

    def __enter__(self):
        self.dir = tempfile.TemporaryDirectory()
        self.t = kt.Tuner(os.path.join(self.dir.name, "tune.json"))
        self.old = kg.TUNER, kc.TUNER, ks.TUNER
        kg.TUNER = kc.TUNER = ks.TUNER = self.t
        return self.t

    def __exit__(self, *a):
        kg.TUNER, kc.TUNER, ks.TUNER = self.old
        self.dir.cleanup()


def mags(shape, mag):
    return torch.randn(*shape, device=dev, dtype=f64) * mag


# ----------------------------------------------------------------- 1. scale + gate
def test_gate_scale():
    x = torch.full((1, 1, 1, 16), 256.0, device=dev)
    w = torch.full((16, 32), 1 / 16, device=dev, dtype=f16)
    out = torch.empty(1, 1, 1, 16, device=dev)
    kg.fused_gemm(x, w, n=16, k=16, m=1, out=out, gate=True)
    check("gemm fp32-A gate: 256*1/16 x16 squared = 65536", bool((out == 65536).all()), f"got {out.flatten()[0].item()}")
    b = torch.randn(32, device=dev)
    kg.fused_gemm(x, w, n=16, k=16, m=1, out=out, gate=True, bias=b)
    ref = (256.0 + b[:16].double()) * (256.0 + b[16:].double())
    check_rows("gemm fp32-A gate + bias", out.view(1, 16), ref.view(1, 16), 1e-6)
    # explicit static a_scale on the fp16 path: both accumulators must be un-scaled
    xh = torch.full((1, 1, 1, 16), 2.0, device=dev, dtype=f16)
    kg.fused_gemm(xh, w, n=16, k=16, m=1, out=out, gate=True, a_scale=1 / 256)
    check("gemm static a_scale gate (2*1/16 x16)^2 = 4", bool((out == 4).all()), f"got {out.flatten()[0].item()}")
    # random gated fp32 GEMM vs fp64, large and tiny rows
    for mag in (2.0 ** -18, 6e4):
        x = mags((2, 9, 11, 96), mag)
        w = (torch.randn(96, 160, device=dev) * 0.1).to(f16)
        bias = torch.randn(160, device=dev) * mag * 0.3
        out = torch.empty(2, 9, 11, 80, device=dev)
        kg.fused_gemm(x.float(), w, n=80, k=96, m=2 * 9 * 11, out=out, gate=True, bias=bias)
        r = x.float().double() @ w.double() + bias.double()
        check_rows(f"gemm fp32-A gate vs fp64 |x|~{mag:.1e}", out, r[..., :80] * r[..., 80:], 5e-3)


# ------------------------------------------------------------ 2. operand range
def test_range():
    x = torch.full((1, 1, 1, 16), 2.0 ** -18, device=dev)
    w = torch.full((16, 16), 256.0, device=dev, dtype=f16)
    out = torch.empty(1, 1, 1, 16, device=dev)
    kg.fused_gemm(x, w, n=16, k=16, m=1, out=out)
    check("gemm 16 x 2^-18 . 16 x 256 = 2^-6", bool((out == 2.0 ** -6).all()), f"got {out.flatten()[0].item()}")
    n, h, wd = 2, 10, 14
    for mag in (2.0 ** -18, 1.0, 6e4, 1e7):
        # dense PRO 0 (plain linear; wide N exercises the tiled kernel since A is fp32)
        x = mags((n, h, wd, 128), mag)
        w = (torch.randn(128, 96, device=dev) * 0.1).to(f16)
        out = torch.empty(n, h, wd, 96, device=dev)
        kg.fused_gemm(x.float(), w, n=96, k=128, m=n * h * wd, out=out)
        check_rows(f"gemm fp32-A dense vs fp64 |x|~{mag:.1e}", out, x.float().double() @ w.double())
        # 2x2 stride-2 down conv (A_MODE 1)
        xd = mags((n, h, wd, 32), mag)
        wd4 = (torch.randn(64, 32, 2, 2, device=dev) * 0.1).to(f16)
        out = torch.empty(n, h // 2, wd // 2, 64, device=dev)
        kg.fused_gemm(xd.float(), wd4.permute(2, 3, 1, 0).reshape(128, 64).contiguous(), n=64, k=128,
                      m=n * (h // 2) * (wd // 2), out=out, geom=(h, wd, 1, 1, 0, 0), a_mode=1)
        ref = F.conv2d(xd.float().double().permute(0, 3, 1, 2), wd4.double(), stride=2).permute(0, 2, 3, 1)
        check_rows(f"gemm fp32-A down (a_mode 1) vs fp64 |x|~{mag:.1e}", out, ref)
        # pixel-shuffle up (C_MODE 1)
        wu = (torch.randn(128, 4 * 16, device=dev) * 0.1).to(f16)
        out = torch.empty(n, 2 * h, 2 * wd, 16, device=dev)
        kg.fused_gemm(x.float(), wu, n=64, k=128, m=n * h * wd, out=out, geom=(h, wd, 1, 1, 0, 0), c_mode=1)
        y = (x.float().double() @ wu.double()).view(n, h, wd, 2, 2, 16).permute(0, 1, 3, 2, 4, 5)
        check_rows(f"gemm fp32-A up (c_mode 1) vs fp64 |x|~{mag:.1e}", out, y.reshape(n, 2 * h, 2 * wd, 16))
        # dense 3x3 conv, fp32 input
        xc = mags((n, h, wd, 48), mag)
        wc = (torch.randn(32, 48, 3, 3, device=dev) * 0.1).to(f16)
        bc = torch.zeros(32, device=dev)
        got = kc.conv3x3(xc.float(), kc.pad_cols(wc.permute(2, 3, 1, 0).reshape(-1, 32).contiguous()), bc, 32,
                         out_dtype=f32)
        ref = F.conv2d(xc.float().double().permute(0, 3, 1, 2), wc.double(), padding=1).permute(0, 2, 3, 1)
        check_rows(f"conv3x3 fp32-in vs fp64 |x|~{mag:.1e}", got, ref)
    # per-row scale: tiny and huge rows in the same tile
    x = mags((1, 4, 16, 128), 1.0)
    x[:, ::2] *= 2.0 ** -18
    x[:, 1::2] *= 6e4
    w = (torch.randn(128, 96, device=dev) * 0.1).to(f16)
    out = torch.empty(1, 4, 16, 96, device=dev)
    kg.fused_gemm(x.float(), w, n=96, k=128, m=64, out=out)
    check_rows("gemm fp32-A mixed 2^-18 / 6e4 rows vs fp64", out, x.float().double() @ w.double())


# ------------------------------------------------------------ 3. box mean precision
def test_boxmean():
    x = torch.tensor([[1000.0, -1000.0], [1000.5, -1000.0]], device=dev).view(1, 2, 2, 1).expand(1, 2, 2, 8)
    got = ks.boxmean(x.contiguous().to(f16), 2, 2, 1, 1)
    check("boxmean [[1000,-1000],[1000.5,-1000]] = 0.125", bool((got.float() == 0.125).all()),
          f"got {got.flatten()[0].item()}")
    # large opposite-sign columns + small signal, vs an fp64 box mean of the same fp16 input
    n, h, w, c = 2, 37, 53, 24
    sign = torch.where(torch.arange(w, device=dev) % 2 == 0, 1.0, -1.0).view(1, 1, w, 1)
    x = (1000.0 * sign + torch.rand(n, h, w, c, device=dev)).to(f16)
    for k1, k2 in ((2, 2), (8, 6), (20, 31)):
        oh, ow = h - k1 + 1, w - k2 + 1
        got = ks.boxmean(x, k1, k2, oh, ow).double()
        ref = F.avg_pool2d(x.double().permute(0, 3, 1, 2), (k1, k2), stride=1).permute(0, 2, 3, 1)
        # fp16 output rounding (2^-11 relative) + fp32 running-prefix rounding (prefix <= max|x| * L);
        # an fp16 intermediate rounds 1000.x column means to 0.5 ulp = 0.25
        bound = ref.abs() * 2.0 ** -11 + 2.0 ** -22 * 1001 * max(h, w)
        err = ((got - ref).abs() - bound).max().item()
        check(f"boxmean cancellation k={k1}x{k2} vs fp64", err <= 0,
              f"max|d|={(got - ref).abs().max().item():.3e} (bound {2.0 ** -22 * 1001 * max(h, w):.1e} + 2^-11|ref|)")


# --------------------------------------------------------- 4. out aliasing residual
def test_alias():
    """zero weights, unit bias, residual 0, out = residual: the result is 1 whether the
    launch tunes (cold: many benchmark launches) or hits the cache (warm)."""
    with fresh_tuner() as t:
        for phase in ("cold", "warm"):
            r = torch.zeros(1, 16, 24, 32, device=dev, dtype=f16)
            x = torch.randn(1, 16, 24, 48, device=dev, dtype=f16)
            wk = kc.pad_cols(torch.zeros(9 * 48, 32, device=dev, dtype=f16))
            kc.conv3x3(x, wk, torch.ones(32, device=dev), 32, residual=r, out=r)
            check(f"conv3x3 out=residual ({phase} tune) = 1", bool((r == 1).all()), f"max {r.max().item()}")
            for k, n in ((256, 32), (64, 96)):  # tiled kernel / short-K wide kernel
                r = torch.zeros(1, 16, 24, n, device=dev, dtype=f16)
                x = torch.randn(1, 16, 24, k, device=dev, dtype=f16)
                kg.fused_gemm(x, torch.zeros(k, n, device=dev, dtype=f16), n=n, k=k, m=16 * 24, out=r,
                              bias=torch.ones(n, device=dev), residual=r)
                check(f"gemm k={k} out=residual ({phase} tune) = 1", bool((r == 1).all()), f"max {r.max().item()}")
            if phase == "cold":
                check("alias: 3 entries tuned cold", len(t.cache) == 3, str(sorted(t.cache)))
    a = torch.zeros(4, 8, device=dev)
    check("aliased(): slices of one buffer / disjoint tensors",
          kt.aliased(a[:, :4], a) and kt.aliased(a[:, 4:], a[:, :4]) and not kt.aliased(a[:2], a[2:])
          and not kt.aliased(a, torch.zeros(4, device=dev)) and not kt.aliased(a, None))


# ------------------------------------------------------------------ 5. tuning cache
def _noop(c):
    pass


_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
from clarus.kernels import tuning as kt
t = kt.Tuner(sys.argv[2])
for i in range(int(sys.argv[4])):
    t.pick("k", (sys.argv[3], i), [{"v": i}], None)
"""


def test_tuning():
    env_old = os.environ.get("CLARUS_TUNE_CACHE")
    cwd = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as d:
            # a. relative path without a directory component
            os.chdir(d)
            os.environ["CLARUS_TUNE_CACHE"] = "tune.json"
            t = kt.Tuner()
            t.pick("k", (1,), [{"v": 1}], _noop)
            sub = os.path.join(d, "sub")
            os.makedirs(sub)
            os.chdir(sub)  # resolved to an absolute path on first use: later saves stay put
            t.pick("k", (2,), [{"v": 2}], _noop)
            import json
            ok = os.path.exists(os.path.join(d, "tune.json")) and not os.path.exists(os.path.join(sub, "tune.json"))
            check("CLARUS_TUNE_CACHE=tune.json saves into the cwd", ok and len(json.load(open("../tune.json"))) == 2)
            os.chdir(d)

            # b. per-target state: two GPU models in one process
            os.environ.pop("CLARUS_TUNE_CACHE")
            old_dir, old_name = kt._CACHE_DIR, kt._device_name
            kt._CACHE_DIR = d
            name = ["GPU A"]
            kt._device_name = lambda device=None: name[0]
            try:
                t = kt.Tuner()
                ca = t.pick("k", (7,), [{"v": "a"}], _noop)
                name[0] = "GPU B"
                cb = t.pick("k", (7,), [{"v": "b"}], _noop)
                check("tuner state partitioned by device target", ca == {"v": "a"} and cb == {"v": "b"}, f"{ca} {cb}")
                fa = os.path.join(d, f"tune_GPU-A_triton{kt.triton.__version__}.json")
                fb = os.path.join(d, f"tune_GPU-B_triton{kt.triton.__version__}.json")
                check("one cache file per target", os.path.exists(fa) and os.path.exists(fb))
                # pinned file: first target owns it, others get a sibling file
                pinned = os.path.join(d, "pinned.json")
                t = kt.Tuner(pinned)
                name[0] = "GPU A"
                t.pick("k", (8,), [{"v": "a"}], _noop)
                name[0] = "GPU B"
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cb = t.pick("k", (8,), [{"v": "b"}], _noop)
                check("pinned cache: second target uses a sibling file",
                      cb == {"v": "b"} and t.path_for() != pinned and os.path.exists(t.path_for()), t.path_for())
            finally:
                kt._CACHE_DIR, kt._device_name = old_dir, old_name

            # c. concurrent writers merge instead of overwriting each other
            p = os.path.join(d, "shared.json")
            t1, t2 = kt.Tuner(p), kt.Tuner(p)
            t1.pick("k", ("one",), [{"v": 1}], _noop)
            t2.pick("k", ("two",), [{"v": 2}], _noop)   # t2 loaded before seeing "one" on disk
            t1.pick("k", ("three",), [{"v": 3}], _noop)  # t1's stale view lacks "two"
            import json
            keys = set(json.load(open(p)))
            check("stale tuners merge on save", keys == {"k|one", "k|two", "k|three"}, str(sorted(keys)))
            p = os.path.join(d, "procs.json")
            procs = [subprocess.Popen([sys.executable, "-c", _CHILD, os.path.join(ROOT, "python"), p, f"p{j}", "40"],
                                      env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}) for j in range(4)]
            rc = [q.wait() for q in procs]
            keys = set(json.load(open(p)))
            check("4 concurrent processes x 40 entries all kept", rc == [0] * 4 and len(keys) == 160,
                  f"rc={rc} entries={len(keys)}")
    finally:
        os.chdir(cwd)
        if env_old is None:
            os.environ.pop("CLARUS_TUNE_CACHE", None)
        else:
            os.environ["CLARUS_TUNE_CACHE"] = env_old


# ------------------------------------------------------------- 6. tuning key dtype
def test_key_dtype():
    with fresh_tuner() as t:
        w = (torch.randn(256, 96, device=dev) * 0.1).to(f16)
        for adt in (f16, f32):  # K > 128: both run the tiled kernel
            x = torch.randn(1, 16, 16, 256, device=dev).to(adt)
            kg.fused_gemm(x, w, n=96, k=256, m=256, out=torch.empty(1, 16, 16, 96, device=dev, dtype=f16))
        keys = sorted(k for k in t.cache if k.startswith("gemm3|"))
        check("gemm key separates fp16 / fp32 A", len(keys) == 2, str(keys))


if __name__ == "__main__":
    test_gate_scale()
    test_range()
    test_boxmean()
    test_alias()
    test_tuning()
    test_key_dtype()
    print("FAILED:" if FAIL else "ALL OK", FAIL)
    sys.exit(1 if FAIL else 0)
