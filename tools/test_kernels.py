"""Unit tests: every Triton kernel vs a PyTorch fp32 reference.

  python tools/test_kernels.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from clarus.kernels import conv3x3 as kc  # noqa: E402
from clarus.kernels import gemm as kg  # noqa: E402
from clarus.kernels import spatial as ks  # noqa: E402
from clarus.ops.torch_ref import TorchOps, box_geometry  # noqa: E402
from clarus.ops.triton_cuda import _conv_kn, _taps  # noqa: E402

dev = "cuda"
dt = torch.float16
torch.manual_seed(0)
FAIL = []


def check(name, got, ref, tol=2e-2):
    got, ref = got.float(), ref.float()
    err = (got - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    ok = err / scale < tol
    print(f"{'OK ' if ok else 'BAD'} {name:28s} max|d|={err:.3e} rel={err / scale:.2e}")
    if not ok:
        FAIL.append(name)


def rnd(*s, dtype=dt):
    return torch.randn(*s, device=dev).to(dtype)


def ln(x, eps=1e-6):
    return TorchOps._ln(x, eps)


def small_expand(s, pt, pl, h, w):
    oh, ow = s.shape[1:3]
    ih = (torch.arange(h, device=dev) - pt).clamp(0, oh - 1)
    iw = (torch.arange(w, device=dev) - pl).clamp(0, ow - 1)
    return s.index_select(1, ih).index_select(2, iw)


def test_gemm():
    n, h, w, k, c = 2, 13, 21, 96, 80
    x = rnd(n, h, w, k)
    wt = rnd(k, 2 * c) * 0.1
    b = torch.randn(2 * c, device=dev)
    sc = torch.randn(c, device=dev)
    res = rnd(n, h, w, c)
    m = n * h * w
    out = torch.empty(n, h, w, c, device=dev, dtype=dt)
    kg.fused_gemm(x, wt, n=c, k=k, m=m, out=out, bias=b, gate=True, scale=sc, residual=res,
                  pro=1, stats=kg.row_stats(x))
    r = ln(x) @ wt.float() + b
    r = r[..., :c] * r[..., c:]
    check("gemm ln+gate+scale+res", out, res.float() + r * sc)

    out = torch.empty(n, h, w, 2 * c, device=dev, dtype=dt)
    kg.fused_gemm(x, wt, n=2 * c, k=k, m=m, out=out, bias=b, act=1)
    check("gemm gelu", out, F.gelu(x.float() @ wt.float() + b))

    k1, k2, oh, ow, pt, pl = box_geometry(h, w, 8)
    s = rnd(n, oh, ow, k)
    wt2 = wt[:, :c].contiguous()
    out = torch.empty(n, h, w, c, device=dev, dtype=dt)
    kg.fused_gemm(x, wt2, n=c, k=k, m=m, out=out, pro=2, small=s, geom=(h, w, oh, ow, pt, pl))
    check("gemm mul-small", out, (x.float() * small_expand(s, pt, pl, h, w).float()) @ wt2.float())

    # stride-2 2x2 conv
    hh, ww = 14, 22
    xi = rnd(n, hh, ww, 32)
    w4 = rnd(64, 32, 2, 2) * 0.1
    bb = torch.randn(64, device=dev)
    out4 = torch.empty(n, hh // 2, ww // 2, 64, device=dev, dtype=dt)
    kg.fused_gemm(xi, _conv_kn(w4, dt), n=64, k=128, m=n * (hh // 2) * (ww // 2), out=out4, bias=bb,
                  geom=(hh, ww, 1, 1, 0, 0), a_mode=1)
    check("gemm space-to-depth (down)", out4,
          F.conv2d(xi.permute(0, 3, 1, 2).float(), w4.float(), bb, stride=2).permute(0, 2, 3, 1))

    # pixel shuffle up
    wu = rnd(4 * 48, 32) * 0.1  # [4C', C] torch layout, o = c*4+dy*2+dx
    wu_kn = wu.view(48, 2, 2, 32).permute(1, 2, 0, 3).reshape(192, 32).t().contiguous()
    resu = rnd(n, 2 * hh, 2 * ww, 48)
    outu = torch.empty(n, 2 * hh, 2 * ww, 48, device=dev, dtype=dt)
    kg.fused_gemm(xi, wu_kn, n=192, k=32, m=n * hh * ww, out=outu, residual=resu,
                  geom=(hh, ww, 1, 1, 0, 0), c_mode=1)
    ref = F.pixel_shuffle(F.conv2d(xi.permute(0, 3, 1, 2).float(), wu.float()[:, :, None, None]), 2)
    check("gemm pixel-shuffle (up)", outu, ref.permute(0, 2, 3, 1) + resu.float())



def test_spatial():
    n, h, w, c = 2, 37, 53, 48
    x = rnd(n, h, w, c)
    for k in (8, 40, 64):
        k1, k2, oh, ow, pt, pl = box_geometry(h, w, k)
        got = ks.boxmean(x, k1, k2, oh, ow)
        ref = TorchOps(dt, dev).boxmean(x, k).data
        check(f"boxmean k={k}", got, ref, tol=5e-3)
    x2 = rnd(n, h, w, 2 * c)
    w2 = rnd(2 * c, 1, 3, 3)
    b2 = torch.randn(2 * c, device=dev)
    got = ks.dwconv(x2, _taps(w2), b2, 3, 3, gate=True)
    r = F.conv2d(x2.permute(0, 3, 1, 2).float(), w2.float(), b2, padding=1, groups=2 * c).permute(0, 2, 3, 1)
    check("dwconv3x3 gate", got, r[..., :c] * r[..., c:])
    got = ks.dwconv(x, _taps(w2[:c]), b2[:c].contiguous(), 3, 3)
    r = F.conv2d(x.permute(0, 3, 1, 2).float(), w2[:c].float(), b2[:c], padding=1, groups=c)
    check("dwconv3x3 plain", got, r.permute(0, 2, 3, 1))
    got = ks.dwconv(x, _taps(w2[:c]), b2[:c].contiguous(), 3, 3, dil=2)
    r = F.conv2d(x.permute(0, 3, 1, 2).float(), w2[:c].float(), b2[:c], padding=2, dilation=2, groups=c)
    check("dwconv3x3 dil2", got, r.permute(0, 2, 3, 1))

    check("ln rows", ks.ln_rows(x), ln(x))
    check("rms rows", ks.ln_rows(x, rms=True), TorchOps._ln(x, 1e-6, rms=True))
    k1, k2, oh, ow, pt, pl = box_geometry(h, w, 20)
    s = rnd(n, oh, ow, c)
    check("mul_small", ks.mul_small(x, s, pt, pl), x.float() * small_expand(s, pt, pl, h, w).float())
    st = kg.row_stats(x)
    xf = x.float().reshape(-1, c)
    check("row_stats mean", st[:, 0], xf.mean(-1))
    check("row_stats rstd", st[:, 1], torch.rsqrt(xf.var(-1, unbiased=False) + 1e-6))


def test_conv():
    n, h, w = 2, 19, 27
    a = rnd(n, h, w, 64)
    wt = rnd(27, 64, 3, 3) * 0.1
    bias = torch.randn(27, device=dev)
    biasp = torch.cat([bias, bias.new_zeros(5)])
    got = kc.conv3x3(a, kc.pad_cols(_conv_kn(wt, dt)), biasp, 32, out_dtype=torch.float32)[..., :27]
    ref = F.conv2d(a.permute(0, 3, 1, 2).float(), wt.float(), bias, padding=1)
    check("conv3x3 (cout 27, padded)", got, ref.permute(0, 2, 3, 1))
    x3 = torch.rand(n, 3, h, w, device=dev)
    wi = rnd(64, 3, 3, 3) * 0.3
    bi = torch.randn(64, device=dev)
    ref = F.conv2d(x3, wi.float(), bi, padding=1).permute(0, 2, 3, 1)
    check("intro3x3 (NCHW in)", kc.intro3x3(x3, _conv_kn(wi, torch.float32), bi, torch.float32), ref)
    xe = rnd(n, h, w, 64).float() * 300.0
    we = rnd(3, 64, 3, 3) * 0.1
    be = torch.randn(3, device=dev)
    ref = F.conv2d(xe.permute(0, 3, 1, 2), we.float(), be, padding=1) + x3
    got = kc.conv3x3(xe, kc.pad_cols(_conv_kn(we, dt)), be, 3, residual=x3, out_dtype=torch.float32, nchw_out=True)
    check("conv3x3 ending fp32-in NCHW+res", got, ref)
    xd = rnd(n, h, w, 32).float() * 1000.0
    wd = rnd(64, 32, 2, 2) * 0.1
    bd = torch.randn(64, device=dev)
    out = torch.empty(n, h // 2, w // 2, 64, device=dev, dtype=torch.float32)
    xd = xd[:, : h // 2 * 2, : w // 2 * 2].contiguous()
    hh, ww = xd.shape[1:3]
    kg.fused_gemm(xd, _conv_kn(wd, dt), n=64, k=128, m=n * (hh // 2) * (ww // 2), out=out, bias=bd,
                  geom=(hh, ww, 1, 1, 0, 0), a_mode=1)
    check("down fp32-in (scaled fp16)", out,
          F.conv2d(xd.permute(0, 3, 1, 2), wd.float(), bd, stride=2).permute(0, 2, 3, 1))
    xl = rnd(n, h, w, 96)
    wl = rnd(96, 64) * 0.1
    bl = torch.randn(64, device=dev)
    out = torch.empty(n, h, w, 64, device=dev, dtype=dt)
    kg.fused_gemm(xl, wl, n=64, k=96, m=n * h * w, out=out, bias=bl, pro=1)
    check("gemm inline-LN (K<=128)", out, ln(xl) @ wl.float() + bl)


def test_generic_ops():
    """Ops added for Restormer / RRDBNet."""
    from clarus.kernels import attention as ka
    n, h, w = 2, 21, 34
    # conv3x3 with channel-slice input / output, LeakyReLU, scale, residual
    buf = rnd(n, h, w, 96)
    wt = rnd(32, 64, 3, 3) * 0.05
    b = torch.randn(32, device=dev)
    res = rnd(n, h, w, 32)
    kc.conv3x3(buf[..., :64], kc.pad_cols(_conv_kn(wt, dt)), b, 32, out=buf[..., 64:], act=1, out_scale=0.2,
               residual=res)
    ref = F.leaky_relu(F.conv2d(buf[..., :64].permute(0, 3, 1, 2).float(), wt.float(), b, padding=1), 0.2) * 0.2
    check("conv3x3 slice+lrelu+scale+res", buf[..., 64:], ref.permute(0, 2, 3, 1) + res.float())
    # conv3x3 with fused nearest x2 upsample
    x = rnd(n, h, w, 48)
    wt2 = rnd(40, 48, 3, 3) * 0.05
    b2 = torch.randn(40, device=dev)
    got = kc.conv3x3(x, kc.pad_cols(_conv_kn(wt2, dt)), b2, 40, up2=True, act=1)
    up = F.interpolate(x.permute(0, 3, 1, 2).float(), scale_factor=2, mode="nearest")
    check("conv3x3 up2+lrelu", got, F.leaky_relu(F.conv2d(up, wt2.float(), b2, padding=1), 0.2).permute(0, 2, 3, 1))
    # depthwise conv with GELU gate (Restormer GDFN), odd channel count
    c = 127
    xg = rnd(n, h, w, 2 * c)
    wd = rnd(2 * c, 1, 3, 3)
    bd = torch.randn(2 * c, device=dev)
    got = ks.dwconv(xg, _taps(wd), bd, 3, 3, gate="gelu")
    r = F.conv2d(xg.permute(0, 3, 1, 2).float(), wd.float(), bd, padding=1, groups=2 * c).permute(0, 2, 3, 1)
    check("dwconv3x3 gelu gate (C=254)", got, F.gelu(r[..., :c]) * r[..., c:])
    # MDTA Gram + norms
    heads, ch = 4, 48
    qkv = rnd(n, h, w, 3 * heads * ch)
    g, sq, sk = ka.gram(qkv, heads)
    q = qkv[..., :heads * ch].reshape(n, h * w, heads, ch).float()
    k = qkv[..., heads * ch:2 * heads * ch].reshape(n, h * w, heads, ch).float()
    check("gram q^T k", g, torch.einsum("nphi,nphj->nhij", q, k), tol=1e-3)
    check("gram |q|^2", sq, q.square().sum(1), tol=1e-3)
    check("gram |k|^2", sk, k.square().sum(1), tol=1e-3)
    # GEMM with a row stride (channel slice input) and LayerNorm eps
    xs = rnd(n, h, w, 3 * 64)[..., 128:]
    wl = rnd(64, 80) * 0.1
    out = torch.empty(n, h, w, 80, device=dev, dtype=dt)
    kg.fused_gemm(xs, wl, n=80, k=64, m=n * h * w, out=out, lda=xs.stride(2))
    check("gemm lda (channel slice)", out, xs.float() @ wl.float())
    x2 = rnd(n, h, w, 96) * 0.01
    wln = rnd(96, 48) * 0.1
    o2 = torch.empty(n, h, w, 48, device=dev, dtype=dt)
    kg.fused_gemm(x2, wln, n=48, k=96, m=n * h * w, out=o2, pro=1, ln_eps=1e-5)
    check("gemm inline LN eps=1e-5", o2, ln(x2, 1e-5) @ wln.float())
    # intro with a non-power-of-2 channel count
    xi = torch.rand(n, 3, h, w, device=dev)
    wi = rnd(48, 3, 3, 3) * 0.3
    bi = torch.randn(48, device=dev)
    check("intro3x3 C=48", kc.intro3x3(xi, _conv_kn(wi, torch.float32), bi, torch.float32),
          F.conv2d(xi, wi.float(), bi, padding=1).permute(0, 2, 3, 1))


def test_freq_ssm_ops():
    """Ops added for FFTformer / EVSSM, through the triton backend vs the fp32 torch reference."""
    from clarus.ops.torch_ref import grid_tiles
    from clarus.ops.triton_cuda import TritonOps
    T, R = TritonOps(dt, torch.device(dev)), TorchOps(torch.float32, torch.device(dev))

    def r32(*s, sc=1.0):
        return rnd(*s).float() * sc

    for h, w, k, n_out, ln_mode, res in [(64, 96, 48, 288, True, False), (40, 64, 192, 256, True, False),
                                         (90, 45, 144, 48, False, True), (52, 76, 96, 96, "rms", True)]:
        x, f = r32(1, h, w, k), torch.randn(n_out, 8, 5, device=dev)
        wl, b = r32(k, n_out, sc=k ** -0.5), torch.randn(n_out, device=dev)
        rr = r32(1, h, w, n_out) if res else None
        check(f"linear_patch_filter {h}x{w} k{k} ln={ln_mode} res={res}",
              T.linear_patch_filter(x.to(dt), wl.to(dt), b, f, ln=ln_mode, ln_eps=1e-5,
                                    residual=rr.to(dt) if res else None),
              R.linear_patch_filter(x, wl, b, f, ln=ln_mode, ln_eps=1e-5, residual=rr))
    x = r32(2, 64, 96, 288)
    check("patch_corr (q, k slices)", T.patch_corr(x.to(dt), 96), R.patch_corr(x, 96))
    for k in (48, 192):
        x, wl, b = r32(2, 20, 30, k), r32(k, 80, sc=k ** -0.5), torch.randn(80, device=dev)
        check(f"linear rms k={k}", T.linear(x.to(dt), wl.to(dt), b, ln="rms"), R.linear(x, wl, b, ln="rms"))
    x, hid = r32(2, 20, 30, 96), r32(2, 20, 30, 288)
    lw, lb, wl, b, res = torch.randn(96, device=dev), torch.randn(96, device=dev), r32(96, 48, sc=0.1), \
        torch.randn(48, device=dev), r32(2, 20, 30, 48)
    check("ln_mul_linear (m = channel slice)",
          T.ln_mul_linear(x.to(dt), lw, lb, hid.to(dt)[..., 192:], wl.to(dt), b, residual=res.to(dt)),
          R.ln_mul_linear(x, lw, lb, hid[..., 192:], wl, b, residual=res))
    check("ln_mul_linear gelu(m), fp32 x", T.ln_mul_linear(x * 1e4, lw, lb, x.to(dt), wl.to(dt), m_act="gelu"),
          R.ln_mul_linear(x * 1e4, lw, lb, x, wl, m_act="gelu"))
    x, wd, bd = r32(3, 40, 50, 96), torch.randn(96, 1, 3, 3, device=dev), torch.randn(96, device=dev)
    check("dwconv act=gelu", T.dwconv(x.to(dt), wd, bd, act="gelu"), R.dwconv(x, wd, bd, act="gelu"))
    x, w1, b1 = r32(3, 24, 40, 19), torch.randn(19, 7, device=dev), torch.randn(19, device=dev)
    check("seq_dwconv1d", T.seq_dwconv1d(x, w1, b1), R.seq_dwconv1d(x, w1, b1))
    for t, h, w, d, r in [(2, 48, 64, 96, 3), (8, 96, 96, 192, 6), (1, 300, 300, 32, 2)]:
        u, dbl = r32(t, h, w, d), r32(t, h, w, r + 16)
        dtw, dtb, dsk = r32(d, r, sc=0.5), torch.randn(d, device=dev) - 2, torch.randn(d, device=dev)
        A = -torch.rand(d, 8, device=dev) * 4 - 0.5
        check(f"selective_scan {t}x{h}x{w} d{d}", T.selective_scan(u.to(dt), dbl, dtw, dtb, A, dsk),
              R.selective_scan(u.to(dt).float(), dbl, dtw, dtb, A, dsk), tol=1e-3)
    for h, w, k in [(352, 640, 384), (180, 320, 96), (100, 104, 96)]:
        x = rnd(2, h, w, 48)
        for orient in (0, 1, 2):
            oh, ow = (w, h) if orient == 2 else (h, w)
            k1, k2, idx = grid_tiles(oh, ow, k)
            parts = rnd(2 * len(idx), k1, k2, 48)
            check(f"tiles gather {h}x{w} o{orient}", T.tiles_gather(x, k1, k2, idx, orient),
                  R.tiles_gather(x, k1, k2, idx, orient), tol=1e-6)
            check(f"tiles merge {h}x{w} o{orient}", T.tiles_merge(x, parts, k1, k2, idx, orient),
                  R.tiles_merge(x.float(), parts.float(), k1, k2, idx, orient), tol=2e-3)
    for h, w in [(92, 160), (45, 81)]:
        x = rnd(2, h, w, 96)
        for sc in (0.5, 2):
            check(f"resize_bilinear {h}x{w} x{sc}", T.resize_bilinear(x, sc), R.resize_bilinear(x.float(), sc))


if __name__ == "__main__":
    test_gemm()
    test_spatial()
    test_conv()
    test_generic_ops()
    test_freq_ssm_ops()
    print("FAILED:" if FAIL else "ALL OK", FAIL)
    sys.exit(1 if FAIL else 0)
