"""Mamba selective scan (one direction, z = None) over NHWC crops, chunk-parallel.

For crop t, channel d, state n along the raster sequence l:
  delta_l = softplus(dt_raw_l . dt_w[d] + dt_bias[d])
  h_l = exp(delta_l A[d, n]) h_{l-1} + delta_l B_l[n] u_l ;   y_l = sum_n C_l[n] h_l + D[d] u_l

Each lane runs one channel's recurrence step by step (work-efficient, all in registers). The
sequence is split into S segments so that enough programs run in parallel:
  1. scan8_kernel(SUMMARY): per segment, the end state from h = 0 and the total decay
                            exp(A * sum delta)
  2. carry_kernel:          sequential combine over the S segments -> start state of each one
  3. scan8_kernel:          per segment from its start state -> y
(1 and 2 are skipped when one segment already gives enough parallelism.)
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .tuning import TUNER

LOG2E = 1.4426950408889634


@triton.jit
def _softplus(x):
    return tl.where(x <= 20.0, libdevice.log1p(tl.exp(x)), x)


@triton.jit
def carry_kernel(PD, EH, H0, S, ND, BLOCK: tl.constexpr):
    """H0[t, s] = PD[t, s-1] * H0[t, s-1] + EH[t, s-1], H0[t, 0] = 0 (rows of NS*D values)."""
    t = tl.program_id(0)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = i < ND
    h = tl.zeros((BLOCK,), dtype=tl.float32)
    base = t.to(tl.int64) * S * ND
    for s in range(S):
        tl.store(H0 + base + s * ND + i, h, mask=m)
        h = tl.load(PD + base + s * ND + i, mask=m, other=0.0) * h + tl.load(EH + base + s * ND + i, mask=m, other=0.0)


@triton.jit
def scan8_kernel(U, DBL, DTW, DTB, A2, DP, H0, Y, PD, EH, L, D, SEG,
                 NC: tl.constexpr, R: tl.constexpr, HAS_H0: tl.constexpr, SUMMARY: tl.constexpr,
                 BLOCK_D: tl.constexpr, UNROLL: tl.constexpr):
    """d_state = 8. One step per iteration; each lane owns one channel and keeps its 8 states in
    named registers (1-D tensors only: no layout conversions inside the loop; B / C / dt_raw of a
    step are scalars shared by all lanes). Parallelism: crops x channel blocks x segments.
    SUMMARY: write the segment's total decay / end state (PD, EH) instead of y."""
    pid = tl.program_id(0)
    nseg = tl.cdiv(L, SEG)
    t = pid // nseg
    s = pid % nseg
    ds = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = ds < D
    dtb = tl.load(DTB + ds, mask=dm, other=0.0)
    a0 = tl.load(A2 + ds * 8 + 0, mask=dm, other=0.0)
    a1 = tl.load(A2 + ds * 8 + 1, mask=dm, other=0.0)
    a2 = tl.load(A2 + ds * 8 + 2, mask=dm, other=0.0)
    a3 = tl.load(A2 + ds * 8 + 3, mask=dm, other=0.0)
    a4 = tl.load(A2 + ds * 8 + 4, mask=dm, other=0.0)
    a5 = tl.load(A2 + ds * 8 + 5, mask=dm, other=0.0)
    a6 = tl.load(A2 + ds * 8 + 6, mask=dm, other=0.0)
    a7 = tl.load(A2 + ds * 8 + 7, mask=dm, other=0.0)
    if not SUMMARY:
        dsk = tl.load(DP + ds, mask=dm, other=0.0)
    hb = pid.to(tl.int64) * 8 * D + ds
    if HAS_H0:
        h0 = tl.load(H0 + hb + 0 * D, mask=dm, other=0.0)
        h1 = tl.load(H0 + hb + 1 * D, mask=dm, other=0.0)
        h2 = tl.load(H0 + hb + 2 * D, mask=dm, other=0.0)
        h3 = tl.load(H0 + hb + 3 * D, mask=dm, other=0.0)
        h4 = tl.load(H0 + hb + 4 * D, mask=dm, other=0.0)
        h5 = tl.load(H0 + hb + 5 * D, mask=dm, other=0.0)
        h6 = tl.load(H0 + hb + 6 * D, mask=dm, other=0.0)
        h7 = tl.load(H0 + hb + 7 * D, mask=dm, other=0.0)
    else:
        h0 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        h1 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        h2 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        h3 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        h4 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        h5 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        h6 = tl.zeros((BLOCK_D,), dtype=tl.float32)
        h7 = tl.zeros((BLOCK_D,), dtype=tl.float32)
    dsum = tl.zeros((BLOCK_D,), dtype=tl.float32)
    l0 = s * SEG
    l1 = tl.minimum(l0 + SEG, L)
    for lb in range(l0, l1, UNROLL):
        for i in tl.static_range(UNROLL):
            l = lb + i
            ok = l < l1
            rb = DBL + (t.to(tl.int64) * L + l) * NC
            uv = tl.load(U + (t.to(tl.int64) * L + l) * D + ds, mask=dm & ok, other=0.0).to(tl.float32)
            acc = dtb
            for r in tl.static_range(R):
                acc += tl.load(rb + r, mask=ok, other=0.0) * tl.load(DTW + ds * R + r, mask=dm, other=0.0)
            delta = tl.where(ok, _softplus(acc), 0.0)
            du = delta * uv
            h0 = tl.exp2(delta * a0) * h0 + du * tl.load(rb + R + 0, mask=ok, other=0.0)
            h1 = tl.exp2(delta * a1) * h1 + du * tl.load(rb + R + 1, mask=ok, other=0.0)
            h2 = tl.exp2(delta * a2) * h2 + du * tl.load(rb + R + 2, mask=ok, other=0.0)
            h3 = tl.exp2(delta * a3) * h3 + du * tl.load(rb + R + 3, mask=ok, other=0.0)
            h4 = tl.exp2(delta * a4) * h4 + du * tl.load(rb + R + 4, mask=ok, other=0.0)
            h5 = tl.exp2(delta * a5) * h5 + du * tl.load(rb + R + 5, mask=ok, other=0.0)
            h6 = tl.exp2(delta * a6) * h6 + du * tl.load(rb + R + 6, mask=ok, other=0.0)
            h7 = tl.exp2(delta * a7) * h7 + du * tl.load(rb + R + 7, mask=ok, other=0.0)
            if SUMMARY:
                dsum += delta
            else:
                y = dsk * uv
                y += h0 * tl.load(rb + R + 8 + 0, mask=ok, other=0.0)
                y += h1 * tl.load(rb + R + 8 + 1, mask=ok, other=0.0)
                y += h2 * tl.load(rb + R + 8 + 2, mask=ok, other=0.0)
                y += h3 * tl.load(rb + R + 8 + 3, mask=ok, other=0.0)
                y += h4 * tl.load(rb + R + 8 + 4, mask=ok, other=0.0)
                y += h5 * tl.load(rb + R + 8 + 5, mask=ok, other=0.0)
                y += h6 * tl.load(rb + R + 8 + 6, mask=ok, other=0.0)
                y += h7 * tl.load(rb + R + 8 + 7, mask=ok, other=0.0)
                tl.store(Y + (t.to(tl.int64) * L + l) * D + ds, y.to(Y.dtype.element_ty), mask=dm & ok)
    if SUMMARY:
        tl.store(PD + hb + 0 * D, tl.exp2(a0 * dsum), mask=dm)
        tl.store(EH + hb + 0 * D, h0, mask=dm)
        tl.store(PD + hb + 1 * D, tl.exp2(a1 * dsum), mask=dm)
        tl.store(EH + hb + 1 * D, h1, mask=dm)
        tl.store(PD + hb + 2 * D, tl.exp2(a2 * dsum), mask=dm)
        tl.store(EH + hb + 2 * D, h2, mask=dm)
        tl.store(PD + hb + 3 * D, tl.exp2(a3 * dsum), mask=dm)
        tl.store(EH + hb + 3 * D, h3, mask=dm)
        tl.store(PD + hb + 4 * D, tl.exp2(a4 * dsum), mask=dm)
        tl.store(EH + hb + 4 * D, h4, mask=dm)
        tl.store(PD + hb + 5 * D, tl.exp2(a5 * dsum), mask=dm)
        tl.store(EH + hb + 5 * D, h5, mask=dm)
        tl.store(PD + hb + 6 * D, tl.exp2(a6 * dsum), mask=dm)
        tl.store(EH + hb + 6 * D, h6, mask=dm)
        tl.store(PD + hb + 7 * D, tl.exp2(a7 * dsum), mask=dm)
        tl.store(EH + hb + 7 * D, h7, mask=dm)


def selective_scan(u, dbl, dt_w, dt_b, a2, dsk, out=None, sm_count=56):
    """u [T, h, w, D] (fp16/bf16); dbl [T, h, w, R + 2*NS] fp32; dt_w [D, R]; dt_b, dsk [D];
    a2 [D, NS] = A * log2(e) -> y [T, h, w, D] fp32."""
    t, hh, ww, d = u.shape
    L = hh * ww
    nc = dbl.shape[-1]
    r = dt_w.shape[1]
    ns = a2.shape[1]
    if out is None:
        out = torch.empty(u.shape, device=u.device, dtype=torch.float32)

    assert ns == 8, "scan8_kernel is written for d_state = 8"

    def launch(cf):
        bd, nw = cf["BLOCK_D"], cf["num_warps"]
        nd = triton.cdiv(d, bd)
        seg = max(16, triton.cdiv(L, max(1, cf["PROGS"] // (t * nd))))
        nseg = triton.cdiv(L, seg)
        h0 = pd = eh = None
        common = dict(NC=nc, R=r, BLOCK_D=bd, UNROLL=cf["UNROLL"], num_warps=nw)
        if nseg > 1:
            pd = torch.empty((t * nseg, ns, d), device=u.device, dtype=torch.float32)
            eh = torch.empty_like(pd)
            scan8_kernel[(t * nseg, nd)](u, dbl, dt_w, dt_b, a2, dsk, pd, out, pd, eh, L, d, seg,
                                         HAS_H0=False, SUMMARY=True, **common)
            h0 = torch.empty_like(pd)
            carry_kernel[(t, triton.cdiv(ns * d, 256))](pd, eh, h0, nseg, ns * d, BLOCK=256, num_warps=4)
        scan8_kernel[(t * nseg, nd)](u, dbl, dt_w, dt_b, a2, dsk, h0 if h0 is not None else out, out,
                                     out, out, L, d, seg, HAS_H0=h0 is not None, SUMMARY=False, **common)

    bd_max = max(16, triton.next_power_of_2(d))
    cands = []
    for bd, nw, pg, un in [(32, 1, 1024, 1), (32, 1, 2048, 1), (32, 1, 4096, 1), (32, 1, 8192, 1), (32, 1, 2048, 2),
                           (32, 1, 4096, 2), (64, 2, 2048, 1), (32, 1, 512, 1), (32, 1, 256, 1)]:
        cf = dict(BLOCK_D=min(bd, bd_max), num_warps=nw, PROGS=pg, UNROLL=un)   # clamp, never filter to nothing
        if cf not in cands:
            cands.append(cf)
    key = (L.bit_length(), t, d, r, ns, str(u.dtype))
    launch(TUNER.pick("sscan8", key, cands, launch))
    return out


# ------------------------------------------------------- overlapping crops (EVSSM grids)
@triton.jit
def _src(oy, ox, SH, SW, ORIENT: tl.constexpr):
    """oriented (y, x) -> source (y, x): 0 identity, 1 flip H and W, 2 transpose."""
    if ORIENT == 1:
        return SH - 1 - oy, SW - 1 - ox
    elif ORIENT == 2:
        return ox, oy
    else:
        return oy, ox


@triton.jit
def crops_gather_kernel(X, IDX, OUT, SH, SW, C, NT, K1, K2,
                        ORIENT: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)                       # (img * NT + tile) * K1 + row
    yy = pid % K1
    it = pid // K1
    img = it // NT
    tt = it % NT
    xs = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    cs = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    i = tl.load(IDX + 2 * tt)
    j = tl.load(IDX + 2 * tt + 1)
    sy, sx = _src(i + yy, j + xs, SH, SW, ORIENT)
    m = (xs < K2)[:, None] & (cs < C)[None, :]
    v = tl.load(X + ((img.to(tl.int64) * SH + sy) * SW + sx)[:, None] * C + cs[None, :], mask=m, other=0.0)
    tl.store(OUT + ((it.to(tl.int64) * K1 + yy) * K2 + xs)[:, None] * C + cs[None, :], v, mask=m)


@triton.jit
def crops_merge_kernel(X, P, IDX, OUT, SH, SW, OH, OW, C, NT, K1, K2,
                       ORIENT: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)                       # img * OH + oriented row
    img = pid // OH
    oy = pid % OH
    xs = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    cs = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    xm = xs < OW
    cm = cs < C
    acc = tl.zeros((BLOCK_W, BLOCK_C), dtype=tl.float32)
    cnt = tl.zeros((BLOCK_W,), dtype=tl.float32)
    for tt in range(NT):
        i = tl.load(IDX + 2 * tt)
        j = tl.load(IDX + 2 * tt + 1)
        if (oy >= i) & (oy < i + K1):
            inside = xm & (xs >= j) & (xs < j + K2)
            off = (((img * NT + tt).to(tl.int64) * K1 + (oy - i)) * K2 + (xs - j))[:, None] * C + cs[None, :]
            acc += tl.load(P + off, mask=inside[:, None] & cm[None, :], other=0.0).to(tl.float32)
            cnt += inside.to(tl.float32)
    sy, sx = _src(oy, xs, SH, SW, ORIENT)
    m = xm[:, None] & cm[None, :]
    base = tl.load(X + ((img.to(tl.int64) * SH + sy) * SW + sx)[:, None] * C + cs[None, :], mask=m, other=0.0)
    out = base.to(tl.float32) + acc / tl.maximum(cnt, 1.0)[:, None]
    tl.store(OUT + ((img.to(tl.int64) * OH + oy) * OW + xs)[:, None] * C + cs[None, :],
             out.to(OUT.dtype.element_ty), mask=m)


def _crop_blocks(c):
    bc = min(64, triton.next_power_of_2(c))
    return max(8, min(64, 4096 // bc)), bc


def crops_gather(x, idx_t, nt, k1, k2, orient):
    n, sh, sw, c = x.shape
    out = torch.empty((n * nt, k1, k2, c), device=x.device, dtype=x.dtype)
    bw, bc = _crop_blocks(c)
    crops_gather_kernel[(n * nt * k1, triton.cdiv(k2, bw), triton.cdiv(c, bc))](
        x, idx_t, out, sh, sw, c, nt, k1, k2, ORIENT=orient, BLOCK_W=bw, BLOCK_C=bc, num_warps=4)
    return out


def crops_merge(x, parts, idx_t, nt, k1, k2, orient):
    n, sh, sw, c = x.shape
    oh, ow = (sw, sh) if orient == 2 else (sh, sw)
    out = torch.empty((n, oh, ow, c), device=x.device, dtype=x.dtype)
    bw, bc = _crop_blocks(c)
    crops_merge_kernel[(n * oh, triton.cdiv(ow, bw), triton.cdiv(c, bc))](
        x, parts, idx_t, out, sh, sw, oh, ow, c, nt, k1, k2, ORIENT=orient, BLOCK_W=bw, BLOCK_C=bc, num_warps=4)
    return out


# ------------------------------------------------- depthwise 1-D conv along sequences
@triton.jit
def seq_dwconv1d_kernel(X, Wt, B, OUT, M, SEQ, C, KS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cs = tl.arange(0, BLOCK_C)
    cm = cs < C
    pos = rm % SEQ
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32) + tl.load(B + cs, mask=cm, other=0.0)[None, :]
    for k in tl.static_range(KS):
        sh = k - KS // 2
        ok = (rm < M) & (pos + sh >= 0) & (pos + sh < SEQ)
        v = tl.load(X + (rm.to(tl.int64) + sh)[:, None] * C + cs[None, :], mask=ok[:, None] & cm[None, :], other=0.0)
        acc += v * tl.load(Wt + k * C + cs, mask=cm, other=0.0)[None, :]
    tl.store(OUT + rm.to(tl.int64)[:, None] * C + cs[None, :], acc, mask=(rm < M)[:, None] & cm[None, :])


def seq_dwconv1d(x, wt, b):
    """x [T, h, w, C] fp32, wt [k, C] -> fp32, conv along each crop's raster sequence."""
    t, hh, ww, c = x.shape
    m = t * hh * ww
    out = torch.empty_like(x)
    bc = triton.next_power_of_2(c)

    def launch(cf):
        bm = cf["BLOCK_M"]
        seq_dwconv1d_kernel[(triton.cdiv(m, bm),)](x, wt, b, out, m, hh * ww, c, KS=wt.shape[0], BLOCK_M=bm,
                                                   BLOCK_C=bc, num_warps=cf["num_warps"])

    cands = [dict(BLOCK_M=bm, num_warps=nw) for bm, nw in [(16, 4), (32, 2), (16, 2), (32, 4), (64, 4), (8, 1), (16, 1)]]
    launch(TUNER.pick("dwconv1d", ((m.bit_length()), c, wt.shape[0]), cands, launch))
    return out
