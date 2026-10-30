"""Fused NHWC GEMM for 1x1 / 2x2-s2 / pixel-shuffle convolutions.

out[M, N] = epilogue( prologue(A)[M, K] @ B[K, N] )

prologue (PRO):
  0 none
  1 LN      : (a - mean[m]) * rstd[m]                (row stats from row_stats())
  2 MUL     : a * S[small(m), k]                      (TLC local-average gather, SCA)
  4 LN inline (K <= BLOCK_K), 6 RMS norm inline: a * rsqrt(mean(a^2) + eps)
  5 LN-MUL  : ((a - mean[m]) * rstd[m] * LNW[k] + LNB[k]) * act(M2[m, k])   (act: none / GELU)
A addressing (A_MODE): 0 dense rows, 1 space-to-depth 2x2 (stride-2 conv)
C addressing (C_MODE): 0 dense rows, 1 depth-to-space 2x2 (PixelShuffle)
epilogue: +bias, SimpleGate (B has 2N columns), GELU, *scale, +residual.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .spatial import _gelu as gelu  # exact-erf GELU with a fast erf
from .tuning import TUNER, cfg, p2


@triton.jit
def small_index(rm, H, W, OH, OW, PT, PL):
    hw = H * W
    img = rm // hw
    r = rm % hw
    y = r // W
    x = r % W
    sy = tl.minimum(tl.maximum(y - PT, 0), OH - 1)
    sx = tl.minimum(tl.maximum(x - PL, 0), OW - 1)
    return (img * OH + sy) * OW + sx


@triton.jit
def fused_gemm_kernel(
    A, B, C, BIAS, SCALE, R, S, STATS, M2, LNW, LNB,
    M, N, K,
    H, W, OH, OW, PT, PL,
    a_scale, ln_eps, LDA, LDM2,
    PRO: tl.constexpr, A_MODE: tl.constexpr, C_MODE: tl.constexpr, M2_ACT: tl.constexpr,
    HAS_BIAS: tl.constexpr, GATE: tl.constexpr, ACT: tl.constexpr, HAS_SCALE: tl.constexpr,
    HAS_RES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n = tl.cdiv(N, BLOCK_N)
    group = GROUP_M * num_n
    gid = pid // group
    first_m = gid * GROUP_M
    gsize = tl.minimum(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % gsize
    pid_n = (pid % group) // gsize

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mmask = rm < M
    nmask = rn < N
    rm64 = rm.to(tl.int64)
    rk = tl.arange(0, BLOCK_K)
    LDB = 2 * N if GATE else N

    cin = K // 4
    if A_MODE == 1:
        # rows are output pixels at (H//2, W//2); A is NHWC [*, H, W, K//4]
        wo = W // 2
        ho = H // 2
        img = rm // (ho * wo)
        rr = rm % (ho * wo)
        yo = rr // wo
        xo = rr % wo
        a_row = ((img.to(tl.int64) * H + 2 * yo) * W + 2 * xo) * cin
    else:
        a_row = rm64 * LDA

    if PRO == 1 or PRO == 5:
        mean = tl.load(STATS + rm64 * 2, mask=mmask, other=0.0)
        rstd = tl.load(STATS + rm64 * 2 + 1, mask=mmask, other=0.0)
    if PRO == 2:
        s_row = small_index(rm, H, W, OH, OW, PT, PL).to(tl.int64)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if GATE:
        acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if A_MODE == 1:
        # 2x2 stride-2 conv: K = (dy, dx, cin); one contiguous cin slab per quadrant
        for q in tl.static_range(4):
            qoff = ((q // 2) * W + (q % 2)) * cin
            for k0 in range(0, cin, BLOCK_K):
                kk = k0 + rk
                km = kk < cin
                a = tl.load(A + a_row[:, None] + qoff + kk[None, :], mask=mmask[:, None] & km[None, :], other=0.0)
                a = (a.to(tl.float32) * a_scale).to(B.dtype.element_ty)
                b = tl.load(B + (q * cin + kk)[:, None] * LDB + rn[None, :], mask=km[:, None] & nmask[None, :],
                            other=0.0)
                acc = tl.dot(a, b, acc)
    if A_MODE == 0:
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + rk
            kmask = kk < K
            amask = mmask[:, None] & kmask[None, :]
            a = tl.load(A + a_row[:, None] + kk[None, :], mask=amask, other=0.0)
            if PRO == 1:
                a = ((a.to(tl.float32) - mean[:, None]) * rstd[:, None]).to(B.dtype.element_ty)
            elif PRO == 4:
                # whole row in one K tile (K <= BLOCK_K): exact two-pass LayerNorm in registers
                af = a.to(tl.float32)
                mu = tl.sum(af, axis=1) / K
                dv = tl.where(amask, af - mu[:, None], 0.0)
                rs = tl.rsqrt(tl.sum(dv * dv, axis=1) / K + ln_eps)
                a = (dv * rs[:, None]).to(B.dtype.element_ty)
            elif PRO == 6:
                af = a.to(tl.float32)
                rs = tl.rsqrt(tl.sum(af * af, axis=1) / K + ln_eps)
                a = (af * rs[:, None]).to(B.dtype.element_ty)
            elif PRO == 5:
                lw = tl.load(LNW + kk, mask=kmask, other=0.0)
                lb = tl.load(LNB + kk, mask=kmask, other=0.0)
                mv = tl.load(M2 + rm64[:, None] * LDM2 + kk[None, :], mask=amask, other=0.0).to(tl.float32)
                if M2_ACT == 1:
                    mv = gelu(mv)
                nv = (a.to(tl.float32) - mean[:, None]) * rstd[:, None] * lw[None, :] + lb[None, :]
                a = (nv * mv).to(B.dtype.element_ty)
            elif PRO == 2:
                sv = tl.load(S + s_row[:, None] * K + kk[None, :], mask=amask, other=0.0)
                a = (a.to(tl.float32) * sv.to(tl.float32)).to(B.dtype.element_ty)
            if PRO == 0:
                # raw fp32 residual-stream operands are pre-scaled by 2^-8 so they fit fp16
                a = (a.to(tl.float32) * a_scale).to(B.dtype.element_ty)
            else:
                a = a.to(B.dtype.element_ty)
            bmask = kmask[:, None] & nmask[None, :]
            b = tl.load(B + kk[:, None] * LDB + rn[None, :], mask=bmask, other=0.0)
            acc = tl.dot(a, b, acc)
            if GATE:
                b2 = tl.load(B + kk[:, None] * LDB + N + rn[None, :], mask=bmask, other=0.0)
                acc2 = tl.dot(a, b2, acc2)

    if PRO == 0:
        acc = acc * (1.0 / a_scale)
    if HAS_BIAS:
        acc += tl.load(BIAS + rn, mask=nmask, other=0.0)[None, :]
        if GATE:
            acc2 += tl.load(BIAS + N + rn, mask=nmask, other=0.0)[None, :]
    if GATE:
        acc = acc * acc2
    if ACT == 1:
        acc = gelu(acc)
    cmask = mmask[:, None] & nmask[None, :]
    if C_MODE == 1:
        # PixelShuffle(2): N = (dy, dx, cout); each program's N tile lies in one quadrant
        cout = N // 4
        img = rm // (H * W)
        rr = rm % (H * W)
        y = rr // W
        x = rr % W
        c_row = ((img.to(tl.int64) * (2 * H) + 2 * y) * (2 * W) + 2 * x) * cout
        n0 = pid_n * BLOCK_N
        q = n0 // cout
        c_col = ((q // 2) * (2 * W) + (q % 2)) * cout + (n0 % cout) + tl.arange(0, BLOCK_N)
        c_off = c_row[:, None] + c_col[None, :]
    else:
        c_off = rm64[:, None] * N + rn[None, :]
    if HAS_SCALE:
        acc = acc * tl.load(SCALE + rn, mask=nmask, other=0.0)[None, :]
    if HAS_RES:
        acc += tl.load(R + c_off, mask=cmask, other=0.0).to(tl.float32)
    tl.store(C + c_off, acc.to(C.dtype.element_ty), mask=cmask)


@triton.jit
def gemm_wide_kernel(A, B, C, BIAS, SCALE, R, M, N, K, ln_eps, LDA,
                     PRO: tl.constexpr, GATE: tl.constexpr, ACT: tl.constexpr, HAS_BIAS: tl.constexpr,
                     HAS_SCALE: tl.constexpr, HAS_RES: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """Short-K GEMM (K <= BLOCK_K) whose program owns BLOCK_M full output rows: the A tile is
    loaded and normalized (PRO 4 LayerNorm / 6 RMS norm / 0 none) once and kept in registers
    while the loop walks over all N columns. The tiled kernel re-reads and re-normalizes A for
    every N tile, which made wide-N projections (qkv, FFN in) 2-3x slower than their bandwidth."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    mm = rm < M
    km = rk < K
    am = mm[:, None] & km[None, :]
    rm64 = rm.to(tl.int64)
    a = tl.load(A + rm64[:, None] * LDA + rk[None, :], mask=am, other=0.0).to(tl.float32)
    if PRO == 4:
        mu = tl.sum(a, axis=1) / K
        a = tl.where(am, a - mu[:, None], 0.0)
        a = a * tl.rsqrt(tl.sum(a * a, axis=1) / K + ln_eps)[:, None]
    elif PRO == 6:
        a = a * tl.rsqrt(tl.sum(a * a, axis=1) / K + ln_eps)[:, None]
    a = a.to(B.dtype.element_ty)
    LDB = 2 * N if GATE else N
    rn0 = tl.arange(0, BLOCK_N)
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + rn0
        nm = rn < N
        bm = km[:, None] & nm[None, :]
        acc = tl.dot(a, tl.load(B + rk[:, None] * LDB + rn[None, :], mask=bm, other=0.0))
        if HAS_BIAS:
            acc += tl.load(BIAS + rn, mask=nm, other=0.0)[None, :]
        if GATE:
            acc2 = tl.dot(a, tl.load(B + rk[:, None] * LDB + N + rn[None, :], mask=bm, other=0.0))
            if HAS_BIAS:
                acc2 += tl.load(BIAS + N + rn, mask=nm, other=0.0)[None, :]
            acc = acc * acc2
        if ACT == 1:
            acc = gelu(acc)
        if HAS_SCALE:
            acc = acc * tl.load(SCALE + rn, mask=nm, other=0.0)[None, :]
        cm = mm[:, None] & nm[None, :]
        c_off = rm64[:, None] * N + rn[None, :]
        if HAS_RES:
            acc += tl.load(R + c_off, mask=cm, other=0.0).to(tl.float32)
        tl.store(C + c_off, acc.to(C.dtype.element_ty), mask=cm)


@triton.jit
def row_stats_kernel(X, OUT, M, K, eps, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """Exact two-pass mean / rstd of each row of X[M, K] (K <= BLOCK_K)."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    mask = (rm < M)[:, None] & (rk < K)[None, :]
    x = tl.load(X + rm.to(tl.int64)[:, None] * K + rk[None, :], mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=1) / K
    d = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / K
    rstd = tl.rsqrt(var + eps)
    rmask = rm < M
    tl.store(OUT + rm.to(tl.int64) * 2, mean, mask=rmask)
    tl.store(OUT + rm.to(tl.int64) * 2 + 1, rstd, mask=rmask)


def row_stats(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    k = x.shape[-1]
    m = x.numel() // k
    out = torch.empty((m, 2), device=x.device, dtype=torch.float32)
    bk = triton.next_power_of_2(k)
    bm = max(1, min(64, 8192 // bk))
    row_stats_kernel[(triton.cdiv(m, bm),)](x, out, m, k, eps, BLOCK_M=bm, BLOCK_K=bk,
                                             num_warps=4 if bk <= 256 else 8)
    return out


_BASE = [
    (128, 128, 32, 4, 4), (256, 128, 32, 8, 3), (128, 256, 32, 8, 3), (128, 128, 32, 8, 3),
    (64, 128, 32, 4, 4), (128, 64, 32, 4, 4), (128, 64, 64, 4, 3), (64, 64, 32, 2, 5),
    (128, 32, 32, 4, 4), (64, 32, 32, 2, 4), (128, 16, 32, 4, 4), (64, 16, 64, 2, 3),
]


def _candidates(m, n, k, gate, pro=0, c_mode=0):
    bn_max = max(16, min(256, p2(n)))
    if c_mode:
        cout = n // 4
        assert cout % 16 == 0, "pixel-shuffle GEMM needs output channels % 16 == 0"
        bn_max = min(bn_max, cout & -cout)
    bk_max = max(16, min(64, p2(k // 4 if pro == -1 else k)))
    out = []
    if pro in (4, 6):
        for bm, bn, w in [(128, 64, 8), (64, 64, 4), (64, 128, 8), (128, 128, 8), (32, 64, 4), (64, 32, 4)]:
            if bm * p2(k) > 128 * 128 or (gate and bm * bn > 64 * 64):
                continue
            c = cfg(bm, min(bn, bn_max), max(16, p2(k)), w, 2, GROUP_M=8)
            if c not in out:
                out.append(c)
        return out
    for bm, bn, bk, w, st in _BASE:
        bn, bk = min(bn, bn_max), min(bk, bk_max)
        if bm > max(64, p2(m)):
            continue
        if gate and bn * bm > 128 * 128:
            continue  # two accumulators
        if c_mode and (n // 4) % bn:
            continue
        c = cfg(bm, bn, bk, w, st, GROUP_M=8)
        if c not in out:
            out.append(c)
    return out


def fused_gemm(a, b, *, n, k, m, out, bias=None, scale=None, residual=None, small=None, stats=None,
               m2=None, lnw=None, lnb=None, m2_act=0, geom=(1, 1, 1, 1, 0, 0), a_scale=1.0, ln_eps=1e-6,
               lda=None, pro=0, a_mode=0, c_mode=0, gate=False, act=0):
    dummy = out
    h, w, oh, ow, pt, pl = geom
    if pro == 1 and k <= 128:
        pro = 4
    if (pro in (0, 4, 6) and a_mode == 0 and c_mode == 0 and k <= 128 and a.dtype != torch.float32
            and n > 64 and small is None and stats is None and m2 is None):
        return _gemm_wide(a, b, n=n, k=k, m=m, out=out, bias=bias, scale=scale, residual=residual, ln_eps=ln_eps,
                          lda=lda, pro=pro, gate=gate, act=act)
    if pro == 0 and a.dtype == torch.float32:
        a_scale = 1.0 / 256.0
    args = (
        a, b, out,
        bias if bias is not None else dummy,
        scale if scale is not None else dummy,
        residual if residual is not None else dummy,
        small if small is not None else dummy,
        stats if stats is not None else dummy,
        m2 if m2 is not None else dummy,
        lnw if lnw is not None else dummy,
        lnb if lnb is not None else dummy,
        m, n, k,
        h, w, oh, ow, pt, pl,
        float(a_scale), float(ln_eps), int(lda if lda is not None else k),
        int(m2.stride(-2) if m2 is not None and m2.dim() >= 2 else k),
    )
    flags = dict(PRO=pro, A_MODE=a_mode, C_MODE=c_mode, M2_ACT=m2_act,
                 HAS_BIAS=bias is not None, GATE=gate, ACT=act, HAS_SCALE=scale is not None,
                 HAS_RES=residual is not None)

    def launch(c):
        grid = (triton.cdiv(m, c["BLOCK_M"]) * triton.cdiv(n, c["BLOCK_N"]),)
        fused_gemm_kernel[grid](*args, **flags, **c)

    # the two zeros keep the key layout of earlier versions, so cached tuning stays valid
    key = (m.bit_length(), n, k, pro, a_mode, c_mode, 0, int(gate), act, 0, str(b.dtype))
    cands = _candidates(m, n, k, gate, -1 if a_mode else pro, c_mode)
    launch(TUNER.pick("gemm2", key, cands, launch))
    return out


def _gemm_wide(a, b, *, n, k, m, out, bias, scale, residual, ln_eps, lda, pro, gate, act):
    dummy = out
    bk = max(16, p2(k))
    bn_max = max(16, min(256, p2(n)))

    def launch(c):
        gemm_wide_kernel[(triton.cdiv(m, c["BLOCK_M"]),)](
            a, b, out, bias if bias is not None else dummy, scale if scale is not None else dummy,
            residual if residual is not None else dummy, m, n, k, float(ln_eps), int(lda if lda is not None else k),
            PRO=pro, GATE=gate, ACT=act, HAS_BIAS=bias is not None, HAS_SCALE=scale is not None,
            HAS_RES=residual is not None, BLOCK_M=c["BLOCK_M"], BLOCK_N=c["BLOCK_N"], BLOCK_K=bk,
            num_warps=c["num_warps"], num_stages=c["num_stages"])

    cands = []
    for bm, bn, nw, st in [(128, 128, 4, 2), (64, 64, 2, 2), (64, 128, 4, 2), (128, 64, 8, 2), (64, 64, 4, 2),
                           (128, 128, 8, 2), (64, 128, 8, 2), (32, 64, 2, 2), (128, 64, 4, 3), (32, 128, 4, 2),
                           (256, 64, 8, 2), (64, 256, 8, 2), (128, 32, 4, 3), (64, 32, 2, 3)]:
        if gate and bm * bn > 128 * 64:
            continue
        c = dict(BLOCK_M=bm, BLOCK_N=min(bn, bn_max), num_warps=nw, num_stages=st)
        if c not in cands:
            cands.append(c)
    key = (m.bit_length(), n, k, pro, int(gate), act, int(bias is not None), int(scale is not None),
           int(residual is not None), str(b.dtype))
    launch(TUNER.pick("gemm_wide3", key, cands, launch))
    return out
