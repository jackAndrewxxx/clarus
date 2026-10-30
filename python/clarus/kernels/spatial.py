"""Spatial NHWC kernels: depthwise conv, TLC box mean, row LayerNorm, local-mean gather,
bilinear resize."""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .tuning import TUNER


# ----------------------------------------------------------------- depthwise
@triton.jit
def _gelu(x):
    """Exact (erf) GELU with erf from Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7, far below the
    fp16 / bf16 output precision): a handful of FMAs and one exp instead of libdevice erf."""
    z = tl.abs(x) * 0.7071067811865476
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    e = 1.0 - poly * tl.exp(-z * z)
    erf = tl.where(x >= 0, e, -e)
    return 0.5 * x * (1.0 + erf)


def _gate_code(gate) -> int:
    """False -> 0 (none), True / "mul" -> 1 (x1 * x2), "gelu" -> 2 (gelu(x1) * x2)."""
    if gate is False or gate is None:
        return 0
    if gate is True or gate == "mul":
        return 1
    if gate == "gelu":
        return 2
    raise ValueError(f"unknown gate {gate!r}")


@triton.jit
def dwconv_kernel(X, Wt, BIAS, OUT, H, W, C, COUT,
                  KH: tl.constexpr, KW: tl.constexpr, DIL: tl.constexpr,
                  GATE: tl.constexpr, ACT: tl.constexpr,
                  BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr):
    """X [N,H,W,C]; Wt [KH*KW, C] fp32; OUT [N,H,W,COUT] (COUT = C//2 if GATE)."""
    pid_row = tl.program_id(0)  # n*H*cdiv(W) + ...
    pid_c = tl.program_id(1)
    nwb = tl.cdiv(W, BLOCK_W)
    row = pid_row // nwb
    wb = pid_row % nwb
    n = row // H
    y = row % H
    xs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cs < COUT
    acc = tl.zeros((BLOCK_W, BLOCK_C), dtype=tl.float32)
    if GATE:
        acc2 = tl.zeros((BLOCK_W, BLOCK_C), dtype=tl.float32)
    base = X + n.to(tl.int64) * H * W * C
    for ky in tl.static_range(KH):
        yy = y + (ky - KH // 2) * DIL
        ymask = (yy >= 0) & (yy < H)
        for kx in tl.static_range(KW):
            xx = xs + (kx - KW // 2) * DIL
            pmask = ymask & (xx >= 0) & (xx < W)
            m2 = pmask[:, None] & cmask[None, :]
            ptr = base + (yy * W + xx).to(tl.int64)[:, None] * C + cs[None, :]
            v = tl.load(ptr, mask=m2, other=0.0).to(tl.float32)
            wv = tl.load(Wt + (ky * KW + kx) * C + cs, mask=cmask, other=0.0)
            acc += v * wv[None, :]
            if GATE:
                v2 = tl.load(ptr + COUT, mask=m2, other=0.0).to(tl.float32)
                wv2 = tl.load(Wt + (ky * KW + kx) * C + COUT + cs, mask=cmask, other=0.0)
                acc2 += v2 * wv2[None, :]
    acc += tl.load(BIAS + cs, mask=cmask, other=0.0)[None, :]
    if GATE:
        acc2 += tl.load(BIAS + COUT + cs, mask=cmask, other=0.0)[None, :]
        if GATE == 2:
            acc = _gelu(acc)
        acc = acc * acc2
    if ACT:
        acc = _gelu(acc)
    omask = (xs < W)[:, None] & cmask[None, :]
    optr = OUT + ((row.to(tl.int64) * W + xs)[:, None]) * COUT + cs[None, :]
    tl.store(optr, acc.to(OUT.dtype.element_ty), mask=omask)


@triton.jit
def _row3(base, off, C, mm, m0, mp, rv):
    """The three horizontally shifted [BLOCK_W, BLOCK_C] tiles of one input row. `off` and the
    masks are precomputed per program, so a row costs one pointer add per element and load."""
    p = base + off
    vm = tl.load(p - C, mask=mm & rv, other=0.0).to(tl.float32)
    v0 = tl.load(p, mask=m0 & rv, other=0.0).to(tl.float32)
    vp = tl.load(p + C, mask=mp & rv, other=0.0).to(tl.float32)
    return vm, v0, vp


@triton.jit
def dw3x3_rows_kernel(X, Wt, BIAS, OUT, H, W, C, COUT, ROWS,
                      GATE: tl.constexpr, ACT: tl.constexpr,
                      BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr):
    """Depthwise 3x3 (pad 1) with a rolling 3-row line buffer; optional SimpleGate / scale."""
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)
    nwb = tl.cdiv(W, BLOCK_W)
    nrb = tl.cdiv(H, ROWS)
    wb = pid % nwb
    t = pid // nwb
    rb = t % nrb
    n = t // nrb
    xs = wb * BLOCK_W + tl.arange(0, BLOCK_W)
    cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cs < COUT
    xmask = xs < W
    base_n = X + n.to(tl.int64) * H * W * C
    out_n = OUT + n.to(tl.int64) * H * W * COUT
    a00 = tl.load(Wt + 0 * C + cs, mask=cmask, other=0.0)[None, :]
    a01 = tl.load(Wt + 1 * C + cs, mask=cmask, other=0.0)[None, :]
    a02 = tl.load(Wt + 2 * C + cs, mask=cmask, other=0.0)[None, :]
    a10 = tl.load(Wt + 3 * C + cs, mask=cmask, other=0.0)[None, :]
    a11 = tl.load(Wt + 4 * C + cs, mask=cmask, other=0.0)[None, :]
    a12 = tl.load(Wt + 5 * C + cs, mask=cmask, other=0.0)[None, :]
    a20 = tl.load(Wt + 6 * C + cs, mask=cmask, other=0.0)[None, :]
    a21 = tl.load(Wt + 7 * C + cs, mask=cmask, other=0.0)[None, :]
    a22 = tl.load(Wt + 8 * C + cs, mask=cmask, other=0.0)[None, :]
    bias = tl.load(BIAS + cs, mask=cmask, other=0.0)[None, :]
    if GATE:
        g00 = tl.load(Wt + 0 * C + COUT + cs, mask=cmask, other=0.0)[None, :]
        g01 = tl.load(Wt + 1 * C + COUT + cs, mask=cmask, other=0.0)[None, :]
        g02 = tl.load(Wt + 2 * C + COUT + cs, mask=cmask, other=0.0)[None, :]
        g10 = tl.load(Wt + 3 * C + COUT + cs, mask=cmask, other=0.0)[None, :]
        g11 = tl.load(Wt + 4 * C + COUT + cs, mask=cmask, other=0.0)[None, :]
        g12 = tl.load(Wt + 5 * C + COUT + cs, mask=cmask, other=0.0)[None, :]
        g20 = tl.load(Wt + 6 * C + COUT + cs, mask=cmask, other=0.0)[None, :]
        g21 = tl.load(Wt + 7 * C + COUT + cs, mask=cmask, other=0.0)[None, :]
        g22 = tl.load(Wt + 8 * C + COUT + cs, mask=cmask, other=0.0)[None, :]
        gbias = tl.load(BIAS + COUT + cs, mask=cmask, other=0.0)[None, :]
    y0 = rb * ROWS
    y1 = tl.minimum(y0 + ROWS, H)
    acc_a = tl.zeros((BLOCK_W, BLOCK_C), dtype=tl.float32)
    acc_b = tl.zeros((BLOCK_W, BLOCK_C), dtype=tl.float32)
    gacc_a = tl.zeros((BLOCK_W, BLOCK_C), dtype=tl.float32)
    gacc_b = tl.zeros((BLOCK_W, BLOCK_C), dtype=tl.float32)
    m2base = xmask[:, None] & cmask[None, :]
    off = (xs * C)[:, None] + cs[None, :]
    ooff = (xs * COUT)[:, None] + cs[None, :]
    mm = m2base & (xs >= 1)[:, None]
    mp = m2base & (xs + 1 < W)[:, None]
    # software pipelining: row r + 1 is loaded before row r's result is stored, so the loads
    # are not serialized behind the store (the compiler cannot prove OUT does not alias X)
    rv = (y0 - 1 >= 0) & (y0 - 1 < H)
    rb_ptr = base_n + (y0 - 1).to(tl.int64) * W * C
    vm, v0, vp = _row3(rb_ptr, off, C, mm, m2base, mp, rv)
    if GATE:
        um, u0, up = _row3(rb_ptr + COUT, off, C, mm, m2base, mp, rv)
    for r in range(y0 - 1, y1 + 1):
        nr = r + 1
        nrv = (nr < H) & (nr <= y1)
        n_ptr = base_n + nr.to(tl.int64) * W * C
        nvm, nv0, nvp = _row3(n_ptr, off, C, mm, m2base, mp, nrv)
        if GATE:
            num, nu0, nup = _row3(n_ptr + COUT, off, C, mm, m2base, mp, nrv)
        out_v = acc_a + (a20 * vm + a21 * v0 + a22 * vp)
        acc_a = acc_b + (a10 * vm + a11 * v0 + a12 * vp)
        acc_b = a00 * vm + a01 * v0 + a02 * vp
        res = out_v + bias
        if GATE:
            gout = gacc_a + (g20 * um + g21 * u0 + g22 * up)
            gacc_a = gacc_b + (g10 * um + g11 * u0 + g12 * up)
            gacc_b = g00 * um + g01 * u0 + g02 * up
            if GATE == 2:
                res = _gelu(res)
            res = res * (gout + gbias)
            um, u0, up = num, nu0, nup
        vm, v0, vp = nvm, nv0, nvp
        if ACT:
            res = _gelu(res)
        yo = r - 1
        tl.store(out_n + yo.to(tl.int64) * W * COUT + ooff, res.to(OUT.dtype.element_ty), mask=m2base & (yo >= y0))


def dwconv(x, wt, bias, kh, kw, dil=1, gate=False, out=None, act=None):
    """gate: False | True / "mul" (SimpleGate) | "gelu" (gelu(x1) * x2, Restormer GDFN);
    act: None | "gelu" (applied to the output, EVSSM SS2D)."""
    gate = _gate_code(gate)
    act = 1 if act == "gelu" else 0
    n, h, w, c = x.shape
    cout = c // 2 if gate else c
    if out is None:
        out = torch.empty((n, h, w, cout), device=x.device, dtype=x.dtype)
    if kh == 3 and kw == 3 and dil == 1:
        flags = dict(GATE=gate, ACT=act)

        def launch(cf):
            bw, bc, rows = cf["BLOCK_W"], cf["BLOCK_C"], cf["ROWS"]
            grid = (n * triton.cdiv(h, rows) * triton.cdiv(w, bw), triton.cdiv(cout, bc))
            dw3x3_rows_kernel[grid](x, wt, bias, out, h, w, c, cout, rows, **flags,
                                    BLOCK_W=bw, BLOCK_C=bc, num_warps=cf["num_warps"])

        cands = []
        for bw, bc, rows, nw in [(32, 32, 16, 4), (32, 64, 16, 4), (16, 64, 32, 4), (64, 32, 16, 4),
                                 (16, 128, 16, 4), (32, 64, 32, 8), (8, 128, 32, 4), (64, 64, 8, 8),
                                 (16, 32, 32, 2), (32, 32, 64, 4), (16, 32, 64, 4), (16, 64, 64, 4),
                                 (32, 64, 64, 4), (16, 64, 64, 2), (8, 64, 64, 2)]:
            cf = dict(BLOCK_W=bw, BLOCK_C=min(bc, triton.next_power_of_2(cout)), ROWS=rows, num_warps=nw)
            if cf not in cands:
                cands.append(cf)
        # size-bucketed key: new image shapes reuse tuned configs (the 0 keeps the earlier key layout)
        key = ((n * h * w).bit_length(), w.bit_length(), c, min(gate, 1), 0, str(x.dtype)) + ((act,) if act else ())
        launch(TUNER.pick("dw3x3p", key, cands, launch))
        return out
    bc = min(64, triton.next_power_of_2(cout))
    bw = max(8, min(64, 2048 // bc))
    grid = (n * h * triton.cdiv(w, bw), triton.cdiv(cout, bc))
    dwconv_kernel[grid](x, wt, bias, out, h, w, c, cout, KH=kh, KW=kw, DIL=dil, GATE=gate, ACT=act,
                        BLOCK_W=bw, BLOCK_C=bc, num_warps=4)
    return out


# ------------------------------------------------------------------ box mean
@triton.jit
def box_axis_kernel(X, OUT, L, K, OL, I, SXO, SXA, SOO, SOA, scale,
                    BLOCK_A: tl.constexpr, BLOCK_I: tl.constexpr):
    """Sliding-window sum of length K along one axis, all rows of a tile at once.

    X[o, a, i] (strides SXO, SXA, 1) -> OUT[o, j, i] = scale * sum_{a=j}^{j+K-1} X[o, a, i],
    j in [0, OL). Two running inclusive-prefix cursors (lag at j-1, lead at j+K-1)
    advance by BLOCK_A rows per step using tl.cumsum, all accumulation in fp32.
    """
    pid_i = tl.program_id(0)
    o = tl.program_id(1).to(tl.int64)
    ii = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    imask = ii < I
    ra = tl.arange(0, BLOCK_A)
    xb = X + o * SXO
    ob = OUT + o * SOO
    lead = tl.zeros((BLOCK_I,), dtype=tl.float32)
    for a0 in range(0, K - 1, BLOCK_A):
        rows = a0 + ra
        m = (rows < K - 1)[:, None] & imask[None, :]
        v = tl.load(xb + rows.to(tl.int64)[:, None] * SXA + ii[None, :], mask=m, other=0.0).to(tl.float32)
        lead += tl.sum(v, axis=0)
    lag = tl.zeros((BLOCK_I,), dtype=tl.float32)
    for j0 in range(0, OL, BLOCK_A):
        rows = j0 - 1 + ra
        m = ((rows >= 0) & (rows < L))[:, None] & imask[None, :]
        v = tl.load(xb + rows.to(tl.int64)[:, None] * SXA + ii[None, :], mask=m, other=0.0).to(tl.float32)
        q_lag = lag[None, :] + tl.cumsum(v, axis=0)
        lag += tl.sum(v, axis=0)
        rows = j0 + K - 1 + ra
        m = (rows < L)[:, None] & imask[None, :]
        v = tl.load(xb + rows.to(tl.int64)[:, None] * SXA + ii[None, :], mask=m, other=0.0).to(tl.float32)
        q_lead = lead[None, :] + tl.cumsum(v, axis=0)
        lead += tl.sum(v, axis=0)
        orow = j0 + ra
        om = (orow < OL)[:, None] & imask[None, :]
        tl.store(ob + orow.to(tl.int64)[:, None] * SOA + ii[None, :],
                 ((q_lead - q_lag) * scale).to(OUT.dtype.element_ty), mask=om)


def boxmean(x, k1, k2, oh, ow, scratch=None, out=None):
    """TLC local mean: [N,H,W,C] -> [N,OH,OW,C], window k1 x k2, valid positions only."""
    n, h, w, c = x.shape
    if scratch is None:
        # column means (not sums) are stored, so the activation dtype has ample range
        scratch = torch.empty((n, oh, w, c), device=x.device, dtype=x.dtype)
    if out is None:
        out = torch.empty((n, oh, ow, c), device=x.device, dtype=x.dtype)

    def launch_v(cf):  # along H; inner = W*C contiguous
        inner = w * c
        grid = (triton.cdiv(inner, cf["BLOCK_I"]), n)
        box_axis_kernel[grid](x, scratch, h, k1, oh, inner, h * inner, inner, oh * inner, inner, 1.0 / k1, **cf)

    def launch_h(cf):  # along W; inner = C
        grid = (triton.cdiv(c, cf["BLOCK_I"]), n * oh)
        box_axis_kernel[grid](scratch, out, w, k2, ow, c, w * c, c, ow * c, c, 1.0 / k2, **cf)

    key = ((n * h * w).bit_length(), h.bit_length(), w.bit_length(), c, k1.bit_length(), k2.bit_length(),
           str(x.dtype), str(scratch.dtype))
    cv = [dict(BLOCK_A=a, BLOCK_I=i, num_warps=nw)
          for a, i, nw in [(32, 64, 4), (16, 128, 4), (32, 128, 4), (64, 64, 4), (16, 64, 2), (8, 256, 4),
                           (16, 256, 4), (32, 32, 2)]]
    launch_v(TUNER.pick("boxvb", key, cv, launch_v))
    bi = triton.next_power_of_2(c)
    ch = []
    for a, i, nw in [(32, 64, 4), (64, 64, 4), (16, 128, 4), (32, 128, 4), (64, 32, 4), (128, 16, 4),
                     (64, 16, 2), (32, 256, 4), (16, 256, 4)]:
        cf = dict(BLOCK_A=a, BLOCK_I=min(i, bi), num_warps=nw)
        if cf not in ch:
            ch.append(cf)
    launch_h(TUNER.pick("boxhb", key, ch, launch_h))
    return out


# ---------------------------------------------------------- row LayerNorms
@triton.jit
def ln_rows_kernel(X, OUT, M, K, eps, RMS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """Rows of X[M, K] normalized: LayerNorm without affine (affine folded elsewhere), or
    x / rms(x) (RMS)."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    mask = (rm < M)[:, None] & (rk < K)[None, :]
    ptr = rm.to(tl.int64)[:, None] * K + rk[None, :]
    x = tl.load(X + ptr, mask=mask, other=0.0).to(tl.float32)
    if RMS:
        d = x
    else:
        d = tl.where(mask, x - (tl.sum(x, axis=1) / K)[:, None], 0.0)
    out = d * tl.rsqrt(tl.sum(d * d, axis=1) / K + eps)[:, None]
    tl.store(OUT + ptr, out.to(OUT.dtype.element_ty), mask=mask)


def ln_rows(x, rms=False, out=None, eps=1e-6, out_dtype=None):
    n, h, w, k = x.shape
    m = n * h * w
    if out is None:
        out = torch.empty((n, h, w, k), device=x.device, dtype=out_dtype or x.dtype)
    bk = triton.next_power_of_2(k)
    bm = max(1, min(64, 4096 // bk))
    ln_rows_kernel[(triton.cdiv(m, bm),)](x, out, m, k, eps, RMS=rms, BLOCK_M=bm, BLOCK_K=bk,
                                         num_warps=4 if bk <= 256 else 8)
    return out


# ------------------------------------------------- x * expand(local average)
@triton.jit
def mul_small_kernel(X, S, OUT, M, H, W, C, OH, OW, PT, PL,
                     BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (rm < M)[:, None] & (cs < C)[None, :]
    hw = H * W
    img = rm // hw
    r = rm % hw
    sy = tl.minimum(tl.maximum(r // W - PT, 0), OH - 1)
    sx = tl.minimum(tl.maximum(r % W - PL, 0), OW - 1)
    srow = ((img * OH + sy) * OW + sx).to(tl.int64)
    x = tl.load(X + rm.to(tl.int64)[:, None] * C + cs[None, :], mask=mask, other=0.0).to(tl.float32)
    s = tl.load(S + srow[:, None] * C + cs[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(OUT + rm.to(tl.int64)[:, None] * C + cs[None, :], (x * s).to(OUT.dtype.element_ty), mask=mask)


def mul_small(x, s, pt, pl, out=None):
    n, h, w, c = x.shape
    _, oh, ow, _ = s.shape
    m = n * h * w
    if out is None:
        out = torch.empty_like(x)
    bc = min(128, triton.next_power_of_2(c))
    bm = max(1, 4096 // bc)
    mul_small_kernel[(triton.cdiv(m, bm), triton.cdiv(c, bc))](x, s, out, m, h, w, c, oh, ow, pt, pl,
                                                              BLOCK_M=bm, BLOCK_C=bc, num_warps=4)
    return out


# ------------------------------------------------------------ bilinear resize
@triton.jit
def resize_bilinear_kernel(X, OUT, H, W, OH, OW, C, inv_scale,
                           BLOCK_W: tl.constexpr, BLOCK_C: tl.constexpr):
    """NHWC F.interpolate(mode="bilinear", align_corners=False, scale_factor=1/inv_scale)."""
    pid = tl.program_id(0)                      # n * OH + oy
    n = pid // OH
    oy = pid % OH
    xs = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
    cs = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    sy = tl.maximum((oy + 0.5) * inv_scale - 0.5, 0.0)
    y0 = sy.to(tl.int32)
    y1 = tl.minimum(y0 + 1, H - 1)
    ly = sy - y0
    sx = tl.maximum((xs + 0.5) * inv_scale - 0.5, 0.0)
    x0 = sx.to(tl.int32)
    x1 = tl.minimum(x0 + 1, W - 1)
    lx = (sx - x0)[:, None]
    m = (xs < OW)[:, None] & (cs < C)[None, :]
    base = X + n.to(tl.int64) * H * W * C
    r0 = base + y0.to(tl.int64) * W * C
    r1 = base + y1.to(tl.int64) * W * C
    v00 = tl.load(r0 + x0[:, None] * C + cs[None, :], mask=m, other=0.0).to(tl.float32)
    v01 = tl.load(r0 + x1[:, None] * C + cs[None, :], mask=m, other=0.0).to(tl.float32)
    v10 = tl.load(r1 + x0[:, None] * C + cs[None, :], mask=m, other=0.0).to(tl.float32)
    v11 = tl.load(r1 + x1[:, None] * C + cs[None, :], mask=m, other=0.0).to(tl.float32)
    top = v00 + (v01 - v00) * lx
    bot = v10 + (v11 - v10) * lx
    out = top + (bot - top) * ly
    tl.store(OUT + ((n.to(tl.int64) * OH + oy) * OW + xs)[:, None] * C + cs[None, :],
             out.to(OUT.dtype.element_ty), mask=m)


def resize_bilinear(x, scale, out_dtype=None):
    n, h, w, c = x.shape
    oh, ow = int(h * scale), int(w * scale)
    out = torch.empty((n, oh, ow, c), device=x.device, dtype=out_dtype or x.dtype)
    bc = min(64, triton.next_power_of_2(c))
    bw = max(8, min(64, 4096 // bc))
    resize_bilinear_kernel[(n * oh, triton.cdiv(ow, bw), triton.cdiv(c, bc))](
        x, out, h, w, oh, ow, c, 1.0 / scale, BLOCK_W=bw, BLOCK_C=bc, num_warps=4)
    return out
