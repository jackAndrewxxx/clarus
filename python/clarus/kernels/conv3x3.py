"""Dense 3x3 convolutions (stride 1, pad 1) for NHWC activations.

conv3x3():  implicit GEMM, fp32 inputs pre-scaled by 2^-8 into fp16, weights padded to a
            multiple of 16 output channels, and an NCHW fp32 output mode with fused
            `+ inp` residual (the network's final `ending` conv).
intro3x3(): 3 -> C conv reading the NCHW fp32 image directly (27 FMAs / output).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .tuning import TUNER, cfg, p2


@triton.jit
def conv3x3_kernel(X1, Wk, BIAS, RES, OUT, M, H, W, C1, COUT, LDW, a_scale, out_scale, slope,
                   LDX1, LDO, LDR,
                   HAS_RES: tl.constexpr, OUT_NCHW: tl.constexpr, A_FP32: tl.constexpr,
                   ACT: tl.constexpr, UP2: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """out = act(conv3x3(x1) + bias) * out_scale (+ res).

    H, W are output sizes; with UP2 the inputs are (H/2, W/2) and read through a nearest x2
    upsample. LDX1 / LDO / LDR are per-pixel strides (elements), so inputs and outputs
    can be channel slices of a wider NHWC buffer (dense blocks without concatenation).
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mmask = rm < M
    nmask = rn < COUT
    wmask = rn < LDW
    hw = H * W
    img = rm // hw
    r = rm % hw
    y = r // W
    x = r % W
    if UP2:
        HS = H // 2
        WS = W // 2
    else:
        HS = H
        WS = W
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # one flat loop over (tap, K chunk) so the software pipeliner (num_stages) overlaps the loads
    # of the next chunk with the current MMA even when C1 fits in a single chunk
    nk = tl.cdiv(C1, BLOCK_K)
    for it in range(0, 9 * nk):
        t = it // nk
        k0 = (it % nk) * BLOCK_K
        yy = y + (t // 3) - 1
        xx = x + (t % 3) - 1
        valid = mmask & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        if UP2:
            yy = yy // 2
            xx = xx // 2
        pix = img.to(tl.int64) * (HS * WS) + yy * WS + xx
        kk = k0 + rk
        a = tl.load(X1 + pix[:, None] * LDX1 + kk[None, :], mask=valid[:, None] & (kk < C1)[None, :], other=0.0)
        if A_FP32:   # raw fp32 residual-stream operands are pre-scaled by 2^-8 to fit fp16
            a = (a * a_scale).to(Wk.dtype.element_ty)
        else:
            a = a.to(Wk.dtype.element_ty)
        b = tl.load(Wk + (t * C1 + kk)[:, None] * LDW + rn[None, :],
                    mask=(kk < C1)[:, None] & wmask[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    if A_FP32:
        acc = acc * (1.0 / a_scale)
    acc += tl.load(BIAS + rn, mask=nmask, other=0.0)[None, :]
    if ACT == 1:
        acc = tl.where(acc >= 0, acc, acc * slope)
    acc = acc * out_scale
    omask = mmask[:, None] & nmask[None, :]
    if OUT_NCHW:
        optr = (img.to(tl.int64) * COUT * hw + r)[:, None] + rn[None, :].to(tl.int64) * hw
        rptr = optr
    else:
        optr = rm.to(tl.int64)[:, None] * LDO + rn[None, :]
        rptr = rm.to(tl.int64)[:, None] * LDR + rn[None, :]
    if HAS_RES:
        acc += tl.load(RES + rptr, mask=omask, other=0.0).to(tl.float32)
    tl.store(OUT + optr, acc.to(OUT.dtype.element_ty), mask=omask)


def _pix_stride(t: torch.Tensor) -> int:
    """Per-pixel stride of an NHWC tensor that may be a channel slice of a wider buffer."""
    n, h, w, c = t.shape
    s = t.stride()
    if s[3] != 1 or s[1] != w * s[2] or (n > 1 and s[0] != h * w * s[2]):
        raise ValueError(f"NHWC tensor with unsupported strides {s}")
    return s[2]


def pad_cols(wk: torch.Tensor) -> torch.Tensor:
    """Zero-pad a [K, N] weight to N % 16 == 0 so B-tile rows are aligned."""
    n = wk.shape[1]
    np_ = max(16, (n + 15) // 16 * 16)
    if np_ == n:
        return wk.contiguous()
    out = torch.zeros((wk.shape[0], np_), device=wk.device, dtype=wk.dtype)
    out[:, :n] = wk
    return out


def conv3x3(x1, wk, bias, cout, residual=None, out=None, out_dtype=None, nchw_out=False,
            act=0, out_scale=1.0, up2=False, slope=0.2):
    """x1 [N,H,W,C1]; wk [(tap, C1), LDW >= cout] (see pad_cols).

    act: 0 none, 1 LeakyReLU(slope). up2: inputs are nearest-upsampled x2 on the fly.
    x1 / out / residual may be channel slices of wider NHWC buffers.
    """
    n, h, w, c1 = x1.shape
    if up2:
        h, w = 2 * h, 2 * w
    ldw = wk.shape[1]
    m = n * h * w
    if out is None:
        shape = (n, cout, h, w) if nchw_out else (n, h, w, cout)
        out = torch.empty(shape, device=x1.device, dtype=out_dtype or x1.dtype)
    a_scale = 1.0 / 256.0 if x1.dtype == torch.float32 else 1.0
    flags = dict(HAS_RES=residual is not None, OUT_NCHW=nchw_out, ACT=int(act), UP2=bool(up2),
                 A_FP32=x1.dtype == torch.float32)
    ldx1 = _pix_stride(x1)
    ldo = 0 if nchw_out else _pix_stride(out)
    ldr = _pix_stride(residual) if (residual is not None and not nchw_out) else 0

    def launch(c):
        grid = (triton.cdiv(m, c["BLOCK_M"]), triton.cdiv(cout, c["BLOCK_N"]))
        conv3x3_kernel[grid](x1, wk, bias, residual if residual is not None else out, out,
                             m, h, w, c1, cout, ldw, a_scale, float(out_scale), float(slope), ldx1, ldo, ldr,
                             **flags, **c)

    cands = []
    for a in [(128, 32, 32, 4, 3), (128, 32, 64, 4, 3), (64, 32, 64, 4, 3), (256, 32, 32, 8, 3),
              (128, 64, 32, 4, 3), (64, 64, 64, 4, 3), (128, 16, 32, 4, 3), (128, 16, 64, 4, 3),
              (256, 16, 32, 8, 3), (64, 16, 64, 4, 3), (128, 64, 64, 8, 3), (64, 32, 32, 4, 4),
              (128, 128, 32, 8, 3), (128, 64, 32, 4, 4), (128, 64, 64, 4, 4), (128, 32, 64, 4, 4),
              (256, 64, 32, 8, 4), (64, 64, 32, 4, 5), (128, 128, 64, 8, 3), (256, 32, 64, 8, 3), (64, 32, 64, 4, 4),
              (128, 32, 32, 4, 5)]:
        bm, bn, bk, nw, ns = a
        c = cfg(bm, min(bn, max(16, p2(ldw))), min(bk, max(16, p2(c1))), nw, ns)
        if c not in cands:
            cands.append(c)
    key = (m.bit_length(), c1, 0, cout, ldw, int(residual is not None), int(nchw_out),
           str(x1.dtype), str(wk.dtype), str(out.dtype)) + ((int(act), int(up2)) if (act or up2) else ())
    launch(TUNER.pick("conv3x3g", key, cands, launch))
    return out


@triton.jit
def intro_kernel(X, Wt, BIAS, OUT, M, H, W, COUT, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr):
    """X NCHW fp32 [N,3,H,W]; Wt [27, COUT] fp32 ordered (ky, kx, c); OUT NHWC [N,H,W,COUT]."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    nmask = rn < COUT
    mmask = rm < M
    hw = H * W
    img = rm // hw
    r = rm % hw
    y = r // W
    x = r % W
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for t in tl.static_range(9):
        yy = y + (t // 3) - 1
        xx = x + (t % 3) - 1
        valid = mmask & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        for c in tl.static_range(3):
            v = tl.load(X + (img.to(tl.int64) * 3 + c) * hw + yy * W + xx, mask=valid, other=0.0)
            acc += v[:, None] * tl.load(Wt + (t * 3 + c) * COUT + rn, mask=nmask, other=0.0)[None, :]
    acc += tl.load(BIAS + rn, mask=nmask, other=0.0)[None, :]
    tl.store(OUT + rm.to(tl.int64)[:, None] * COUT + rn[None, :], acc.to(OUT.dtype.element_ty),
             mask=mmask[:, None] & nmask[None, :])


def intro3x3(inp, wt, bias, out_dtype):
    n, _, h, w = inp.shape
    cout = wt.shape[1]
    m = n * h * w
    out = torch.empty((n, h, w, cout), device=inp.device, dtype=out_dtype)
    bm = 64
    intro_kernel[(triton.cdiv(m, bm),)](inp, wt, bias, out, m, h, w, cout, BLOCK_N=p2(cout), BLOCK_M=bm,
                                        num_warps=4)
    return out
