"""Per-patch frequency-domain ops of FFTformer / EVSSM as dense DFT products on tensor cores.

The p x p patch rfft2 / irfft2 pair is linear:

  patch_filter:  y = I @ (g_c * (T @ x))      T, I: [p*p, p*p] real basis of the spectrum,
                                              g_c: the filter per real coefficient
  patch_corr:    y = Ir @ Re(P) + Ii @ Im(P),  P = (Tr q + i Ti q) * (Tr k + i Ti k)
                                              (= circular correlation, FSAS)

A program works on whole p x p patches (p*p pixels x a block of channels, NHWC) and runs the
DFT products with tl.dot (fp16 operands, fp32 accumulation); the filter is fused behind the
1x1 conv that produces its input (ln_gemm_patch_filter). Operands are normalized per channel
before each product so they always fit fp16.
"""
from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

from .tuning import TUNER


@functools.lru_cache(maxsize=None)
def _dft64(p: int):
    """float64 (Tr, Ti [nb, p*p], Ir, Ii [p*p, nb]) with nb = p * (p//2 + 1)."""
    nb = p * (p // 2 + 1)
    eye = torch.eye(p * p, dtype=torch.float64).view(p * p, p, p)
    f = torch.fft.rfft2(eye).reshape(p * p, nb)
    basis = torch.eye(nb, dtype=torch.float64).view(nb, p, p // 2 + 1).to(torch.complex128)
    ir = torch.fft.irfft2(basis, s=(p, p)).reshape(nb, p * p).t()
    ii = torch.fft.irfft2(1j * basis, s=(p, p)).reshape(nb, p * p).t()
    return f.real.t().contiguous(), f.imag.t().contiguous(), ir.contiguous(), ii.contiguous()


def _real_coeffs(p: int):
    """The p*p independent real coefficients of a real patch's rfft2: (u, v, 0=Re / 1=Im).
    Columns v = 0 and v = p/2 are Hermitian along u, so only u <= p/2 is kept there."""
    out = []
    for v in range(p // 2 + 1):
        for u in range(p):
            if v in (0, p // 2):
                if u in (0, p // 2):
                    out.append((u, v, 0))
                elif u < p // 2:
                    out += [(u, v, 0), (u, v, 1)]
            else:
                out += [(u, v, 0), (u, v, 1)]
    return out


@functools.lru_cache(maxsize=None)
def _real_basis(p: int):
    """float64 T [p*p coeffs, p*p px] (rfft2 restricted to the real coefficients) and its
    inverse I [p*p px, p*p coeffs] (irfft2 of the Hermitian-completed half spectrum)."""
    coef = _real_coeffs(p)
    eye = torch.eye(p * p, dtype=torch.float64).view(p * p, p, p)
    f = torch.fft.rfft2(eye)
    t = torch.stack([f[:, u, v].real if part == 0 else f[:, u, v].imag for u, v, part in coef])
    inv = torch.zeros(p * p, p * p, dtype=torch.float64)
    for j, (u, v, part) in enumerate(coef):
        y = torch.zeros(p, p // 2 + 1, dtype=torch.complex128)
        val = 1.0 if part == 0 else 1j
        y[u, v] = val
        if v in (0, p // 2) and u not in (0, p // 2):
            y[(p - u) % p, v] = complex(val).conjugate()
        inv[:, j] = torch.fft.irfft2(y, s=(p, p)).reshape(p * p)
    return t, inv


def filter_mats(p, device, dtype):
    """T [64, 64] / 8 and I [64, 64] * 8 in the real-coefficient basis (p = 8)."""
    t, inv = _real_basis(p)
    return (t / 8).to(device, dtype).contiguous(), (inv * 8).to(device, dtype).contiguous()


def filter_coeffs(f: torch.Tensor):
    """f [C, p, p//2+1] -> g [p*p, C] fp32, one gain per real coefficient.

    irfft2 only sees the Hermitian part of the v = 0 and v = p/2 columns (it takes the real
    part along W after the complex inverse along H), so a real filter that is not symmetric
    there, f(u) != f(p-u), acts exactly like its average (f(u) + f(p-u)) / 2."""
    p = f.shape[1]
    ff = f.float()
    rows = []
    for u, v, _ in _real_coeffs(p):
        if v in (0, p // 2) and u not in (0, p // 2):
            rows.append(0.5 * (ff[:, u, v] + ff[:, (p - u) % p, v]))
        else:
            rows.append(ff[:, u, v])
    return torch.stack(rows).contiguous()


def corr_mats(p, device, dtype):
    """(Tr, Ti) [64, 64] / 8 (bins zero-padded to 64 rows) and (Ir, Ii) [64, 64]."""
    tr, ti, ir, ii = _dft64(p)
    nb = tr.shape[0]

    def rows(m):
        o = torch.zeros(64, p * p, dtype=torch.float64)
        o[:nb] = m
        return (o / 8).to(device, dtype).contiguous()

    def cols(m):
        o = torch.zeros(p * p, 64, dtype=torch.float64)
        o[:, :nb] = m
        return o.to(device, dtype).contiguous()
    return rows(tr), rows(ti), cols(ir), cols(ii)


@triton.jit
def _patch_ptrs(n, py, px, H, W, LDX, P: tl.constexpr):
    pix = tl.arange(0, P * P)
    return ((n.to(tl.int64) * H + py * P + pix // P) * W + px * P + pix % P) * LDX


@triton.jit
def _colnorm(s):
    """per-column scale so that s / scale is within [-1, 1]."""
    return tl.maximum(tl.max(tl.abs(s), axis=0), 1e-30)


@triton.jit
def patch_corr_kernel(X, TR, TI, IR, II, OUT, H, W, C, NPX, LDX,
                      P: tl.constexpr, PPP: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)
    nhp = H // P
    nblk = tl.cdiv(NPX, PPP)
    n = pid // (nhp * nblk)
    rest = pid % (nhp * nblk)
    py = rest // nblk
    px0 = (rest % nblk) * PPP
    cs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    cmask = cs < C
    rb = tl.arange(0, 64)
    rp = tl.arange(0, P * P)
    tr = tl.load(TR + rb[:, None] * (P * P) + rp[None, :])
    ti = tl.load(TI + rb[:, None] * (P * P) + rp[None, :])
    ir = tl.load(IR + rp[:, None] * 64 + rb[None, :])
    ii = tl.load(II + rp[:, None] * 64 + rb[None, :])
    for q in range(PPP):
        px = px0 + q
        if px < NPX:
            off = _patch_ptrs(n, py, px, H, W, LDX, P)
            qv = tl.load(X + off[:, None] + cs[None, :], mask=cmask[None, :], other=0.0).to(tr.dtype)
            kv = tl.load(X + off[:, None] + C + cs[None, :], mask=cmask[None, :], other=0.0).to(tr.dtype)
            qr = tl.dot(tr, qv)
            qi = tl.dot(ti, qv)
            kr = tl.dot(tr, kv)
            ki = tl.dot(ti, kv)
            pr = qr * kr - qi * ki
            pi = qr * ki + qi * kr
            sc = tl.maximum(_colnorm(pr), _colnorm(pi))
            y = tl.dot(ir, (pr / sc[None, :]).to(tr.dtype))
            y = tl.dot(ii, (pi / sc[None, :]).to(tr.dtype), y) * (sc * 64.0)[None, :]
            ooff = ((n.to(tl.int64) * H + py * P + rp // P) * W + px * P + rp % P) * C
            tl.store(OUT + ooff[:, None] + cs[None, :], y.to(OUT.dtype.element_ty), mask=cmask[None, :])


def patch_corr(x, c, mats, p=8, out=None):
    n, h, w, _ = x.shape
    if out is None:
        out = torch.empty((n, h, w, c), device=x.device, dtype=torch.float32)
    npx = w // p
    tr, ti, ir, ii = mats

    def launch(cf):
        bc, ppp = cf["BLOCK_C"], cf["PPP"]
        grid = (n * (h // p) * triton.cdiv(npx, ppp), triton.cdiv(c, bc))
        patch_corr_kernel[grid](x, tr, ti, ir, ii, out, h, w, c, npx, x.stride(2), P=p, PPP=ppp, BLOCK_C=bc,
                                num_warps=cf["num_warps"])

    cands = [dict(BLOCK_C=bc, PPP=ppp, num_warps=nw) for bc, ppp, nw in
             [(32, 4, 4), (64, 4, 4), (64, 4, 8), (32, 8, 4), (16, 4, 4), (32, 2, 4), (64, 2, 8)]
             if bc <= max(16, triton.next_power_of_2(c))]
    key = ((n * h * w).bit_length(), c, str(x.dtype))
    launch(TUNER.pick("patch_corr", key, cands, launch))
    return out


@triton.jit
def ln_gemm_patch_filter_kernel(A, B, BIAS, T2, I2, FQ, R, OUT, H, W, K, N, NPX, ln_eps,
                                LN: tl.constexpr, HAS_RES: tl.constexpr, P: tl.constexpr, PPP: tl.constexpr,
                                BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    """patch_filter(LN(A) @ B + bias) (+ residual) for PPP p x p patches (along x) x BLOCK_N output
    channels per program (FFTformer DFFN project_in + filter, EVSSM EDFFN project_out + filter).

    Computed transposed, Y^T = ((B^T A^T) T^T * g^T) I^T with channels as rows, so each product's
    result is the *left* operand of the next one and stays in registers (no shared-memory round
    trip between the chained MMAs); per-channel normalization is then a row reduction. Patches
    crossing the bottom / right edge read reflected rows (= reflect padding) and store the
    valid part. LN != 0 needs K <= BLOCK_K (one K tile)."""
    pid = tl.program_id(0)
    nhp = tl.cdiv(H, P)
    nblk = tl.cdiv(NPX, PPP)
    n = pid // (nhp * nblk)
    rest = pid % (nhp * nblk)
    py = rest // nblk
    px0 = (rest % nblk) * PPP
    rp = tl.arange(0, P * P)
    rk = tl.arange(0, BLOCK_K)
    rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = rn < N
    tt = tl.load(T2 + rp[None, :] * (P * P) + rp[:, None])            # T^T [px, coef]
    it = tl.load(I2 + rp[None, :] * (P * P) + rp[:, None])            # I^T [coef, px]
    gt = tl.load(FQ + rp[None, :] * N + rn[:, None], mask=nm[:, None], other=0.0)   # g^T [BN, coef]
    bias = tl.load(BIAS + rn, mask=nm, other=0.0)
    yy = py * P + rp // P
    ry = tl.where(yy < H, yy, 2 * H - 2 - yy)
    for q in range(PPP):
        px = px0 + q
        if px < NPX:
            xx = px * P + rp % P
            valid = (yy < H) & (xx < W)
            row = (n.to(tl.int64) * H + ry) * W + tl.where(xx < W, xx, 2 * W - 2 - xx)
            orow = (n.to(tl.int64) * H + yy) * W + xx
            acc = tl.zeros((BLOCK_N, P * P), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                kk = k0 + rk
                km = kk < K
                at = tl.load(A + row[None, :] * K + kk[:, None], mask=km[:, None], other=0.0).to(tl.float32)
                if LN == 1:
                    mu = tl.sum(at, axis=0) / K
                    at = tl.where(km[:, None], at - mu[None, :], 0.0)
                if LN != 0:
                    at = at * tl.rsqrt(tl.sum(at * at, axis=0) / K + ln_eps)[None, :]
                bt = tl.load(B + kk[None, :] * N + rn[:, None], mask=km[None, :] & nm[:, None], other=0.0)
                acc = tl.dot(bt, at.to(bt.dtype), acc)
            acc += bias[:, None]
            s0 = tl.maximum(tl.max(tl.abs(acc), axis=1), 1e-30)
            sp = tl.dot((acc / s0[:, None]).to(tt.dtype), tt) * (gt * s0[:, None])
            sc = tl.maximum(tl.max(tl.abs(sp), axis=1), 1e-30)
            y = tl.dot((sp / sc[:, None]).to(tt.dtype), it) * sc[:, None]
            optr = orow[None, :] * N + rn[:, None]
            om = valid[None, :] & nm[:, None]
            if HAS_RES:
                y += tl.load(R + optr, mask=om, other=0.0).to(tl.float32)
            tl.store(OUT + optr, y.to(OUT.dtype.element_ty), mask=om)


def ln_gemm_patch_filter(x, w, bias, fq, t2, i2, ln=0, ln_eps=1e-5, residual=None, p=8):
    n, h, wd, k = x.shape
    nout = w.shape[1]
    out = torch.empty((n, h, wd, nout), device=x.device, dtype=x.dtype)
    npx = triton.cdiv(wd, p)
    bk = max(16, triton.next_power_of_2(k)) if ln else 64
    assert ln == 0 or k <= 128, "inline LN needs the whole row in one K tile"

    def launch(cf):
        bn, ppp = cf["BLOCK_N"], cf["PPP"]
        grid = (n * triton.cdiv(h, p) * triton.cdiv(npx, ppp), triton.cdiv(nout, bn))
        ln_gemm_patch_filter_kernel[grid](x, w, bias, t2, i2, fq, residual if residual is not None else out, out,
                                          h, wd, k, nout, npx, ln_eps, LN=ln, HAS_RES=residual is not None, P=p,
                                          PPP=ppp, BLOCK_K=bk, BLOCK_N=bn, num_warps=cf["num_warps"])

    bn_max = max(16, triton.next_power_of_2(nout))
    cands = []
    for bn, ppp, nw in [(64, 4, 4), (64, 8, 4), (32, 4, 4), (64, 4, 8), (128, 4, 8), (32, 8, 4), (64, 16, 4),
                        (128, 8, 8), (64, 2, 4)]:
        cf = dict(BLOCK_N=min(bn, bn_max), PPP=ppp, num_warps=nw)   # clamp, never filter to nothing
        if cf not in cands:
            cands.append(cf)
    key = ((n * h * wd).bit_length(), k, nout, ln, int(residual is not None), str(x.dtype))
    launch(TUNER.pick("ln_gemm_pfT", key, cands, launch))
    return out
