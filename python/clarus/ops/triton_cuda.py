"""Triton backend for NVIDIA GPUs (fp16 / bf16 activations, NHWC).

Implements every op of the reference backend with fused kernels. Weights arrive in the
canonical layout and are re-laid out lazily (`Backend.cached`) on first use, which happens
in the eager warm-up pass before CUDA-graph capture. Calls a kernel cannot handle raise
`Unsupported` and fall through to the next backend (normally `torch`).
"""
from __future__ import annotations

import torch

from . import Backend, BackendUnavailable, Unsupported
from .torch_ref import Small, box_geometry

try:
    from ..kernels import conv3x3 as kc
    from ..kernels import gemm as kg
    from ..kernels import spatial as ks
    from ..kernels import attention as ka
    from ..kernels import freq as kf
    from ..kernels import ssm as kss
    _IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - triton missing / unsupported platform
    _IMPORT_ERROR = e


def _taps(w4d: torch.Tensor) -> torch.Tensor:
    """Depthwise [C, 1, kh, kw] or single-output [1, C, kh, kw] conv weight -> [kh*kw, C] fp32."""
    w = w4d[:, 0] if w4d.shape[1] == 1 else w4d[0]
    return w.reshape(w.shape[0], -1).t().contiguous().float()


def _conv_kn(w4d: torch.Tensor, dtype) -> torch.Tensor:
    """dense conv weight [Cout, Cin, kh, kw] -> [(ky, kx, cin), Cout]."""
    return w4d.permute(2, 3, 1, 0).reshape(-1, w4d.shape[0]).contiguous().to(dtype)


def _dense(*ts) -> bool:
    """All given tensors (None skipped) are contiguous: the kernels index them as dense arrays."""
    return all(t is None or t.is_contiguous() for t in ts)


def _same(t, shape) -> bool:
    """t is None, or contiguous with exactly `shape` (residuals / outputs indexed like the output)."""
    return t is None or (t.is_contiguous() and tuple(t.shape) == tuple(shape))


def _pix_rows(t: torch.Tensor) -> bool:
    """NHWC tensor addressed as pixel * stride(2) + channel (a channel slice of a wider buffer
    is fine): unit channel stride and uniform pixel stride across rows and images."""
    n, h, w, _ = t.shape
    s = t.stride()
    return s[3] == 1 and (h == 1 or s[1] == w * s[2]) and (n == 1 or s[0] == h * w * s[2])


class TritonOps(Backend):
    name = "triton"

    def __init__(self, dtype, device):
        super().__init__(dtype, device)
        if _IMPORT_ERROR is not None:
            raise BackendUnavailable(f"triton kernels failed to import: {_IMPORT_ERROR}")
        if torch.device(device).type != "cuda":
            raise BackendUnavailable(f"needs a CUDA device, got {device}")
        if dtype not in (torch.float16, torch.bfloat16):
            raise BackendUnavailable(f"runs fp16/bf16 activations, got {dtype}")
        cap = torch.cuda.get_device_capability(torch.device(device))
        if dtype == torch.bfloat16 and cap < (8, 0):
            raise BackendUnavailable(f"bf16 kernels need compute capability >= 8.0, device has {cap[0]}.{cap[1]}")

    # ------------------------------------------------------------- re-layouts
    def _taps(self, w):
        return self.cached(w, "taps", _taps)

    def _kn(self, w, dtype=None, pad=False):
        dt = dtype or self.dtype

        def conv(t):
            kn = _conv_kn(t, dt)
            return kc.pad_cols(kn) if pad else kn
        return self.cached(w, f"kn{dt}{pad}", conv)

    # ------------------------------------------------------------------ GEMMs
    def linear(self, x, w, bias=None, ln=False, act=None, gate=False, scale=None,
               residual=None, mul=None, out_dtype=None, ln_eps=1e-6):
        if isinstance(x, Small):
            return x.with_data(self.linear(x.data, w, bias, ln, act, gate, scale, residual, mul, out_dtype, ln_eps))
        if ln and mul is not None:
            raise Unsupported("linear: LayerNorm and mul prologues together")
        n, h, wd, k = x.shape
        m = n * h * wd
        nout = w.shape[1] // 2 if gate else w.shape[1]
        if not (_dense(x, w, bias, scale) and _same(residual, (n, h, wd, nout))
                and (mul is None or (mul.data.is_contiguous() and tuple(mul.data.shape[::3]) == (n, k)))):
            raise Unsupported("linear: contiguous x / weights / mul map, residual of the output's shape")
        out = torch.empty((n, h, wd, nout), device=x.device, dtype=out_dtype or self.dtype)
        kw = dict(n=nout, k=k, m=m, out=out, bias=bias, gate=gate, act=1 if act == "gelu" else 0,
                  geom=(h, wd, 1, 1, 0, 0), ln_eps=ln_eps, scale=scale, residual=residual)
        if ln and k <= 128:
            # LayerNorm / RMS norm computed inside the GEMM (whole row in one K tile)
            kw.update(pro=6 if ln == "rms" else 1)
        elif ln:
            # normalize once into the compute dtype, then run a plain tensor-core GEMM
            x = ks.ln_rows(x, rms=ln == "rms", out_dtype=self.dtype, eps=ln_eps)
        elif mul is not None and (k >= 256 or x.dtype == torch.float32):
            # compute-bound regime: a separate gather-multiply keeps the GEMM main loop dense;
            # fp32 x also goes here so the GEMM's range-preserving fp32 prologue applies
            x = ks.mul_small(x, mul.data, mul.pt, mul.pl)
        elif mul is not None:
            oh, ow = mul.data.shape[1:3]
            kw.update(pro=2, small=mul.data, geom=(h, wd, oh, ow, mul.pt, mul.pl))
        return kg.fused_gemm(x, w, **kw)

    def down(self, x, w4d, b, out_dtype=None):
        if tuple(w4d.shape[-2:]) != (2, 2) or not _dense(x, b):
            raise Unsupported("down: only 2x2 stride-2 convs, contiguous input")
        n, h, wd, c = x.shape
        wk = self._kn(w4d)
        out = torch.empty((n, h // 2, wd // 2, wk.shape[1]), device=x.device, dtype=out_dtype or self.dtype)
        return kg.fused_gemm(x, wk, n=wk.shape[1], k=4 * c, m=n * (h // 2) * (wd // 2), out=out,
                             bias=b, geom=(h, wd, 1, 1, 0, 0), a_mode=1)

    def up(self, x, w, residual=None, out_dtype=None):
        n, h, wd, c = x.shape
        if (w.shape[1] // 4) % 16 or not (_dense(x, w) and _same(residual, (n, 2 * h, 2 * wd, w.shape[1] // 4))):
            raise Unsupported("up: output channels a multiple of 16, contiguous x, residual of the output's shape")
        out = torch.empty((n, 2 * h, 2 * wd, w.shape[1] // 4), device=x.device, dtype=out_dtype or self.dtype)
        return kg.fused_gemm(x, w, n=w.shape[1], k=c, m=n * h * wd, out=out, residual=residual,
                             geom=(h, wd, 1, 1, 0, 0), c_mode=1)

    # ---------------------------------------------------------------- spatial
    def boxmean(self, x, k) -> Small:
        if not x.is_contiguous() or x.dtype != self.dtype:
            raise Unsupported("boxmean: contiguous input in the compute dtype")
        n, h, w, c = x.shape
        k1, k2, oh, ow, pt, pl = box_geometry(h, w, k)
        return Small(ks.boxmean(x, k1, k2, oh, ow), pt, pl, h, w)

    def dwconv(self, x, w, b, dil=1, gate=False, act=None):
        kh, kw = w.shape[-2:]
        if (kh, kw) != (3, 3) or b is None or x.dtype != self.dtype or not _dense(x, b):
            raise Unsupported("dwconv: only contiguous 3x3 windows with bias, x in the compute dtype")
        return ks.dwconv(x, self._taps(w), b, 3, 3, dil=dil, gate=gate, act=act)

    def intro(self, inp, w, b, out_dtype=None):
        if tuple(w.shape[1:]) != (3, 3, 3) or w.shape[0] > 256 or b is None:
            raise Unsupported("intro: 3-channel input, 3x3, <= 256 output channels, bias")
        return kc.intro3x3(inp.contiguous(), self._kn(w, torch.float32), b.float().contiguous(),
                           out_dtype or self.dtype)

    def ending(self, x, w, b, inp):
        n, h, wd, _ = x.shape
        if (tuple(w.shape[-2:]) != (3, 3) or tuple(inp.shape) != (n, w.shape[0], h, wd) or b is None
                or not _pix_rows(x)):
            raise Unsupported("ending: 3x3 with global residual of the output's shape, NHWC rows")
        return kc.conv3x3(x, self._kn(w, pad=True), b.float().contiguous(), cout=w.shape[0],
                          residual=inp.contiguous(), out_dtype=torch.float32, nchw_out=True)

    def conv3x3(self, x, w, b=None, out_dtype=None, act=None, residual=None, out_scale=1.0, up2=False,
                out=None, slope=0.2):
        n, h, wd, _ = x.shape
        oshape = (n, 2 * h, 2 * wd, w.shape[0]) if up2 else (n, h, wd, w.shape[0])
        if act not in (None, "lrelu") or tuple(w.shape[-2:]) != (3, 3) or w.shape[1] != x.shape[-1]:
            raise Unsupported("conv3x3 kernel: 3x3, activation none / lrelu")
        # NHWC with a uniform pixel stride (channel slices of wider buffers are fine); the kernel
        # wrapper rejects anything else with ValueError, which would not fall back
        if not (_pix_rows(x) and all(t is None or (tuple(t.shape) == oshape and _pix_rows(t)) for t in (residual, out))):
            raise Unsupported("conv3x3 kernel: NHWC tensors with a uniform pixel stride, residual / out of the output's shape")
        bias = b if b is not None else self.cached(w, "zbias", lambda t: torch.zeros(t.shape[0], device=t.device))
        return kc.conv3x3(x, self._kn(w, pad=True), bias.float().contiguous(), cout=w.shape[0], residual=residual,
                          out=out, out_dtype=out_dtype or self.dtype, act=1 if act == "lrelu" else 0,
                          out_scale=out_scale, up2=up2, slope=slope)

    def channel_attention(self, qkv, heads, temperature):
        if not qkv.is_contiguous():
            raise Unsupported("channel_attention: contiguous qkv only")
        g, sq, sk = ka.gram(qkv, heads)
        nq = sq.clamp_min(1e-24).rsqrt()
        nk = sk.clamp_min(1e-24).rsqrt()
        g = g * nq[..., :, None] * nk[..., None, :] * temperature.view(1, heads, 1, 1).float()
        return torch.softmax(g, dim=-1)

    def linear_per_image(self, x, w, residual=None, out_dtype=None, bias=None):
        n, h, wd, k = x.shape
        nout = w.shape[-1]
        # each image is its own GEMM launch (x[i:i+1]), so x's batch stride is free
        if (x.stride(-1) != 1 or x.stride(1) != wd * x.stride(2) or tuple(w.shape[:2]) != (n, k)
                or not (_dense(bias) and _same(residual, (n, h, wd, nout)))):
            raise Unsupported("linear_per_image: NHWC rows with a uniform pixel stride, residual of the output's shape")
        out = torch.empty((n, h, wd, nout), device=x.device, dtype=out_dtype or self.dtype)
        wk = w.to(self.dtype)
        for i in range(n):  # one GEMM per image (graph-captured, so the loop costs nothing at replay)
            kg.fused_gemm(x[i:i + 1], wk[i].contiguous(), n=nout, k=k, m=h * wd, out=out[i:i + 1], bias=bias,
                          residual=residual[i:i + 1] if residual is not None else None,
                          geom=(h, wd, 1, 1, 0, 0), lda=x.stride(2))
        return out

    # ------------------------------------------------ frequency / state-space ops
    def patch_corr(self, x, c, p=8):
        n, h, w, cx = x.shape
        # the kernel takes x's batch / row / pixel strides, so only unit channel stride is required
        if p != 8 or h % 8 or w % 8 or x.stride(-1) != 1 or cx < 2 * c:
            raise Unsupported("patch_corr: 8x8 patches, H and W multiples of 8, unit channel stride, >= 2c channels")
        key = ("corr_mats", self.dtype)
        mats = self._cache.get(key)
        if mats is None:
            mats = self._cache[key] = kf.corr_mats(8, x.device, self.dtype)
        return kf.patch_corr(x, c, mats)

    def ln_mul_linear(self, x, ln_w, ln_b, m, w, bias=None, m_act=None, residual=None, ln_eps=1e-5,
                      out_dtype=None):
        n, h, wd, k = x.shape
        if (not _dense(x, ln_w, ln_b, w, bias) or tuple(m.shape) != tuple(x.shape) or not _pix_rows(m)
                or not _same(residual, (n, h, wd, w.shape[1]))):
            raise Unsupported("ln_mul_linear: contiguous x, NHWC m with a uniform pixel stride, "
                              "residual of the output's shape")
        stats = kg.row_stats(x, ln_eps)
        out = torch.empty((n, h, wd, w.shape[1]), device=x.device, dtype=out_dtype or self.dtype)
        return kg.fused_gemm(x, w, n=w.shape[1], k=k, m=n * h * wd, out=out, bias=bias, residual=residual,
                             stats=stats, m2=m, lnw=ln_w, lnb=ln_b, m2_act=1 if m_act == "gelu" else 0,
                             geom=(h, wd, 1, 1, 0, 0), pro=5)

    def seq_dwconv1d(self, x, w, b):
        if not x.is_contiguous() or x.dtype != torch.float32 or w.shape[-1] % 2 == 0 or b is None:
            raise Unsupported("seq_dwconv1d: contiguous fp32 input, odd kernel, bias")
        return kss.seq_dwconv1d(x, self.cached(w, "taps1d", lambda v: v.float().t().contiguous()), b.float().contiguous())

    def selective_scan(self, u, dbl, dt_w, dt_bias, A, D):
        r, ns = dt_w.shape[1], A.shape[1]
        if (not (u.is_contiguous() and dbl.is_contiguous() and dbl.dtype == torch.float32) or ns != 8
                or tuple(dbl.shape[:3]) != tuple(u.shape[:3]) or dbl.shape[-1] < r + 2 * ns):
            raise Unsupported("selective_scan: contiguous inputs, fp32 dbl, d_state 8 (scan8_kernel)")
        a2 = self.cached(A, "a2", lambda t: (t.float() * kss.LOG2E).contiguous())
        return kss.selective_scan(u, dbl, dt_w.float().contiguous(), dt_bias.float().contiguous(), a2,
                                  D.float().contiguous())

    def _idx(self, idx, device):
        key = ("crop_idx", tuple(idx), str(device))
        t = self._cache.get(key)
        if t is None:
            t = self._cache[key] = torch.tensor(idx, dtype=torch.int32, device=device).flatten().contiguous()
        return t

    def tiles_gather(self, x, k1, k2, idx, orient=0):
        if not x.is_contiguous():
            raise Unsupported("tiles_gather: contiguous input")
        return kss.crops_gather(x, self._idx(idx, x.device), len(idx), k1, k2, orient)

    def tiles_merge(self, base, parts, k1, k2, idx, orient=0):
        if (not (base.is_contiguous() and parts.is_contiguous()) or parts.shape[-1] != base.shape[-1]
                or base.dtype != self.dtype):
            raise Unsupported("tiles_merge: contiguous inputs, same channel count, base in the compute dtype")
        return kss.crops_merge(base, parts, self._idx(idx, base.device), len(idx), k1, k2, orient)

    def resize_bilinear(self, x, scale):
        if scale not in (0.5, 2) or not x.is_contiguous():
            raise Unsupported("resize_bilinear: scale 0.5 / 2, contiguous input")
        return ks.resize_bilinear(x, scale, out_dtype=self.dtype)

    def linear_patch_filter(self, x, w, bias, f, ln=False, ln_eps=1e-6, residual=None):
        n, h, wd, k = x.shape
        if (tuple(f.shape[1:]) != (8, 5) or f.shape[0] != w.shape[1] or min(h, wd) < 9 or bias is None
                or x.dtype != self.dtype or not _dense(x, w, bias) or not _same(residual, (n, h, wd, w.shape[1]))):
            raise Unsupported("linear_patch_filter: 8x8 patches, H, W >= 9, contiguous, bias, x in the compute "
                              "dtype (the kernel writes x.dtype), residual of the output's shape")
        t2, i2 = self.cached(f, "fmats", lambda t: kf.filter_mats(8, t.device, self.dtype))
        fq = self.cached(f, "fcoef", kf.filter_coeffs)
        code = {False: 0, True: 1, "rms": 2}[ln]
        if code and k > 128:  # normalize once, then the plain fused GEMM + filter
            x = ks.ln_rows(x, rms=ln == "rms", out_dtype=self.dtype, eps=ln_eps)
            code = 0
        return kf.ln_gemm_patch_filter(x, w, bias, fq, t2, i2, ln=code, ln_eps=ln_eps, residual=residual)
