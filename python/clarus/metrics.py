"""On-device PSNR / SSIM, numerically matching basicsr.metrics (calculate_psnr, calculate_ssim).

Both reference metrics run on the uint8 images produced by tensor2img:
  PSNR: float64 MSE over H*W*3, max_value = 1 if out.max() <= 1 else 255.
  SSIM: `_ssim_3d`: 11x11x11 Gaussian (sigma 1.5) over (H, W, C) with replicate
        padding, C1=(0.01*L)^2, C2=(0.03*L)^2, mean of the SSIM map.
The 3-D Gaussian is the outer product of three 1-D kernels, so it is applied as
three separable 1-D passes with clamped indices (identical to replicate padding).
The sum of squared errors is accumulated exactly in integers.
"""
from __future__ import annotations

import math
import weakref

import numpy as np
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - devices without triton use TorchMetrics
    triton = None


def gaussian_1d(size: int = 11, sigma: float = 1.5):
    # cv2.getGaussianKernel(11, 1.5) in float64
    half = size // 2
    g = [math.exp(-((i - half) ** 2) / (2 * sigma * sigma)) for i in range(size)]
    s = sum(g)
    return [v / s for v in g]


def _f64(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu", torch.float64).numpy()
    return np.asarray(x, dtype=np.float64)


def psnr_peak(pred) -> float:
    """basicsr's peak rule: 1.0 if pred.max() <= 1 else 255.0 (the device metrics use the same)."""
    return 1.0 if _f64(pred).max() <= 1 else 255.0


def psnr(pred, gt) -> float:
    """basicsr calculate_psnr(pred, gt, crop_border=0) on host: float64 MSE, peak psnr_peak(pred),
    inf when identical. pred / gt: numpy arrays or torch tensors (any device, any layout, same
    shape) holding 0..255 values (uint8 or the rounded float of tensor2img)."""
    a, b = _f64(pred), _f64(gt)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * np.log10(psnr_peak(a) / np.sqrt(mse)))


if triton is not None:
    @triton.jit
    def ssd_max_kernel(A, B, SSD, MX, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < N
        a = tl.load(A + offs, mask=m, other=0).to(tl.int32)
        b = tl.load(B + offs, mask=m, other=0).to(tl.int32)
        d = a - b
        tl.store(SSD + pid, tl.sum(d * d, axis=0))
        tl.store(MX + pid, tl.max(a, axis=0))


    @triton.jit
    def _clampi(i, n):
        return tl.minimum(tl.maximum(i, 0), n - 1)


    @triton.jit
    def ssim_h_kernel(A, B, T, G, CM, H, W, OFF: tl.constexpr, BLOCK_W: tl.constexpr):
        """Channel + horizontal Gaussian of (a, b, a^2, b^2, ab) for channel c.

        A, B: u8 [H, W, 3]; G: fp32 [11]; CM: fp32 [3, 3] channel-mixing matrix; T: fp32 [5, 3, H, W].
        """
        y = tl.program_id(0)
        xs = tl.program_id(1) * BLOCK_W + tl.arange(0, BLOCK_W)
        c = tl.program_id(2)
        xm = xs < W
        row = y.to(tl.int64) * W * 3
        wa = tl.load(CM + c * 3 + 0)
        wb = tl.load(CM + c * 3 + 1)
        wc = tl.load(CM + c * 3 + 2)
        s_a = tl.zeros((BLOCK_W,), dtype=tl.float32)
        s_b = tl.zeros((BLOCK_W,), dtype=tl.float32)
        s_aa = tl.zeros((BLOCK_W,), dtype=tl.float32)
        s_bb = tl.zeros((BLOCK_W,), dtype=tl.float32)
        s_ab = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for t in tl.static_range(11):
            gt = tl.load(G + t)
            xx = _clampi(xs + (t - 5), W)
            p = A + row + xx.to(tl.int64) * 3
            q = B + row + xx.to(tl.int64) * 3
            # moments are taken about OFF (128): mathematically identical, far less fp32 cancellation
            a0 = tl.load(p + 0, mask=xm, other=0).to(tl.float32) - OFF
            a1 = tl.load(p + 1, mask=xm, other=0).to(tl.float32) - OFF
            a2 = tl.load(p + 2, mask=xm, other=0).to(tl.float32) - OFF
            b0 = tl.load(q + 0, mask=xm, other=0).to(tl.float32) - OFF
            b1 = tl.load(q + 1, mask=xm, other=0).to(tl.float32) - OFF
            b2 = tl.load(q + 2, mask=xm, other=0).to(tl.float32) - OFF
            s_a += gt * (wa * a0 + wb * a1 + wc * a2)
            s_b += gt * (wa * b0 + wb * b1 + wc * b2)
            s_aa += gt * (wa * a0 * a0 + wb * a1 * a1 + wc * a2 * a2)
            s_bb += gt * (wa * b0 * b0 + wb * b1 * b1 + wc * b2 * b2)
            s_ab += gt * (wa * a0 * b0 + wb * a1 * b1 + wc * a2 * b2)
        hw = H * W
        base = y.to(tl.int64) * W + xs
        tl.store(T + (0 * 3 + c) * hw + base, s_a, mask=xm)
        tl.store(T + (1 * 3 + c) * hw + base, s_b, mask=xm)
        tl.store(T + (2 * 3 + c) * hw + base, s_aa, mask=xm)
        tl.store(T + (3 * 3 + c) * hw + base, s_bb, mask=xm)
        tl.store(T + (4 * 3 + c) * hw + base, s_ab, mask=xm)


    @triton.jit
    def ssim_v_kernel(T, OUT, G, CC, H, W, OFF: tl.constexpr, BLOCK_W: tl.constexpr):
        """Vertical Gaussian + SSIM map; one partial sum per program. T: fp32 [5, 3, H, W]; CC: [C1, C2]."""
        y = tl.program_id(0)
        c = tl.program_id(1)
        pid_x = tl.program_id(2)
        xs = pid_x * BLOCK_W + tl.arange(0, BLOCK_W)
        xm = xs < W
        hw = H * W
        mu1 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        mu2 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        s11 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        s22 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        s12 = tl.zeros((BLOCK_W,), dtype=tl.float32)
        for t in tl.static_range(11):
            gt = tl.load(G + t)
            yy = _clampi(y + (t - 5), H)
            off = yy.to(tl.int64) * W + xs
            mu1 += gt * tl.load(T + (0 * 3 + c) * hw + off, mask=xm, other=0.0)
            mu2 += gt * tl.load(T + (1 * 3 + c) * hw + off, mask=xm, other=0.0)
            s11 += gt * tl.load(T + (2 * 3 + c) * hw + off, mask=xm, other=0.0)
            s22 += gt * tl.load(T + (3 * 3 + c) * hw + off, mask=xm, other=0.0)
            s12 += gt * tl.load(T + (4 * 3 + c) * hw + off, mask=xm, other=0.0)
        c1 = tl.load(CC)
        c2 = tl.load(CC + 1)
        # centered moments -> variances / covariance, then restore the means
        v1 = s11 - mu1 * mu1
        v2 = s22 - mu2 * mu2
        v12 = s12 - mu1 * mu2
        mu1 = mu1 + OFF
        mu2 = mu2 + OFF
        m11 = mu1 * mu1
        m22 = mu2 * mu2
        m12 = mu1 * mu2
        ssim = ((2 * m12 + c1) * (2 * v12 + c2)) / ((m11 + m22 + c1) * (v1 + v2 + c2))
        ssim = tl.where(xm, ssim, 0.0)
        nx = tl.num_programs(2)
        tl.store(OUT + (y * 3 + c) * nx + pid_x, tl.sum(ssim, axis=0))



class TritonMetrics:
    def __init__(self, device):
        self.device = device
        g = gaussian_1d()
        # effective 3x3 channel-mixing matrix of an 11-tap kernel over a replicate-padded length-3 axis
        cm = [[0.0] * 3 for _ in range(3)]
        for c in range(3):
            for t in range(-5, 6):
                cm[c][min(max(c + t, 0), 2)] += g[t + 5]
        self.g = torch.tensor(g, device=device, dtype=torch.float32)
        self.cm = torch.tensor(cm, device=device, dtype=torch.float32).flatten()
        self.c_hi = torch.tensor([(0.01 * 255) ** 2, (0.03 * 255) ** 2], device=device)
        self.c_lo = torch.tensor([0.01 ** 2, 0.03 ** 2], device=device)
        # live scratch per (h, w), shared by the callers that hold it; freed with the last holder
        self._live = weakref.WeakValueDictionary()

    def scratch(self, h: int, w: int) -> "_Scratch":
        """Device scratch for [h, w, 3] images (~60 B/pixel). The caller owns it: keep it alive as
        long as anything (e.g. a captured graph) may use it, pass it to compute(). Callers asking
        for the same shape while one is alive share it (they must run on one stream)."""
        s = self._live.get((h, w))
        if s is None:
            s = _Scratch(h, w, self.device)
            self._live[(h, w)] = s
        return s

    def compute(self, out_u8: torch.Tensor, gt_u8: torch.Tensor, res: torch.Tensor, scratch=None):
        """out_u8, gt_u8: contiguous [H, W, 3] uint8. Writes res[0] = PSNR, res[1] = SSIM (float64).

        `scratch`: from scratch(H, W); allocated per call when omitted. No host synchronization:
        safe inside CUDA-graph capture (the graph then references `scratch`, keep it alive).
        """
        h, w, _ = out_u8.shape
        b = scratch if scratch is not None else self.scratch(h, w)
        if (b.h, b.w) != (h, w):
            raise ValueError(f"metrics scratch is for {b.h}x{b.w}, image is {h}x{w}")
        n = h * w * 3
        ssd_max_kernel[(b.nb,)](out_u8, gt_u8, b.ssd, b.mx, n, BLOCK=b.block, num_warps=8)
        ssd = b.ssd.sum(dtype=torch.int64).double()
        small = b.mx.max() <= 1
        maxv = torch.where(small, 1.0, 255.0).double()
        mse = ssd / n
        psnr = 20.0 * torch.log10(maxv / torch.sqrt(mse))
        res[0] = torch.where(mse == 0, torch.full_like(psnr, float("inf")), psnr)
        cc = torch.where(small, self.c_lo, self.c_hi)
        ssim_h_kernel[(h, triton.cdiv(w, b.bw), 3)](out_u8, gt_u8, b.tmp, self.g, self.cm, h, w,
                                                    OFF=128.0, BLOCK_W=b.bw, num_warps=4)
        part = b.part
        ssim_v_kernel[(h, 3, b.nx)](b.tmp, part, self.g, cc, h, w, OFF=128.0, BLOCK_W=b.bw, num_warps=4)
        res[1] = part.double().sum() / n


class _Scratch:
    """TritonMetrics device buffers for one image shape (a class so it can be weakly cached)."""

    def __init__(self, h, w, device):
        self.h, self.w = h, w
        self.block, self.bw = 4096, 128
        self.nb = triton.cdiv(h * w * 3, self.block)
        self.nx = triton.cdiv(w, self.bw)
        self.ssd = torch.empty(self.nb, device=device, dtype=torch.int32)
        self.mx = torch.empty(self.nb, device=device, dtype=torch.int32)
        self.tmp = torch.empty((5, 3, h, w), device=device, dtype=torch.float32)
        self.part = torch.empty((h * 3 * self.nx,), device=device, dtype=torch.float32)


class TorchMetrics:
    """Same math as TritonMetrics with plain torch ops (any device; fp32 SSIM, exact SSE)."""

    def __init__(self, device):
        self.device = device
        g = gaussian_1d()
        cm = [[0.0] * 3 for _ in range(3)]
        for c in range(3):
            for t in range(-5, 6):
                cm[c][min(max(c + t, 0), 2)] += g[t + 5]
        self.cm = torch.tensor(cm, device=device, dtype=torch.float32)
        k = torch.tensor(g, device=device, dtype=torch.float32)
        self.kh = k.view(1, 1, 1, 11)
        self.kv = k.view(1, 1, 11, 1)

    def scratch(self, h, w):
        return None  # works on per-call temporaries

    def compute(self, out_u8, gt_u8, res, scratch=None):
        h, w, _ = out_u8.shape
        n = h * w * 3
        d = out_u8.int() - gt_u8.int()
        ssd = (d * d).sum(dtype=torch.int64).to(res.dtype)
        small = out_u8.max() <= 1
        maxv = torch.where(small, 1.0, 255.0).to(res.dtype)
        mse = ssd / n
        psnr = 20.0 * torch.log10(maxv / torch.sqrt(mse))
        res[0] = torch.where(mse == 0, torch.full_like(psnr, float("inf")), psnr)
        a = out_u8.float() - 128.0
        b = gt_u8.float() - 128.0
        maps = torch.stack((a, b, a * a, b * b, a * b))                    # [5, H, W, 3]
        maps = torch.einsum("khwc,dc->kdhw", maps, self.cm).reshape(15, 1, h, w)
        maps = F.conv2d(F.pad(maps, (5, 5, 0, 0), mode="replicate"), self.kh)
        maps = F.conv2d(F.pad(maps, (0, 0, 5, 5), mode="replicate"), self.kv).view(5, 3, h, w)
        mu1, mu2, s11, s22, s12 = maps
        v1, v2, v12 = s11 - mu1 * mu1, s22 - mu2 * mu2, s12 - mu1 * mu2
        mu1, mu2 = mu1 + 128.0, mu2 + 128.0
        c1 = torch.where(small, 0.01 ** 2, (0.01 * 255) ** 2)
        c2 = torch.where(small, 0.03 ** 2, (0.03 * 255) ** 2)
        ssim = ((2 * mu1 * mu2 + c1) * (2 * v12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (v1 + v2 + c2))
        res[1] = ssim.to(res.dtype).sum() / n


def make_metrics(device: torch.device, backend: str = "auto"):
    """'triton' on CUDA when available, else 'torch'."""
    if backend == "auto":
        backend = "triton" if (triton is not None and torch.device(device).type == "cuda") else "torch"
    if backend == "triton":
        return TritonMetrics(device)
    if backend == "torch":
        return TorchMetrics(device)
    raise ValueError(f"unknown metrics backend {backend!r}")
