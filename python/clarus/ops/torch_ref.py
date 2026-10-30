"""PyTorch reference backend: every op, any device torch supports (cuda, npu, cpu, ...).

It defines the semantics every other backend must reproduce, and is the fallback for
ops a faster backend does not implement. Math runs in fp32 and is cast back to the
activation dtype at op boundaries. Activations are NHWC; weights canonical (see ops/__init__).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import Backend


@dataclass
class Small:
    """TLC local-average map before replicate padding.

    data: [N, h', w', C]; full-resolution value at (y, x) is
    data[clamp(y - pt, 0, h'-1), clamp(x - pl, 0, w'-1)].
    """

    data: torch.Tensor
    pt: int
    pl: int
    H: int
    W: int

    def with_data(self, data):
        return Small(data, self.pt, self.pl, self.H, self.W)


def box_geometry(h: int, w: int, k):
    """Valid-window geometry of a (kh, kw) local mean (int k means square)."""
    kh, kw = (k, k) if isinstance(k, int) else k
    k1, k2 = min(h, kh), min(w, kw)
    oh, ow = h - k1 + 1, w - k2 + 1
    return k1, k2, oh, ow, (h - oh) // 2, (w - ow) // 2


def nchw(x):
    return x.permute(0, 3, 1, 2)


def nhwc(x):
    return x.permute(0, 2, 3, 1).contiguous()


def _orient(x, orient):
    """NHWC view / copy: 0 identity, 1 flip H and W, 2 transpose H and W."""
    if orient == 1:
        return torch.flip(x, dims=(1, 2))
    if orient == 2:
        return x.transpose(1, 2)
    return x


def grid_tiles(h: int, w: int, k: int):
    """EVSSM EVS.grids(): (k1, k2, [(i, j), ...]) of the overlapping k x k crops covering h x w."""
    k1, k2 = min(h, k), min(w, k)
    nr, nc = (h - 1) // k1 + 1, (w - 1) // k2 + 1
    step_j = k2 if nc == 1 else math.ceil((w - k2) / (nc - 1) - 1e-8)
    step_i = k1 if nr == 1 else math.ceil((h - k1) / (nr - 1) - 1e-8)
    idx, i, last_i = [], 0, False
    while i < h and not last_i:
        j, last_j = 0, False
        if i + k1 >= h:
            i, last_i = h - k1, True
        while j < w and not last_j:
            if j + k2 >= w:
                j, last_j = w - k2, True
            idx.append((i, j))
            j += step_j
        i += step_i
    return k1, k2, idx


class TorchOps(Backend):
    name = "torch"

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def expand(s: Small) -> torch.Tensor:
        dev = s.data.device
        oh, ow = s.data.shape[1], s.data.shape[2]
        ih = (torch.arange(s.H, device=dev) - s.pt).clamp_(0, oh - 1)
        iw = (torch.arange(s.W, device=dev) - s.pl).clamp_(0, ow - 1)
        return s.data.index_select(1, ih).index_select(2, iw)

    @staticmethod
    def _ln(x, eps=1e-6, rms=False):
        x = x.float()
        if rms:  # no mean subtraction (EVSSM's "WithBias_LayerNorm")
            return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        mu = x.mean(-1, keepdim=True)
        var = (x - mu).square().mean(-1, keepdim=True)
        return (x - mu) * torch.rsqrt(var + eps)

    # ------------------------------------------------------------- primitives
    def linear(self, x, w, bias=None, ln=False, act=None, gate=False, scale=None,
               residual=None, mul=None, out_dtype=None, ln_eps=1e-6):
        """1x1 conv as GEMM, w [K, N] (2N columns when gate=True).

        out = epilogue(prologue(x) @ w): prologue = LayerNorm (ln=True; ln="rms": x / rms(x),
        no mean subtraction) or * expand(mul);
        epilogue = +bias, SimpleGate, GELU, *scale, +residual. x may be a `Small` (applied to
        the local-mean map).
        """
        if isinstance(x, Small):
            return x.with_data(self.linear(x.data, w, bias, ln, act, gate, scale, residual))
        a = x.float()
        if mul is not None:
            a = a * self.expand(mul).float()
        if ln:
            a = self._ln(a, ln_eps, rms=ln == "rms")
        out = torch.matmul(a.to(w.dtype), w).float()
        if bias is not None:
            out = out + bias
        if gate:
            o1, o2 = out.chunk(2, dim=-1)
            out = o1 * o2
        if act == "gelu":
            out = F.gelu(out)
        if scale is not None:
            out = out * scale
        if residual is not None:
            out = residual.float() + out
        return out.to(out_dtype or self.dtype)

    def boxmean(self, x, k) -> Small:
        """Local mean over a (kh, kw) window (TLC); returns the un-padded valid map."""
        n, h, w, c = x.shape
        k1, k2, oh, ow, pt, pl = box_geometry(h, w, k)
        s = x.float().cumsum(dim=2).cumsum(dim=1)
        s = F.pad(s, (0, 0, 1, 0, 1, 0))
        out = s[:, k1:, k2:] + s[:, :-k1, :-k2] - s[:, :-k1, k2:] - s[:, k1:, :-k2]
        out = out / (k1 * k2)
        return Small(out.to(self.dtype), pt, pl, h, w)

    def dwconv(self, x, w, b, dil=1, gate=False, act=None):
        """Depthwise conv, w [C, 1, kh, kw], 'same' padding; optional gate (True / "mul":
        x1 * x2, "gelu": gelu(x1) * x2 on the two channel halves) and act ("gelu", no gate)."""
        kh, kw = w.shape[-2:]
        out = F.conv2d(nchw(x).float(), w.float(), b, padding=(dil * (kh // 2), dil * (kw // 2)),
                       dilation=dil, groups=x.shape[-1])
        out = out.permute(0, 2, 3, 1)
        if gate:
            o1, o2 = out.chunk(2, dim=-1)
            out = (F.gelu(o1) if gate == "gelu" else o1) * o2
        if act == "gelu":
            out = F.gelu(out)
        return out.contiguous().to(self.dtype)

    def intro(self, inp, w, b, out_dtype=None):
        return nhwc(F.conv2d(inp.float(), w.float(), b, padding=1)).to(out_dtype or self.dtype)

    def down(self, x, w4d, b, out_dtype=None):
        return nhwc(F.conv2d(nchw(x).float(), w4d.float(), b, stride=2)).to(out_dtype or self.dtype)

    def up(self, x, w, residual=None, out_dtype=None):
        """1x1 conv to 4C' (w [C, 4C'], N ordered (dy, dx, c')) + PixelShuffle(2) (+ residual)."""
        n, h, wd, _ = x.shape
        y = torch.matmul(x.float(), w.float())
        c = y.shape[-1] // 4
        y = y.view(n, h, wd, 2, 2, c).permute(0, 1, 3, 2, 4, 5).reshape(n, 2 * h, 2 * wd, c)
        if residual is not None:
            y = y + residual.float()
        return y.to(out_dtype or self.dtype)

    def ending(self, x, w, b, inp):
        """Final 3x3 conv + global residual -> fp32 NCHW."""
        return F.conv2d(nchw(x).float(), w.float(), b, padding=1) + inp

    # ------------------------------------------------- generic ops (Restormer, SR, ...)
    def conv3x3(self, x, w, b=None, out_dtype=None, act=None, residual=None, out_scale=1.0, up2=False,
                out=None):
        """Dense 3x3 conv, stride 1, 'same' padding: NHWC -> NHWC; w [Cout, Cin, 3, 3].

        y = act(conv(up(x)) + b) * out_scale (+ residual); act: None | "lrelu" (LeakyReLU 0.2);
        up2: nearest x2 upsample of x first. `out` may be a channel slice of a wider buffer
        (written in place and returned), and so may `x` / `residual`.
        """
        xi = nchw(x).float()
        if up2:
            xi = F.interpolate(xi, scale_factor=2, mode="nearest")
        y = F.conv2d(xi, w.float(), b, padding=1).permute(0, 2, 3, 1)
        if act == "lrelu":
            y = F.leaky_relu(y, 0.2)
        if out_scale != 1.0:
            y = y * out_scale
        if residual is not None:
            y = y + residual.float()
        if out is not None:
            out.copy_(y)
            return out
        return y.contiguous().to(out_dtype or self.dtype)

    def add_scaled(self, a, b, alpha, out_dtype=None, out=None):
        """a * alpha + b (elementwise, NHWC); `out` may alias b (in-place)."""
        y = a.float() * alpha + b.float()
        if out is not None:
            out.copy_(y)
            return out
        return y.to(out_dtype or self.dtype)

    def pixel_unshuffle(self, x, r=2):
        """NHWC PixelUnshuffle(r): out channel = c * r * r + dy * r + dx (torch order)."""
        n, h, w, c = x.shape
        return x.view(n, h // r, r, w // r, r, c).permute(0, 1, 3, 5, 2, 4).reshape(n, h // r, w // r, r * r * c)

    def pixel_shuffle(self, x):
        """NHWC PixelShuffle(2): input channel = c * 4 + dy * 2 + dx (torch order)."""
        n, h, w, c4 = x.shape
        c = c4 // 4
        return x.view(n, h, w, c, 2, 2).permute(0, 1, 4, 2, 5, 3).reshape(n, 2 * h, 2 * w, c)

    def channel_attention(self, qkv, heads, temperature):
        """Restormer MDTA attention map. qkv [N, H, W, 3C] (q | k | v on channels).

        Per image and head: A = softmax((q / |q|) (k / |k|)^T * t) over the c = C / heads
        channels, the norms taken over all pixels. Returns fp32 [N, heads, c, c].
        """
        n, h, w, c3 = qkv.shape
        c = c3 // 3
        ch = c // heads
        q = qkv[..., :c].reshape(n, h * w, heads, ch).float()
        k = qkv[..., c:2 * c].reshape(n, h * w, heads, ch).float()
        g = torch.einsum("nphi,nphj->nhij", q, k)
        nq = q.square().sum(1).clamp_min(1e-24).rsqrt()          # [n, heads, ch]
        nk = k.square().sum(1).clamp_min(1e-24).rsqrt()
        g = g * nq[..., :, None] * nk[..., None, :] * temperature.view(1, heads, 1, 1).float()
        return torch.softmax(g, dim=-1)

    def linear_per_image(self, x, w, residual=None, out_dtype=None, bias=None):
        """x [N, H, W, K] (may be a channel slice), w [N, K, Nout] per image -> [N, H, W, Nout] (+ bias, + residual)."""
        n, h, wd, k = x.shape
        out = torch.bmm(x.reshape(n, h * wd, k).float(), w.float()).view(n, h, wd, -1)
        if bias is not None:
            out = out + bias
        if residual is not None:
            out = out + residual.float()
        return out.to(out_dtype or self.dtype)

    # --------------------------------------- frequency / state-space ops (FFTformer, EVSSM)
    def resize_bilinear(self, x, scale):
        """F.interpolate(bilinear, align_corners=False) by `scale` (0.5 or 2), NHWC."""
        y = F.interpolate(nchw(x).float(), scale_factor=scale, mode="bilinear", align_corners=False)
        return nhwc(y).to(self.dtype)

    def patch_filter(self, x, f, residual=None, out_dtype=None):
        """Per p x p patch and channel: irfft2(rfft2(patch) * f[c]) (+ residual).

        x NHWC [N, H, W, C]; f [C, p, p//2+1] real (FFTformer DFFN / EVSSM EDFFN). Sizes that
        are not multiples of p are reflect-padded first and the result cropped (EVSSM)."""
        n, h, w, c = x.shape
        p = f.shape[1]
        xi = nchw(x).float()
        ph, pw = (-h) % p, (-w) % p
        if ph or pw:
            xi = F.pad(xi, (0, pw, 0, ph), mode="reflect")
        hh, ww = h + ph, w + pw
        xp = xi.reshape(n, c, hh // p, p, ww // p, p).permute(0, 1, 2, 4, 3, 5)
        y = torch.fft.irfft2(torch.fft.rfft2(xp) * f.float().view(1, c, 1, 1, p, p // 2 + 1), s=(p, p))
        y = y.permute(0, 2, 4, 3, 5, 1).reshape(n, hh, ww, c)[:, :h, :w]
        if residual is not None:
            y = y + residual.float()
        return y.contiguous().to(out_dtype or self.dtype)

    def linear_patch_filter(self, x, w, bias, f, ln=False, ln_eps=1e-6, residual=None):
        """patch_filter(linear(x, w, bias, ln), f, residual) (FFTformer DFFN project_in + filter,
        EVSSM EDFFN project_out + filter)."""
        return self.patch_filter(self.linear(x, w, bias, ln=ln, ln_eps=ln_eps), f, residual=residual)

    def patch_corr(self, x, c, p=8):
        """FFTformer FSAS: per p x p patch, irfft2(rfft2(q) * rfft2(k)) with q = x[..., :c],
        k = x[..., c:2c] (H, W multiples of p) -> fp32 [N, H, W, c] (values reach ~1e7; a
        LayerNorm follows)."""
        n, h, w, _ = x.shape

        def patches(t):
            return nchw(t).float().reshape(n, c, h // p, p, w // p, p).permute(0, 1, 2, 4, 3, 5)
        y = torch.fft.irfft2(torch.fft.rfft2(patches(x[..., :c])) * torch.fft.rfft2(patches(x[..., c:2 * c])),
                             s=(p, p))
        return y.permute(0, 2, 4, 3, 5, 1).reshape(n, h, w, c).contiguous()

    def ln_mul_linear(self, x, ln_w, ln_b, m, w, bias=None, m_act=None, residual=None, ln_eps=1e-5,
                      out_dtype=None):
        """((LN(x) * ln_w + ln_b) * act(m)) @ w (+ bias, + residual); m has x's shape (may be a
        channel slice), act None | "gelu" (FFTformer FSAS output, EVSSM SS2D output)."""
        a = self._ln(x, ln_eps) * ln_w + ln_b
        mm = m.float()
        a = a * (F.gelu(mm) if m_act == "gelu" else mm)
        out = torch.matmul(a.to(w.dtype), w).float()
        if bias is not None:
            out = out + bias
        if residual is not None:
            out = out + residual.float()
        return out.to(out_dtype or self.dtype)

    def seq_dwconv1d(self, x, w, b):
        """Depthwise 1-D conv along each image's raster-order pixel sequence (zero padded at the
        sequence ends): x [T, h, w, C], w [C, k], b [C] -> fp32 [T, h, w, C] (EVSSM x_conv)."""
        t, h, wd, c = x.shape
        y = F.conv1d(x.reshape(t, h * wd, c).transpose(1, 2).float(), w.float()[:, None], b,
                     padding=w.shape[-1] // 2, groups=c)
        return y.transpose(1, 2).reshape(t, h, wd, c).contiguous()

    def selective_scan(self, u, dbl, dt_w, dt_bias, A, D, chunk=32):
        """Mamba selective scan along each image's raster sequence (one direction, z=None).

        u [T, h, w, d]; dbl [T, h, w, r + 2n] = (dt_raw | B | C); dt_w [d, r]; A [d, n] (negative);
        delta = softplus(dt_raw @ dt_w^T + dt_bias); h_l = exp(delta A) h_{l-1} + delta B_l u_l;
        y_l = C_l . h_l + D u_l -> fp32 (values reach ~1e7; a LayerNorm follows).
        Chunked closed form (exact up to rounding)."""
        t, hh, ww, d = u.shape
        L, r, ns = hh * ww, dt_w.shape[1], A.shape[1]
        u = u.reshape(t, L, d).float()
        dbl = dbl.reshape(t, L, -1).float()
        delta = F.softplus(dbl[..., :r] @ dt_w.float().t() + dt_bias.float())
        Bm, Cm = dbl[..., r:r + ns], dbl[..., r + ns:r + 2 * ns]
        A = A.float()
        state = torch.zeros(t, d, ns, device=u.device)
        ys = []
        for s in range(0, L, chunk):
            dl, uc = delta[:, s:s + chunk], u[:, s:s + chunk]
            c = dl.shape[1]
            S = (dl[..., None] * A).cumsum(1)                                   # [t, c, d, n]
            b = (dl * uc)[..., None] * Bm[:, s:s + c, None, :]
            diff = S[:, :, None] - S[:, None, :]                                # [t, c(l), c(s), d, n]
            tri = torch.ones(c, c, dtype=torch.bool, device=u.device).tril()
            wgt = torch.where(tri[None, :, :, None, None], diff, float("-inf")).exp()
            hs = S.exp() * state[:, None] + torch.einsum("tlsdn,tsdn->tldn", wgt, b)
            ys.append(torch.einsum("tldn,tln->tld", hs, Cm[:, s:s + c]))
            state = hs[:, -1]
        y = torch.cat(ys, 1) + u * D.float()
        return y.view(t, hh, ww, d)

    def tiles_gather(self, x, k1, k2, idx, orient=0):
        """Crops o(x)[:, i:i+k1, j:j+k2] for (i, j) in idx -> [N * len(idx), k1, k2, C],
        image-major (EVSSM grids()); o = identity (0), flip H and W (1) or transpose H/W (2)."""
        x = _orient(x, orient)
        return torch.stack([x[:, i:i + k1, j:j + k2] for i, j in idx], 1).reshape(-1, k1, k2, x.shape[-1]).contiguous()

    def tiles_merge(self, base, parts, k1, k2, idx, orient=0):
        """o(base) + mean over the tiles covering each pixel of parts (EVSSM grids_inverse)."""
        base = _orient(base, orient)
        n, h, w, c = base.shape
        parts = parts.view(n, len(idx), k1, k2, c)
        acc = torch.zeros((n, h, w, c), device=base.device, dtype=torch.float32)
        cnt = torch.zeros((1, h, w, 1), device=base.device, dtype=torch.float32)
        for t, (i, j) in enumerate(idx):
            acc[:, i:i + k1, j:j + k2] += parts[:, t].float()
            cnt[:, i:i + k1, j:j + k2] += 1.0
        return (base.float() + acc / cnt).to(self.dtype).contiguous()
