"""Channel (transposed) attention statistics for Restormer's MDTA.

gram(): per image and head, the c x c Gram matrix q^T k over all pixels plus the squared
norms of every q / k channel, as a split-K reduction straight from the NHWC qkv tensor
(fp16 inputs, fp32 accumulation). The softmax over the tiny c x c maps is left to torch.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def gram_kernel(QKV, G, NQ, NK, P, LD, C, CH, SPLIT_P, NSPLIT,
                BLOCK_P: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)          # image * heads + head
    sp = tl.program_id(1)           # pixel split
    heads = C // CH
    img = pid // heads
    h = pid % heads
    base = QKV + img.to(tl.int64) * P * LD
    ci = tl.arange(0, BLOCK_C)
    cm = ci < CH
    acc = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    sq = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sk = tl.zeros((BLOCK_C,), dtype=tl.float32)
    p0 = sp * SPLIT_P
    p1 = tl.minimum(p0 + SPLIT_P, P)
    rp0 = tl.arange(0, BLOCK_P)
    for p in range(p0, p1, BLOCK_P):
        rp = p + rp0
        m = (rp < p1)[:, None] & cm[None, :]
        off = rp.to(tl.int64)[:, None] * LD + ci[None, :]
        q = tl.load(base + off + h * CH, mask=m, other=0.0)
        k = tl.load(base + off + C + h * CH, mask=m, other=0.0)
        acc = tl.dot(tl.trans(q), k, acc)
        qf = q.to(tl.float32)
        kf = k.to(tl.float32)
        sq += tl.sum(qf * qf, axis=0)
        sk += tl.sum(kf * kf, axis=0)
    o = (pid * NSPLIT + sp).to(tl.int64)
    tl.store(G + o * BLOCK_C * BLOCK_C + ci[:, None] * BLOCK_C + ci[None, :], acc)
    tl.store(NQ + o * BLOCK_C + ci, sq)
    tl.store(NK + o * BLOCK_C + ci, sk)


def gram(qkv: torch.Tensor, heads: int, sm_count: int = 56):
    """qkv [N, H, W, 3C] contiguous -> (G [N, heads, c, c], |q|^2 [N, heads, c], |k|^2 [N, heads, c]) fp32."""
    n, h, w, c3 = qkv.shape
    c = c3 // 3
    ch = c // heads
    p = h * w
    bc = max(16, triton.next_power_of_2(ch))
    bp = 64
    target = max(1, (4 * sm_count) // (n * heads))
    split = max(bp, triton.cdiv(triton.cdiv(p, target), bp) * bp)
    nsplit = triton.cdiv(p, split)
    dev = qkv.device
    g = torch.empty((n * heads, nsplit, bc, bc), device=dev, dtype=torch.float32)
    nq = torch.empty((n * heads, nsplit, bc), device=dev, dtype=torch.float32)
    nk = torch.empty((n * heads, nsplit, bc), device=dev, dtype=torch.float32)
    gram_kernel[(n * heads, nsplit)](qkv, g, nq, nk, p, c3, c, ch, split, nsplit, BLOCK_P=bp, BLOCK_C=bc,
                                     num_warps=4, num_stages=3)
    g = g.sum(1)[:, :ch, :ch].view(n, heads, ch, ch)
    return g, nq.sum(1)[:, :ch].view(n, heads, ch), nk.sum(1)[:, :ch].view(n, heads, ch)
