"""Compare on-GPU PSNR/SSIM with basicsr.metrics on real GoPro outputs (dev tool)."""
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import GT, LQ  # noqa: E402
from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim  # noqa: E402
from clarus.metrics import TorchMetrics, TritonMetrics  # noqa: E402

m = (TorchMetrics if "--torch" in sys.argv else TritonMetrics)(torch.device("cuda"))
if "--no-tf32" in sys.argv:
    torch.backends.cudnn.allow_tf32 = False
names = sorted(os.listdir(LQ))[::150]
worst_p = worst_s = 0.0
for name in names:
    a = cv2.imread(os.path.join(LQ, name))  # use the blurred input as the "restored" image
    b = cv2.imread(os.path.join(GT, name))
    p_ref = calculate_psnr(a, b, 0)
    s_ref = calculate_ssim(a, b, 0)
    res = torch.zeros(2, device="cuda", dtype=torch.float64)
    m.compute(torch.from_numpy(a[:, :, ::-1].copy()).cuda(), torch.from_numpy(b[:, :, ::-1].copy()).cuda(), res)
    p, s = res.tolist()
    worst_p = max(worst_p, abs(p - p_ref))
    worst_s = max(worst_s, abs(s - s_ref))
    print(f"{name}: psnr {p:.10f} ref {p_ref:.10f} | ssim {s:.8f} ref {s_ref:.8f}")
print(f"max |dpsnr|={worst_p:.3e} max |dssim|={worst_s:.3e}")

if "--fp64" in sys.argv:
    # float64 ground truth of the same 3-D SSIM (CPU), two images
    from basicsr.metrics.psnr_ssim import _generate_3d_gaussian_kernel
    k = _generate_3d_gaussian_kernel().double()
    for name in names[:2]:
        a = torch.from_numpy(cv2.imread(os.path.join(LQ, name)).astype(np.float64))
        b = torch.from_numpy(cv2.imread(os.path.join(GT, name)).astype(np.float64))
        f = lambda x: k(x[None, None])[0, 0]
        with torch.no_grad():
            mu1, mu2 = f(a), f(b)
            s1, s2, s12 = f(a * a) - mu1 ** 2, f(b * b) - mu2 ** 2, f(a * b) - mu1 * mu2
            C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
            ss = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        res = torch.zeros(2, device="cuda", dtype=torch.float64)
        m.compute(torch.from_numpy(cv2.imread(os.path.join(LQ, name))[:, :, ::-1].copy()).cuda(),
                  torch.from_numpy(cv2.imread(os.path.join(GT, name))[:, :, ::-1].copy()).cuda(), res)
        print(f"fp64 truth {name}: {ss.mean().item():.8f}  gpu: {res[1].item():.8f}")
