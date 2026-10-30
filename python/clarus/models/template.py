"""Template for adding a model to the engine — copy this file and fill in the TODOs.

It is also a working example: a fixed (weight-free) unsharp-mask "restorer", loaded as a
plugin, so the whole path (config -> registry -> ops dispatch -> worker -> engine) can be
exercised without a checkpoint:

    clarus eval --lq LQ --gt GT --set model.name=example_unsharp \\
        --set model.plugins=[clarus.models.template] --set model.amount=0.6

Checklist for a real model (see the clarus-adapt skill for the full workflow):
  1. Arch:    read the network description from config (`network_g` / `arch_from`), reject
              options you do not implement with a clear error.
  2. Weights: load the checkpoint and fold it once into the canonical layout
              (dense convs [Cout, Cin, kh, kw], depthwise [C, 1, kh, kw], 1x1 / linear [K, N]).
  3. Forward: NHWC activations in `self.dtype`, every heavy op through `self.ops.<op>(...)`;
              capture-safe (no .item(), no host syncs, no data-dependent Python branches).
  4. IOSpec:  channels, scale (super-resolution), pad multiple.
  5. Parity:  compare against the training-code network (tools/parity.py pattern).
"""
from __future__ import annotations

import math

import torch

from . import IOSpec, RestorationModel, register_model


@register_model("example_unsharp")
class ExampleUnsharp(RestorationModel):
    """y = x + amount * (x - gaussian5x5(x)), as one depthwise conv through the ops layer."""

    def __init__(self, cfg, ops, device, dtype):
        super().__init__(cfg, ops, device, dtype)
        # 1) "arch" from config (a real model reads network_g / arch_from here)
        amount = float(cfg.get("amount", 0.5))
        sigma = float(cfg.get("sigma", 1.0))
        # 2) "weights", already in the canonical depthwise layout [C, 1, kh, kw]
        g = torch.tensor([math.exp(-(i - 2) ** 2 / (2 * sigma * sigma)) for i in range(5)])
        k2 = torch.outer(g, g) / g.sum() ** 2
        delta = torch.zeros(5, 5)
        delta[2, 2] = 1.0
        kernel = (1 + amount) * delta - amount * k2       # x + a(x - blur) as one filter
        self.w = kernel.expand(3, 1, 5, 5).contiguous().to(device, dtype)
        self.b = torch.zeros(3, device=device)
        # 4) IO contract
        self.io = IOSpec(in_channels=3, out_channels=3, scale=1, pad_multiple=1)

    def forward(self, x):
        # 3) NHWC in the compute dtype; dwconv is a dispatched op (5x5 -> torch fallback on CUDA,
        #    since the Triton dwconv kernel only handles 3x3: shows up in the report as dwconv->torch)
        h = x.permute(0, 2, 3, 1).contiguous().to(self.dtype)
        y = self.ops.dwconv(h, self.w, self.b)
        return y.permute(0, 3, 1, 2).float()


@register_model("example_bilinear_x2")
class ExampleBilinearX2(RestorationModel):
    """Super-resolution IO example (IOSpec.scale = 2): bilinear 2x upsampling, no weights.

    Exercises the scale-aware paths (slot sizes, GT size check, output size) end to end.
    """

    def __init__(self, cfg, ops, device, dtype):
        super().__init__(cfg, ops, device, dtype)
        self.io = IOSpec(in_channels=3, out_channels=3, scale=2, pad_multiple=1)

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
