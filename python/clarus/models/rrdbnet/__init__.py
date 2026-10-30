"""RRDBNet (ESRGAN / Real-ESRGAN generator, basicsr `archs/rrdbnet_arch.py`), x4 / x2 / x1.

model config keys:
  checkpoint:  .pth with `params_ema` / `params` (e.g. RealESRGAN_x4plus.pth)
  network_g:   {type: RRDBNet, num_in_ch: 3, num_out_ch: 3, scale: 4, num_feat: 64,
                num_block: 23, num_grow_ch: 32}

Inference layout: every residual dense block works inside one NHWC buffer of
num_feat + 4 * num_grow_ch channels. Each conv reads a channel prefix of the buffer and
writes its output into the next slice, so the dense concatenations cost no copies.
The 0.2 residual scale of the fifth conv is fused into that conv's epilogue, and the
nearest x2 upsampling is fused into the following conv's input addressing.
"""
from __future__ import annotations

import os

import torch

from .. import IOSpec, RestorationModel, register_model
from .._fold import resolve_network_g

DEFAULTS = dict(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32)


@register_model("rrdbnet", "realesrgan", "esrgan")
class RRDBNet(RestorationModel):
    def __init__(self, cfg, ops, device, dtype):
        super().__init__(cfg, ops, device, dtype)
        net = dict(DEFAULTS)
        net.update(resolve_network_g(cfg, cfg.get("_config_dir", "")))
        if net.pop("type", "RRDBNet") != "RRDBNet":
            raise ValueError("network_g.type must be RRDBNet")
        if net["num_in_ch"] != 3 or net["num_out_ch"] != 3:
            raise ValueError("RRDBNet: only 3-channel in / out is supported by the RGB8 IO path")
        if net["scale"] not in (1, 2, 4):
            raise ValueError("RRDBNet scale must be 1, 2 or 4")
        self.net_cfg = net
        ckpt = cfg["checkpoint"]
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(cfg.get("_config_dir", ""), ckpt)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        key = cfg.get("params_key") or ("params_ema" if "params_ema" in sd else "params" if "params" in sd else None)
        sd = sd[key] if key else sd
        self.w = self._load(sd, net)
        self.io = IOSpec(in_channels=3, out_channels=3, scale=net["scale"], pad_multiple=1)

    def _load(self, sd, net):
        used = set()
        w = {}
        for k, v in sd.items():
            used.add(k)
            keep = k.endswith(".bias")
            w[k] = v.float().to(self.device, torch.float32 if keep else self.dtype).contiguous()
        need = ["conv_first.weight", "conv_body.weight", "conv_up1.weight", "conv_up2.weight", "conv_hr.weight",
                "conv_last.weight"] + [f"body.{i}.rdb{j}.conv{c}.weight" for i in range(net["num_block"])
                                       for j in (1, 2, 3) for c in range(1, 6)]
        missing = [k for k in need if k not in w]
        extra = [k for k in w if not any(k.startswith(p.rsplit(".", 1)[0]) for p in need)]
        if missing or extra:
            raise ValueError(f"checkpoint does not match network_g: missing {missing[:4]}, unexpected {extra[:4]}")
        return w

    def _conv(self, x, name, **kw):
        return self.ops.conv3x3(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"], **kw)

    @torch.no_grad()
    def forward(self, inp):
        o, net = self.ops, self.net_cfg
        nf, g = net["num_feat"], net["num_grow_ch"]
        n, _, h, wd = inp.shape
        x = inp.permute(0, 2, 3, 1).to(self.dtype).contiguous()
        if net["scale"] == 2:
            x = o.pixel_unshuffle(x)
        elif net["scale"] == 1:
            x = o.pixel_unshuffle(x, 4)
        hh, ww = x.shape[1:3]
        feat = self._conv(x, "conv_first")
        # dense buffers [x | x1 | x2 | x3 | x4]; b0[:nf] holds the RRDB input / output
        b0, b1, b2 = (torch.empty((n, hh, ww, nf + 4 * g), device=x.device, dtype=self.dtype) for _ in range(3))
        b0[..., :nf].copy_(feat)

        def rdb(src, dst, p):
            for c in range(1, 5):
                self._conv(src[..., :nf + (c - 1) * g], f"{p}.conv{c}", act="lrelu",
                           out=src[..., nf + (c - 1) * g:nf + c * g])
            # 0.2 * conv5(dense) + x, written into the next block's input slice
            self._conv(src, f"{p}.conv5", out_scale=0.2, residual=src[..., :nf], out=dst[..., :nf])

        for i in range(net["num_block"]):
            rdb(b0, b1, f"body.{i}.rdb1")
            rdb(b1, b2, f"body.{i}.rdb2")
            rdb(b2, b1, f"body.{i}.rdb3")
            o.add_scaled(b1[..., :nf], b0[..., :nf], 0.2, out=b0[..., :nf])   # RRDB: 0.2 * y + x
        cur = b0[..., :nf]
        body = self._conv(cur, "conv_body", residual=feat)  # feat + conv_body(body(feat))
        f = self._conv(body, "conv_up1", act="lrelu", up2=True)
        f = self._conv(f, "conv_up2", act="lrelu", up2=True)
        f = self._conv(f, "conv_hr", act="lrelu")
        out = self._conv(f, "conv_last", out_dtype=torch.float32)
        return out.permute(0, 3, 1, 2)

    def describe(self):
        d = super().describe()
        d["network_g"] = self.net_cfg
        return d
