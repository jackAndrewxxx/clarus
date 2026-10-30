"""NAFNet / NAFNetLocal (Chen et al., ECCV 2022): official `basicsr/models/archs/NAFNet_arch.py`.

model config keys:
  checkpoint:  basicsr .pth (`params` / `params_ema`), e.g. NAFNet-GoPro-width64.pth
  arch_from:   basicsr yml whose `network_g` describes the network (training or test yml)
  network_g:   inline network_g dict; merged over arch_from (shallow, per key)
  options:
    residual_fp32_from_level: 3   # pyramid level from which the residual stream is fp32
                                  # (0 = all fp32, i.e. --safe; >= levels+1 = never)

NAFNetLocal is TLC (Local_Base.convert): every AdaptiveAvgPool2d(1) becomes a local mean whose
window is 1.5x the feature size at `train_size`. Inference layout: NHWC; LayerNorm affine
folded into the following 1x1 conv; SimpleGate fused into the depthwise conv; the channel
attention (SCA) multiply fused into conv3's prologue; beta / gamma kept as fp32 epilogue scales
(values as small as 4e-6 would underflow if folded into fp16 weights).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import IOSpec, RestorationModel, register_model
from .._fold import Folder, kn, ln_fold, load_state_dict, resolve_network_g

TYPES = {"NAFNet": False, "NAFNetLocal": True}


@register_model("nafnet")
class NAFNet(RestorationModel):
    def __init__(self, cfg, ops, device, dtype):
        super().__init__(cfg, ops, device, dtype)
        net = resolve_network_g(cfg, cfg.get("_config_dir", ""))
        kind = net.get("type", "NAFNet")
        if kind not in TYPES:
            raise ValueError(f"network_g.type {kind!r} not supported; supported: {sorted(TYPES)}")
        if net.get("img_channel", 3) != 3:
            raise ValueError("img_channel != 3 is not supported by the RGB8 IO path")
        if TYPES[kind] and net.get("fast_imp"):
            raise ValueError("TLC fast_imp=True is not implemented")
        self.net_cfg = net
        self.enc = [int(v) for v in net.get("enc_blk_nums", [])]
        self.dec = [int(v) for v in net.get("dec_blk_nums", [])]
        self.mid = int(net.get("middle_blk_num", 1))
        ts = net.get("train_size", (1, 3, 256, 256))
        self.tlc = (int(ts[-2]), int(ts[-1])) if TYPES[kind] else None
        self.res_fp32_level = int((cfg.get("options") or {}).get("residual_fp32_from_level", 3))
        f = Folder(load_state_dict(cfg), device, dtype)
        self._fold(f)
        f.check_consumed()
        self.w = f.t
        self.padder = 2 ** len(self.enc)
        self.io = IOSpec(in_channels=3, out_channels=3, scale=1, pad_multiple=self.padder)

    # ---------------------------------------------------------------- weights
    def _block(self, f, src, dst):
        w, b = ln_fold(f.get(f"{src}.conv1.weight"), f.get(f"{src}.conv1.bias"),
                       f.get(f"{src}.norm1.weight"), f.get(f"{src}.norm1.bias"))
        f.put(f"{dst}.conv1.w", w)
        f.put(f"{dst}.conv1.bias", b, True)
        f.put(f"{dst}.dw.w", f.get(f"{src}.conv2.weight"))                    # [2C, 1, 3, 3]
        f.put(f"{dst}.dw.bias", f.get(f"{src}.conv2.bias"), True)
        f.put(f"{dst}.sca.w", kn(f.get(f"{src}.sca.1.weight")))
        f.put(f"{dst}.sca.bias", f.get(f"{src}.sca.1.bias"), True)
        f.put(f"{dst}.conv3.w", kn(f.get(f"{src}.conv3.weight")))
        f.put(f"{dst}.conv3.bias", f.get(f"{src}.conv3.bias"), True)
        f.put(f"{dst}.conv3.scale", f.get(f"{src}.beta").flatten(), True)
        w, b = ln_fold(f.get(f"{src}.conv4.weight"), f.get(f"{src}.conv4.bias"),
                       f.get(f"{src}.norm2.weight"), f.get(f"{src}.norm2.bias"))
        f.put(f"{dst}.conv4.w", w)
        f.put(f"{dst}.conv4.bias", b, True)
        f.put(f"{dst}.conv5.w", kn(f.get(f"{src}.conv5.weight")))
        f.put(f"{dst}.conv5.bias", f.get(f"{src}.conv5.bias"), True)
        f.put(f"{dst}.conv5.scale", f.get(f"{src}.gamma").flatten(), True)

    def _fold(self, f):
        f.put("intro.w", f.get("intro.weight"), True)
        f.put("intro.bias", f.get("intro.bias"), True)
        f.put("ending.w", f.get("ending.weight"))
        f.put("ending.bias", f.get("ending.bias"), True)
        for i, n in enumerate(self.enc):
            for j in range(n):
                self._block(f, f"encoders.{i}.{j}", f"enc{i}.{j}")
            f.put(f"down{i}.w", f.get(f"downs.{i}.weight"))                  # [2C, C, 2, 2]
            f.put(f"down{i}.bias", f.get(f"downs.{i}.bias"), True)
        for j in range(self.mid):
            self._block(f, f"middle_blks.{j}", f"mid.{j}")
        for i, n in enumerate(self.dec):
            w = f.get(f"ups.{i}.0.weight").flatten(1)      # [4C', C]; out channel o = c*4 + dy*2 + dx
            cout = w.shape[0] // 4
            # reorder N to (dy, dx, c): the GEMM output [P, 2, 2, C'] is NHWC after PixelShuffle
            f.put(f"up{i}.w", w.view(cout, 2, 2, -1).permute(1, 2, 0, 3).reshape(4 * cout, -1).t())
            for j in range(n):
                self._block(f, f"decoders.{i}.{j}", f"dec{i}.{j}")

    # ---------------------------------------------------------------- forward
    def _rdt(self, level):
        return torch.float32 if level >= self.res_fp32_level else self.dtype

    def _tlc_kernel(self, level):
        """(kh, kw) of the local mean replacing AdaptiveAvgPool2d(1) at this level (global
        without TLC): feature size at train_size * base_size // train_size, base = 1.5x."""
        if self.tlc is None:
            return (1 << 30, 1 << 30)
        return tuple((t >> level) * int(t * 1.5) // t for t in self.tlc)

    def _nafblock(self, x, p, level):
        o, w = self.ops, self.w
        t = o.linear(x, w[f"{p}.conv1.w"], w[f"{p}.conv1.bias"], ln=True)
        t = o.dwconv(t, w[f"{p}.dw.w"], w[f"{p}.dw.bias"], gate=True)
        sca = o.linear(o.boxmean(t, self._tlc_kernel(level)), w[f"{p}.sca.w"], w[f"{p}.sca.bias"])
        y = o.linear(t, w[f"{p}.conv3.w"], w[f"{p}.conv3.bias"], scale=w[f"{p}.conv3.scale"], residual=x,
                     mul=sca, out_dtype=self._rdt(level))
        u = o.linear(y, w[f"{p}.conv4.w"], w[f"{p}.conv4.bias"], ln=True, gate=True)
        return o.linear(u, w[f"{p}.conv5.w"], w[f"{p}.conv5.bias"], scale=w[f"{p}.conv5.scale"], residual=y,
                        out_dtype=self._rdt(level))

    @torch.no_grad()
    def forward(self, inp):
        o, w = self.ops, self.w
        n, _, h, wd = inp.shape
        ph, pw = (self.padder - h % self.padder) % self.padder, (self.padder - wd % self.padder) % self.padder
        if ph or pw:
            inp = F.pad(inp, (0, pw, 0, ph))                                  # zeros, like check_image_size
        x = o.intro(inp, w["intro.w"], w["intro.bias"], out_dtype=self._rdt(0))
        skips = []
        for i, nb in enumerate(self.enc):
            for j in range(nb):
                x = self._nafblock(x, f"enc{i}.{j}", i)
            skips.append(x)
            x = o.down(x, w[f"down{i}.w"], w[f"down{i}.bias"], out_dtype=self._rdt(i + 1))
        for j in range(self.mid):
            x = self._nafblock(x, f"mid.{j}", len(self.enc))
        for i, nb in enumerate(self.dec):
            level = len(self.enc) - 1 - i
            x = o.up(x, w[f"up{i}.w"], residual=skips[level], out_dtype=self._rdt(level))
            for j in range(nb):
                x = self._nafblock(x, f"dec{i}.{j}", level)
        out = o.ending(x, w["ending.w"], w["ending.bias"], inp)
        return out[:, :, :h, :wd]

    def describe(self):
        d = super().describe()
        d["network_g"] = self.net_cfg
        return d
