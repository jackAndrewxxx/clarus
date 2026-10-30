"""Restormer (Zamir et al., CVPR 2022): official `basicsr/models/archs/restormer_arch.py`.

model config keys:
  checkpoint:  official .pth (`params`)
  network_g:   the official `network_g` (dim, num_blocks, num_refinement_blocks, heads,
               ffn_expansion_factor, bias, LayerNorm_type); or `arch_from: Options/<task>.yml`
  options:
    residual_fp32_from_level: 99  # pyramid level from which the residual stream is fp32
                                  # (measured peak |x| on GoPro is ~35, so fp16 is safe; 0 = all fp32)

Inference layout: NHWC activations; LayerNorm affine folded into the following 1x1 conv
(qkv, project_in); MDTA attention is evaluated as a per-image channel-mixing matrix that is
multiplied into project_out, so attention costs one Gram reduction and one GEMM per block.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .. import IOSpec, RestorationModel, register_model
from .._fold import resolve_network_g

DEFAULTS = dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias",
                dual_pixel_task=False)


def _kn(w):
    """[N, K, 1, 1] -> [K, N]"""
    return w.reshape(w.shape[0], -1).t().contiguous()


@register_model("restormer")
class Restormer(RestorationModel):
    def __init__(self, cfg, ops, device, dtype):
        super().__init__(cfg, ops, device, dtype)
        net = dict(DEFAULTS)
        net.update(resolve_network_g(cfg, cfg.get("_config_dir", "")))
        if net.pop("type", "Restormer") != "Restormer":
            raise ValueError("network_g.type must be Restormer")
        if net["LayerNorm_type"] != "WithBias":
            raise ValueError("Restormer LayerNorm_type BiasFree is not implemented")
        if net["dual_pixel_task"] or net["inp_channels"] != 3 or net["out_channels"] != 3:
            raise ValueError("Restormer dual-pixel / non-RGB variants are not implemented")
        self.net_cfg = net
        ckpt = cfg["checkpoint"]
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(cfg.get("_config_dir", ""), ckpt)
        sd = torch.load(ckpt, map_location="cpu", weights_only=True)
        sd = sd.get("params", sd.get("params_ema", sd))
        sd = {k.removeprefix("module."): v.float() for k, v in sd.items()}
        self.res_fp32_level = int((cfg.get("options") or {}).get("residual_fp32_from_level", 99))
        self.w, self.blocks = self._fold(sd, net)
        self.io = IOSpec(in_channels=3, out_channels=3, scale=1, pad_multiple=1)  # pads (reflect) itself

    # ---------------------------------------------------------------- weights
    def _fold(self, sd, net):
        used = set()

        def g(k):
            if k not in sd:
                raise ValueError(f"checkpoint has no tensor {k!r} (network_g does not match the checkpoint?)")
            used.add(k)
            return sd[k]

        def opt(k):
            return g(k) if k in sd else None

        dev, dt = self.device, self.dtype
        t = {}

        def put(name, v, keep_fp32=False):
            t[name] = v.to(device=dev, dtype=torch.float32 if keep_fp32 else dt).contiguous()

        def ln_fold(w, b, ln_w, ln_b):
            w = w.flatten(1)
            b2 = w @ ln_b + (b if b is not None else 0.0)
            return (w * ln_w[None, :]).t().contiguous(), b2

        blocks = {}

        def block(src, dst, heads):
            w, b = ln_fold(g(f"{src}.attn.qkv.weight"), opt(f"{src}.attn.qkv.bias"),
                           g(f"{src}.norm1.body.weight"), g(f"{src}.norm1.body.bias"))
            put(f"{dst}.qkv.w", w)
            put(f"{dst}.qkv.bias", b, True)
            dw = g(f"{src}.attn.qkv_dwconv.weight")
            put(f"{dst}.qkv_dw.w", dw)
            put(f"{dst}.qkv_dw.bias", opt(f"{src}.attn.qkv_dwconv.bias") if f"{src}.attn.qkv_dwconv.bias" in sd
                else torch.zeros(dw.shape[0]), True)
            put(f"{dst}.temp", g(f"{src}.attn.temperature").flatten(), True)
            put(f"{dst}.proj.w", _kn(g(f"{src}.attn.project_out.weight")), True)   # fp32: mixed per image
            pb = opt(f"{src}.attn.project_out.bias")
            put(f"{dst}.proj.bias", pb if pb is not None else torch.zeros(t[f"{dst}.proj.w"].shape[1]), True)
            w, b = ln_fold(g(f"{src}.ffn.project_in.weight"), opt(f"{src}.ffn.project_in.bias"),
                           g(f"{src}.norm2.body.weight"), g(f"{src}.norm2.body.bias"))
            dw = g(f"{src}.ffn.dwconv.weight")
            db = opt(f"{src}.ffn.dwconv.bias")
            db = db if db is not None else torch.zeros(dw.shape[0])
            wo = _kn(g(f"{src}.ffn.project_out.weight"))                  # [hidden, C]
            # pad the hidden width (e.g. int(96 * 2.66) = 255) to a multiple of 16 with zero
            # channels: both gate halves then start 16-aligned, so loads stay vectorized
            hid = wo.shape[0]
            hp = (hid + 15) // 16 * 16
            idx = torch.cat([torch.arange(hid), hp + torch.arange(hid)])
            wp = torch.zeros(w.shape[0], 2 * hp)
            wp[:, idx] = w
            bp = torch.zeros(2 * hp)
            bp[idx] = b
            dwp = torch.zeros(2 * hp, 1, 3, 3)
            dwp[idx] = dw
            dbp = torch.zeros(2 * hp)
            dbp[idx] = db
            wop = torch.zeros(hp, wo.shape[1])
            wop[:hid] = wo
            put(f"{dst}.ffn_in.w", wp)
            put(f"{dst}.ffn_in.bias", bp, True)
            put(f"{dst}.ffn_dw.w", dwp)
            put(f"{dst}.ffn_dw.bias", dbp, True)
            put(f"{dst}.ffn_out.w", wop)
            ob = opt(f"{src}.ffn.project_out.bias")
            put(f"{dst}.ffn_out.bias", ob if ob is not None else torch.zeros(t[f"{dst}.ffn_out.w"].shape[1]), True)
            blocks[dst] = heads

        nb, hd = net["num_blocks"], net["heads"]
        stages = [("encoder_level1", nb[0], hd[0]), ("encoder_level2", nb[1], hd[1]),
                  ("encoder_level3", nb[2], hd[2]), ("latent", nb[3], hd[3]),
                  ("decoder_level3", nb[2], hd[2]), ("decoder_level2", nb[1], hd[1]),
                  ("decoder_level1", nb[0], hd[0]), ("refinement", net["num_refinement_blocks"], hd[0])]
        for name, n, h in stages:
            for i in range(n):
                block(f"{name}.{i}", f"{name}.{i}", h)
        put("embed.w", g("patch_embed.proj.weight"), True)
        put("embed.bias", opt("patch_embed.proj.bias") if "patch_embed.proj.bias" in sd
            else torch.zeros(sd["patch_embed.proj.weight"].shape[0]), True)
        for name in ("down1_2", "down2_3", "down3_4", "up4_3", "up3_2", "up2_1"):
            put(f"{name}.w", g(f"{name}.body.0.weight"))
        for lvl in (3, 2):
            w = g(f"reduce_chan_level{lvl}.weight").flatten(1)      # [C, 2C]: cat(up, enc)
            c = w.shape[0]
            put(f"reduce{lvl}.up.w", w[:, :c].t().contiguous())
            put(f"reduce{lvl}.enc.w", w[:, c:].t().contiguous())
            rb = opt(f"reduce_chan_level{lvl}.bias")
            put(f"reduce{lvl}.bias", rb if rb is not None else torch.zeros(c), True)
        put("output.w", g("output.weight"))
        ob = opt("output.bias")
        put("output.bias", ob if ob is not None else torch.zeros(3), True)
        unused = [k for k in sd if k not in used]
        if unused:
            raise ValueError(f"checkpoint tensors not consumed: {unused[:8]}")
        self.stages = stages
        return t, blocks

    # ---------------------------------------------------------------- forward
    def _rdt(self, level):
        return torch.float32 if level >= self.res_fp32_level else self.dtype

    def _block(self, x, p, heads, level):
        o, w = self.ops, self.w
        qkv = o.linear(x, w[f"{p}.qkv.w"], w[f"{p}.qkv.bias"], ln=True, ln_eps=1e-5)
        qkv = o.dwconv(qkv, w[f"{p}.qkv_dw.w"], w[f"{p}.qkv_dw.bias"])
        a = o.channel_attention(qkv, heads, w[f"{p}.temp"])                 # [N, heads, c, c] fp32
        n, _, ch, _ = a.shape
        c = heads * ch
        # out = blockdiag(A_h) applied to v, then project_out: fold into one per-image matrix
        bd = torch.zeros(n, c, c, device=a.device, dtype=torch.float32)
        for h in range(heads):
            bd[:, h * ch:(h + 1) * ch, h * ch:(h + 1) * ch] = a[:, h].transpose(1, 2)
        mix = bd @ w[f"{p}.proj.w"]                                         # [N, C, C]
        x = o.linear_per_image(qkv[..., 2 * c:], mix, residual=x, out_dtype=self._rdt(level), bias=w[f"{p}.proj.bias"])
        u = o.linear(x, w[f"{p}.ffn_in.w"], w[f"{p}.ffn_in.bias"], ln=True, ln_eps=1e-5)
        u = o.dwconv(u, w[f"{p}.ffn_dw.w"], w[f"{p}.ffn_dw.bias"], gate="gelu")
        return o.linear(u, w[f"{p}.ffn_out.w"], w[f"{p}.ffn_out.bias"], residual=x, out_dtype=self._rdt(level))

    def _stage(self, x, name, n, heads, level):
        for i in range(n):
            x = self._block(x, f"{name}.{i}", heads, level)
        return x

    @torch.no_grad()
    def forward(self, inp):
        o, w = self.ops, self.w
        n, _, h, wd = inp.shape
        ph, pw = (8 - h % 8) % 8, (8 - wd % 8) % 8
        x_in = F.pad(inp, (0, pw, 0, ph), mode="reflect") if (ph or pw) else inp
        s = {name: (nb, hd) for name, nb, hd in self.stages}
        x = o.intro(x_in, w["embed.w"], w["embed.bias"], out_dtype=self._rdt(0))
        e1 = self._stage(x, "encoder_level1", *s["encoder_level1"], 0)
        x = o.pixel_unshuffle(o.conv3x3(e1, w["down1_2.w"], out_dtype=self._rdt(1)))
        e2 = self._stage(x, "encoder_level2", *s["encoder_level2"], 1)
        x = o.pixel_unshuffle(o.conv3x3(e2, w["down2_3.w"], out_dtype=self._rdt(2)))
        e3 = self._stage(x, "encoder_level3", *s["encoder_level3"], 2)
        x = o.pixel_unshuffle(o.conv3x3(e3, w["down3_4.w"], out_dtype=self._rdt(3)))
        x = self._stage(x, "latent", *s["latent"], 3)
        x = o.pixel_shuffle(o.conv3x3(x, w["up4_3.w"], out_dtype=self._rdt(2)))
        x = o.linear(x, w["reduce3.up.w"], w["reduce3.bias"],
                     residual=o.linear(e3, w["reduce3.enc.w"], out_dtype=torch.float32),
                     out_dtype=self._rdt(2))
        x = self._stage(x, "decoder_level3", *s["decoder_level3"], 2)
        x = o.pixel_shuffle(o.conv3x3(x, w["up3_2.w"], out_dtype=self._rdt(1)))
        x = o.linear(x, w["reduce2.up.w"], w["reduce2.bias"],
                     residual=o.linear(e2, w["reduce2.enc.w"], out_dtype=torch.float32),
                     out_dtype=self._rdt(1))
        x = self._stage(x, "decoder_level2", *s["decoder_level2"], 1)
        x = o.pixel_shuffle(o.conv3x3(x, w["up2_1.w"], out_dtype=self._rdt(0)))
        x = torch.cat([x, e1.to(x.dtype)], dim=-1)
        x = self._stage(x, "decoder_level1", *s["decoder_level1"], 0)
        x = self._stage(x, "refinement", *s["refinement"], 0)
        out = o.ending(x, w["output.w"], w["output.bias"], x_in)
        return out[:, :, :h, :wd]

    def describe(self):
        d = super().describe()
        d["network_g"] = self.net_cfg
        return d
