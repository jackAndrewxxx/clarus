"""EVSSM (Kong et al., CVPR 2025): official `models/EVSSM.py` (GoPro / RealBlur weights).

model config keys:
  checkpoint:  official .pth (`params`, e.g. net_g_GoPro.pth)
  network_g:   {type: EVSSM, dim, num_blocks, ffn_expansion_factor, bias}

Every EVS block re-orients the feature map (transpose for even block index, 180-degree flip for
odd), splits it into overlapping k x k crops (k = 384 / 192 / 96 per level), runs a one-direction
selective scan (Mamba) over each crop's raster sequence and averages the crops back. Here the
orientation is folded into the crop gather / merge addressing, the norm affine into in_proj,
and out_norm + `* gelu(z)` into the out_proj GEMM prologue. The input is reflect-padded to a
multiple of 4 (the official network requires it).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import IOSpec, RestorationModel, register_model
from .._fold import Folder, kn, ln_fold, load_state_dict, resolve_network_g
from ...ops.torch_ref import grid_tiles

DEFAULTS = dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[6, 6, 12], num_refinement_blocks=4,
                ffn_expansion_factor=3, bias=False)
PATCH = (384, 192, 96)  # crop size per pyramid level (hard-coded upstream)


@register_model("evssm")
class EVSSM(RestorationModel):
    def __init__(self, cfg, ops, device, dtype):
        super().__init__(cfg, ops, device, dtype)
        net = dict(DEFAULTS)
        net.update(resolve_network_g(cfg, cfg.get("_config_dir", "")))
        if net.pop("type", "EVSSM") != "EVSSM":
            raise ValueError("network_g.type must be EVSSM")
        if net["inp_channels"] != 3 or net["out_channels"] != 3:
            raise ValueError("EVSSM: only 3-channel in / out is supported")
        self.net_cfg = net
        f = Folder(load_state_dict(cfg), device, dtype)
        d, nb = net["dim"], net["num_blocks"]
        self.stages = [("encoder_level1", nb[0], 0), ("encoder_level2", nb[1], 1), ("encoder_level3", nb[2], 2),
                       ("decoder_level3", nb[2], 2), ("decoder_level2", nb[1], 1), ("decoder_level1", nb[0], 0)]
        for name, n, lvl in self.stages:
            for i in range(n):
                self._block(f, f"{name}.block{i}", d * 2 ** lvl)
        f.put("embed.w", f.get("patch_embed.proj.weight"), True)
        f.put("embed.bias", f.opt("patch_embed.proj.bias", (d,)), True)
        for name in ("down1_2", "down2_3", "up3_2", "up2_1"):
            f.put(f"{name}.w", f.get(f"{name}.body.1.weight"))
        f.put("output.w", f.get("output.weight"))
        f.put("output.bias", f.opt("output.bias", (3,)), True)
        f.check_consumed()
        self.w = f.t
        self.io = IOSpec(in_channels=3, out_channels=3, scale=1, pad_multiple=1)  # pads (reflect) itself

    def _block(self, f, p, dim):
        a = f"{p}.attn"
        w, b = ln_fold(f.get(f"{a}.in_proj.weight"), f.opt(f"{a}.in_proj.bias"),
                       f.get(f"{p}.norm1.body.weight"), f.get(f"{p}.norm1.body.bias"))
        di = w.shape[1] // 2
        f.put(f"{p}.in_x.w", w[:, :di])
        f.put(f"{p}.in_x.bias", b[:di], True)
        f.put(f"{p}.in_z.w", w[:, di:])
        f.put(f"{p}.in_z.bias", b[di:], True)
        f.put(f"{p}.conv.w", f.get(f"{a}.conv2d.weight"))
        f.put(f"{p}.conv.bias", f.opt(f"{a}.conv2d.bias", (di,)), True)
        f.put(f"{p}.xproj.w", f.get(f"{a}.x_proj_weight")[0].t())           # [di, r + 2n]
        xc = f.get(f"{a}.x_conv.weight")[:, 0]                               # [r + 2n, k]
        f.put(f"{p}.xconv.w", xc, True)
        f.put(f"{p}.xconv.bias", f.opt(f"{a}.x_conv.bias", (xc.shape[0],)), True)
        f.put(f"{p}.dt.w", f.get(f"{a}.dt_projs_weight")[0], True)          # [di, r]
        f.put(f"{p}.dt.bias", f.get(f"{a}.dt_projs_bias")[0], True)
        f.put(f"{p}.A", -torch.exp(f.get(f"{a}.A_logs")), True)             # [di, n]
        f.put(f"{p}.D", f.get(f"{a}.Ds"), True)
        f.put(f"{p}.onorm.w", f.get(f"{a}.out_norm.weight"), True)
        f.put(f"{p}.onorm.b", f.get(f"{a}.out_norm.bias"), True)
        f.put(f"{p}.out.w", kn(f.get(f"{a}.out_proj.weight")))
        f.put(f"{p}.out.bias", f.opt(f"{a}.out_proj.bias", (dim,)), True)
        w, b = ln_fold(f.get(f"{p}.ffn.project_in.weight"), f.opt(f"{p}.ffn.project_in.bias"),
                       f.get(f"{p}.norm2.body.weight"), f.get(f"{p}.norm2.body.bias"))
        f.put(f"{p}.ffn_in.w", w)
        f.put(f"{p}.ffn_in.bias", b, True)
        dw = f.get(f"{p}.ffn.dwconv.weight")
        f.put(f"{p}.ffn_dw.w", dw)
        f.put(f"{p}.ffn_dw.bias", f.opt(f"{p}.ffn.dwconv.bias", (dw.shape[0],)), True)
        f.put(f"{p}.ffn_out.w", kn(f.get(f"{p}.ffn.project_out.weight")))
        f.put(f"{p}.ffn_out.bias", f.opt(f"{p}.ffn.project_out.bias", (dim,)), True)
        fft = f.get(f"{p}.ffn.fft")
        f.put(f"{p}.ffn_fft", fft.reshape(fft.shape[0], *fft.shape[-2:]), True)

    # ---------------------------------------------------------------- forward
    def _run_block(self, x, p, i, patch):
        o, w = self.ops, self.w
        orient = 2 if i % 2 == 0 else 1                                       # transpose / flip H and W
        h, wd = (x.shape[2], x.shape[1]) if orient == 2 else x.shape[1:3]
        k1, k2, idx = grid_tiles(h, wd, patch)
        t = o.tiles_gather(x, k1, k2, idx, orient)
        xs = o.linear(t, w[f"{p}.in_x.w"], w[f"{p}.in_x.bias"], ln="rms", ln_eps=1e-6)
        z = o.linear(t, w[f"{p}.in_z.w"], w[f"{p}.in_z.bias"], ln="rms", ln_eps=1e-6)
        u = o.dwconv(xs, w[f"{p}.conv.w"], w[f"{p}.conv.bias"], act="gelu")
        dbl = o.linear(u, w[f"{p}.xproj.w"], out_dtype=torch.float32)
        dbl = o.seq_dwconv1d(dbl, w[f"{p}.xconv.w"], w[f"{p}.xconv.bias"])
        y = o.selective_scan(u, dbl, w[f"{p}.dt.w"], w[f"{p}.dt.bias"], w[f"{p}.A"], w[f"{p}.D"])
        a = o.ln_mul_linear(y, w[f"{p}.onorm.w"], w[f"{p}.onorm.b"], z, w[f"{p}.out.w"], w[f"{p}.out.bias"],
                            m_act="gelu", ln_eps=1e-5)
        x = o.tiles_merge(x, a, k1, k2, idx, orient)
        u = o.linear(x, w[f"{p}.ffn_in.w"], w[f"{p}.ffn_in.bias"], ln="rms", ln_eps=1e-6)
        u = o.dwconv(u, w[f"{p}.ffn_dw.w"], w[f"{p}.ffn_dw.bias"], gate="gelu")
        return o.linear_patch_filter(u, w[f"{p}.ffn_out.w"], w[f"{p}.ffn_out.bias"], w[f"{p}.ffn_fft"], residual=x)

    def _stage(self, x, name):
        n, lvl = next((s[1], s[2]) for s in self.stages if s[0] == name)
        for i in range(n):
            x = self._run_block(x, f"{name}.block{i}", i, PATCH[lvl])
        return x

    @torch.no_grad()
    def forward(self, inp):
        o, w = self.ops, self.w
        n, _, h, wd = inp.shape
        ph, pw = (4 - h % 4) % 4, (4 - wd % 4) % 4
        x_in = F.pad(inp, (0, pw, 0, ph), mode="reflect") if (ph or pw) else inp
        e1 = self._stage(o.intro(x_in, w["embed.w"], w["embed.bias"]), "encoder_level1")
        e2 = self._stage(o.conv3x3(o.resize_bilinear(e1, 0.5), w["down1_2.w"]), "encoder_level2")
        x = self._stage(o.conv3x3(o.resize_bilinear(e2, 0.5), w["down2_3.w"]), "encoder_level3")
        x = self._stage(x, "decoder_level3")
        x = self._stage(o.conv3x3(o.resize_bilinear(x, 2), w["up3_2.w"], residual=e2), "decoder_level2")
        x = self._stage(o.conv3x3(o.resize_bilinear(x, 2), w["up2_1.w"], residual=e1), "decoder_level1")
        out = o.ending(x, w["output.w"], w["output.bias"], x_in)
        return out[:, :, :h, :wd]

    def describe(self):
        d = super().describe()
        d["network_g"] = self.net_cfg
        return d
