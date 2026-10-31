"""FFTformer (Kong et al., CVPR 2023): official `basicsr/models/archs/fftformer_arch.py`.

model config keys:
  checkpoint:  official .pth (plain state dict, e.g. fftformer_GoPro.pth)
  params_key:  state-dict key in the checkpoint (default: plain dict, else params, then params_ema)
  network_g:   {type: fftformer, dim, num_blocks, num_refinement_blocks, ffn_expansion_factor, bias}

Inference layout: NHWC; LayerNorm affine folded into the following 1x1 conv; the per-8x8-patch
FFT filter (DFFN) and the FFT correlation (FSAS) are evaluated as small dense DFT products
(`patch_filter`, `patch_corr`); FSAS's LayerNorm + `v *` run in the project_out GEMM prologue
(`ln_mul_linear`); Fuse's conv2 + split-add is folded into one GEMM. The input is reflect-padded
to a multiple of 32 like the official test script (needs H, W >= 17).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .. import IOSpec, RestorationModel, register_model
from .._fold import Folder, check_reflect_pad, check_state_dict, kn, ln_fold, load_checkpoint, \
    pad_gated_hidden, resolve_network_g, scatter_rows

DEFAULTS = dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[6, 6, 12, 8], num_refinement_blocks=4,
                ffn_expansion_factor=3, bias=False)
PARAMS_KEYS = ("params", "params_ema")


def _conv(s, name, co, ci, k, bias):
    s[f"{name}.weight"] = (co, ci, k, k)
    if bias:
        s[f"{name}.bias"] = (co,)


def _block_shapes(s, p, c, ffn, bias, att):
    """TransformerBlock(c, ffn, bias, att) (WithBias LayerNorms, 8x8 patches)."""
    hid = int(c * ffn)
    if att:
        s[f"{p}.norm1.body.weight"] = s[f"{p}.norm1.body.bias"] = (c,)
        _conv(s, f"{p}.attn.to_hidden", 6 * c, c, 1, bias)
        _conv(s, f"{p}.attn.to_hidden_dw", 6 * c, 1, 3, bias)
        _conv(s, f"{p}.attn.project_out", c, 2 * c, 1, bias)
        s[f"{p}.attn.norm.body.weight"] = s[f"{p}.attn.norm.body.bias"] = (2 * c,)
    s[f"{p}.norm2.body.weight"] = s[f"{p}.norm2.body.bias"] = (c,)
    _conv(s, f"{p}.ffn.project_in", 2 * hid, c, 1, bias)
    _conv(s, f"{p}.ffn.dwconv", 2 * hid, 1, 3, bias)
    s[f"{p}.ffn.fft"] = (2 * hid, 1, 1, 8, 5)
    _conv(s, f"{p}.ffn.project_out", c, hid, 1, bias)


def state_dict_shapes(net):
    """Exact {key: shape} of the upstream fftformer state dict."""
    d, nb, ffn, b = net["dim"], net["num_blocks"], net["ffn_expansion_factor"], net["bias"]
    s = {}
    _conv(s, "patch_embed.proj", d, 3, 3, False)
    for name, n, c, att in (("encoder_level1", nb[0], d, False), ("encoder_level2", nb[1], 2 * d, False),
                            ("encoder_level3", nb[2], 4 * d, False), ("decoder_level3", nb[2], 4 * d, True),
                            ("decoder_level2", nb[1], 2 * d, True), ("decoder_level1", nb[0], d, True),
                            ("refinement", net["num_refinement_blocks"], d, True)):
        for i in range(n):
            _block_shapes(s, f"{name}.{i}", c, ffn, b, att)
    for name, c in (("down1_2", d), ("down2_3", 2 * d)):
        _conv(s, f"{name}.body.1", 2 * c, c, 3, False)
    for name, c in (("up3_2", 4 * d), ("up2_1", 2 * d)):
        _conv(s, f"{name}.body.1", c // 2, c, 3, False)
    _conv(s, "reduce_chan_level2", 2 * d, 4 * d, 1, b)                      # unused upstream
    for name, n in (("fuse2", 2 * d), ("fuse1", d)):                         # Fuse(n_feat)
        _block_shapes(s, f"{name}.att_channel", 2 * n, 2.66, False, False)   # TransformerBlock defaults
        _conv(s, f"{name}.conv", 2 * n, 2 * n, 1, True)
        _conv(s, f"{name}.conv2", 2 * n, 2 * n, 1, True)
    _conv(s, "output", 3, d, 3, b)
    return s


@register_model("fftformer")
class FFTformer(RestorationModel):
    def __init__(self, cfg, ops, device, dtype):
        super().__init__(cfg, ops, device, dtype)
        net = dict(DEFAULTS)
        net.update(resolve_network_g(cfg, cfg.get("_config_dir", "")))
        if net.pop("type", "fftformer") not in ("fftformer", "FFTformer"):
            raise ValueError("network_g.type must be fftformer")
        if net["inp_channels"] != 3 or net["out_channels"] != 3:
            raise ValueError("FFTformer: only 3-channel in / out is supported")
        self.net_cfg = net
        self.params_key, sd = load_checkpoint(cfg, PARAMS_KEYS)
        check_state_dict(sd, state_dict_shapes(net), "fftformer network_g")
        f = Folder(sd, device, dtype,
                   ignore=("reduce_chan_level2.weight", "reduce_chan_level2.bias"))  # unused upstream
        self.blocks = {}
        self._fold(f, net)
        f.check_consumed()
        self.w = f.t
        self.io = IOSpec(in_channels=3, out_channels=3, scale=1, pad_multiple=1)  # pads (reflect) itself

    # ---------------------------------------------------------------- weights
    def _block(self, f, src, dst, dim, att):
        if att:
            w, b = ln_fold(f.get(f"{src}.attn.to_hidden.weight"), f.opt(f"{src}.attn.to_hidden.bias"),
                           f.get(f"{src}.norm1.body.weight"), f.get(f"{src}.norm1.body.bias"))
            f.put(f"{dst}.hid.w", w)
            f.put(f"{dst}.hid.bias", b, True)
            f.put(f"{dst}.hid_dw.w", f.get(f"{src}.attn.to_hidden_dw.weight"))
            f.put(f"{dst}.hid_dw.bias", f.opt(f"{src}.attn.to_hidden_dw.bias", (6 * dim,)), True)
            f.put(f"{dst}.attn_ln.w", f.get(f"{src}.attn.norm.body.weight"), True)
            f.put(f"{dst}.attn_ln.b", f.get(f"{src}.attn.norm.body.bias"), True)
            f.put(f"{dst}.proj.w", kn(f.get(f"{src}.attn.project_out.weight")))
            f.put(f"{dst}.proj.bias", f.opt(f"{src}.attn.project_out.bias", (dim,)), True)
        w, b = ln_fold(f.get(f"{src}.ffn.project_in.weight"), f.opt(f"{src}.ffn.project_in.bias"),
                       f.get(f"{src}.norm2.body.weight"), f.get(f"{src}.norm2.body.bias"))
        wo = kn(f.get(f"{src}.ffn.project_out.weight"))                     # [hidden, C]
        hp, idx = pad_gated_hidden(wo.shape[0])                               # 16-aligned gate halves
        wp = torch.zeros(w.shape[0], 2 * hp)
        wp[:, idx] = w
        f.put(f"{dst}.ffn_in.w", wp)
        f.put(f"{dst}.ffn_in.bias", scatter_rows(b, idx, 2 * hp), True)
        fft = f.get(f"{src}.ffn.fft")
        f.put(f"{dst}.ffn_fft", scatter_rows(fft.reshape(fft.shape[0], *fft.shape[-2:]), idx, 2 * hp), True)
        f.put(f"{dst}.ffn_dw.w", scatter_rows(f.get(f"{src}.ffn.dwconv.weight"), idx, 2 * hp))
        f.put(f"{dst}.ffn_dw.bias", scatter_rows(f.opt(f"{src}.ffn.dwconv.bias", (len(idx),)), idx, 2 * hp), True)
        f.put(f"{dst}.ffn_out.w", scatter_rows(wo, torch.arange(wo.shape[0]), hp))
        f.put(f"{dst}.ffn_out.bias", f.opt(f"{src}.ffn.project_out.bias", (dim,)), True)
        self.blocks[dst] = att

    def _fold(self, f, net):
        d, nb = net["dim"], net["num_blocks"]
        self.stages = [("encoder_level1", nb[0], d, False), ("encoder_level2", nb[1], 2 * d, False),
                       ("encoder_level3", nb[2], 4 * d, False), ("decoder_level3", nb[2], 4 * d, True),
                       ("decoder_level2", nb[1], 2 * d, True), ("decoder_level1", nb[0], d, True),
                       ("refinement", net["num_refinement_blocks"], d, True)]
        for name, n, dim, att in self.stages:
            for i in range(n):
                self._block(f, f"{name}.{i}", f"{name}.{i}", dim, att)
        for p, n in (("fuse2", 2 * d), ("fuse1", d)):
            w = f.get(f"{p}.conv.weight").flatten(1)                          # [2n, 2n]: cat(up, skip)
            f.put(f"{p}.up.w", w[:, :n].t().contiguous())
            f.put(f"{p}.skip.w", w[:, n:].t().contiguous())
            f.put(f"{p}.bias", f.opt(f"{p}.conv.bias", (2 * n,)), True)
            self._block(f, f"{p}.att_channel", f"{p}.att", 2 * n, False)
            w2 = f.get(f"{p}.conv2.weight").flatten(1)                        # out = e + d (split halves)
            b2 = f.opt(f"{p}.conv2.bias", (2 * n,))
            f.put(f"{p}.out.w", (w2[:n] + w2[n:]).t().contiguous())
            f.put(f"{p}.out.bias", b2[:n] + b2[n:], True)
        f.put("embed.w", f.get("patch_embed.proj.weight"), True)
        f.put("embed.bias", f.opt("patch_embed.proj.bias", (d,)), True)
        for name in ("down1_2", "down2_3", "up3_2", "up2_1"):
            f.put(f"{name}.w", f.get(f"{name}.body.1.weight"))
        f.put("output.w", f.get("output.weight"))
        f.put("output.bias", f.opt("output.bias", (3,)), True)

    # ---------------------------------------------------------------- forward
    def _run_block(self, x, p):
        o, w = self.ops, self.w
        if self.blocks[p]:
            c2 = w[f"{p}.attn_ln.w"].shape[0]
            hid = o.linear(x, w[f"{p}.hid.w"], w[f"{p}.hid.bias"], ln=True, ln_eps=1e-5)
            hid = o.dwconv(hid, w[f"{p}.hid_dw.w"], w[f"{p}.hid_dw.bias"])
            corr = o.patch_corr(hid, c2)                                      # q = [:c2], k = [c2:2c2]
            x = o.ln_mul_linear(corr, w[f"{p}.attn_ln.w"], w[f"{p}.attn_ln.b"], hid[..., 2 * c2:],
                                w[f"{p}.proj.w"], w[f"{p}.proj.bias"], residual=x, ln_eps=1e-5)
        u = o.linear_patch_filter(x, w[f"{p}.ffn_in.w"], w[f"{p}.ffn_in.bias"], w[f"{p}.ffn_fft"], ln=True,
                                  ln_eps=1e-5)
        u = o.dwconv(u, w[f"{p}.ffn_dw.w"], w[f"{p}.ffn_dw.bias"], gate="gelu")
        return o.linear(u, w[f"{p}.ffn_out.w"], w[f"{p}.ffn_out.bias"], residual=x)

    def _stage(self, x, name):
        n = next(s[1] for s in self.stages if s[0] == name)
        for i in range(n):
            x = self._run_block(x, f"{name}.{i}")
        return x

    def _fuse(self, up, skip, p):
        o, w = self.ops, self.w
        x = o.linear(up, w[f"{p}.up.w"], w[f"{p}.bias"], residual=o.linear(skip, w[f"{p}.skip.w"]))
        x = self._run_block(x, f"{p}.att")
        return o.linear(x, w[f"{p}.out.w"], w[f"{p}.out.bias"])

    @torch.no_grad()
    def forward(self, inp):
        o, w = self.ops, self.w
        n, _, h, wd = inp.shape
        ph, pw = (32 - h % 32) % 32, (32 - wd % 32) % 32
        check_reflect_pad(h, wd, 32, "FFTformer")
        x_in = F.pad(inp, (0, pw, 0, ph), mode="reflect") if (ph or pw) else inp
        e1 = self._stage(o.intro(x_in, w["embed.w"], w["embed.bias"]), "encoder_level1")
        e2 = self._stage(o.conv3x3(o.resize_bilinear(e1, 0.5), w["down1_2.w"]), "encoder_level2")
        e3 = self._stage(o.conv3x3(o.resize_bilinear(e2, 0.5), w["down2_3.w"]), "encoder_level3")
        x = self._stage(e3, "decoder_level3")
        x = self._fuse(o.conv3x3(o.resize_bilinear(x, 2), w["up3_2.w"]), e2, "fuse2")
        x = self._stage(x, "decoder_level2")
        x = self._fuse(o.conv3x3(o.resize_bilinear(x, 2), w["up2_1.w"]), e1, "fuse1")
        x = self._stage(self._stage(x, "decoder_level1"), "refinement")
        out = o.ending(x, w["output.w"], w["output.bias"], x_in)
        return out[:, :, :h, :wd]

    def describe(self):
        d = super().describe()
        d["network_g"] = self.net_cfg
        d["params_key"] = self.params_key
        return d
