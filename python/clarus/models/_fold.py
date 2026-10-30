"""Config / checkpoint loading and weight-folding helpers shared by the model adapters."""
from __future__ import annotations

import os

import torch
import yaml


def resolve_network_g(cfg: dict, base_dir: str = "") -> dict:
    """model.network_g, merged over the `network_g` of the basicsr yml named by model.arch_from."""
    net: dict = {}
    src = cfg.get("arch_from")
    if src:
        path = src if os.path.isabs(src) else os.path.join(base_dir, src)
        with open(path) as f:
            doc = yaml.safe_load(f)
        if "network_g" not in doc:
            raise ValueError(f"{path}: no network_g section")
        net.update(doc["network_g"])
    net.update(cfg.get("network_g") or {})
    if not net:
        raise ValueError("model config needs arch_from (basicsr yml) and/or network_g")
    return net


def load_state_dict(cfg: dict) -> dict:
    """model.checkpoint (relative to the config dir) -> fp32 state dict ('params' / 'params_ema' / plain)."""
    ckpt = cfg["checkpoint"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(cfg.get("_config_dir", ""), ckpt)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    key = cfg.get("params_key") or ("params" if "params" in sd else "params_ema" if "params_ema" in sd else None)
    sd = sd[key] if key else sd
    return {k.removeprefix("module."): v.float() for k, v in sd.items()}


class Folder:
    """Pulls tensors out of a state dict (tracking which were used) into a dict of device tensors."""

    def __init__(self, sd: dict, device, dtype, ignore=()):
        self.sd, self.device, self.dtype = sd, device, dtype
        self.used = set(ignore)
        self.t: dict[str, torch.Tensor] = {}

    def get(self, k):
        if k not in self.sd:
            raise ValueError(f"checkpoint has no tensor {k!r} (network_g does not match the checkpoint?)")
        self.used.add(k)
        return self.sd[k]

    def opt(self, k, shape=None):
        """Optional tensor (bias=False convs); zeros of `shape` when missing."""
        if k in self.sd:
            return self.get(k)
        return None if shape is None else torch.zeros(shape)

    def put(self, name, v, fp32=False):
        self.t[name] = v.to(device=self.device, dtype=torch.float32 if fp32 else self.dtype).contiguous()

    def check_consumed(self):
        unused = [k for k in self.sd if k not in self.used]
        if unused:
            raise ValueError(f"checkpoint tensors not consumed: {unused[:8]}")


def ln_fold(w, b, ln_w, ln_b):
    """1x1 conv / linear [N, K(,1,1)] (+ bias) after a LayerNorm affine -> ([K, N], bias [N])."""
    w = w.flatten(1)
    b2 = w @ ln_b + (b if b is not None else 0.0)
    return (w * ln_w[None, :]).t().contiguous(), b2


def kn(w):
    """[N, K(,1,1)] -> [K, N]."""
    return w.reshape(w.shape[0], -1).t().contiguous()


def pad_gated_hidden(hid: int, mult: int = 16):
    """Index map that places both halves of a 2*hid gated tensor at multiples of `mult`:
    returns (padded hidden hp, index tensor of length 2*hid into 2*hp)."""
    hp = (hid + mult - 1) // mult * mult
    return hp, torch.cat([torch.arange(hid), hp + torch.arange(hid)])


def scatter_rows(v, idx, rows):
    """Zero tensor with `rows` leading entries, v placed at idx."""
    out = torch.zeros((rows, *v.shape[1:]), dtype=v.dtype)
    out[idx] = v
    return out
