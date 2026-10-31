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


def select_state_dict(raw, params_key=None, default_keys=("params", "params_ema")) -> tuple[str | None, dict]:
    """Pick the state dict out of a loaded checkpoint -> (key used or None, state dict).

    An explicit `params_key` must exist. Otherwise a dict holding tensors directly is the state
    dict itself (key None), else the first of `default_keys` present is used. A leading
    "module." (DataParallel) is stripped when every key has it."""
    if not isinstance(raw, dict):
        raise TypeError(f"checkpoint is a {type(raw).__name__}, expected a dict")
    if params_key is not None:
        if params_key not in raw:
            raise KeyError(f"params_key {params_key!r} not in checkpoint; top-level keys: {sorted(raw)[:8]}")
        key = params_key
    elif any(torch.is_tensor(v) for v in raw.values()):
        key = None
    else:
        key = next((k for k in default_keys if k in raw), None)
        if key is None:
            raise KeyError(f"checkpoint has none of {list(default_keys)} and no tensors at the top level; "
                           f"top-level keys: {sorted(raw)[:8]} (set model.params_key)")
    sd = raw if key is None else raw[key]
    if not isinstance(sd, dict):
        raise TypeError(f"checkpoint[{key!r}] is a {type(sd).__name__}, expected a state dict")
    if sd and all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return key, sd


def load_checkpoint(cfg: dict, params_keys=("params", "params_ema")) -> tuple[str | None, dict]:
    """model.checkpoint (relative to the config dir) -> (params key used, fp32 state dict); the
    key is chosen by select_state_dict(raw, model.params_key, params_keys)."""
    ckpt = cfg["checkpoint"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(cfg.get("_config_dir", ""), ckpt)
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    key, sd = select_state_dict(raw, cfg.get("params_key"), params_keys)
    return key, {k: v.float() for k, v in sd.items()}


def load_state_dict(cfg: dict, params_keys=("params", "params_ema")) -> dict:
    """model.checkpoint -> fp32 state dict (see load_checkpoint)."""
    return load_checkpoint(cfg, params_keys)[1]


def check_state_dict(sd: dict, expected: dict, what: str = "network_g") -> None:
    """Exact contract check: `sd` must have exactly the keys of `expected` ({key: shape}) with
    those shapes; ValueError listing a few missing / unexpected / mis-shaped tensors otherwise."""
    missing = [k for k in expected if k not in sd]
    unexpected = [k for k in sd if k not in expected]
    shapes = [f"{k}: checkpoint {tuple(sd[k].shape)} vs expected {tuple(expected[k])}"
              for k in expected if k in sd and tuple(sd[k].shape) != tuple(expected[k])]
    if missing or unexpected or shapes:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing (e.g. {missing[:4]})")
        if unexpected:
            parts.append(f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})")
        if shapes:
            parts.append(f"{len(shapes)} with wrong shape (e.g. {shapes[:3]})")
        raise ValueError(f"checkpoint does not match the configured {what}: " + "; ".join(parts))


def check_reflect_pad(h: int, w: int, multiple: int, model: str) -> None:
    """Official test code reflect-pads H, W up to `multiple`; PyTorch rejects a reflect pad >= the
    input size, so tiny inputs fail upstream too. Raise a clear error naming the minimum size."""
    for n in (h, w):
        pad = (-n) % multiple
        if pad and pad >= n:
            raise ValueError(f"{model}: input {h}x{w} is too small: it is reflect-padded to a multiple of "
                             f"{multiple} like the official code, which needs H and W >= {multiple // 2 + 1}")


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
