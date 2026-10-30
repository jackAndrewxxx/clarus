"""Build a served model from the engine config (the same YAML the Rust engine reads).

    cfg = load_config("configs/nafnet_gopro_width64.yaml")
    model, ops, dev = build(cfg["model"], cfg["runtime"])
    y = model.forward(x)          # x: [N, 3, H, W] fp32 in [0, 1] on dev.torch_device

Only the `model:` and `runtime:` sections matter here; `scheduler:` / `io:` belong to Rust.
"""
from __future__ import annotations

import copy
import os

import torch
import yaml

from .device import get_device
from .models import create_model
from .ops import Ops

DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

RUNTIME_DEFAULTS = {
    "device": "cuda:0",
    "device_imports": [],
    "dtype": "fp16",
    "backends": ["triton", "torch"],
    "op_overrides": {},
    "graphs": "auto",
    "max_graphs": 128,
    "metrics_backend": "auto",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("model", {})["_config_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def runtime_config(rt: dict | None) -> dict:
    out = copy.deepcopy(RUNTIME_DEFAULTS)
    out.update({k: v for k, v in (rt or {}).items() if v is not None})
    if out["dtype"] not in DTYPES:
        raise ValueError(f"runtime.dtype must be one of {sorted(DTYPES)}, got {out['dtype']!r}")
    return out


def build(model_cfg: dict, runtime_cfg: dict | None = None, log=print):
    rt = runtime_config(runtime_cfg)
    dev = get_device(rt["device"], rt["device_imports"])
    dev.set_current()
    dtype = DTYPES[rt["dtype"]]
    ops = Ops(rt["backends"], dtype, dev.torch_device, rt["op_overrides"], log=log)
    model = create_model(model_cfg, ops, dev.torch_device, dtype)
    return model, ops, dev


def build_from_config(path: str, model_overrides: dict | None = None, **runtime_overrides):
    """Tools helper: load YAML, apply overrides (e.g. dtype="fp32", backends=["torch"])."""
    cfg = load_config(path)
    rt = dict(cfg.get("runtime") or {})
    rt.update({k: v for k, v in runtime_overrides.items() if v is not None})
    model_cfg = dict(cfg["model"])
    for k, v in (model_overrides or {}).items():
        if isinstance(v, dict):
            model_cfg[k] = {**(model_cfg.get(k) or {}), **v}
        else:
            model_cfg[k] = v
    return build(model_cfg, rt)
