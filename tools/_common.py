"""Shared helpers for the dev / validation tools (all take --config like the engine)."""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# A basicsr checkout that contains the network's architecture, e.g. a clone of
# megvii-research/NAFNet for the NAFNet configs. Only the reference tools import it, to build
# the original network and metrics; the engine itself never does.
BASICSR_ROOT = os.environ.get("CLARUS_BASICSR")
sys.path.insert(0, os.path.join(ROOT, "python"))
if BASICSR_ROOT:
    sys.path.insert(0, BASICSR_ROOT)

from clarus.runtime import DTYPES, build_from_config, load_config  # noqa: E402,F401

DEFAULT_CONFIG = os.path.join(ROOT, "configs", "nafnet_gopro_width64.yaml")
# evaluation pair used by the tools: directories of degraded / ground-truth PNGs with matching
# names (e.g. the GoPro test set's blur / sharp folders)
LQ = os.environ.get("CLARUS_LQ", "")
GT = os.environ.get("CLARUS_GT", "")


def add_config_args(ap):
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--backends", default=None, help="comma list, e.g. triton,torch or torch")
    ap.add_argument("--dtype", default=None, help="fp16 | bf16 | fp32")
    ap.add_argument("--device", default=None, help="e.g. cuda:0, cpu")
    ap.add_argument("--res-level", type=int, default=None, help="model.options.residual_fp32_from_level")
    ap.add_argument("--op-override", action="append", default=[], help="op=backend, e.g. selective_scan=torch")
    return ap


def build_engine_model(args):
    """-> (model, ops, device) built exactly like the engine worker, with CLI overrides."""
    mo = {"options": {"residual_fp32_from_level": args.res_level}} if args.res_level is not None else None
    overrides = dict(o.split("=", 1) for o in args.op_override) or None
    return build_from_config(args.config, model_overrides=mo,
                             backends=args.backends.split(",") if args.backends else None,
                             dtype=args.dtype, device=args.device, op_overrides=overrides)


def build_reference(config_path: str, device="cuda"):
    """The original basicsr network for the config's network_g + checkpoint (fp32)."""
    if not BASICSR_ROOT:
        raise SystemExit("set CLARUS_BASICSR to a basicsr checkout with the architecture (e.g. megvii-research/NAFNet)")
    from basicsr.models.archs import define_network
    from clarus.models._fold import resolve_network_g

    cfg = load_config(config_path)["model"]
    net = define_network(dict(resolve_network_g(cfg, cfg["_config_dir"])))
    ckpt = cfg["checkpoint"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(cfg["_config_dir"], ckpt)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("params", sd.get("params_ema", sd))
    net.load_state_dict(sd, strict=True)
    return net.to(device).eval()


def load_rgb(path, device="cuda"):
    img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(img.copy()).permute(2, 0, 1)[None].to(device)
