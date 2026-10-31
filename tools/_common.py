"""Shared helpers for the dev / validation tools (all take --config like the engine)."""
from __future__ import annotations

import glob
import importlib
import importlib.util
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The checkout that holds the config's original network, one per model family:
#   nafnet     a clone of megvii-research/NAFNet     restormer  a clone of swz30/Restormer
#   fftformer  a clone of kkkls/FFTformer            evssm      a clone of kkkls/EVSSM
#   rrdbnet    a directory containing the pip `basicsr` package (basicsr.archs.rrdbnet_arch)
# EVSSM also needs `mamba_ssm` importable (CLARUS_MAMBA_PIP may name a directory containing it).
# Only the reference tools import it, to build the original network and metrics; the engine never does.
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


def repo_revision(path: str) -> str:
    """git HEAD of a reference checkout (+ '-dirty' if tracked files are modified), or the pip version."""
    try:
        rev = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        pass
    dist = sorted(glob.glob(os.path.join(path, "basicsr-*.dist-info")))
    if dist:
        return "pip " + os.path.basename(dist[-1])[: -len(".dist-info")]
    return "unknown (not a git checkout)"


def _load_file(repo: str, rel: str):
    """Import a standalone arch file by path (FFTformer's `basicsr` has no __init__.py, so a
    regular `basicsr` elsewhere on sys.path would shadow it)."""
    path = os.path.join(repo, rel)
    spec = importlib.util.spec_from_file_location("_clarus_ref_" + os.path.basename(rel)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Each official test script, reduced to its tensor path (input: fp32 RGB [N, 3, H, W] in [0, 1];
# output before clamp / quantization, which every tool applies itself).

def _fwd_nafnet(net, x):
    # basicsr/test.py -> ImageRestorationModel.test (options/test/GoPro/NAFNet-width64.yml: val.grids
    # false) = net_g(lq); NAFNet.forward zero-pads to 2**len(encoders) itself and crops; NAFNetLocal
    # is the TLC network (local average pools, set up in its __init__)
    return net(x)


def _fwd_restormer(net, x):
    # Motion_Deblurring/test.py: reflect pad to a multiple of 8 (its formula), crop [:h, :w]
    factor = 8
    h, w = x.shape[2:]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    return net(F.pad(x, (0, padw, 0, padh), "reflect"))[:, :, :h, :w]


def _fwd_fftformer(net, x):
    # test.py: reflect pad to a multiple of 32, crop [:h, :w]
    h, w = x.shape[2:]
    h_n, w_n = (32 - h % 32) % 32, (32 - w % 32) % 32
    return net(F.pad(x, (0, w_n, 0, h_n), mode="reflect"))[:, :, :h, :w]


def _fwd_evssm(net, x):
    # test.py: models/EVSSM.py (overlapping-crop scan in EVS.grids, which asserts batch 1), no
    # padding (the pad / crop lines are commented out), batch_size=1
    h, w = x.shape[2:]
    if h % 4 or w % 4:
        raise ValueError(f"official EVSSM test path does not pad; {h}x{w} is not a multiple of 4 "
                         "(its down / up sampling would not line up)")
    return torch.cat([net(x[i:i + 1]) for i in range(x.shape[0])])


def _fwd_rrdbnet(net, x, scale):
    # Real-ESRGAN inference_realesrgan.py defaults (tile 0, pre_pad 0) -> RealESRGANer.pre_process:
    # reflect mod-pad to 2 (scale 2) / 4 (scale 1), post_process crops mod_pad * scale
    mod = {2: 2, 1: 4}.get(scale)
    if mod is None:
        return net(x)
    h, w = x.shape[2:]
    ph, pw = (mod - h % mod if h % mod else 0), (mod - w % mod if w % mod else 0)
    y = net(F.pad(x, (0, pw, 0, ph), "reflect"))
    return y[:, :, :y.shape[2] - ph * scale, :y.shape[3] - pw * scale]


def _ref_arch(name, net_g, repo):
    """-> (network instance, forward(net, x), arch source, official test path)"""
    kw = {k: v for k, v in net_g.items() if k != "type"}
    if name == "nafnet":
        from basicsr.models.archs import define_network
        return define_network(dict(net_g)), _fwd_nafnet, "basicsr/models/archs/NAFNet_arch.py", \
            "basicsr/test.py (val.grids: false)"
    if name == "restormer":
        cls = importlib.import_module("basicsr.models.archs.restormer_arch").Restormer
        return cls(**kw), _fwd_restormer, "basicsr/models/archs/restormer_arch.py", "Motion_Deblurring/test.py"
    if name == "fftformer":
        mod = _load_file(repo, "basicsr/models/archs/fftformer_arch.py")
        return mod.fftformer(**kw), _fwd_fftformer, "basicsr/models/archs/fftformer_arch.py", "test.py"
    if name == "evssm":
        if os.environ.get("CLARUS_MAMBA_PIP"):
            sys.path.insert(0, os.environ["CLARUS_MAMBA_PIP"])
        mod = _load_file(repo, "models/EVSSM.py")      # what test.py imports (not basicsr's EVSSM_arch)
        return mod.EVSSM(**kw), _fwd_evssm, "models/EVSSM.py", "test.py"
    if name == "rrdbnet":
        cls = importlib.import_module("basicsr.archs.rrdbnet_arch").RRDBNet
        scale = kw.get("scale", 4)
        return cls(**kw), (lambda net, x: _fwd_rrdbnet(net, x, scale)), "basicsr/archs/rrdbnet_arch.py", \
            "Real-ESRGAN inference_realesrgan.py (tile 0, pre_pad 0, fp32)"
    raise SystemExit(f"no reference wrapper for model {name!r}")


class Reference:
    """The original network (fp32) wrapped in its official test script's pre / post-processing."""

    def __init__(self, net, fwd, info):
        self.net, self._fwd, self.info = net, fwd, info

    def __call__(self, x):
        return self._fwd(self.net, x)

    def describe(self) -> str:
        i = self.info
        return (f"reference {i['model']}: {i['repo']} @ {i['revision']} ({i['arch']}, as in {i['test_path']}; "
                f"weights {os.path.basename(i['checkpoint'])}[{i['params_key']}])")


def build_reference(config_path: str, device="cuda") -> Reference:
    """The original network for the config's network_g + checkpoint, called like the official test."""
    if not BASICSR_ROOT:
        raise SystemExit("set CLARUS_BASICSR to the reference checkout of the config's model "
                         "(e.g. megvii-research/NAFNet for the NAFNet configs)")
    from clarus.models import _MODELS, available_models
    from clarus.models._fold import resolve_network_g, select_state_dict

    cfg = load_config(config_path)["model"]
    available_models()                                   # registers the built-in models
    if cfg.get("name") not in _MODELS:
        raise SystemExit(f"unknown model.name {cfg.get('name')!r}")
    cls = _MODELS[cfg["name"]]
    params_keys = sys.modules[cls.__module__].PARAMS_KEYS   # the engine adapter's checkpoint keys
    net_g = resolve_network_g(cfg, cfg["_config_dir"])
    net, fwd, arch, test_path = _ref_arch(cls.name, net_g, BASICSR_ROOT)
    ckpt = cfg["checkpoint"]
    if not os.path.isabs(ckpt):
        ckpt = os.path.join(cfg["_config_dir"], ckpt)
    raw = torch.load(ckpt, map_location="cpu", weights_only=True)
    key, sd = select_state_dict(raw, cfg.get("params_key"), default_keys=params_keys)
    net.load_state_dict(sd, strict=True)
    info = {"model": cls.name, "repo": os.path.abspath(BASICSR_ROOT), "revision": repo_revision(BASICSR_ROOT),
            "arch": arch, "test_path": test_path, "checkpoint": ckpt, "params_key": key}
    return Reference(net.to(device).eval(), fwd, info)


def load_rgb(path, device="cuda"):
    img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(img.copy()).permute(2, 0, 1)[None].to(device)
