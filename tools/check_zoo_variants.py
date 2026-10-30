"""Random-weight structural variants of every built-in model family (NAFNet, Restormer, RRDBNet,
FFTformer, EVSSM) vs their official architectures: torch/fp32 must match to ~1e-6, the fast
backends to fp16 accuracy.

  CLARUS_NAFNET_REPO=/path/to/NAFNet CLARUS_RESTORMER_REPO=/path/to/Restormer \\
  CLARUS_BASICSR_PIP=/path/with/pip-basicsr CLARUS_FFTFORMER_REPO=/path/to/FFTformer \\
  CLARUS_EVSSM_REPO=/path/to/EVSSM python tools/check_zoo_variants.py [--device cuda:0]

CLARUS_NAFNET_REPO: a clone of megvii-research/NAFNet. CLARUS_RESTORMER_REPO: a clone of swz30/Restormer. CLARUS_BASICSR_PIP: a directory containing
the pip `basicsr` package (for basicsr.archs.rrdbnet_arch). CLARUS_FFTFORMER_REPO /
CLARUS_EVSSM_REPO: clones of kkkls/FFTformer and kkkls/EVSSM (EVSSM also needs `mamba_ssm`
importable; CLARUS_MAMBA_PIP may name a directory that contains it). A family is skipped when
its reference is not available.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
import tempfile

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
from clarus.runtime import build_from_config  # noqa: E402

NAFNET = {
    "nafnet_global_pool": dict(type="NAFNet", width=16, enc_blk_nums=[1, 1, 1, 2], middle_blk_num=1,
                               dec_blk_nums=[1, 1, 1, 1]),
    "nafnetlocal_tlc": dict(type="NAFNetLocal", width=32, enc_blk_nums=[1, 2, 1], middle_blk_num=2,
                            dec_blk_nums=[2, 1, 1], train_size=[1, 3, 32, 32]),   # local means < feature size
}
RESTORMER = {
    "restormer_tiny": dict(type="Restormer", dim=24, num_blocks=[1, 1, 2, 1], num_refinement_blocks=1,
                           heads=[1, 2, 2, 4], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias"),
    "restormer_bias_ffn2": dict(type="Restormer", dim=32, num_blocks=[1, 2, 1, 2], num_refinement_blocks=2,
                                heads=[2, 2, 4, 8], ffn_expansion_factor=2.0, bias=True, LayerNorm_type="WithBias"),
}
FFTFORMER = {
    "fftformer_tiny": dict(type="fftformer", dim=16, num_blocks=[1, 2, 1, 1], num_refinement_blocks=1,
                           ffn_expansion_factor=2.66, bias=False),
    "fftformer_bias": dict(type="fftformer", dim=24, num_blocks=[2, 1, 2, 1], num_refinement_blocks=2,
                           ffn_expansion_factor=3, bias=True),
}
EVSSM = {
    "evssm_tiny": dict(type="EVSSM", dim=16, num_blocks=[1, 2, 3], ffn_expansion_factor=2, bias=False),
    "evssm_bias": dict(type="EVSSM", dim=32, num_blocks=[2, 1, 2], ffn_expansion_factor=3, bias=True),
}
RRDB = {
    "rrdbnet_x4_small": dict(type="RRDBNet", num_in_ch=3, num_out_ch=3, scale=4, num_feat=32, num_block=2, num_grow_ch=16),
    "rrdbnet_x2": dict(type="RRDBNet", num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=2, num_grow_ch=32),
    "rrdbnet_x1": dict(type="RRDBNet", num_in_ch=3, num_out_ch=3, scale=1, num_feat=48, num_block=1, num_grow_ch=24),
}


def randomise(net):
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if (p.ndim == 1 and "norm" in name and name.endswith("weight")) or name.endswith(".fft"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=g))
            elif p.ndim <= 1 or "temperature" in name:
                p.copy_(0.5 + 0.1 * torch.randn(p.shape, generator=g) if "temperature" in name
                        else 0.1 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(torch.randn(p.shape, generator=g) / p[0].numel() ** 0.5)


def ref_class(kind):
    # the references ship packages called `basicsr` / `models`; drop whichever was imported before
    for m in [m for m in sys.modules if m.split(".")[0] in ("basicsr", "models")]:
        del sys.modules[m]
    if kind == "nafnet":
        repo = os.environ.get("CLARUS_NAFNET_REPO")
        if not repo:
            return None
        sys.path.insert(0, repo)
        mod = importlib.import_module("basicsr.models.archs.NAFNet_arch")
        return {"NAFNet": mod.NAFNet, "NAFNetLocal": mod.NAFNetLocal}
    if kind in ("fftformer", "evssm"):
        repo = os.environ.get(f"CLARUS_{kind.upper()}_REPO")
        if not repo:
            return None
        if kind == "evssm" and os.environ.get("CLARUS_MAMBA_PIP"):
            sys.path.insert(0, os.environ["CLARUS_MAMBA_PIP"])
        # standalone files, loaded by path: FFTformer's `basicsr` has no __init__.py, so a
        # regular `basicsr` package elsewhere on sys.path would shadow it
        rel = "basicsr/models/archs/fftformer_arch.py" if kind == "fftformer" else "models/EVSSM.py"
        spec = importlib.util.spec_from_file_location(f"_clarus_ref_{kind}", os.path.join(repo, rel))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.fftformer if kind == "fftformer" else mod.EVSSM
    if kind == "restormer":
        repo = os.environ.get("CLARUS_RESTORMER_REPO")
        if not repo:
            return None
        sys.path.insert(0, repo)
        return importlib.import_module("basicsr.models.archs.restormer_arch").Restormer
    pip = os.environ.get("CLARUS_BASICSR_PIP")
    if not pip:
        return None
    sys.path.insert(0, pip)
    return importlib.import_module("basicsr.archs.rrdbnet_arch").RRDBNet


def run(kind, variants, model_name, dev, hw, ok_all):
    cls = ref_class(kind)
    if cls is None:
        print(f"skip {kind}: reference not configured")
        return ok_all
    tmp = tempfile.mkdtemp(prefix=f"clarus_{kind}_")
    for name, net_g in variants.items():
        kw = {k: v for k, v in net_g.items() if k != "type"}
        if "train_size" in kw:
            kw["train_size"] = tuple(kw["train_size"])
        ref = (cls[net_g["type"]] if isinstance(cls, dict) else cls)(**kw)
        randomise(ref)
        ckpt = os.path.join(tmp, f"{name}.pth")
        torch.save({"params": ref.state_dict()}, ckpt)
        ref = ref.to(dev).eval()
        cfg = os.path.join(tmp, f"{name}.yaml")
        with open(cfg, "w") as f:
            yaml.safe_dump({"model": {"name": model_name, "checkpoint": ckpt, "network_g": net_g},
                            "runtime": {"device": dev}}, f)
        x = torch.rand(2 if kind != "evssm" else 1, 3, *hw, device=dev)   # EVSSM's crop code needs batch 1
        with torch.no_grad():
            y_ref = ref(x)
        scale = y_ref.abs().max().item()
        line = [f"{name:22s}"]
        for label, kw2 in (("torch/fp32", dict(backends=["torch"], dtype="fp32")),
                           ("triton,torch/fp16", dict(backends=["triton", "torch"], dtype="fp16"))):
            model, ops, _ = build_from_config(cfg, **kw2)
            y = model.forward(x)
            rel = (y.float() - y_ref).abs().max().item() / scale
            tol = 1e-4 if label.startswith("torch") else 2e-2
            ok = rel < tol and y.shape == y_ref.shape
            ok_all &= ok
            fb = sorted(op for op, by in ops.report()["ops"].items() if "torch" in by and ops.backends[0].name != "torch")
            line.append(f"{label}: rel {rel:.1e} {'OK' if ok else 'BAD'}" + (f" (torch: {', '.join(fb)})" if fb else ""))
        print(" | ".join(line), flush=True)
    return ok_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ok = run("nafnet", NAFNET, "nafnet", args.device, (75, 101), True)        # first: its basicsr is a namespace pkg
    ok = run("restormer", RESTORMER, "restormer", args.device, (64, 96), ok)   # Restormer needs H, W % 8 == 0
    ok = run("rrdbnet", RRDB, "rrdbnet", args.device, (36, 52), ok)
    ok = run("fftformer", FFTFORMER, "fftformer", args.device, (64, 96), ok)  # sizes the reference accepts as is
    ok = run("evssm", EVSSM, "evssm", args.device, (64, 96), ok)
    print("ALL OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
