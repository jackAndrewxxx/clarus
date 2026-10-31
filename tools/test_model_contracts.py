"""Model-adapter contract tests (CPU, fp32, torch backend; ~1 min): checkpoint selection and
exact key / shape validation, NAFNetLocal TLC windows, and input-size handling, each against
the official code. Reference checkouts come from the same variables as
tools/check_zoo_variants.py (CLARUS_NAFNET_REPO, CLARUS_RESTORMER_REPO, CLARUS_BASICSR_PIP,
CLARUS_FFTFORMER_REPO, CLARUS_EVSSM_REPO, CLARUS_MAMBA_PIP).

    PYTHONPATH=python python tools/test_model_contracts.py [-k name]
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile
import traceback
import types

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

def _env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"set {name} (see the module docstring)")
    return v


REPO = {k: (lambda k=k: _env(f"CLARUS_{k.upper()}_REPO")) for k in ("nafnet", "restormer", "fftformer", "evssm")}
RRDB_PIP = os.environ.get("CLARUS_BASICSR_PIP", "")
MAMBA_PIP = os.environ.get("CLARUS_MAMBA_PIP", "")

from clarus.models._fold import select_state_dict  # noqa: E402
from clarus.runtime import build  # noqa: E402

TMP = tempfile.mkdtemp(prefix="clarus_contracts_")
RT = {"device": "cpu", "dtype": "fp32", "backends": ["torch"], "graphs": False}
RTOL = 1e-5
_n = [0]


# ------------------------------------------------------------------------------------ helpers
def build_model(name, network_g, ckpt, **extra):
    _n[0] += 1
    path = os.path.join(TMP, f"{name}_{_n[0]}.pth")
    torch.save(ckpt, path)
    cfg = {"name": name, "checkpoint": path, "network_g": dict(network_g), **extra}
    return build(cfg, RT, log=lambda *a, **k: None)[0]


def randomize(net, seed):
    """Random weights everywhere (upstream inits zero beta / gamma, which would hide bugs)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for k, p in net.named_parameters():
            base = 1.0 if ("norm" in k and k.endswith("weight")) else 0.0
            p.copy_(base + 0.2 * torch.randn(p.shape, generator=g))
    return net.eval()


def sd_of(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items()}


def import_basicsr(root, module):
    """Import `module` from the `basicsr` under `root` (NAFNet's and pip's packages share the name)."""
    for k in [k for k in sys.modules if k == "basicsr" or k.startswith("basicsr.")]:
        del sys.modules[k]
    sys.path.insert(0, root)
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(root)


def load_file(path):
    spec = importlib.util.spec_from_file_location("_ref_" + os.path.basename(path)[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def raises(exc, fn, *needles):
    try:
        fn()
    except exc as e:
        msg = str(e)
        for s in needles:
            assert s in msg, f"{exc.__name__} message lacks {s!r}: {msg}"
        return msg
    raise AssertionError(f"expected {exc.__name__}")


def close(a, b, what, rtol=RTOL):
    """max |a - b| <= rtol * max(1, max |b|) (random weights give outputs up to ~1e3)."""
    assert a.shape == b.shape, f"{what}: shape {tuple(a.shape)} vs {tuple(b.shape)}"
    d = (a.float() - b.float()).abs().max().item()
    tol = rtol * max(1.0, b.abs().max().item())
    assert d <= tol, f"{what}: max |diff| {d:.3e} > {tol:.3e}"
    return d


def official_pad_crop(net, x, mult, scale=1):
    """Official test-script handling: reflect-pad H, W to `mult`, run, crop to H*scale, W*scale."""
    h, w = x.shape[-2:]
    xp = F.pad(x, (0, (-w) % mult, 0, (-h) % mult), mode="reflect")
    with torch.no_grad():
        return net(xp)[:, :, :h * scale, :w * scale]


def rand_img(h, w, seed=0):
    return torch.rand((1, 3, h, w), generator=torch.Generator().manual_seed(seed))


# -------------------------------------------------------------------------------------- tests
def test_select_state_dict():
    t = torch.zeros(1)
    assert select_state_dict({"a": t}) == (None, {"a": t})
    two = {"params": {"a": t}, "params_ema": {"b": t}}
    assert select_state_dict(two)[0] == "params"
    assert select_state_dict(two, None, ("params_ema", "params"))[0] == "params_ema"
    assert select_state_dict(two, "params", ("params_ema", "params"))[0] == "params"
    raises(KeyError, lambda: select_state_dict(two, "nope"), "nope")
    raises(KeyError, lambda: select_state_dict({"a": t}, "params"), "params")
    raises(KeyError, lambda: select_state_dict({"meta": {"a": t}}), "params")
    assert set(select_state_dict({"params": {"module.a": t, "module.b": t}})[1]) == {"a", "b"}
    assert set(select_state_dict({"module.a": t, "b": t})[1]) == {"module.a", "b"}   # not every key
    for m in ("nafnet", "restormer", "rrdbnet", "fftformer", "evssm"):
        keys = importlib.import_module(f"clarus.models.{m}").PARAMS_KEYS
        assert keys == (("params_ema", "params") if m == "rrdbnet" else ("params", "params_ema")), (m, keys)


NAF_G = dict(width=4, enc_blk_nums=[1, 1], middle_blk_num=1, dec_blk_nums=[1, 1])


def _naf_level(name, n_enc):
    """Pyramid level of an upstream module name (encoders.i / middle_blks / decoders.i)."""
    part = name.split(".")
    if part[0] == "encoders":
        return int(part[1])
    if part[0] == "middle_blks":
        return n_enc
    return n_enc - 1 - int(part[1])


def test_nafnet_tlc():
    arch = import_basicsr(REPO["nafnet"](), "basicsr.models.archs.NAFNet_arch")
    x = rand_img(80, 96)
    for ts in ((1, 3, 33, 33), (1, 3, 32, 32), (1, 3, 33, 45)):
        ref = randomize(arch.NAFNetLocal(img_channel=3, train_size=ts, **NAF_G), 1)
        pools = {n: tuple(m.kernel_size) for n, m in ref.named_modules() if type(m).__name__ == "AvgPool2d"}
        model = build_model("nafnet", {"type": "NAFNetLocal", "train_size": list(ts), **NAF_G}, {"params": sd_of(ref)})
        for n, k in pools.items():
            got = tuple(model._tlc_kernel(_naf_level(n, 2)))
            assert got == k, f"train {ts[-2:]} {n}: window {got} vs upstream {k}"
        if ts[-1] == 33:
            assert {tuple(k) for k in pools.values()} == {(53, 53), (26, 26), (13, 13)}, pools
        with torch.no_grad():
            d = close(model.forward(x), ref(x), f"NAFNetLocal train {ts[-2:]}")
        print(f"  train {ts[-2:]}: windows {sorted(set(pools.values()), reverse=True)} match, max diff {d:.1e}")


def test_nafnet_contract():
    arch = import_basicsr(REPO["nafnet"](), "basicsr.models.archs.NAFNet_arch")
    sd = sd_of(randomize(arch.NAFNet(img_channel=3, **NAF_G), 2))
    m = build_model("nafnet", {"type": "NAFNet", **NAF_G}, {"params": sd})
    assert m.describe()["network_g"]["width"] == 4 and m.describe()["params_key"] == "params"
    assert build_model("nafnet", {"type": "NAFNet", **NAF_G}, {"params_ema": sd}).describe()["params_key"] == \
        "params_ema"
    bad = lambda g, s=sd: build_model("nafnet", {"type": "NAFNet", **{**NAF_G, **g}}, {"params": s})  # noqa: E731
    raises(ValueError, lambda: bad({"width": 99}), "wrong shape", "intro.weight")
    raises(ValueError, lambda: bad({"enc_blk_nums": [1, 2]}), "missing", "encoders.1.1")
    raises(ValueError, lambda: bad({}, {**sd, "intro.garbage": torch.zeros(1)}), "unexpected", "intro.garbage")
    raises(ValueError, lambda: bad({}, {k: v for k, v in sd.items() if k != "ending.bias"}), "missing", "ending.bias")
    raises(ValueError, lambda: bad({"dec_blk_nums": [1]}), "same length")


def _rrdb(scale, seed, **kw):
    arch = import_basicsr(RRDB_PIP or _env("CLARUS_BASICSR_PIP"), "basicsr.archs.rrdbnet_arch")
    g = dict(num_in_ch=3, num_out_ch=3, scale=scale, num_feat=8, num_block=1, num_grow_ch=4, **kw)
    return randomize(arch.RRDBNet(**g), seed), {"type": "RRDBNet", **g}


def test_rrdb_contract():
    ref, g = _rrdb(2, 3)
    sd = sd_of(ref)
    sd_b = sd_of(_rrdb(2, 4)[0])
    ok = build_model("rrdbnet", g, {"params_ema": sd})
    assert ok.describe()["params_key"] == "params_ema"
    bias = "body.0.rdb1.conv1.bias"
    raises(ValueError, lambda: build_model("rrdbnet", g, {k: v for k, v in sd.items() if k != bias}), "missing", bias)
    raises(ValueError, lambda: build_model("rrdbnet", g, {**sd, "conv_first.garbage": torch.zeros(1)}),
           "unexpected", "conv_first.garbage")
    raises(ValueError, lambda: build_model("rrdbnet", {**g, "num_grow_ch": 8}, sd), "wrong shape")
    raises(ValueError, lambda: build_model("rrdbnet", {**g, "num_block": 2}, sd), "missing", "body.1.")
    raises(ValueError, lambda: build_model("rrdbnet", {**g, "scale": 4}, sd), "wrong shape", "conv_first.weight")
    # params selection: default params_ema, explicit params_key honoured, unknown key rejected
    both = {"params": sd, "params_ema": sd_b}
    m = build_model("rrdbnet", g, both)
    assert m.describe()["params_key"] == "params_ema" and torch.equal(m.w["conv_first.weight"], sd_b["conv_first.weight"])
    m = build_model("rrdbnet", g, both, params_key="params")
    assert m.describe()["params_key"] == "params" and torch.equal(m.w["conv_first.weight"], sd["conv_first.weight"])
    raises(KeyError, lambda: build_model("rrdbnet", g, both, params_key="nope"), "nope")


def test_rrdb_pad_crop():
    """scale 2 / 1: RealESRGANer.pre_process reflect-pads to 2 / 4, post_process crops."""
    x = rand_img(7, 9)
    for scale, mult in ((2, 2), (1, 4), (4, 1)):
        ref, g = _rrdb(scale, 5)
        m = build_model("rrdbnet", g, {"params_ema": sd_of(ref)})
        assert m.io.pad_multiple == 1                  # the worker must not zero-pad first
        d = close(m.forward(x), official_pad_crop(ref, x, mult, scale), f"RRDBNet x{scale} 7x9")
        print(f"  x{scale} 7x9 -> {7 * scale}x{9 * scale}: max diff {d:.1e}")
    ref, g = _rrdb(1, 5)
    m = build_model("rrdbnet", g, {"params_ema": sd_of(ref)})
    raises(RuntimeError, lambda: official_pad_crop(ref, rand_img(2, 9), 4))    # upstream fails too
    raises(ValueError, lambda: m.forward(rand_img(2, 9)), "RRDBNet", ">= 3")
    close(m.forward(rand_img(3, 3)), official_pad_crop(ref, rand_img(3, 3), 4), "RRDBNet x1 3x3")


RST_G = dict(inp_channels=3, out_channels=3, dim=8, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1,
             heads=[1, 2, 2, 4], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="WithBias")


def _restormer(seed, **kw):
    arch = load_file(os.path.join(REPO["restormer"](), "basicsr/models/archs/restormer_arch.py"))
    g = {**RST_G, **kw}
    return randomize(arch.Restormer(**g), seed), {"type": "Restormer", **g}


def test_restormer():
    ref_a, g = _restormer(6)
    ref_b, _ = _restormer(7)
    sd_a, sd_b = sd_of(ref_a), sd_of(ref_b)
    x = rand_img(13, 21)
    both = {"params": sd_a, "params_ema": sd_b}
    m = build_model("restormer", g, both)
    assert m.describe()["params_key"] == "params"
    close(m.forward(x), official_pad_crop(ref_a, x, 8), "Restormer default params")
    m = build_model("restormer", g, both, params_key="params_ema")
    assert m.describe()["params_key"] == "params_ema"
    close(m.forward(x), official_pad_crop(ref_b, x, 8), "Restormer params_key=params_ema")
    raises(KeyError, lambda: build_model("restormer", g, both, params_key="nope"), "nope")
    m = build_model("restormer", g, {"params": {"module." + k: v for k, v in sd_a.items()}})
    close(m.forward(x), official_pad_crop(ref_a, x, 8), "Restormer module. prefix")
    ref_c, g_c = _restormer(8, bias=True)
    close(build_model("restormer", g_c, {"params": sd_of(ref_c)}).forward(x), official_pad_crop(ref_c, x, 8),
          "Restormer bias=True")
    raises(ValueError, lambda: build_model("restormer", g_c, {"params": sd_a}), "missing", ".bias")
    raises(ValueError, lambda: build_model("restormer", g, {"params": sd_of(ref_c)}), "unexpected", ".bias")
    raises(ValueError, lambda: build_model("restormer", {**g, "heads": [1, 2, 4, 4]}, {"params": sd_a}),
           "wrong shape", "temperature")
    raises(ValueError, lambda: build_model("restormer", {**g, "ffn_expansion_factor": 2}, {"params": sd_a}),
           "wrong shape", "ffn.project_in")
    # tiny inputs: official reflect pad to 8 needs H, W >= 5
    raises(RuntimeError, lambda: official_pad_crop(ref_a, rand_img(4, 9), 8))
    m = build_model("restormer", g, {"params": sd_a})
    raises(ValueError, lambda: m.forward(rand_img(4, 9)), "Restormer", ">= 5")
    raises(ValueError, lambda: m.forward(rand_img(9, 3)), "Restormer", ">= 5")
    close(m.forward(rand_img(5, 5)), official_pad_crop(ref_a, rand_img(5, 5), 8), "Restormer 5x5")


FFT_G = dict(inp_channels=3, out_channels=3, dim=8, num_blocks=[1, 1, 1, 1], num_refinement_blocks=1,
             ffn_expansion_factor=3, bias=False)


def test_fftformer():
    arch = load_file(os.path.join(REPO["fftformer"](), "basicsr/models/archs/fftformer_arch.py"))
    ref = randomize(arch.fftformer(**FFT_G), 9)
    g = {"type": "fftformer", **FFT_G}
    m = build_model("fftformer", g, sd_of(ref))                              # plain state dict
    assert m.describe()["params_key"] is None
    x = rand_img(17, 40)
    close(m.forward(x), official_pad_crop(ref, x, 32), "FFTformer 17x40")
    ref_b = randomize(arch.fftformer(**{**FFT_G, "bias": True}), 10)
    close(build_model("fftformer", {**g, "bias": True}, {"params": sd_of(ref_b)}).forward(x),
          official_pad_crop(ref_b, x, 32), "FFTformer bias=True")
    raises(ValueError, lambda: build_model("fftformer", {**g, "ffn_expansion_factor": 2}, sd_of(ref)),
           "wrong shape", "ffn.project_in")
    raises(ValueError, lambda: build_model("fftformer", {**g, "dim": 16}, sd_of(ref)), "wrong shape")
    raises(ValueError, lambda: build_model("fftformer", {**g, "bias": True}, sd_of(ref)), "missing", ".bias")
    # tiny inputs: official reflect pad to 32 needs H, W >= 17
    raises(RuntimeError, lambda: official_pad_crop(ref, rand_img(16, 40), 32))
    raises(ValueError, lambda: m.forward(rand_img(16, 40)), "FFTformer", ">= 17")
    raises(ValueError, lambda: m.forward(rand_img(40, 5)), "FFTformer", ">= 17")


def _evssm_arch():
    """models/EVSSM.py with mamba_ssm's pure-torch selective_scan_ref (CPU)."""
    extra = [MAMBA_PIP] if MAMBA_PIP else []
    sys.path[:0] = extra
    try:
        ssi = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
        ref_scan = ssi.selective_scan_ref
    except Exception:                                   # no mamba: shapes-only reference
        ssi, ref_scan = types.ModuleType("mamba_ssm.ops.selective_scan_interface"), None
        ssi.selective_scan_fn = ssi.selective_scan_ref = None
        for name in ("mamba_ssm", "mamba_ssm.ops"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["mamba_ssm.ops.selective_scan_interface"] = ssi
    finally:
        for p in extra:
            sys.path.remove(p)
    return load_file(os.path.join(REPO["evssm"](), "models/EVSSM.py")), ref_scan


def test_evssm():
    arch, ref_scan = _evssm_arch()
    from clarus.models.evssm import MIN_SIZE, state_dict_shapes
    for bias in (False, True):
        kw = dict(inp_channels=3, out_channels=3, dim=16, num_blocks=[1, 2, 1], ffn_expansion_factor=3, bias=bias)
        ref = randomize(arch.EVSSM(**kw), 11)
        with torch.no_grad():                           # keep the scan stable: A = -exp(A_logs) < 0
            for k, p in ref.named_parameters():
                if k.endswith("A_logs"):
                    p.uniform_(-1.0, 1.0)
        sd = sd_of(ref)
        exp = state_dict_shapes({**kw, "num_refinement_blocks": 4})
        assert {k: tuple(v.shape) for k, v in sd.items()} == {k: tuple(v) for k, v in exp.items()}, bias
        m = build_model("evssm", {"type": "EVSSM", **kw}, {"params": sd})
        x = rand_img(MIN_SIZE, 22)
        y = m.forward(x)
        if ref_scan is not None:
            for mod in ref.modules():
                if hasattr(mod, "selective_scan"):
                    mod.selective_scan = ref_scan
            close(y, official_pad_crop(ref, x, 4), f"EVSSM bias={bias} {MIN_SIZE}x22")
            if not bias:                                # upstream EDFFN reflect pad fails below MIN_SIZE
                raises(RuntimeError, lambda: official_pad_crop(ref, rand_img(MIN_SIZE - 1, 22), 4))
        raises(ValueError, lambda: m.forward(rand_img(MIN_SIZE - 1, 22)), "EVSSM", f">= {MIN_SIZE}")
    print(f"  upstream numerics compared: {ref_scan is not None}")


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    only = sys.argv[sys.argv.index("-k") + 1] if "-k" in sys.argv else ""
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and only in n]
    failed = []
    for name, fn in tests:
        print(f"{name} ...", flush=True)
        try:
            fn()
            print(f"{name} OK", flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(name)
            print(f"{name} FAILED", flush=True)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed" + (f"; failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()