# clarus

**English** | [简体中文](README.zh-CN.md)

A from-scratch inference and evaluation engine for image-restoration networks, built along
the lines of vLLM / SGLang. A Rust process owns scheduling, IO and reporting. The model runs
in an embedded Python worker: thin glue whose operators are dispatched to fused Triton
kernels on NVIDIA GPUs and to a PyTorch reference implementation everywhere else. One YAML
file selects the model, device, operator backends and batching.

Built in, with their official weights: NAFNet (any basicsr `network_g` of the NAFNet /
NAFNetLocal family, no code changes), Restormer, Real-ESRGAN (RRDBNet), FFTformer and EVSSM.
New networks plug in as Python classes, and new hardware starts on the PyTorch backend and
gains fast kernels one operator at a time.

- **7.2–8.4× faster end to end** than each model's official test script on one A30, with the same PSNR / SSIM
- **Rust core**: YAML config, PNG IO thread pool (`png` + `rayon`), shape-bucketed scheduler, PyO3 host
- **Per-operator backend dispatch**: Triton → PyTorch fallback, mixable op by op, fallbacks reported every run
- **Static mode**: fixed shapes and batch sizes, all device graphs captured at start-up, bit-reproducible batching
- **Validation tools** for every layer, plus an agent skill that walks through adapting the engine

## Contents

- [Results](#results)
- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Configuration](#configuration)
- [Adapting it to your needs](#adapting-it-to-your-needs)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [For AI agents](#for-ai-agents)

## Results

One NVIDIA A30. Each baseline is the model's own official inference / test script, run
unchanged; times are process start to last result.

| model (task) | data | official script | official | clarus | speed-up | quality, official → clarus |
|---|---|---|---|---|---|---|
| NAFNet-GoPro-width64 (deblur) | GoPro test, 1111 × 720p | `basicsr/test.py` (fp32) | 517.9 s | 65.5 s (`eval`) | **7.9×** | PSNR / SSIM 33.7103 / 0.9668 → 33.7103 / 0.9668 |
| Restormer (motion deblur) | GoPro test, 1111 × 720p | `Motion_Deblurring/test.py` (fp32, writes PNGs) | 1802.1 s | 227.1 s (`infer --save`) | **7.9×** | PSNR 32.9177 → 32.9177; 0.9% of pixels differ, by ≤ 2 levels |
| FFTformer (deblur) | GoPro test, 1111 × 720p | `test.py` (fp32, writes PNGs) | 2743.9 s | 381.4 s (`eval --save`) | **7.2×** | PSNR / SSIM 34.2115 / 0.9694 → 34.2113 / 0.9694; 1.2% of pixels differ, by ≤ 5 levels |
| EVSSM (deblur, Mamba) | GoPro test, 1111 × 720p | `test.py` (fp32, mamba_ssm CUDA scan, writes PNGs) | 3972 s | 473.0 s (`eval --save`) | **8.4×** | PSNR / SSIM 34.5106 / 0.9713 → 34.5103 / 0.9713; 2.4% of pixels differ, by ≤ 4 levels |
| Real-ESRGAN x4plus (×4 SR) | Urban100 LR×4, 100 images in 37 sizes | `inference_realesrgan.py` (fp16 by default) | 20.7 s | 12.0 s (`infer --save`) | **1.7×** | PSNR vs HR 21.0561 → 21.0568; closer to an fp32 reference than the official fp16 run (5.2% vs 7.2% of pixels differ) |

Single forward pass (batch 1): NAFNet 720p 251.0 → 54.3 ms; Restormer 720p 1482 → 199 ms;
RRDBNet at 256×384 input 250 (official fp32) / 167 (official fp16) → 84 ms; FFTformer 720p
2030 → 330 ms; EVSSM 720p 3149 → 407 ms. FFTformer / EVSSM PSNR / SSIM above are both
measured with `tools/score_outputs.py` on the saved PNGs (`clarus eval` itself reports
34.2113 / 0.9692 and 34.5103 / 0.9711; the SSIM gap is basicsr's TF32 SSIM on the GPU).
Real-ESRGAN gains least: its 3×3 convolutions are already cuDNN-fp16 territory, and clarus
wins mostly from NHWC layout and fusing the dense-block concatenations, LeakyReLU and the
upsampling into the convolutions.

Reproduce any row with `clarus eval --config <config> --report r.json` (see [Validation](#validation)); run logs are not tracked in git.

## Quick start

### Requirements

| | tested version | needed for |
|---|---|---|
| Python | 3.11 (shared `libpython`) | model worker |
| PyTorch | 2.6.0 + CUDA 12.4 | always |
| Triton | 3.2.0 | the fast CUDA backend (optional elsewhere) |
| PyYAML | 6.0 | config files |
| Rust | 1.92 (edition 2021) | building the engine |
| numpy, opencv-python, the basicsr repo | | validation tools only |

The engine itself does not import basicsr; the network structure comes from the config
(`model.network_g`, or `model.arch_from` pointing at a training yml). Only the reference tools in
`tools/` build the original basicsr network: point `CLARUS_BASICSR` at a basicsr checkout that
contains the architecture (for the NAFNet configs, a clone of megvii-research/NAFNet), and
`CLARUS_LQ` / `CLARUS_GT` at an evaluation pair (e.g. the GoPro test set's blur / sharp folders).

### Build

```bash
git clone git@github.com:<you>/clarus.git && cd clarus/engine
# PyO3 links the interpreter named by PYO3_PYTHON (default set in engine/.cargo/config.toml)
PYO3_PYTHON=/path/to/venv/bin/python cargo build --release      # add --offline once crates are cached
```

The interpreter's home, site-packages and the `python/` package path are baked into the
binary; override them at run time with `CLARUS_PYTHONHOME`, `CLARUS_SITE` and `CLARUS_PYTHONPATH`.

### Run

```bash
B=./target/release/clarus

# evaluation: PSNR / SSIM with the same definition as basicsr calculate_psnr / calculate_ssim
CUDA_VISIBLE_DEVICES=0 $B eval --lq data/blur --gt data/sharp --report report.json [--save out/]

# inference only: writes restored PNGs
$B infer --input data/blur --output out/

# another model / device / fixed shapes: config or flags only
$B eval --config ../configs/restormer_motion_deblurring.yaml --lq ... --gt ...
$B eval --device cpu --dtype fp32 --lq ... --gt ...
$B infer --input ... --output ... --shapes 720x1280 --batch-sizes 2
$B eval ... --set runtime.op_overrides.selective_scan=torch
$B --help
```

Inputs are the `.png` files in `--lq` / `--input`. Ground truth is matched by file name in
`--gt`. Any PNG works: 1–16 bit, gray, gray + alpha, RGB, RGBA, palette, Adam7 interlaced.
Images are decoded to RGB8 like `cv2.IMREAD_COLOR`, can have any size, and are
zero-padded internally to the model's multiple (16 for NAFNet).

### Reading the output

```
[clarus] eval nafnet on NVIDIA A30 | 1111 images, shapes [[720, 1280]], batch sizes [1, 2], graphs true | load 2.3s, warm-up 1.0s
==== clarus eval (nafnet) ====
images            1111 (0 failed)                    # failed = decode errors / rejected shapes / missing GT size
PSNR / SSIM       33.7103 / 0.9668   (psnr 33.710309, ssim 0.966790)
pipeline wall     64.04 s  -> 17.35 img/s (57.6 ms/img)   # first decode -> last result
end-to-end wall   67.43 s (scan 0.03, load 2.32, warm-up 1.03)   # warm-up = graph capture (+ tuning on first run)
batches           556 (Static, mean size 2.00, 0 padded, mean device latency 344.9 ms)   # latency includes queueing
decode / encode   24.7 / 0.0 ms per image (16 IO threads)
op fallbacks      ...                                # only printed if some op ran on a non-primary backend
report            report.json
```

`--report` writes JSON with `settings` (the effective config), `model` (the arch the worker
built), `psnr` / `ssim`, `timing`, `stats` (batches, padding, decode / encode times),
`worker` (peak memory, captured graphs, per-op backend counts) and `images` (per image:
`name`, `h`, `w`, `psnr`, `ssim`, `nonfinite`, `decode_ms`, `batch_ms`, `batch`, `error`).

The first run on a new GPU or image size tunes kernels, which takes tens of seconds to a
few minutes. The results are cached in `cache/tune_<gpu>_triton<version>.json`, and later
warm-ups take seconds.

## How it works

```
clarus (Rust)
 ├─ config.rs    YAML + --set overrides; model/runtime go to the worker, scheduler/io stay here
 ├─ main.rs      scan PNG headers -> shape set, admission, warm-up, summary + JSON report
 ├─ engine.rs    scheduler: shape buckets, graph batch sizes (+ padding), static / dynamic batching
 ├─ io.rs        rayon pool: decode PNG straight into host slots, encode outputs
 └─ pyworker.rs  PyO3 host; one JSON string per call (info / warmup / submit / poll / wait / shutdown)
        │
        ▼
clarus (Python)
 ├─ worker.py    host slots, one captured device graph per (batch, H, W), metrics, completion events
 ├─ runtime.py   config -> (model, ops, device)
 ├─ models/      registry, RestorationModel / IOSpec, nafnet/ restormer/ rrdbnet/ fftformer/ evssm/, template.py
 ├─ ops/         dispatcher; torch_ref.py (reference, every op), triton_cuda.py (fused kernels)
 ├─ device/      CudaDevice | AcceleratorDevice (any torch.<type>, e.g. torch_npu) | CpuDevice
 ├─ kernels/     Triton kernels + persistent tuner
 └─ metrics.py   PSNR / SSIM on the device (Triton or PyTorch)
```

**One batch, start to finish.**
1. A Rust IO thread decodes the PNG into a host slot. The slot is page-locked memory owned
   by the worker, so pixels never become Python objects.
2. The scheduler groups decoded images of the same shape into one of the configured
   batch sizes and calls `submit`.
3. The worker queues everything on one device stream: copy to the device, `u8 → fp32 / 255`
   with zero padding, the network, clamp / round to u8, PSNR / SSIM, and copy back. For
   each (batch, H, W) this whole sequence is one captured device graph, so Python does a
   single graph replay per batch.
4. `poll` / `wait` return the metrics. The slot is freed, or handed to an IO thread to
   encode the output PNG.

Slots are the back-pressure unit: decoding never runs further ahead than
`max(batch_sizes) × (inflight + 2)` images.

**Batching modes.** `static` (the default) fixes every batch before the run: per shape, files
are taken in name order, and decodes are issued batch by batch, so outputs are
bit-identical across runs. `dynamic` forms batches from whatever has finished decoding.
A full batch goes out at once; a partial batch goes out only when fewer than two batches
are queued on the device.

**Where the speed-ups come from** (details in the source):
1. Inference-only rewrites with folded weights. LayerNorm affines are merged into the next
   1×1 conv; split / concatenate / add patterns become one GEMM with split weights; tiny
   residual scales (NAFNet's beta / gamma) stay in fp32 epilogues.
2. One fused NHWC GEMM kernel serves 1×1 convs, 2×2 stride-2 convs and PixelShuffle.
   Prologues: LayerNorm / RMS norm, TLC local-mean gather, LayerNorm × gate. Epilogues: bias,
   SimpleGate, GELU, scale, residual.
3. NAFNet's TLC local mean uses sliding prefix cursors over the valid region only, instead
   of a full-image fp32 double cumsum plus pad.
4. Restormer's channel attention is a split-K Gram reduction plus one per-image GEMM folded
   into project_out. FFTformer / EVSSM's 8×8-patch FFT filters are 64×64 real DFT products on
   tensor cores, fused behind the 1×1 conv that feeds them.
5. EVSSM's selective scan runs one channel per lane with segment-parallel carries (faster
   than the official mamba_ssm kernel at these shapes); the per-block transpose / flip and
   overlapping crops are folded into gather / merge addressing.
6. Depthwise convs use line buffers with fused gates; concatenations become channel slices
   of one buffer (RRDBNet dense blocks).
7. Mixed precision: fp16 activations; fp32 where values exceed fp16 (NAFNet's deepest
   residual stream, FFT correlation and scan outputs that feed a LayerNorm).
8. Device graphs plus a persistent per-GPU tuning cache.
9. Metrics on the device. PSNR uses exact integer SSE; SSIM uses a separable 11×11×11
   Gaussian with centred moments.

## Project layout

```
clarus/
├── README.md / README.zh-CN.md
├── configs/
│   ├── nafnet_gopro_width64.yaml     default config (official NAFNet GoPro weights); every key documented inline
│   ├── restormer_motion_deblurring.yaml  official Restormer motion-deblurring weights
│   ├── realesrgan_x4plus.yaml        official Real-ESRGAN x4plus (RRDBNet, x4 SR)
│   ├── fftformer_gopro.yaml          official FFTformer GoPro weights
│   └── evssm_gopro.yaml              official EVSSM GoPro weights (Mamba / selective scan)
├── checkpoints/                      put / symlink weights here (not tracked by git)
├── engine/                           Rust crate `clarus`
│   ├── .cargo/config.toml            PYO3_PYTHON default
│   ├── build.rs                      rpath to libpython, baked default paths
│   └── src/{main,config,engine,io,png,pyworker}.rs
├── python/clarus/                   model worker package
│   ├── worker.py  runtime.py  metrics.py
│   ├── device/__init__.py
│   ├── ops/{__init__,torch_ref,triton_cuda}.py
│   ├── models/{__init__,_fold,template}.py, models/{nafnet,restormer,rrdbnet,fftformer,evssm}/
│   └── kernels/{gemm,conv3x3,spatial,attention,freq,ssm,tuning}.py
├── tools/                            validation, benchmarking, debugging (all take --config)
├── .claude/skills/clarus-adapt/      agent skill (SKILL.md + references/), for Claude Code
├── .agents/skills/clarus-adapt/      the same skill, for agents that read .agents/skills
└── cache/                            kernel tuning results (per GPU model + Triton version)
```

The two skill directories hold identical copies; edit both (or copy one over the other) when changing the skill.

## Configuration

One YAML file with four sections. Relative paths resolve against the config file's directory.
Any key can be overridden with `--set section.key=value` (the value is parsed as YAML).

| key | default | meaning |
|---|---|---|
| `model.name` | — | registered model: `nafnet`, `restormer`, `rrdbnet` (aliases `realesrgan`, `esrgan`), `fftformer`, `evssm`; `example_unsharp` / `example_bilinear_x2` from the template plugin |
| `model.plugins` | `[]` | Python modules imported before the model is built (external models) |
| `model.checkpoint` | — | weights file (basicsr `params` / `params_ema` dicts are accepted) |
| `model.network_g` | — | basicsr `network_g` dict describing the network (copy it from the training yml) |
| `model.arch_from` | — | alternatively, a basicsr yml to read `network_g` from (use the *training* yml); `network_g` keys then override it |
| `model.options.*` | | model-specific, documented in each model's module docstring (e.g. NAFNet's `residual_fp32_from_level`: pyramid level from which the residual stream is fp32, `0` = all, i.e. `--safe`) |
| `runtime.device` | `cuda:0` | `cuda:N`, `cpu`, or any torch device such as `npu:0` |
| `runtime.device_imports` | `[]` | modules that register the device with torch, e.g. `[torch_npu]` |
| `runtime.dtype` | `fp16` | activation dtype `fp16` / `bf16` / `fp32` (fp32 → PyTorch backend only) |
| `runtime.backends` | `[triton, torch]` | operator dispatch order; an op the first backend cannot handle falls through |
| `runtime.op_overrides` | `{}` | pin an op to one backend, e.g. `{selective_scan: torch}` |
| `runtime.graphs` | `auto` | capture device graphs: `auto` / `true` / `false` |
| `runtime.metrics_backend` | `auto` | `triton` on CUDA, `torch` elsewhere |
| `scheduler.batch_sizes` | `[1, 2]` | graph batch sizes; a short batch is padded to the next size |
| `scheduler.batching` | `static` | `static` (reproducible) / `dynamic` |
| `scheduler.shapes` | `auto` | `auto` = the shapes found in the input set, or a fixed list `[[h, w], ...]` captured at start-up |
| `scheduler.unknown_shape` | `reject` | for shapes outside a fixed list: `reject` / `lazy` (prepare on first use) |
| `scheduler.inflight` | `3` | batches queued on the device |
| `io.threads` | `16` | PNG decode / encode threads |
| `io.png_level` | `1` | output compression: 0 none, 1 fast, 2–6 balanced, 7+ high |

Command-line shortcuts (`clarus --help`): `--config`, `--set`, `--ckpt`, `--device`
(`1` keeps the device type), `--dtype`, `--backends a,b`, `--no-graph`, `--safe`,
`--batch-sizes 1,2`, `--max-batch N`, `--batching`, `--shapes HxW,...`, `--inflight`,
`--io-threads`, `--png-level`, `--limit N`, `--verbose`.

| environment variable | effect |
|---|---|
| `PYO3_PYTHON` (build time) | interpreter whose libpython / site-packages the engine uses |
| `CLARUS_CONFIG` | default config file |
| `CLARUS_PYTHONHOME`, `CLARUS_SITE`, `CLARUS_PYTHONPATH` | override the baked interpreter home, site-packages and `clarus` path |
| `CLARUS_TUNE_DIR`, `CLARUS_TUNE_CACHE` | tuning cache directory / exact file |
| `CUDA_VISIBLE_DEVICES` | as usual |

## Adapting it to your needs

Start with the smallest change that fits. The skill in `.claude/skills/clarus-adapt/` has
the full procedures.

**1. Another checkpoint of a built-in family (config only).**
For example any NAFNet / NAFNetLocal with any width, depths and levels, and TLC with any
`train_size` or a global pool; or a Restormer / RRDBNet / FFTformer / EVSSM checkpoint
with a different `network_g`.
```bash
cp configs/nafnet_gopro_width64.yaml configs/mine.yaml
# edit model.checkpoint and model.network_g (copied from the yml the checkpoint was trained with)
python tools/parity.py --config configs/mine.yaml --images 12 --odd       # accuracy vs the original network
engine/target/release/clarus eval --config configs/mine.yaml --lq ... --gt ...
```
Options the runtime does not implement are rejected at load with their name. A
config / checkpoint mismatch fails in weight folding, either with "checkpoint has no
tensor ..." or with "tensors not consumed ...".

**2. Your own dataset.** Put LQ and GT PNGs with identical file names in two directories.
GT must be `scale ×` the LQ size. Mixed image sizes are fine: each shape gets its own
graphs. Use `--limit N` for a quick check.

**3. A fixed-shape deployment.** `--shapes 720x1280,1080x1920 --batch-sizes 4` (or the
`scheduler:` keys) captures every graph at start-up. Inputs of other shapes are rejected,
or prepared on first use with `unknown_shape: lazy`. Short batches are padded, so only
the configured graphs ever run.

**4. A different network.** Copy `python/clarus/models/template.py`. It is a working
plugin, runnable with
`--set model.name=example_unsharp --set model.plugins=[clarus.models.template]`.
Implement `__init__` (parse the arch, load and fold weights into the canonical layout, set
`IOSpec`) and a capture-safe `forward` that calls `self.ops.<op>`. Register the class with
`@register_model("name")` and load it through `model.plugins`. Any new op goes into
`ops/torch_ref.py` first, so it runs everywhere; add a fast kernel afterwards.

**5. Other hardware.** Start with the PyTorch backend:
`--set runtime.device=npu:0 --set runtime.device_imports=[torch_npu] --backends torch --no-graph`.
Check parity with `tools/parity.py --device npu:0 --backends torch`. Then enable graphs
if the device supports them, and add a backend class (`ops/__init__.Backend`) that
implements the hot ops. Unimplemented ops keep running on PyTorch, and the
`op fallbacks` line shows which ones. CPU is verified this way; NPU has not been tested
here.

**6. A faster op.** Write the kernel against the op's `torch_ref.py` semantics. Raise
`Unsupported` for argument combinations it does not handle, add a case to
`tools/test_kernels.py`, and measure with `tools/kernel_breakdown.py`.

**7. Numerics.** `--safe` keeps the whole residual stream in fp32. It is needed for
out-of-distribution inputs; the engine warns when outputs contain non-finite values. Use
`--dtype bf16` for more range at lower precision, or
`--dtype fp32 --backends torch` for a reference run.

**8. From Python.**
```python
import sys; sys.path.insert(0, "python")
from clarus.runtime import build_from_config
model, ops, dev = build_from_config("configs/nafnet_gopro_width64.yaml")
y = model.forward(x)        # x: [N, 3, H, W] float32 in [0, 1] on dev.torch_device
```
`clarus.worker.ModelWorker` adds slots, graphs and metrics; `tools/worker_smoke.py`
drives it without Rust.

## Validation

| check | command | pass |
|---|---|---|
| kernels vs fp32 PyTorch | `python tools/test_kernels.py` | `ALL OK` |
| model vs original network | `python tools/parity.py --config CFG [--backends torch --dtype fp32]` | ~1e-6 relative (reference backend); max dPSNR < 0.01 dB (fp16) |
| structural variants | `python tools/check_zoo_variants.py` (NAFNet, Restormer, RRDBNet, FFTformer, EVSSM; random weights) | `ALL OK` |
| official script outputs | `python tools/score_outputs.py OUT GT --other CLARUS_OUT` | same metric as `clarus eval`, plus pixel agreement |
| metrics | `python tools/test_metrics.py --no-tf32 --fp64` | PSNR exact; SSIM ~1e-7 from fp64 |
| PNG decoder vs cv2 | `clarus png-hash DIR` vs `python tools/png_hash_cv2.py DIR` | identical |
| engine end to end | `python tools/check_engine.py --device N` | `PASS` |
| per-image vs original pipeline | `tools/reference_eval.py` + `tools/compare_reports.py` | p99 dPSNR < 0.002 dB |
| Rust unit tests | `cd engine && cargo test --release --offline` | all pass |
| kernel / op / model / tool contracts | `python tools/test_kernel_contracts.py`, `test_ops_contracts.py`, `test_model_contracts.py`, `test_tools.py`, `worker_smoke.py --selftest` | all pass |

Every tool exits non-zero on a failure, on missing or extra samples, and when nothing was
checked (e.g. no reference configured). `clarus eval` / `infer` exit 1 when any image failed
(`--allow-failures` keeps exit 0); the report is written either way.

Commands, thresholds and reference numbers: `.claude/skills/clarus-adapt/references/validation.md`.

## Troubleshooting

| message / symptom | cause and fix |
|---|---|
| `unknown model.name 'x'; registered: [...]` | typo, or the plugin module is missing from `model.plugins` |
| `checkpoint has no tensor ...` / `tensors not consumed ...` | `network_g` / `arch_from` does not describe this checkpoint; take it from the yml it was trained with |
| `... not implemented` / `... not supported` while loading | a `network_g` option or network type the runtime lacks; see "A different network" |
| `op backend 'triton' unavailable: ...` | expected on non-CUDA devices or with fp32; the PyTorch backend takes over |
| `op fallbacks  x->torch` | an op ran on the PyTorch backend: no kernel for those arguments. Correct, but slower |
| `untuned kernel shape during graph capture` | a kernel was first used inside capture; make `forward` deterministic in shape and call ops the same way every time |
| `no valid config for ...` | no tile configuration compiled for this problem size; check the candidate list in that kernel |
| very long warm-up | first run on this GPU / image size (tuning); later runs use `cache/` |
| `shape HxW (HxW) not in scheduler.shapes` | fixed-shape mode; add the shape or set `scheduler.unknown_shape: lazy` |
| `missing GT for NAME` / `GT size ... != expected` | GT directory incomplete or not `scale ×` the input size |
| `WARNING: N non-finite output values` | fp16 overflow on unusual inputs; re-run with `--safe` |
| CUDA out of memory right at start-up | another process holds the GPU; choose a free one with `nvidia-smi` |
| build: cannot find libpython / wrong torch at run time | set `PYO3_PYTHON` when building, or `CLARUS_PYTHONHOME` / `CLARUS_SITE` at run time |

## Known limitations

- One device per process. The IO path is PNG → RGB8, 3 channels in and out. Super-resolution
  (`IOSpec.scale > 1`) has only been verified with the synthetic `example_bilinear_x2`.
- The Triton backend has only been tuned and validated on an NVIDIA A30. Other devices run
  the PyTorch backend (eager): CPU is verified, NPU is untested.
- In `dynamic` batching, an image can land in batches of different sizes across runs, so about
  0.5% of its pixels may differ by one grey level. `static` (the default) avoids this.
- The mixed-precision choices (fp32 residual levels, fp32 op outputs, 2⁻⁸ operand scaling)
  were chosen from measured activation ranges of these checkpoints; measure again for a new model.

## For AI agents

The skill `clarus-adapt` ships as identical copies in `.claude/skills/clarus-adapt` and
`.agents/skills/clarus-adapt`, so Claude Code and other agents that read `.agents/skills`
find it without setup. It lists
the ground rules (canonical weights, capture safety, fp16 range, shared GPUs) and links to
the architecture contracts, step-by-step recipes and the validation checklist.

## License

MIT, see [LICENSE](LICENSE). The networks this engine re-implements (NAFNet, Restormer,
Real-ESRGAN, FFTformer, EVSSM) come from their authors' repositories, each under its own
license; no weights are distributed here.
