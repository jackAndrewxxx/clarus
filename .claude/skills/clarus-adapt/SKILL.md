---
name: clarus-adapt
description: Adapt the clarus inference engine (Rust scheduler + embedded Python model worker + Triton kernels) to a new checkpoint, a new restoration model, a new op kernel, or a new device / framework (e.g. Ascend NPU via torch_npu, ROCm, CPU). Use when the user wants to serve or evaluate another model with clarus, port it to other hardware, add or replace a kernel, change batching / fixed-shape settings, or debug an accuracy or performance regression in the engine.
---

# clarus: adapting to new models and devices

The engine is this repository (`clarus`). Read `README.md` for results and usage. This skill covers changing what the engine
serves and where it runs. Detailed references are in `references/` next to this file:

| file | read it when |
|---|---|
| `references/architecture.md` | before any change: layers, files, contracts (canonical weights, op semantics, graph-capture rules, worker protocol) |
| `references/new-model.md` | serving a new checkpoint or a new network |
| `references/new-device.md` | new hardware / framework, new op backend or kernel |
| `references/validation.md` | proving the change is correct and fast; the exact commands and pass thresholds |

## Pick the smallest change

1. **Same network family, different checkpoint or structure** (NAFNet / NAFNetLocal of any
   width / depth / TLC `train_size`, or another Restormer / RRDBNet / FFTformer / EVSSM
   `network_g`): config only. Copy the family's config (e.g. `configs/nafnet_gopro_width64.yaml`),
   set `model.checkpoint` and `model.network_g` (copied from the training yml) or
   `model.arch_from`, run the validation. No code.
2. **Different network**: write a model class (`clarus/models/template.py` is the
   skeleton; `models/restormer/`, `models/rrdbnet/`, `models/fftformer/` and `models/evssm/`
   are complete worked examples: a transformer, an x4 SR GAN generator, a frequency-domain
   transformer and a Mamba (selective scan) network), register it, load it via `model.plugins` or add
   it to the package. Reuse
   existing ops; any op you add needs a `torch` reference implementation first.
3. **Different hardware**: usually `runtime.device` + `runtime.device_imports` +
   `runtime.backends: [torch]` already runs (PyTorch reference backend, eager). Then
   add a device class only if the generic one is not enough, and add a fast backend
   op by op. Ops without a fast kernel keep falling back to `torch`.
4. **Scheduling only** (fixed shapes, batch sizes, determinism): `scheduler:` keys,
   no code. See "Scheduler knobs" below.

## Ground rules (they encode past bugs)

- Never change numerics without running the parity tools; keep the PyTorch reference
  backend (`clarus/ops/torch_ref.py`) as the definition of every op.
- Weights are folded once into the *canonical layout*; backends re-layout lazily via
  `Backend.cached`. Do not make a backend mutate the shared weight dict: ops from
  different backends must be mixable.
- `forward` must be graph-capture safe: no `.item()` / `.cpu()` / `.tolist()` on device
  tensors, no shape- or value-dependent Python branches inside a captured region,
  and nothing that allocates new kernel configs during capture (the tuner raises
  `untuned kernel shape during graph capture` if it happens).
- Values can exceed the fp16 range: NAFNet-width64's deepest residual stream, FFTformer's FFT
  correlation (~1e7) and EVSSM's scan output (~7e6) all overflow fp16 on real images.
  Measure activation ranges for a new model before choosing dtypes
  (`model.options.residual_fp32_from_level` for NAFNet / Restormer).
- A kernel that cannot handle a call raises `clarus.ops.Unsupported`; do not silently
  compute something approximate.
- basicsr's `calculate_ssim` runs with TF32 enabled, so a per-image SSIM gap of up to 4e-3
  against it is noise. The engine's SSIM matches an fp64 ground truth to ~1e-7.
- On shared GPUs, check `nvidia-smi --query-gpu=index,memory.used,utilization.gpu
  --format=csv` and pick an idle card before benchmarking; timings on a busy card are
  meaningless.
- If the machine reaches crates.io / PyPI only through a proxy, export `http_proxy` /
  `https_proxy` for `cargo` / `pip`, and never print or commit proxy credentials.

## Scheduler knobs (config only)

```yaml
scheduler:
  batch_sizes: [1, 2]       # one device graph per (size, H, W); short batches are padded
  batching: static          # static = bit-reproducible batches; dynamic = form as decodes finish
  shapes: auto              # or [[720, 1280], ...]: fixed set, captured at start-up
  unknown_shape: reject     # reject | lazy (prepare a graph on first use, may tune mid-run)
  inflight: 3
```

For a fully fixed-shape deployment, set `shapes` to a list, `batch_sizes` to one size,
and `unknown_shape: reject`. Every batch then replays one of a fixed set of graphs.

## Done means

Run the checks in `references/validation.md` that match the change, and report the
numbers (parity deltas, eval PSNR/SSIM against the official outputs or reference, throughput
before / after, the `op fallbacks` line). Do not report success from a smoke run alone.
