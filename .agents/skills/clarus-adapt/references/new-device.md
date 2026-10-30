# New hardware / framework, new backend, new kernel

## 1. Run on the new device with the reference backend (no kernels needed)

```yaml
runtime:
  device: npu:0                 # any torch device string; cpu works too
  device_imports: [torch_npu]   # modules that register the device with torch
  dtype: fp16                   # fp32 first if fp16 support is unknown
  backends: [torch]             # PyTorch reference implementation of every op
  graphs: false                 # turn on after eager parity passes
  metrics_backend: torch
```

`device.get_device` maps `cuda` to `CudaDevice`, `cpu` to `CpuDevice`, and anything else
to `AcceleratorDevice`. That class drives the generic `torch.<type>` module API:
`Stream`, `stream()`, `Event`, `synchronize`, `set_device`, `get_device_name`,
`max_memory_allocated`, plus `graph` and `*Graph` for capture. If the device's torch
plugin differs, subclass `AcceleratorDevice` (or `Device`), override what differs, and
call `register_device("<type>", Cls)` from a module listed in `runtime.device_imports`.

Things to check on a new device:
- Math-mode flags. `CudaDevice.set_current` turns TF32 off; the generic class does not
  touch such flags, so set the device's equivalent (e.g. reduced-precision matmul modes)
  in your device module. Driving CUDA through `AcceleratorDevice` gives bit-identical
  engine outputs only with TF32 off, and 0/9 identical (dPSNR 2e-6) with it on.
- `torch.fft` (used by the torch `patch_filter` / `patch_corr` ops of FFTformer / EVSSM)
  may be missing or slow on it; the DFT-matrix form in `kernels/freq.py` is a portable
  alternative.
- float64 support (results buffer, PSNR). If it is missing, switch `res` in
  `worker._entry` and `TorchMetrics` to float32 behind a device check.
- Pinned host memory: `AcceleratorDevice.host_buffer` falls back to pageable memory.
  Copies still work, just not asynchronously.
- Graph capture: enable `graphs: auto|true` only after eager parity passes. The model
  must be capture-safe (SKILL.md ground rules).

Validate with `tools/parity.py --device <dev> --backends torch` and a small
`clarus eval --device <dev> --limit 20`. CPU has been verified this way: parity is
~1e-6 relative, and end-to-end metrics are within 0.0013 dB of the CUDA engine.

## 2. Add a fast backend

1. New module, e.g. `clarus/ops/ascend.py`:
   ```python
   from . import Backend, BackendUnavailable, Unsupported, register_backend

   @register_backend("ascend")
   class AscendOps(Backend):
       def __init__(self, dtype, device):
           super().__init__(dtype, device)
           if torch.device(device).type != "npu":
               raise BackendUnavailable("needs an NPU")
       def linear(self, x, w, bias=None, ln=False, act=None, gate=False, scale=None,
                  residual=None, mul=None, out_dtype=None, ln_eps=1e-6):
           if mul is not None:
               raise Unsupported("not yet")      # -> next backend (torch)
           ...
   ```
   Built-in names are resolved in `ops/__init__._BACKENDS`. For a plugin backend, import
   its module through `runtime.device_imports` (the decorator registers it) or call
   `register_backend("name", "pkg.mod:Class")`.
2. List it first: `runtime.backends: [ascend, torch]`.
3. Implement ops in order of time share. Find the share on the reference backend with
   `torch.profiler` on the new device, or `tools/kernel_breakdown.py` on CUDA. Typical at
   720p: NAFNet is mostly fused GEMMs plus depthwise conv and the TLC box mean; EVSSM
   spends about a third in `selective_scan`; FFTformer in the fused 1x1 conv + FFT filter
   and depthwise conv.
4. Weight layouts: convert from the canonical layout with
   `self.cached(w, "tag", fn)`. Never write back into the weight dict.
5. Per-op fallback lets you ship partial backends. The engine prints
   `op fallbacks  dwconv->torch, ...` after every run, and `Ops.report()` gives counts.

## 3. Add or replace a kernel in an existing backend

1. Reference first: the op's semantics are its `torch_ref.py` implementation.
2. Write the kernel; for Triton follow `kernels/*`: an explicit candidate list of configs
   plus `TUNER.pick(name, key, cands, launch)`. Clamp tile sizes to the problem
   (`min(BLOCK, max(16, next_power_of_2(dim)))`) instead of filtering candidates out:
   empty candidate lists crash small models. Use a size-bucketed key so new image
   shapes reuse tuned configs. Tuned configs are stored per GPU model and Triton version
   (`cache/tune_<gpu>_triton<ver>.json`).
3. Guard unsupported arguments with `raise Unsupported(...)`.
4. Add a case to `tools/test_kernels.py` (kernel vs fp32 torch on odd sizes), then run the
   model-level checks.
5. Measure with `tools/kernel_breakdown.py --config ...` (per-launch time, TF/s, GB/s)
   and `tools/bench_model.py --graph`.

## 4. Other frameworks (TensorRT, ONNX Runtime, MindSpore, ...)

Treat the framework as a backend for the whole model rather than per op:
1. Write a model class whose constructor builds or loads the framework engine (e.g. a
   TensorRT plan for each (batch, H, W) the scheduler will use, which `scheduler.shapes` +
   `batch_sizes` make finite) and whose `forward` runs it on torch device tensors.
2. Set `runtime.graphs: false` if the framework manages its own graphs or streams.
3. The Rust engine, scheduler, IO and metrics stay unchanged.
Custom ops (TLC local mean, patch FFT filters, selective scan) need framework plugins;
export the canonical folded weights (the model's `self.w` after folding) rather than the
training graph.
