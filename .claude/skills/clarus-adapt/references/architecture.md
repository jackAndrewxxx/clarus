# Architecture and contracts

## Layers

```
clarus (Rust, engine/src)                       config.rs  YAML + --set overrides
  main.rs    CLI, dataset scan (PNG headers), shape admission, warm-up, report
  engine.rs  scheduler: static / dynamic batching, graph batch sizes + padding, back-pressure
  io.rs      rayon IO pool: PNG decode into host slots, PNG encode from slots
  png.rs     `png` crate wrapper: any PNG -> RGB8 (cv2.IMREAD_COLOR semantics), RGB8 -> PNG
  pyworker.rs PyO3 host of the Python worker (JSON string in / out)
        |  info / warmup / submit / poll / wait / shutdown
        v
clarus (Python, python/clarus)
  worker.py   ModelWorker: host slots, per-(batch,H,W) device graph, metrics, completion events
  runtime.py  build(model_cfg, runtime_cfg) -> (model, ops, device); load_config()
  device/     Device: CudaDevice | AcceleratorDevice (torch.<type>, e.g. npu) | CpuDevice
  models/     registry + RestorationModel/IOSpec; nafnet/ restormer/ rrdbnet/ fftformer/ evssm/; _fold.py; template.py
  ops/        Ops dispatcher; torch_ref.py (reference, every op); triton_cuda.py (fused kernels)
  kernels/    Triton kernels (gemm, conv3x3, spatial, attention, freq, ssm) + tuning.py (persistent tuner)
  metrics.py  PSNR / SSIM: TritonMetrics (CUDA) | TorchMetrics (any device)
```

What is model- or hardware-specific:

| piece | specific to |
|---|---|
| `models/<family>/` | one network family each (any structure of it that the module accepts) |
| `ops/triton_cuda.py`, `kernels/*`, `metrics.TritonMetrics` | NVIDIA (Triton). ROCm Triton may work after re-tuning (unverified) |
| `device/CudaDevice` | CUDA streams / graphs |
| everything else | generic |

## Config flow

The Rust engine reads one YAML file (`configs/*.yaml`, schema documented in
`configs/nafnet_gopro_width64.yaml`), applies `--set` overrides, and sends the
`model:` and `runtime:` sections plus a `worker:` section (slots, max_pixels,
batch_sizes, inflight) to `clarus.worker.create(json)`. `model._config_dir` is
injected so relative paths resolve against the config file. Python tools load the same
YAML via `clarus.runtime.load_config` / `build_from_config`.

## Worker protocol (Python side, one JSON string per call)

| call | request | reply |
|---|---|---|
| `create(cfg)` | `{"model", "runtime", "worker"}` | worker object |
| `info()` | - | slot base / stride, `planes` {lq, gt, out: [offset, cap]}, `scale`, `batch_sizes`, device, graphs, model describe(), ops report |
| `warmup(r)` | `{"shapes": [[h, w]...], "metrics": bool}` | prepare time per (batch, shape) |
| `submit(r)` | `{"id", "h", "w", "slots": [...], "metrics", "output"}` | `"ok"` (async) |
| `poll()` / `wait()` | - | completed batches: `{"id", "results": [{"slot", "psnr", "ssim", "nonfinite"}], "latency_ms"}` |
| `shutdown()` | - | peak memory, graphs, ops report |

`len(slots)` is always one of `batch_sizes`; Rust pads short batches by repeating the
last slot and ignores the extra results. PSNR is `null` when the image is a perfect
reconstruction, because JSON has no Infinity.

## Ops contract

All backends implement a subset of these methods with identical semantics and
return types (the docstrings in `ops/torch_ref.py` are normative):

```
# NAFNet and shared
linear(x, w[K,N], bias, ln, act, gate, scale, residual, mul, out_dtype, ln_eps)   # mul = Small (TLC SCA)
boxmean(x, k|(kh,kw)) -> Small        dwconv(x, w[C,1,kh,kw], b, dil, gate, act)
intro(inp NCHW fp32, w, b, out_dtype) down(x, w[2C,C,2,2], b, out_dtype)
up(x, w[C,4C'] (dy,dx,c') order, residual, out_dtype)   ending(x, w, b, inp) -> fp32 NCHW

# Restormer, RRDBNet
conv3x3(x, w[Co,Ci,3,3], b, act=None|"lrelu", residual, out_scale, up2, out)   # out / x may be channel slices
pixel_shuffle(x) / pixel_unshuffle(x, r)   add_scaled(a, b, alpha, out)
channel_attention(qkv, heads, temperature) -> fp32 [N, heads, c, c]   # Restormer MDTA map
linear_per_image(x, w[N,K,Nout], residual, bias)                       # per-image GEMM
dwconv(..., gate="gelu")

# frequency / state-space ops (FFTformer, EVSSM)
linear(..., ln="rms")                                   # x / rms(x) prologue, no mean subtraction
linear_patch_filter(x, w, bias, f[C,8,5], ln, residual) # irfft2(rfft2(8x8 patch) * f) after a 1x1 conv
patch_corr(x, c) -> fp32                                # irfft2(rfft2(q) * rfft2(k)), q|k = x[..., :2c]
ln_mul_linear(x, ln_w, ln_b, m, w, bias, m_act, residual)  # ((LN(x)*w+b) * act(m)) @ w
dwconv(..., act="gelu")   resize_bilinear(x, 0.5 | 2)
seq_dwconv1d(x fp32, w[C,k], b)                          # depthwise conv along each image's raster sequence
selective_scan(u, dbl, dt_w, dt_bias, A, D) -> fp32      # Mamba scan, one direction, d_state 8 on triton
tiles_gather(x, k1, k2, idx, orient) / tiles_merge(base, parts, k1, k2, idx, orient)   # EVSSM crops
```

- Activations: NHWC, contiguous, compute dtype `Ops.dtype` unless `out_dtype` is given.
- Weights: canonical layout from the model's fold step. 1x1 convs and linears are
  `[K, N]`; spatial convs are 4-D PyTorch layout. Vectors (bias, scale) are fp32.
- Dispatch: `runtime.backends` is a priority list. Per call, the first backend that has
  the method and does not raise `Unsupported` runs it. `runtime.op_overrides` pins an
  op to one backend. `Ops.report()` counts which backend served each op during
  eager calls, and the engine prints the non-primary ones as `op fallbacks`.

## Graph capture rules

`worker._entry` runs the pipeline eagerly (compiles and tunes kernels), runs it again, then
captures it (`Device.capture`). During capture `TUNER.frozen` is set. Anything that would
tune, synchronize, or allocate pinned host memory inside `forward` breaks capture.
Per-backend weight re-layouts happen in the first eager pass and are cached.

## Numerics already decided

- NAFNet: fp16 activations; the residual stream is fp32 from level `residual_fp32_from_level` (3).
  Raw fp32 residual operands entering fp16 GEMMs are scaled by 2^-8.
- FFTformer / EVSSM: `patch_corr` and `selective_scan` return fp32 (their outputs reach
  ~1e7 and feed a LayerNorm).
- LayerNorm affine is folded into the next 1x1 conv; residual scales stay as fp32 epilogue
  vectors, because values as small as 4e-6 underflow in fp16.
- TLC local mean: window = feature size at `train_size` * 1.5 (basicsr `Local_Base`);
  non-Local networks use a global mean.
