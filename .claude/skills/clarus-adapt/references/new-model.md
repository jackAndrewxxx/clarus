# Serving a new checkpoint or a new network

## A. Same family (another checkpoint of a built-in network)

1. Copy the family's config, e.g. `cp configs/nafnet_gopro_width64.yaml configs/<name>.yaml`.
2. Set `model.checkpoint`, and either inline the basicsr `network_g` the checkpoint was
   trained with as `model.network_g`, or point `model.arch_from` at that training yml.
   With `arch_from`, keys in `model.network_g` override single keys. Take the network from
   the *training* yml: test ymls can drift (other checkpoint, other options).
3. Unsupported options fail at load with a message naming the option. A checkpoint /
   config mismatch fails at load in the exact key / shape check, listing missing,
   unexpected and mis-shaped tensors. `model.params_key` picks the weights inside a basicsr
   checkpoint (default: the family's `PARAMS_KEYS` order, e.g. `params_ema` first for RRDBNet).
4. Validate (see validation.md): `tools/parity.py --config configs/<name>.yaml`, then a
   full `clarus eval --config ...` compared with the official outputs or the training log.

What each family accepts is in its module docstring. NAFNet: `NAFNet` / `NAFNetLocal` with
any width, block counts and number of levels, TLC with any `train_size` (not `fast_imp`).

## B. A different network

1. Start from `python/clarus/models/template.py` (a working example, loadable with
   `--set model.name=example_unsharp --set model.plugins=[clarus.models.template]`).
2. Put the model in its own module (inside `clarus/models/<name>/` or anywhere on
   `sys.path`) and decorate the class with `@register_model("<name>")`. If it lives
   outside the package, list the module in `model.plugins`.
3. Constructor `(cfg, ops, device, dtype)`:
   - parse the architecture from `cfg` (`network_g` / `arch_from`, or your own keys)
     and reject unsupported options explicitly;
   - load the checkpoint with `_fold.load_checkpoint(cfg, PARAMS_KEYS)` (define a module-level
     `PARAMS_KEYS`; the reference tools load the same tensors through it), then validate it
     with `_fold.check_state_dict(sd, {key: shape})` built from the parsed architecture, so a
     wrong `network_g` fails at load instead of silently running the checkpoint's shapes;
   - reject input sizes the official code cannot handle (e.g. reflect padding wider than
     the image: `_fold.check_reflect_pad`) with the minimum size in the message;
   - **fold once**: fold norm affines into the following linear
     ops, merge consecutive linear maps, drop training-only heads, and convert to the
     canonical layout (references/architecture.md). Keep tiny scale vectors in fp32;
   - set `self.io = IOSpec(in_channels, out_channels, scale, pad_multiple)`.
4. `forward(x)`: `x` is `[N, 3, H, W]` fp32 in [0, 1] (zero-padded to `pad_multiple` by
   the worker). Return `[N, 3, H*scale, W*scale]` fp32; the worker crops, clamps,
   rounds, and computes metrics. Inside, use NHWC activations in `self.dtype` and call
   heavy work through `self.ops.<op>`. Small per-image vector math (e.g. a tiny MLP on a
   pooled vector) can use plain torch ops.
5. Missing op? Add it to `ops/torch_ref.py` first, with a docstring that defines its
   semantics. That makes the model run everywhere. Only then add a fast version
   (new-device.md, "adding a kernel").
6. Parity: write the equivalent of `tools/_common.build_reference` for your training
   code (the basicsr version works for any basicsr arch), then run the parity loop with
   `--backends torch --dtype fp32` (expect ~1e-6 relative) and with the fast backends
   (fp16: ~1e-3 relative, PSNR delta < 0.01 dB).
7. If the network family has structural variants, add them to
   `tools/check_zoo_variants.py` (random weights, every option exercised).

## Built-in models (worked examples)

| model | module | notes |
|---|---|---|
| NAFNet / NAFNetLocal | `models/nafnet/` | arch from basicsr `network_g`; LayerNorm folding, TLC local mean, SCA fused into conv3, fp32 residual levels |
| Restormer | `models/restormer/` | MDTA attention as a Gram reduction + one per-image GEMM folded into project_out |
| RRDBNet (ESRGAN / Real-ESRGAN) | `models/rrdbnet/` | dense blocks inside one NHWC buffer (convs read / write channel slices), fused 0.2 scale, LeakyReLU, nearest x2 upsampling |
| FFTformer | `models/fftformer/` | 8x8-patch FFT filter / correlation as dense DFT products on tensor cores, fused behind the 1x1 conv; FSAS LayerNorm + `v *` in the project_out prologue |
| EVSSM | `models/evssm/` | per-block transpose / flip and overlapping crops folded into gather / merge kernels; selective scan with one lane per channel and segment-parallel carries |

Lessons from porting them (each one cost real time):
- Pad odd channel widths to a multiple of 16 at fold time with zero channels. Restormer's
  `ffn_expansion_factor = 2.66` gives 255 / 510 / 1021 hidden channels; misaligned gate
  halves made the depthwise kernel 1.6x slower and the GEMMs 2x slower.
- Measure the residual-stream range before picking fp16: Restormer peaks at ~35 on GoPro,
  so fp16 is fine (406 -> 245 ms); NAFNet-width64 needs fp32 at the deepest level.
- Dense concatenations: allocate one buffer and pass channel slices (`out=` / strided
  inputs) instead of `torch.cat`.
- Datasets with many image sizes (SR test sets): one graph per size costs a capture at
  start-up; prefer `runtime.graphs: false`, or raise `runtime.max_graphs` above the number of
  sizes, otherwise the LRU thrashes.
- Check every intermediate's range, not only the residual stream: FFTformer's FFT
  correlation reaches ~1e7 and EVSSM's scan output ~7e6 on real images (both are followed by
  a LayerNorm). In fp16 they overflow to inf; those two ops return fp32.
- A real filter applied to an rfft2 half spectrum is not what it seems: irfft2 only sees the
  Hermitian part of the v = 0 and v = p/2 columns, so the filter acts like its symmetrized
  version there. That makes the 8x8 filter a 64 x 64 real map (`kernels/freq.py`), half the
  work of treating the 40 complex bins as 80 independent reals.
- Elementwise torch ops on a transposed view keep the transposed strides: an op result fed
  to a Triton kernel that assumes contiguous NHWC must be `.contiguous()` (EVSSM crop merge).
- In Triton, a per-lane recurrence needs one layout for every tensor in the loop: 2-D
  [state, channel] tiles made Triton insert shared-memory layout conversions every step
  (2x slower); 1-D per-channel vectors with named state registers removed them
  (`kernels/ssm.py`).
- Random-weight variant tests (`tools/check_zoo_variants.py`) found two bugs the real
  checkpoints did not exercise: a dropped `project_out` bias (bias=True variant) and the
  wrong channel order for PixelUnshuffle(4) (x1 RRDBNet). Add variants for every option.

## IO limits to know

- The IO path is PNG -> RGB8 -> 3 channels in / out. Grayscale or multi-frame models
  need changes in `png.rs` / `io.rs` and the worker's `_pipeline`.
- `IOSpec.scale > 1` (super-resolution) is supported end to end: slot sizes, GT size
  check, and output size. It has only been verified with the synthetic
  `example_bilinear_x2` model in `template.py` (outputs within 1 grey level of a CPU
  bilinear reference). Validate a real SR model the same way.
