# Validation: commands and pass thresholds

All commands run from the repository root with the Python that has torch / triton
(`PY=/path/to/venv/bin/python`). Reference tools need `CLARUS_BASICSR` (basicsr checkout with
the architecture, e.g. a clone of megvii-research/NAFNet for the NAFNet configs) and `CLARUS_LQ` / `CLARUS_GT` (evaluation pair). Choose an idle GPU with
`CUDA_VISIBLE_DEVICES=N` (on shared machines, pick an idle GPU). Every tool accepts `--config CFG`.

| check | command | pass |
|---|---|---|
| kernels vs fp32 torch | `$PY tools/test_kernels.py` | `ALL OK` |
| model semantics (reference backend) | `$PY tools/parity.py --config CFG --backends torch --dtype fp32 --images 3 --odd` | max abs diff ~1e-6, dPSNR 0.0000 |
| fast path accuracy | `$PY tools/parity.py --config CFG --images 12 --odd` | max abs diff ≲ 2e-3, max dPSNR < 0.01 dB |
| mixed backends | `... parity.py --op-override dwconv=torch --op-override boxmean=torch` | same as fast path |
| structural variants (all built-in families) | `CLARUS_NAFNET_REPO=... CLARUS_RESTORMER_REPO=... CLARUS_BASICSR_PIP=... CLARUS_FFTFORMER_REPO=... CLARUS_EVSSM_REPO=... [CLARUS_MAMBA_PIP=...] $PY tools/check_zoo_variants.py` | `ALL OK`; fallbacks listed per variant |
| official script outputs vs GT / vs clarus | `$PY tools/score_outputs.py OFFICIAL_OUT GT --other CLARUS_OUT [--suffix _out]` | same PSNR / SSIM; pixel agreement reported |
| metrics vs basicsr / fp64 | `$PY tools/test_metrics.py --no-tf32 --fp64` | PSNR diff < 1e-12; SSIM ~1e-7 from fp64 |
| PNG decoder vs cv2 | `engine/target/release/clarus png-hash DIR > a; $PY tools/png_hash_cv2.py DIR > b; diff a b` | no diff |
| Rust unit tests | `cd engine && cargo test --release --offline` | all pass |
| engine end to end (mixed shapes, gray, RGBA, saved PNGs) | `$PY tools/check_engine.py --device N` | `PASS` |
| full eval | `engine/target/release/clarus eval --config CFG --lq L --gt G --report r.json` | mean PSNR / SSIM equal to the official outputs (or training-log validation) to 4 decimals |
| per-image vs original code | `$PY tools/reference_eval.py ref.json --config CFG` then `$PY tools/compare_reports.py r.json ref.json` | per-image dPSNR p99 < 0.002 dB |
| determinism | run the same eval twice with `scheduler.batching: static` | identical per-image metrics |
| new device, eager | `$PY tools/parity.py --device DEV --backends torch --images 1 --odd` + `clarus eval --device DEV --limit 20` | as the reference-backend row; metrics within 0.002 dB of CUDA |

Reference numbers on one A30 (GoPro test, 1111 images, 1280x720):

| config | PSNR / SSIM | throughput | original basicsr/test.py |
|---|---|---|---|
| `nafnet_gopro_width64.yaml` | 33.7103 / 0.9668 (official 33.7103 / 0.9668) | 57.6 ms/img, 67.4 s end to end | official repo test.py 517.9 s |
| `restormer_motion_deblurring.yaml` | 32.9177 / 0.9611 (official outputs 32.9177 / 0.9613) | 244.7 ms/img, 276.1 s infer+save | official test.py 1802.1 s |
| `fftformer_gopro.yaml` | 34.2113 / 0.9692 (official outputs, same tool: 34.2115 / 0.9694 vs ours 34.2113 / 0.9694) | 349 ms forward, 403.9 s eval+save | official test.py 2743.9 s |
| `evssm_gopro.yaml` | 34.5103 / 0.9711 (official outputs, same tool: 34.5106 / 0.9713 vs ours 34.5103 / 0.9713) | 433 ms forward, 506.3 s eval+save | official test.py 3972 s |
| `realesrgan_x4plus.yaml` (Urban100 x4) | PSNR vs HR 21.0568 (official fp16 21.0561) | 13.0 s infer+save, 100 images | official inference (fp16) 20.7 s |

Performance workflow: `tools/bench_model.py --config CFG --graph` gives forward latency,
and `tools/kernel_breakdown.py --config CFG` gives per-launch time with TF/s and GB/s
plus a per-kernel-type total. Compare against these numbers before and after a change.
The first run on a new image shape or GPU tunes kernels, which takes tens of seconds to
minutes. Later runs read `cache/tune_<gpu>_triton<ver>.json`.
