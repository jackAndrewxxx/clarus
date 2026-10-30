# clarus

[English](README.md) | **简体中文**

从零写的图像恢复模型推理 / 评估引擎，结构参照 vLLM / SGLang。Rust 进程负责调度、IO 和报告。模型跑在嵌入的 Python worker 里：Python 只做胶水，算子按后端分派，NVIDIA GPU 上走自写的 Triton 融合 kernel，其他设备走 PyTorch 参考实现。用哪个模型、哪种设备、哪些算子后端、怎么组 batch，都由一个 YAML 文件指定。

已内置并可直接加载官方权重的模型：NAFNet（NAFNet / NAFNetLocal 系列只要给出 basicsr 的 `network_g` 就能跑，不用改代码）、Restormer、Real-ESRGAN（RRDBNet）、FFTformer 和 EVSSM。新网络以 Python 类的形式接入；新硬件先用 PyTorch 后端跑通，再逐个算子补快速 kernel。

- 单张 A30 上，**端到端比各模型官方测试脚本快 7.2–8.4 倍**，PSNR / SSIM 一致
- **Rust 核心**：YAML 配置、PNG IO 线程池（`png` + `rayon`）、按 shape 分桶的调度器、PyO3 宿主
- **按算子分派后端**：Triton 不支持的调用自动交给 PyTorch，可以逐个算子混用，每次运行都会列出回退的算子
- **定长模式**：固定 shape 和 batch 大小，启动时捕获全部设备 graph，组批逐位可复现
- 每一层都有对应的**验证工具**，另有一个 agent skill 说明如何适配

## 目录

- [结果](#结果)
- [快速上手](#快速上手)
- [工作原理](#工作原理)
- [目录结构](#目录结构)
- [配置](#配置)
- [按需求适配](#按需求适配)
- [验证](#验证)
- [常见问题](#常见问题)
- [已知限制](#已知限制)
- [给 AI agent](#给-ai-agent)

## 结果

单张 NVIDIA A30。每个基线都是该模型官方的推理 / 测试脚本，原样运行；时间从进程启动算到最后一个结果。

| 模型（任务） | 数据 | 官方脚本 | 官方 | clarus | 加速 | 质量，官方 → clarus |
|---|---|---|---|---|---|---|
| NAFNet-GoPro-width64（去模糊） | GoPro test，1111 张 720p | `basicsr/test.py`（fp32） | 517.9 s | 65.5 s（`eval`） | **7.9×** | PSNR / SSIM 33.7103 / 0.9668 → 33.7103 / 0.9668 |
| Restormer（运动去模糊） | GoPro test，1111 张 720p | `Motion_Deblurring/test.py`（fp32，保存 PNG） | 1802.1 s | 227.1 s（`infer --save`） | **7.9×** | PSNR 32.9177 → 32.9177；0.9% 的像素不同，差值 ≤ 2 |
| FFTformer（去模糊） | GoPro test，1111 张 720p | `test.py`（fp32，保存 PNG） | 2743.9 s | 381.4 s（`eval --save`） | **7.2×** | PSNR / SSIM 34.2115 / 0.9694 → 34.2113 / 0.9694；1.2% 的像素不同，差值 ≤ 5 |
| EVSSM（去模糊，Mamba） | GoPro test，1111 张 720p | `test.py`（fp32，mamba_ssm CUDA 扫描，保存 PNG） | 3972 s | 473.0 s（`eval --save`） | **8.4×** | PSNR / SSIM 34.5106 / 0.9713 → 34.5103 / 0.9713；2.4% 的像素不同，差值 ≤ 4 |
| Real-ESRGAN x4plus（×4 超分） | Urban100 LR×4，100 张，37 种尺寸 | `inference_realesrgan.py`（默认 fp16） | 20.7 s | 12.0 s（`infer --save`） | **1.7×** | 对 HR 的 PSNR 21.0561 → 21.0568；比官方 fp16 更接近 fp32 参考（5.2% vs 7.2% 的像素不同） |

单次前向（batch 1）：NAFNet 720p 251.0 → 53.3 ms；Restormer 720p 1482 → 199 ms；RRDBNet 输入 256×384 时 250（官方 fp32）/ 167（官方 fp16）→ 84 ms；FFTformer 720p 2030 → 330 ms；EVSSM 720p 3149 → 407 ms。FFTformer / EVSSM 的 PSNR / SSIM 都是用 `tools/score_outputs.py` 在保存的 PNG 上算的（`clarus eval` 自己报 34.2113 / 0.9692 和 34.5103 / 0.9711，SSIM 的差来自 basicsr 在 GPU 上用 TF32 算 SSIM）。Real-ESRGAN 加速最少：它的 3×3 卷积本来就是 cuDNN fp16 的强项，clarus 的收益主要来自 NHWC 布局，以及把密集块的拼接、LeakyReLU 和上采样融合进卷积。

每一行都可以用 `clarus eval --config <配置> --report r.json` 复现（见[验证](#验证)）；运行日志不进 git。

## 快速上手

### 环境要求

| | 已验证的版本 | 用途 |
|---|---|---|
| Python | 3.11（需要共享库 `libpython`） | 模型 worker |
| PyTorch | 2.6.0 + CUDA 12.4 | 必需 |
| Triton | 3.2.0 | CUDA 快速后端（其他设备可不装） |
| PyYAML | 6.0 | 读配置 |
| Rust | 1.92（edition 2021） | 编译引擎 |
| numpy、opencv-python、basicsr 仓库 | | 只有验证工具需要 |

引擎本身不导入 basicsr，网络结构来自配置：`model.network_g`，或者用 `model.arch_from` 指向训练 yml。只有 `tools/` 里的对照工具会构建 basicsr 原网络：把 `CLARUS_BASICSR` 指向包含该网络结构的 basicsr 仓库（NAFNet 配置对应 megvii-research/NAFNet 的克隆），`CLARUS_LQ` / `CLARUS_GT` 指向评估数据（例如 GoPro 测试集的 blur / sharp 目录）。

### 编译

```bash
git clone git@github.com:<you>/clarus.git && cd clarus/engine
# PyO3 链接 PYO3_PYTHON 指定的解释器（默认值在 engine/.cargo/config.toml）
PYO3_PYTHON=/path/to/venv/bin/python cargo build --release      # crate 缓存好之后可加 --offline
```

解释器的 home、site-packages 和 `python/` 包路径会写进二进制。运行时可用 `CLARUS_PYTHONHOME`、`CLARUS_SITE`、`CLARUS_PYTHONPATH` 覆盖。

### 运行

```bash
B=./target/release/clarus

# 评估：PSNR / SSIM 的定义与 basicsr 的 calculate_psnr / calculate_ssim 一致
CUDA_VISIBLE_DEVICES=0 $B eval --lq data/blur --gt data/sharp --report report.json [--save out/]

# 只推理，输出恢复后的 PNG
$B infer --input data/blur --output out/

# 换模型 / 换设备 / 定长运行，只改配置或命令行参数
$B eval --config ../configs/restormer_motion_deblurring.yaml --lq ... --gt ...
$B eval --device cpu --dtype fp32 --lq ... --gt ...
$B infer --input ... --output ... --shapes 720x1280 --batch-sizes 2
$B eval ... --set runtime.op_overrides.selective_scan=torch
$B --help
```

输入是 `--lq` / `--input` 目录里的 `.png` 文件，GT 在 `--gt` 目录里按文件名对应。任何 PNG 都可以：1–16 bit、灰度、灰度+alpha、RGB、RGBA、调色板、Adam7 隔行。图像按 `cv2.IMREAD_COLOR` 的方式解码成 RGB8，尺寸不限，内部会用 0 补到模型要求的倍数（NAFNet 是 16）。

### 输出怎么看

```
[clarus] eval nafnet on NVIDIA A30 | 1111 images, shapes [[720, 1280]], batch sizes [1, 2], graphs true | load 2.3s, warm-up 1.0s
==== clarus eval (nafnet) ====
images            1111 (0 failed)                    # failed = 解码失败 / shape 被拒 / GT 尺寸不对
PSNR / SSIM       33.7103 / 0.9668   (psnr 33.710309, ssim 0.966790)
pipeline wall     64.04 s  -> 17.35 img/s (57.6 ms/img)   # 第一次解码到最后一个结果
end-to-end wall   67.43 s (scan 0.03, load 2.32, warm-up 1.03)   # warm-up = 捕获 graph（首次运行还包括调优）
batches           556 (Static, mean size 2.00, 0 padded, mean device latency 344.9 ms)   # 延迟包含排队时间
decode / encode   24.7 / 0.0 ms per image (16 IO threads)
op fallbacks      ...                                # 有算子跑在非首选后端时才打印
report            report.json
```

`--report` 输出的 JSON 包括：`settings`（实际生效的配置）、`model`（worker 构建出的网络结构）、`psnr` / `ssim`、`timing`、`stats`（batch 数、补齐数、解码 / 编码耗时）、`worker`（显存峰值、捕获的 graph、各算子由哪个后端执行），以及 `images`（逐图的 `name`、`h`、`w`、`psnr`、`ssim`、`nonfinite`、`decode_ms`、`batch_ms`、`batch`、`error`）。

在新 GPU 或新图像尺寸上第一次运行会调优 kernel，需要几十秒到几分钟。结果缓存在 `cache/tune_<gpu>_triton<版本>.json`，之后预热只要几秒。

## 工作原理

```
clarus (Rust)
 ├─ config.rs    YAML + --set 覆盖；model/runtime 段交给 worker，scheduler/io 段 Rust 自己用
 ├─ main.rs      扫描 PNG 头得到 shape 集合，准入、预热、汇总输出 + JSON 报告
 ├─ engine.rs    调度器：按 shape 分桶，只用配置的 batch 大小（不足就补齐），static / dynamic 组批
 ├─ io.rs        rayon 线程池：PNG 直接解码进 host slot，编码输出
 └─ pyworker.rs  PyO3 宿主；每次调用传一个 JSON 字符串（info / warmup / submit / poll / wait / shutdown）
        │
        ▼
clarus (Python)
 ├─ worker.py    host slot、每个 (batch, H, W) 一张捕获好的设备 graph、指标、完成事件
 ├─ runtime.py   配置 -> (model, ops, device)
 ├─ models/      注册表、RestorationModel / IOSpec、nafnet/ restormer/ rrdbnet/ fftformer/ evssm/、template.py
 ├─ ops/         分派器；torch_ref.py（参考实现，全部算子）、triton_cuda.py（融合 kernel）
 ├─ device/      CudaDevice | AcceleratorDevice（任意 torch.<type>，如 torch_npu）| CpuDevice
 ├─ kernels/     Triton kernel + 持久化调优
 └─ metrics.py   在设备上算 PSNR / SSIM（Triton 或 PyTorch 实现）
```

**一个 batch 的完整流程：**
1. Rust 的 IO 线程把 PNG 解码进 host slot。slot 是 worker 分配的页锁定内存，像素数据不会变成 Python 对象。
2. 调度器把同 shape、已解码好的图按配置的某个 batch 大小凑成一批，调用 `submit`。
3. worker 把所有步骤排进同一个设备 stream：拷到设备、`u8 → fp32 / 255` 并补 0、网络前向、clamp / round 成 u8、算 PSNR / SSIM、拷回。对每个 (batch, H, W)，这一整串操作捕获成一张设备 graph，所以每个 batch 在 Python 侧只是一次 graph 回放。
4. `poll` / `wait` 拿回指标。slot 随后释放，或交给 IO 线程编码输出 PNG。

slot 是背压单位：解码最多领先 `max(batch_sizes) × (inflight + 2)` 张图。

**两种组批方式：**
- `static`（默认）：运行前就定好每个 batch，每种 shape 内按文件名顺序取图，并按 batch 依次发起解码。因此每次运行的输出逐位一致。
- `dynamic`：用已经解码完成的图组批。凑满的 batch 立即发出；没凑满的只在设备上排队的 batch 少于 2 个时才发出。

**加速从哪里来**（细节见源码）：
1. 推理专用重写 + 权重折叠：LayerNorm 的仿射并入后面的 1×1 conv；“拆分 / 拼接 / 相加”的组合改成一个拆分了权重的 GEMM；很小的残差缩放系数（NAFNet 的 beta / gamma）留在 fp32 epilogue 里。
2. 1×1 conv、2×2 stride-2 conv 和 PixelShuffle 共用一个 NHWC 融合 GEMM kernel。prologue 可以做 LayerNorm / RMS norm、TLC 局部均值 gather、LayerNorm × 门控；epilogue 可以做 bias、SimpleGate、GELU、缩放和残差。
3. NAFNet 的 TLC 局部均值改成在有效区域上滑动的前缀游标，原实现是对整张图做 fp32 双重 cumsum 再 pad。
4. Restormer 的通道注意力是一次 split-K Gram 归约，加一个折进 project_out 的逐图 GEMM。FFTformer / EVSSM 的 8×8 分块 FFT 滤波改成 tensor core 上的 64×64 实数 DFT 乘法，并和前面的 1×1 conv 融合。
5. EVSSM 的选择性扫描每个线程负责一个通道，序列分段并行（在这些尺寸上比官方 mamba_ssm kernel 快）；每个 block 的转置 / 翻转和重叠切块都并进了切块 / 合并 kernel 的寻址。
6. 深度卷积用行缓冲并融合门控；拼接改成同一块缓冲区的通道切片（RRDBNet 的密集块）。
7. 混合精度：激活用 fp16；超出 fp16 范围的地方用 fp32（NAFNet 最深层的残差流、后面接 LayerNorm 的 FFT 相关和扫描输出）。
8. 设备 graph + 按 GPU 型号持久化的调优缓存。
9. 指标在设备上算：PSNR 用整数精确累加 SSE；SSIM 把 11×11×11 高斯拆成可分离卷积，并用中心化的矩计算。

## 目录结构

```
clarus/
├── README.md / README.zh-CN.md
├── configs/
│   ├── nafnet_gopro_width64.yaml     默认配置（官方 NAFNet GoPro 权重），每个字段都有注释
│   ├── restormer_motion_deblurring.yaml  官方 Restormer 运动去模糊权重
│   ├── realesrgan_x4plus.yaml        官方 Real-ESRGAN x4plus（RRDBNet，×4 超分）
│   ├── fftformer_gopro.yaml          官方 FFTformer GoPro 权重
│   └── evssm_gopro.yaml              官方 EVSSM GoPro 权重（Mamba / 选择性扫描）
├── checkpoints/                      权重放这里或软链接到这里（不进 git）
├── engine/                           Rust crate `clarus`
│   ├── .cargo/config.toml            PYO3_PYTHON 默认值
│   ├── build.rs                      libpython 的 rpath，写入默认路径
│   └── src/{main,config,engine,io,png,pyworker}.rs
├── python/clarus/                   模型 worker 包
│   ├── worker.py  runtime.py  metrics.py
│   ├── device/__init__.py
│   ├── ops/{__init__,torch_ref,triton_cuda}.py
│   ├── models/{__init__,_fold,template}.py、models/{nafnet,restormer,rrdbnet,fftformer,evssm}/
│   └── kernels/{gemm,conv3x3,spatial,attention,freq,ssm,tuning}.py
├── tools/                            验证、基准、调试工具（都接受 --config）
├── .claude/skills/clarus-adapt/      agent skill（SKILL.md + references/），给 Claude Code 用
├── .agents/skills/clarus-adapt/      同一个 skill，给读取 .agents/skills 的 agent 用
└── cache/                            kernel 调优结果（按 GPU 型号 + Triton 版本分文件）
```

两个 skill 目录里是完全相同的副本；修改 skill 时两处都要改（或者改完一处后复制到另一处）。

## 配置

一个 YAML 文件分四段，相对路径按配置文件所在目录解析。任何字段都可以用 `--set 段.键=值` 覆盖（值按 YAML 解析）。

| 字段 | 默认值 | 含义 |
|---|---|---|
| `model.name` | — | 注册的模型：`nafnet`、`restormer`、`rrdbnet`（别名 `realesrgan`、`esrgan`）、`fftformer`、`evssm`；模板插件里的 `example_unsharp` / `example_bilinear_x2` |
| `model.plugins` | `[]` | 构建模型前要导入的 Python 模块（包外模型） |
| `model.checkpoint` | — | 权重文件（接受 basicsr 的 `params` / `params_ema` 格式） |
| `model.network_g` | — | 描述网络的 basicsr `network_g`（从训练 yml 里复制） |
| `model.arch_from` | — | 也可以指向一个 basicsr yml，从中读取 `network_g`（请用**训练** yml）；这时 `network_g` 里的键覆盖它 |
| `model.options.*` | | 模型专用选项，见各模型模块的 docstring（如 NAFNet 的 `residual_fp32_from_level`：从第几层金字塔开始残差流用 fp32，`0` = 全部，即 `--safe`） |
| `runtime.device` | `cuda:0` | `cuda:N`、`cpu`，或任意 torch 设备，如 `npu:0` |
| `runtime.device_imports` | `[]` | 向 torch 注册设备的模块，如 `[torch_npu]` |
| `runtime.dtype` | `fp16` | 激活精度 `fp16` / `bf16` / `fp32`（fp32 时只能用 PyTorch 后端） |
| `runtime.backends` | `[triton, torch]` | 算子分派顺序；前一个后端处理不了的调用交给下一个 |
| `runtime.op_overrides` | `{}` | 把某个算子钉在指定后端，如 `{selective_scan: torch}` |
| `runtime.graphs` | `auto` | 是否捕获设备 graph：`auto` / `true` / `false` |
| `runtime.metrics_backend` | `auto` | CUDA 上用 `triton`，其他设备用 `torch` |
| `scheduler.batch_sizes` | `[1, 2]` | 捕获 graph 的 batch 大小；不足的 batch 补齐到下一档 |
| `scheduler.batching` | `static` | `static`（可复现）/ `dynamic` |
| `scheduler.shapes` | `auto` | `auto` = 输入集里出现的 shape；或固定列表 `[[h, w], ...]`，启动时全部捕获 |
| `scheduler.unknown_shape` | `reject` | 固定列表之外的 shape：`reject` 拒绝 / `lazy` 首次遇到时再准备 |
| `scheduler.inflight` | `3` | 设备上排队的 batch 数 |
| `io.threads` | `16` | PNG 解码 / 编码线程数 |
| `io.png_level` | `1` | 输出压缩：0 不压缩、1 快速、2–6 均衡、7 以上高压缩 |

命令行快捷参数（见 `clarus --help`）：`--config`、`--set`、`--ckpt`、`--device`（写 `1` 表示保留设备类型只换编号）、`--dtype`、`--backends a,b`、`--no-graph`、`--safe`、`--batch-sizes 1,2`、`--max-batch N`、`--batching`、`--shapes HxW,...`、`--inflight`、`--io-threads`、`--png-level`、`--limit N`、`--verbose`。

| 环境变量 | 作用 |
|---|---|
| `PYO3_PYTHON`（编译时） | 引擎使用哪个解释器的 libpython / site-packages |
| `CLARUS_CONFIG` | 默认配置文件 |
| `CLARUS_PYTHONHOME`、`CLARUS_SITE`、`CLARUS_PYTHONPATH` | 覆盖写进二进制的解释器 home、site-packages 和 `clarus` 路径 |
| `CLARUS_TUNE_DIR`、`CLARUS_TUNE_CACHE` | 调优缓存目录 / 具体文件 |
| `CUDA_VISIBLE_DEVICES` | 照常使用 |

## 按需求适配

先选改动最小的方案。完整步骤见 `.claude/skills/clarus-adapt/`。

**1. 内置系列的另一个 checkpoint：只改配置。**
例如宽度、深度、层数任意的 NAFNet / NAFNetLocal，TLC 可以用任意 `train_size`，也可以用全局池化；或者 `network_g` 不同的 Restormer / RRDBNet / FFTformer / EVSSM checkpoint。
```bash
cp configs/nafnet_gopro_width64.yaml configs/mine.yaml
# 修改 model.checkpoint 和 model.network_g（从训练这个 checkpoint 用的 yml 里复制）
python tools/parity.py --config configs/mine.yaml --images 12 --odd       # 与原网络对比精度
engine/target/release/clarus eval --config configs/mine.yaml --lq ... --gt ...
```
运行时不支持的选项会在加载时报错，并给出选项名。配置和 checkpoint 对不上时，权重折叠阶段会报 "checkpoint has no tensor ..." 或 "tensors not consumed ..."。

**2. 自己的数据集。** LQ 和 GT 的 PNG 分放两个目录，文件名一一对应。GT 尺寸必须是 LQ 的 `scale` 倍。图像尺寸可以混杂，每种 shape 各有自己的 graph。快速检查可以加 `--limit N`。

**3. 定长部署。** 用 `--shapes 720x1280,1080x1920 --batch-sizes 4`（或 `scheduler:` 段的字段），启动时就捕获全部 graph。其他 shape 的输入会被拒绝；设置 `unknown_shape: lazy` 则在首次遇到时再准备。不足的 batch 会补齐，所以运行中只会用到配置好的那几张 graph。

**4. 别的网络。** 复制 `python/clarus/models/template.py`。它本身就是一个能运行的插件：`--set model.name=example_unsharp --set model.plugins=[clarus.models.template]`。需要实现两部分：
- `__init__`：解析结构，加载权重并折叠成规范布局，设置 `IOSpec`。
- 可被 graph 捕获的 `forward`：所有重计算都通过 `self.ops.<算子>` 调用。

用 `@register_model("名字")` 注册，再通过 `model.plugins` 加载。新算子先写进 `ops/torch_ref.py`，保证在任何设备上都能跑，之后再补快速 kernel。

**5. 其他硬件。** 先用 PyTorch 后端跑通：`--set runtime.device=npu:0 --set runtime.device_imports=[torch_npu] --backends torch --no-graph`，用 `tools/parity.py --device npu:0 --backends torch` 检查精度。之后，设备支持 graph 就打开；再写一个实现热点算子的后端类（继承 `ops/__init__.Backend`）。没实现的算子会继续走 PyTorch，`op fallbacks` 一行会列出它们。CPU 已用这条路径验证过；NPU 在这台机器上没测过。

**6. 加速某个算子。** 按该算子在 `torch_ref.py` 里的语义写 kernel，处理不了的参数组合抛 `Unsupported`。然后在 `tools/test_kernels.py` 里加用例，用 `tools/kernel_breakdown.py` 测耗时。

**7. 数值问题。**
- `--safe`：全部残差流用 fp32。分布外的输入需要它，输出出现非有限值时引擎会告警。
- `--dtype bf16`：数值范围更大，但精度更低。
- `--dtype fp32 --backends torch`：作为参考跑一遍。

**8. 在 Python 里调用。**
```python
import sys; sys.path.insert(0, "python")
from clarus.runtime import build_from_config
model, ops, dev = build_from_config("configs/nafnet_gopro_width64.yaml")
y = model.forward(x)        # x: [N, 3, H, W]，float32，范围 [0, 1]，放在 dev.torch_device 上
```
`clarus.worker.ModelWorker` 在此基础上加了 slot、graph 和指标；`tools/worker_smoke.py` 演示了不经过 Rust 直接驱动它。

## 验证

| 检查项 | 命令 | 通过标准 |
|---|---|---|
| kernel vs fp32 PyTorch | `python tools/test_kernels.py` | `ALL OK` |
| 模型 vs 原网络 | `python tools/parity.py --config CFG [--backends torch --dtype fp32]` | 参考后端相对误差约 1e-6；fp16 下 dPSNR 最大 < 0.01 dB |
| 结构变体 | `python tools/check_zoo_variants.py`（NAFNet、Restormer、RRDBNet、FFTformer、EVSSM；随机权重） | `ALL OK` |
| 官方脚本的输出 | `python tools/score_outputs.py OUT GT --other CLARUS_OUT` | 与 `clarus eval` 同一套指标，并比较像素是否一致 |
| 指标 | `python tools/test_metrics.py --no-tf32 --fp64` | PSNR 精确；SSIM 与 fp64 相差约 1e-7 |
| PNG 解码 vs cv2 | `clarus png-hash DIR` 对比 `python tools/png_hash_cv2.py DIR` | 完全一致 |
| 引擎端到端 | `python tools/check_engine.py --device N` | `PASS` |
| 与原流程逐图对比 | `tools/reference_eval.py` + `tools/compare_reports.py` | dPSNR p99 < 0.002 dB |
| Rust 单测 | `cd engine && cargo test --release --offline` | 全部通过 |

完整命令、标准和参考数字见 `.claude/skills/clarus-adapt/references/validation.md`。

## 常见问题

| 报错 / 现象 | 原因和处理 |
|---|---|
| `unknown model.name 'x'; registered: [...]` | 名字写错，或插件模块没写进 `model.plugins` |
| `checkpoint has no tensor ...` / `tensors not consumed ...` | `network_g` / `arch_from` 描述的不是这个 checkpoint；从训练它时的 yml 里取 |
| 加载时报 `... not implemented` / `... not supported` | 运行时还没实现这个 `network_g` 选项或网络类型；见"别的网络" |
| `op backend 'triton' unavailable: ...` | 在非 CUDA 设备或 fp32 下属于正常情况，由 PyTorch 后端接手 |
| `op fallbacks  x->torch` | 这个算子在这组参数下没有 kernel，跑在 PyTorch 后端上。结果正确，只是慢一些 |
| `untuned kernel shape during graph capture` | 某个 kernel 第一次被用到时已经在捕获阶段；`forward` 的 shape 要确定，每次调用算子的方式要一致 |
| `no valid config for ...` | 这个问题规模下没有能编译的 tile 配置；检查对应 kernel 的候选列表 |
| 预热特别慢 | 在这块 GPU / 这个尺寸上第一次运行，正在调优；之后会用 `cache/` |
| `shape HxW (HxW) not in scheduler.shapes` | 处于定长模式；把这个 shape 加进列表，或设置 `scheduler.unknown_shape: lazy` |
| `missing GT for NAME` / `GT size ... != expected` | GT 目录缺文件，或尺寸不是输入的 `scale` 倍 |
| `WARNING: N non-finite output values` | 特殊输入导致 fp16 溢出；用 `--safe` 重跑 |
| 一启动就 CUDA OOM | GPU 被其他进程占用；用 `nvidia-smi` 找一块空闲卡 |
| 编译找不到 libpython / 运行时 torch 版本不对 | 编译时设置 `PYO3_PYTHON`，或运行时设置 `CLARUS_PYTHONHOME` / `CLARUS_SITE` |

## 已知限制

- 一个进程只用一个设备。IO 路径是 PNG → RGB8，输入输出都是 3 通道。超分（`IOSpec.scale > 1`）只用合成模型 `example_bilinear_x2` 验证过。
- Triton 后端只在 NVIDIA A30 上调优和验证过。其他设备走 PyTorch 后端，按普通方式逐步执行：CPU 已验证，NPU 未实测。
- `dynamic` 组批下，同一张图在不同运行中可能被分进大小不同的 batch，约 0.5% 的像素会差 1 个灰度级。默认的 `static` 没有这个问题。
- 混合精度的选择（哪些层的残差用 fp32、哪些算子输出 fp32、2⁻⁸ 的操作数缩放）是按这几个 checkpoint 实测的激活范围定的，换模型要重新测。

## 给 AI agent

skill `clarus-adapt` 以两份相同的副本放在 `.claude/skills/clarus-adapt` 和 `.agents/skills/clarus-adapt`，Claude Code 和读取 `.agents/skills` 的其他 agent 不用额外配置就能找到它。它写明了几条基本规则（规范权重布局、graph 捕获安全、fp16 数值范围、GPU 可能是共享的），并链接到架构约定、分步做法和验证清单。

## 许可证

MIT 许可证，见 [LICENSE](LICENSE)。引擎重新实现的网络（NAFNet、Restormer、Real-ESRGAN、FFTformer、EVSSM）来自各自作者的仓库，各有其许可证；本仓库不分发任何权重。
