Put model weights here (or symlink them); this directory is not tracked by git.

The example configs expect the official weights of each model:

| file | config | source |
|---|---|---|
| `NAFNet-GoPro-width64.pth` | `configs/nafnet_gopro_width64.yaml` | megvii-research/NAFNet (mirrors on Hugging Face, e.g. `nyanko7/nafnet-models`) |
| `restormer_motion_deblurring.pth` | `configs/restormer_motion_deblurring.yaml` | https://github.com/swz30/Restormer/releases/download/v1.0/motion_deblurring.pth |
| `RealESRGAN_x4plus.pth` | `configs/realesrgan_x4plus.yaml` | https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth |
| `fftformer_GoPro.pth` | `configs/fftformer_gopro.yaml` | `pretrain_model/fftformer_GoPro.pth` in https://github.com/kkkls/FFTformer |
| `EVSSM_GoPro.pth` | `configs/evssm_gopro.yaml` | `net_g_GoPro.pth` from the Google Drive folder linked in https://github.com/kkkls/EVSSM |

Any basicsr checkpoint (`params` / `params_ema`) of a built-in family works; set
`model.checkpoint` and `model.network_g` in your config.
