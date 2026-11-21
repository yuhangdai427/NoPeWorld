# World models

NoPeWorld does not redistribute the six world models. Each head reads the hidden states of one frozen world model, so use the
exact code commit and weight revision below. Get them from their authors (several are gated or non-commercial; check each
licence) and place them under one folder, `NOPE_GENERATORS` (default `./generators`). Symlinks work.

**A GPU with at least 80 GB of memory is required; a 96 GB GPU is recommended, and WorldPlay2 needs it (peak ≈ 86 GiB, ≈ 89 GiB with external depth), as does training on Echo-WM (peak ≈ 88 GiB).** Peak memory below is the highest GPU memory the run's processes used
(the world model, the head and, for external depth, the DA3 process), sampled every 5 s on one 96 GB GPU with default settings.

| World model | id | Backbone | Weights | Python, torch | Peak GPU memory, collision / prediction | external depth |
|---|---|---|---|---|---|---|
| [LingBot-World 2.0](#lingbot-world-20-lingbot_world_2) | `lingbot_world_2` | Wan2.2 | 86.1 GB | 3.10, 2.11.0 | 66.8 / 66.8 GiB | 76.1 GiB |
| [HY-WorldPlay](#hy-worldplay-hy_worldplay) | `hy_worldplay` | HunyuanVideo-1.5 | 56.7 GB | 3.10, 2.8.0 | 52.4 / 52.3 GiB | 71.5 GiB |
| [Echo-WM](#echo-wm-echo_wm) | `echo_wm` | LTX-2.3 | 72.2 GB | 3.11, 2.9.1 | 65.3 / 67.5 GiB | 72.8 GiB |
| [Evoke-Turbo](#evoke-turbo-evoke_turbo) | `evoke_turbo` | Helios | 85.5 GB | 3.10, 2.8.0 | 42.9 GiB / &ndash; | 53.9 GiB |
| [WorldPlay2](#worldplay2-worldplay2) | `worldplay2` | Wan2.2 | 81.7 GB | 3.10, 2.8.0 | 85.8 / 85.8 GiB | 88.6 GiB |
| [Hy-Nomas](#hy-nomas-hy_nomas) | `hy_nomas` | HunyuanVideo-1.5 | 70.0 GB + 16.6 GB text encoder shared with HY-WorldPlay | 3.12, 2.10.0 | 69.3 / 69.3 GiB | 73.7 GiB |

Seconds per route of every method: [INFERENCE.md](INFERENCE.md#time-per-route).

## Folder layout

```
$NOPE_GENERATORS/
  lingbot_world_2/  code/  weights/
  hy_worldplay/     HY-WorldPlay/  weights/
  echo_wm/          code/  weights/
  evoke_turbo/      code/  weights/
  worldplay2/       code/  weights/
  hy_nomas/         hy-nomas/  weights/
  da3/DA3-GIANT-1.1/                                       external_depth method; Task 2 with --recon da3
  vggt_omega/       code/  vggt_omega_1b_416_reproduce.pt  Task 2 evaluator
  sam2/sam2.1_hiera_large.pt                               Task 2 evaluator
```

The exact sub-paths each adapter reads are the `paths:` of `configs/world_models/<id>.yaml`. Each world model runs in its own
Python environment, built with its own install steps; install this package into each one (`pip install -e .`). Sizes count only
the files the adapter loads (1 GB = 10^9 bytes). `hf` is the Hugging Face CLI (`pip install "huggingface_hub[cli]"`).

```bash
export NOPE_GENERATORS=$PWD/generators; G=$NOPE_GENERATORS
```

## LingBot-World 2.0 (`lingbot_world_2`)

- **Code:** [Robbyant/lingbot-world-v2](https://github.com/Robbyant/lingbot-world-v2), commit `a3bd3c96416c77b5a5431a92c14da3c7b30d4cc7`. Only its `wan/` package is used.
- **Weights:** [`robbyant/lingbot-world-v2-14b-causal-fast`](https://huggingface.co/robbyant/lingbot-world-v2-14b-causal-fast), revision `5c33dd40b213598c418fd25bff30fdbd23fd38a7`, the whole snapshot (causal-fast DiT in `transformers/`, UMT5-XXL encoder and tokenizer, Wan2.1 VAE), 86.1 GB.
- **Environment:** Python 3.10, torch 2.11.0 (CUDA 12.8), diffusers 0.39.0, transformers 4.51.3, flash-attn 2.8.3.post1, accelerate 1.14.0, einops 0.8.2, imageio-ffmpeg 0.6.0, opencv-python 4.11.0.86, scipy 1.15.3, numpy 1.26.4. Upstream asks for `pip install -r requirements.txt` (torch ≥ 2.4, transformers ≤ 4.51.3, numpy < 2) and `pip install flash-attn --no-build-isolation`.
- **Licence:** CC BY-NC-SA 4.0 (code and weights).
- **Peak GPU memory:** 66.8 GiB in either mode, 76.1 GiB (external depth).

```bash
git clone https://github.com/Robbyant/lingbot-world-v2 $G/lingbot_world_2/code
git -C $G/lingbot_world_2/code checkout a3bd3c96416c77b5a5431a92c14da3c7b30d4cc7
hf download robbyant/lingbot-world-v2-14b-causal-fast --revision 5c33dd40b213598c418fd25bff30fdbd23fd38a7 \
  --local-dir $G/lingbot_world_2/weights
nopeworld-infer --generator lingbot_world_2 --method nopeworld_collision --task task2 --cases Cabin_Lake__d001 --out runs/lingbot_world_2
```

## HY-WorldPlay (`hy_worldplay`)

- **Code:** [Tencent-Hunyuan/HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay), commit `1588e1336e842b03b0a7860c654ebd7c46bb065e`.
- **Weights** (56.7 GB in all): the adapter reads `weights/ar_distilled_action_model/model.safetensors` and a `weights/runtime/` folder assembled from four repositories:
  - [`tencent/HY-WorldPlay`](https://huggingface.co/tencent/HY-WorldPlay) @ `f4c29235647707b571479a69b569e4166f9f5bf8`: `ar_distilled_action_model/model.safetensors` (34.2 GB; the full action DiT, no base DiT weights needed);
  - [`tencent/HunyuanVideo-1.5`](https://huggingface.co/tencent/HunyuanVideo-1.5) @ `9b49404b3f5df2a8f0b31df27a0c7ab872e7b038`: `vae/`, `scheduler/`, `transformer/480p_i2v/config.json` (5.0 GB);
  - [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) @ `cc594898137f460bfe9f0759e9844b3ce807cfb5` → `runtime/text_encoder/llm/` (16.6 GB; Hy-Nomas reads it from here too);
  - [`black-forest-labs/FLUX.1-Redux-dev`](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) @ `c95859fbf7703ca4d6824b4da4407d7cd0434f81` (gated: accept its licence, then `hf auth login`): `image_encoder/`, `feature_extractor/` → `runtime/vision_encoder/siglip/` (0.86 GB).

  ByT5 and Glyph-SDXL-v2, which the repository's `download_models.py` also fetches, are not loaded.
- **Environment:** Python 3.10, torch 2.8.0 (CUDA 12.8), diffusers 0.35.0, transformers 4.56.0, flash-attn 2.8.3.post1, accelerate 1.15.0, einops 0.8.0, imageio-ffmpeg 0.6.0, opencv-python-headless 4.11.0.86, scipy 1.15.3, numpy 1.26.4, peft 0.17.0, safetensors 0.4.5. Otherwise follow the repository's `requirements.txt`.
- **Licence:** Tencent HY-WorldPlay Community License (it does not apply in the European Union, the United Kingdom and South Korea); HunyuanVideo-1.5: Tencent Hunyuan Community License; Qwen2.5-VL: Apache-2.0; FLUX.1-Redux-dev: FLUX.1 [dev] Non-Commercial License.
- **Peak GPU memory:** 52.4 GiB (collision mode), 52.3 GiB (prediction mode), 71.5 GiB (external depth).

```bash
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay $G/hy_worldplay/HY-WorldPlay
git -C $G/hy_worldplay/HY-WorldPlay checkout 1588e1336e842b03b0a7860c654ebd7c46bb065e
W=$G/hy_worldplay/weights
hf download tencent/HY-WorldPlay ar_distilled_action_model/model.safetensors \
  --revision f4c29235647707b571479a69b569e4166f9f5bf8 --local-dir $W
hf download tencent/HunyuanVideo-1.5 --revision 9b49404b3f5df2a8f0b31df27a0c7ab872e7b038 \
  --include "vae/*" "scheduler/*" "transformer/480p_i2v/config.json" --local-dir $W/runtime
hf download Qwen/Qwen2.5-VL-7B-Instruct --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --local-dir $W/runtime/text_encoder/llm
hf download black-forest-labs/FLUX.1-Redux-dev --revision c95859fbf7703ca4d6824b4da4407d7cd0434f81 \
  --include "image_encoder/*" "feature_extractor/*" --local-dir $W/runtime/vision_encoder/siglip
nopeworld-infer --generator hy_worldplay --method nopeworld_collision --task task2 --cases Cabin_Lake__d001 --out runs/hy_worldplay
```

## Echo-WM (`echo_wm`)

- **Code:** [jd-opensource/JoyAI-Echo](https://github.com/jd-opensource/JoyAI-Echo), commit `a08f1274573d7cd6ec66719f204951f4227a5173`. `echo_wm/code` is the repository's `echo_wm/` subfolder (the one holding `ltx-core/`, `ltx-causal/`, `ltx-pipelines/`, `helpers/`).
- **Weights** (72.2 GB): [`Echo-Team/Echo-WM`](https://huggingface.co/Echo-Team/Echo-WM) @ `04efdf813648f698680c10024f22f351a0273163`, only `echo-wm-flash.safetensors` (47.8 GB; the Flash Preview causal model, not `echo-wm-base`); and [`google/gemma-3-12b-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) @ `68f7ee4fbd59087436ada77ed2d62f373fdd4482` → `weights/gemma-3/` (24.4 GB; gated, accept the Gemma terms first).
- **Environment:** Python 3.11, torch 2.9.1 (CUDA 12.8), transformers 4.57.6, xformers 0.0.33.post2 (no flash-attn, no diffusers), accelerate 1.15.0, einops 0.8.2, imageio-ffmpeg 0.6.0, opencv-python 5.0.0.93, scipy 1.17.1, numpy 2.4.6. This is the repository's own install (`echo_wm/README.md`: torch 2.9.1 from the cu128 index, then `pip install -r requirements.txt`).
- **GPUs with compute capability ≥ 10:** xFormers 0.0.33.post2 has no kernel for the masked attention of the causal model. Our runs send masked calls to PyTorch SDPA with this change in `echo_wm/ltx-core/src/ltx_core/model/transformer/attention.py` (the `DEFAULT` branch of `AttentionFunction`):

  ```diff
               # Default behavior: XFormers if installed else - PyTorch
  +            if mask is not None and q.is_cuda and torch.cuda.get_device_capability(q.device)[0] >= 10:
  +                return PytorchAttention()(q, k, v, heads, mask)
               return (
  ```
- **Licence:** code: LTX-2 Community License Agreement (research use, not for commercial use); Echo-WM weights: see the model card; Gemma 3: Gemma terms of use.
- **Peak GPU memory:** 65.3 GiB (collision mode), 67.5 GiB (prediction mode), 72.8 GiB (external depth).

```bash
git clone https://github.com/jd-opensource/JoyAI-Echo $G/echo_wm/JoyAI-Echo
git -C $G/echo_wm/JoyAI-Echo checkout a08f1274573d7cd6ec66719f204951f4227a5173
ln -s JoyAI-Echo/echo_wm $G/echo_wm/code
hf download Echo-Team/Echo-WM echo-wm-flash.safetensors --revision 04efdf813648f698680c10024f22f351a0273163 \
  --local-dir $G/echo_wm/weights
hf download google/gemma-3-12b-it-qat-q4_0-unquantized --revision 68f7ee4fbd59087436ada77ed2d62f373fdd4482 \
  --local-dir $G/echo_wm/weights/gemma-3
nopeworld-infer --generator echo_wm --method nopeworld_collision --task task2 --cases Cabin_Lake__d001 --out runs/echo_wm
```

## Evoke-Turbo (`evoke_turbo`)

- **Code:** [AlayaLab/Evoke](https://github.com/AlayaLab/Evoke), commit `1d271be45128bbda4263e9e3c7dfc8f6ba260492` (ViGeo's code is vendored in it).
- **Weights** (85.5 GB): [`AlayaLab/Evoke-Turbo`](https://huggingface.co/AlayaLab/Evoke-Turbo) @ `fe5afb975e08bff47f5b23087e8f6f90316aee4d` → `weights/evoke-turbo/transformer/` (57.3 GB); [`AlayaLab/Evoke`](https://huggingface.co/AlayaLab/Evoke) @ `7fa34ecef85754fde6f08996b1ece9d195dcd2f4`, only `evoke-base/` (text encoder, tokenizer, VAE, scheduler; 23.3 GB); [`pkqbajng/ViGeo`](https://huggingface.co/pkqbajng/ViGeo) @ `320bbfa59000df0732afe54511cd7669bb342386`, `vigeo.pt` → `weights/ViGeo1.1/` (5.0 GB). No stage checkpoints, teachers or DA3 weights are needed.
- **Environment:** Python 3.10, torch 2.8.0 (CUDA 12.8), diffusers 0.39.0 (PyPI), transformers 5.3.0, flash-attn 2.8.3.post1, accelerate 1.12.0, einops 0.8.1, imageio-ffmpeg 0.6.0, opencv-python-headless 4.11.0.86, scipy 1.15.3, numpy 1.26.4, peft 0.19.1, huggingface_hub 1.15.0, video-reader-rs 0.4.1. The repository's `requirements.txt` pins torch 2.4.0 (CUDA 12.4), a diffusers 0.39.0.dev0 fork and deepspeed; inference ran with the versions listed here instead.
- **Licence:** code: Apache-2.0; ViGeo: CC BY-NC 4.0 (per the Evoke README); Evoke and Evoke-Turbo weights: see their model cards.
- **Peak GPU memory:** 42.9 GiB (collision mode), 53.9 GiB (external depth). There is no prediction head: Evoke-Turbo keeps no memory between chunks.

```bash
git clone https://github.com/AlayaLab/Evoke $G/evoke_turbo/code
git -C $G/evoke_turbo/code checkout 1d271be45128bbda4263e9e3c7dfc8f6ba260492
W=$G/evoke_turbo/weights
hf download AlayaLab/Evoke-Turbo --revision fe5afb975e08bff47f5b23087e8f6f90316aee4d \
  --include "config.json" "diffusion_pytorch_model*" --local-dir $W/evoke-turbo/transformer
hf download AlayaLab/Evoke --revision 7fa34ecef85754fde6f08996b1ece9d195dcd2f4 --include "evoke-base/*" --local-dir $W
hf download pkqbajng/ViGeo vigeo.pt --revision 320bbfa59000df0732afe54511cd7669bb342386 --local-dir $W/ViGeo1.1
nopeworld-infer --generator evoke_turbo --method nopeworld_collision --task task2 --cases Cabin_Lake__d001 --out runs/evoke_turbo
```

## WorldPlay2 (`worldplay2`)

- **Code:** [WorldPlay2/WorldPlay2](https://github.com/WorldPlay2/WorldPlay2), commit `c5d83e32099116ff3a1437a8a05c764d579f704b`.
- **Weights** (81.7 GB): [`aejion/WorldPlay2-Fast`](https://huggingface.co/aejion/WorldPlay2-Fast) @ `1991d1cd9a3b116c2f6d9513a220c0ee5077ab79`, `{high,low}_noise_model/diffusion_pytorch_model.safetensors` (69.8 GB); and from [`Wan-AI/Wan2.2-I2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) @ `206a9ee1b7bfaaf8f7e4d81335650533490646a3` only the UMT5 encoder, its tokenizer, the VAE and the two expert `config.json` files (11.9 GB; the Wan2.2 DiT shards are not needed, the experts come whole from WorldPlay2-Fast).
- **Environment:** Python 3.10, torch 2.8.0 (CUDA 12.8), diffusers 0.37.0, transformers 4.51.3, flash-attn 2.8.3.post1, accelerate 1.15.0, einops 0.8.2, imageio-ffmpeg 0.6.0, numpy 1.26.4 (no opencv, no scipy before `pip install -e .`). This is within the repository's own `requirements.txt` plus `pip install flash-attn --no-build-isolation`.
- **Licence:** code: CC BY-NC 4.0 (`LICENSE.txt`); WorldPlay2-Fast and Wan2.2-I2V-A14B weights: see their model cards.
- **Peak GPU memory:** 85.8 GiB in either mode, 88.6 GiB (external depth).
- **Kernels:** WorldPlay2 runs its kernels eagerly by default, so a route gives the same frames on every run. `--compile` uses the repository's compiled kernels instead (about as fast on one GPU; runs then vary and decisions can change); see [INFERENCE.md](INFERENCE.md#worldplay2-compiled-kernels).

```bash
git clone https://github.com/WorldPlay2/WorldPlay2 $G/worldplay2/code
git -C $G/worldplay2/code checkout c5d83e32099116ff3a1437a8a05c764d579f704b
W=$G/worldplay2/weights
hf download aejion/WorldPlay2-Fast --revision 1991d1cd9a3b116c2f6d9513a220c0ee5077ab79 \
  --include "*.safetensors" --local-dir $W/WorldPlay2-Fast
hf download Wan-AI/Wan2.2-I2V-A14B --revision 206a9ee1b7bfaaf8f7e4d81335650533490646a3 \
  --include "models_t5_umt5-xxl-enc-bf16.pth" "Wan2.1_VAE.pth" "google/umt5-xxl/*" \
            "high_noise_model/config.json" "low_noise_model/config.json" --local-dir $W/Wan2.2-I2V-A14B
nopeworld-infer --generator worldplay2 --method nopeworld_collision --task task2 --cases Cabin_Lake__d001 --out runs/worldplay2
```

## Hy-Nomas (`hy_nomas`)

- **Code:** [tencent-hunyuan/hy-nomas](https://github.com/tencent-hunyuan/hy-nomas), commit `5ecf99c981432d40a3e6b8788432f007f7acc28b`.
- **Weights** (70.0 GB): [`tencent/Hy-Nomas`](https://huggingface.co/tencent/Hy-Nomas) @ `621df1570098aa1ebf916dac3ac72ae3cbe25caa`: `DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors` (the distilled DiT, run in bf16), `VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors` (decoder) and `VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt` (encoder for the start). The text encoder is Qwen2.5-VL-7B-Instruct, read from `hy_worldplay/weights/runtime/text_encoder/llm` (`paths.text_encoder`); download it there as in the HY-WorldPlay block even if you do not run HY-WorldPlay, or point the YAML at your own copy.
- **Environment:** Python 3.12, torch 2.10.0 (CUDA 13.0), diffusers 0.36.0, transformers 4.57.1, flash-attn 2.8.3 (a CUDA 13 / torch 2.10 build; the repository imports it), accelerate 1.10.1, einops 0.8.0, hydra-core 1.3.8, opencv-python 4.12.0.88, scipy 1.14.1, numpy 2.1.3. The adapter selects PyTorch flex attention (`seg_ar_attention_backend=flex`, one GPU), so FlashAttention 3 and MagiAttention from the repository's `uv sync --extra all` (or its Docker image) are not needed. Hy-Nomas pins `imageio-ffmpeg==0.5.1`, which this package accepts.
- **Licence:** Apache-2.0, with the third-party components listed in its `LICENSE`.
- **Peak GPU memory:** 69.3 GiB in either mode, 73.7 GiB (external depth).

```bash
git clone https://github.com/tencent-hunyuan/hy-nomas $G/hy_nomas/hy-nomas
git -C $G/hy_nomas/hy-nomas checkout 5ecf99c981432d40a3e6b8788432f007f7acc28b
hf download tencent/Hy-Nomas \
  DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors \
  VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors \
  VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt \
  --revision 621df1570098aa1ebf916dac3ac72ae3cbe25caa --local-dir $G/hy_nomas/weights
nopeworld-infer --generator hy_nomas --method nopeworld_collision --task task2 --cases Cabin_Lake__d001 --out runs/hy_nomas
```

## Depth and evaluator models

- **DA3** (external_depth method, `--recon da3`): [`depth-anything/DA3-GIANT-1.1`](https://huggingface.co/depth-anything/DA3-GIANT-1.1) @ `72ee9f89ce4e50d704e9d55ee9c646ec8dc25a19` → `da3/DA3-GIANT-1.1/` (5.4 GB; weights CC BY-NC 4.0); the `depth_anything_3` package from [ByteDance-Seed/depth-anything-3](https://github.com/ByteDance-Seed/depth-anything-3) in the environment named by `NOPE_DEPTH_PYTHON`. Our runs used a copy of that package with additional modules; the official package was not compared.
- **VGGT-Omega** (Task 2 evaluator): [facebookresearch/vggt-omega](https://github.com/facebookresearch/vggt-omega) @ `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` → `vggt_omega/code/` (or installed), checkpoint `vggt_omega_1b_416_reproduce.pt` from [`facebook/VGGT-Omega`](https://huggingface.co/facebook/VGGT-Omega) @ `1041e80fc0e911235d3426b0a3d9a81075111a53` (4.6 GB; gated, request access on its page). Code: FAIR Noncommercial Research License; checkpoint: the terms on its model page.
- **SAM 2.1** (Task 2 evaluator): the `sam2` package from [facebookresearch/sam2](https://github.com/facebookresearch/sam2) @ `2b90b9f5ceec907a1c18123530e92e794ad901a4` and the checkpoint [`sam2.1_hiera_large.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) (0.9 GB) → `sam2/`. Code and checkpoints: Apache-2.0.
- **Environment:** Python 3.11, torch 2.11.0 (CUDA 12.8), transformers 5.17.0, xformers 0.0.35, opencv-python 4.11.0.86, scipy 1.17.1, numpy 1.26.4. `NOPE_DA3_WEIGHTS`, `NOPE_VGGT_OMEGA_WEIGHTS` and `NOPE_SAM2_CHECKPOINT` override the default paths.

## Sharing environments

We used one environment per generator plus one for the depth and evaluator models. LingBot-World 2.0 and WorldPlay2 come from
the same Wan2.2 codebase with the same requirements (Python 3.10, transformers ≤ 4.51.3, numpy < 2, flash-attn), so one environment
is likely to serve both; we have not tested that. The others conflict: HY-WorldPlay pins transformers 4.56.0 and diffusers 0.35.0,
Echo-WM needs torch 2.9.1 with xformers 0.0.33.post2 and transformers < 5, Evoke-Turbo needs transformers 5.3.0, Hy-Nomas needs
Python 3.12 and a CUDA 13 build of torch, and the evaluator uses transformers 5.17.

