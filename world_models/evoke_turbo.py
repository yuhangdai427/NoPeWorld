"""Evoke-Turbo (3-step pyramid, no cache across chunks, camera through a point-cloud warp): teacher-forced replay at step 0 and
generation with a head deciding each chunk, both inside the pipeline's own chunk loop through method overrides."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

from nopeworld.head.candidates import DELTA, NAMES, RHO, candidate_world_poses, intent_fit, path_length, resample_poses
from nopeworld.head.choose import choose
from nopeworld.head.labels import is_decision
from nopeworld.head.model import FieldConfig
from nopeworld.paths import resolve_data
from nopeworld.utils.rays import plucker_field, scale_intrinsics_cover
from nopeworld.utils.runtime import AsyncScorer, Prefetcher, TokenTap, read_video, write_video
from .base import Adapter, HistoryDoesNotFit, check_keep, prompt_sha256, relative_to_roots

HEIGHT, WIDTH = 384, 640
LAT_H, LAT_W = HEIGHT // 8, WIDTH // 8
STRIDE = 4                               # frames per latent
CHUNK = 9                                # latents per chunk
WINDOW = (CHUNK - 1) * STRIDE + 1        # 33 poses of a chunk's warp
CHUNK_FRAMES = CHUNK * STRIDE            # 36 frames between chunks
GRID = (CHUNK, LAT_H // 8, LAT_W // 8)   # step 0: two pyramid halvings of the latent, then the 2 x 2 patch
TOKENS = GRID[0] * GRID[1] * GRID[2]
FPS = 24
TEXT_LEN = 512
SOURCE_K = (640.0, 640.0, 640.0, 360.0)  # the dataset's 1280 x 720 frames, horizontal FOV 90 degrees
KP = scale_intrinsics_cover(1280, 720, 90.0, HEIGHT, WIDTH)   # the head's rays: the token centres over the 384 x 640 picture
MAX_DEG_PER_CHUNK, MAX_TRANS_PER_CHUNK, POSE_SMOOTH_WIN = 10.0, 3.0, 5      # the published command-line's cap and smoothing
BASE, TRANSFORMER, VIGEO = "evoke-base", "evoke-turbo", "ViGeo1.1"
PACKAGE_FOLDERS = ("fps16", "fps24", "default", "lingbot_world_2")      # dataset sub-folders named after the clock or the generator
RUN_CONFIG = {"teacher_encoding": "oneshot", "cloud_frames": "real", "ingest_frames": 36, "allow_retime": False, "dynamic_shifting": True}
HEAD_MUST_MATCH = ("allow_retime", "dynamic_shifting")
GPU = torch.device("cuda")


def import_evoke(code=None):
    """(scripts/inference/infer_single.py, evoke.pipelines.pipeline_evoke) of the repository, which stays read-only."""
    sys.dont_write_bytecode = True
    if code is not None:
        for path in (Path(code), Path(code) / "scripts/inference"):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
    import infer_single
    from evoke.pipelines import pipeline_evoke
    return infer_single, pipeline_evoke


def route_id(package) -> str:
    """The route or case id of a package folder (<route>/fps24, <case>/default, or a folder named after the route)."""
    package = Path(package)
    return package.parent.name if package.name in PACKAGE_FOLDERS else package.name


def lingbot_ks(height_src=720, width_src=1280, source_k=SOURCE_K):
    """The source K after the model's own crop and resize: what its renderer starts from (cx = 319.83 where the head's rays use 320)."""
    fx, fy, cx, cy = source_k
    scale = max(HEIGHT / height_src, WIDTH / width_src)
    return torch.tensor([fx * scale, fy * scale, cx * scale - (round(width_src * scale) - WIDTH) / 2.0,
                         cy * scale - (round(height_src * scale) - HEIGHT) / 2.0], dtype=torch.float32)


def intrinsic_source_size(K, height, width):
    """(height, width) a package's K was calibrated at, by the model loaders' rule: the size its principal point implies wins when both axes
    are more than 10 % off the image's size at one ratio."""
    fx, fy, cx, cy = K
    h_in, w_in = int(round(cy * 2)), int(round(cx * 2))
    off_h, off_w = abs(height - h_in) / max(h_in, 1) > 0.1, abs(width - w_in) / max(w_in, 1) > 0.1
    if off_h and off_w:
        s_h, s_w = h_in / height, w_in / width
        if abs(s_h - s_w) <= 0.05 * max(s_h, s_w):
            return h_in, w_in
    return height, width


def package_intrinsics(pkg):
    """(Ks [4] for the renderer, K for the head's rays, what was read) of a package's camera at the model's 640 x 384. intrinsics_native.npy
    (the K at the start picture's size) comes before intrinsics.npy; the dataset's own camera gives lingbot_ks() and KP exactly."""
    from PIL import Image
    pkg = Path(pkg)
    path = next((pkg / name for name in ("intrinsics_native.npy", "intrinsics.npy") if (pkg / name).exists()), None)
    if path is None:
        return lingbot_ks(), KP, None
    K = tuple(float(v) for v in np.load(path).reshape(-1, 4)[0])
    if abs(K[0] / K[1] - 1.0) > 0.01:
        raise ValueError(f"{path}: fx {K[0]:.2f} and fy {K[1]:.2f} are a stretched picture's, which the centre crop of the frames is not")
    width, height = Image.open(pkg / "handoff.png").size
    h_src, w_src = intrinsic_source_size(K, height, width)
    read = {"file": path.name, "K": [round(v, 4) for v in K], "calibrated_at": [w_src, h_src]}
    if np.allclose(K, SOURCE_K) and (h_src, w_src) == (720, 1280):
        return lingbot_ks(), KP, read
    ks = lingbot_ks(h_src, w_src, K)
    return ks, (float(ks[0]), float(ks[1]), WIDTH / 2.0, HEIGHT / 2.0), read


def check_pixels(row):
    """The grid is 24 fps: a row on frames of another rate would move faster per frame than its poses say."""
    if row.get("pixel_fps", FPS) != FPS:
        raise ValueError(f"{row['pair_id']}: frames at {row['pixel_fps']} fps, Evoke's clock is {FPS} fps")


def crop_box(width, height):
    """[left, top, right, bottom] of a picture that reaches the model: the model loader's centre crop."""
    target = HEIGHT / WIDTH
    if height / width >= target:
        keep = int(width * target)
        top = (height - keep) // 2
        return [0, top, width, top + keep]
    keep = int(height / target)
    left = (width - keep) // 2
    return [left, 0, left + keep, height]


def to_model_frames(rgb, batch=32):
    """[T, H, W, 3] uint8 -> [3, T, 384, 640] fp32 in [-1, 1] on the CPU: the loader's centre crop and bilinear antialiased resize."""
    left, top, right, bottom = crop_box(rgb.shape[2], rgb.shape[1])
    out = []
    for i in range(0, len(rgb), batch):
        part = (torch.from_numpy(rgb[i:i + batch]).float() / 127.5 - 1.0).permute(0, 3, 1, 2)[:, :, top:bottom, left:right]
        out.append(F.interpolate(part, size=(HEIGHT, WIDTH), mode="bilinear", align_corners=False, antialias=True))
    return torch.cat(out).permute(1, 0, 2, 3)


def crop_image(image):
    """The PIL start image with the same centre crop, at the model's size (Lanczos)."""
    from PIL import Image
    return image.crop(tuple(crop_box(*image.size))).resize((WIDTH, HEIGHT), Image.LANCZOS)


def noise_tokens(output):
    """The chunk's noise tokens: the end of the sequence [long | mid | short | noise]."""
    return output[:, -TOKENS:]


def chunk_seed(seed, k):
    """The seed of chunk k of a video or route."""
    return int(hashlib.sha256(f"{seed}:{k}".encode()).hexdigest()[:8], 16)


def vae_stats(vae, device):
    """The pipeline's latent normalisation: z = (sample - mean) * inv_std."""
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
    inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
    return mean, inv_std


def encode_real(vae, frames, *, generator, device):
    """[3, T, H, W] with T = 1 + 4 (L - 1) -> [1, 16, L, 48, 80] fp32: one causal pass over the whole video, the posterior sampled as the
    pipeline samples it."""
    mean, inv_std = vae_stats(vae, device)
    return ((vae.encode(frames[None].to(device, vae.dtype)).latent_dist.sample(generator=generator) - mean) * inv_std).to(torch.float32)


def chained_poses(rows):
    """{frame: [4, 4]} of the real camera at every chunk bound, in the camera frame of frame 0."""
    known, state = {}, np.eye(4)
    for k, row in enumerate(sorted(rows, key=lambda r: r["chunk"])):
        if row["chunk"] != k:
            raise ValueError(f"{row['pair_id']}: rows must hold every chunk (the poses are chained); chunk {k} is missing")
        for frame, relative in zip(row["chunk_bounds"], np.asarray(row["executed_poses"], np.float64)):
            known[frame] = state @ relative
        state = known[row["chunk_bounds"][-1]]
    return known


def real_track(known):
    """[F, 4, 4] float64: the real camera at every frame up to the last bound, interpolated between the bounds."""
    frames = sorted(known)
    return resample_poses(np.stack([known[f] for f in frames]), frames, np.arange(frames[-1] + 1))


def video_track(rows, known=None):
    """[F, 4, 4] float64 in the camera frame of frame 0: the real camera at every frame, from poses.npy next to the frames when it agrees
    with the rows' bounds, else interpolated between them."""
    known = chained_poses(rows) if known is None else known
    path = resolve_data(rows[0]["source_video"]).with_name("poses.npy")
    if path.exists():
        poses = np.load(path).astype(np.float64)
        if len(poses) > max(known):
            relative = np.linalg.inv(poses[0]) @ poses[:max(known) + 1]
            if max(np.abs((relative[f] - pose)[:3]).max() for f, pose in known.items()) < 1e-3:
                return relative
    print(json.dumps({"poses": "interpolated between the chunk bounds", "video": rows[0]["pair_id"]}), flush=True)
    return real_track(known)


def intent_window(row, state):
    """[33, 4, 4] float64: the row's commanded KEEP poses on the real state pose, at every frame of the warp."""
    world = state @ np.asarray(row["candidate_poses"], np.float64)[0, 1:]
    return resample_poses(world, np.arange(CHUNK) * STRIDE, np.arange(WINDOW))


def recipe_argv(args):
    """The argv the published launcher gives infer_single.py, with the local weights; build_pipe only turns it into the recipe."""
    return ["--ckpt_path", str(args.base_ckpt), "--transformer_path", str(args.transformer_ckpt), "--prompt", "unused",
            "--output_folder", "unused", "--sample_type", "v2v", "--is_enable_stage2", "--stage2_num_stages", "3",
            "--stage2_steps", "1", "1", "1", "--stage2_stage_range", "0", str(1 / 3), str(2 / 3), "1",
            "--stage2_warp_compression_mode", "fixed_mem", "--height", str(HEIGHT), "--width", str(WIDTH), "--fps", str(FPS),
            "--num_inference_steps", "3", "--guidance_scale", "1.0", "--vae_decode_type", "persistent", "--no_raw_sink_frames",
            "--use_geometric_state", "--visibility_aware_noise", "--warp_noise_sigma_invisible", "1.0",
            "--warp_noise_sigma_min", "0.0", "--warp_noise_sigma_max", "0.135", "--visible_token_threshold", "0.5",
            "--prefix_idx_mode", "zero", "--warp_rope_mode", "overlap_noise", "--warp_lag_chunks", "0",
            "--geo_recon_backend", "da3", "--geo_cloud_update_n", "12", "--geo_da3_process_res", "644",
            "--geo_depth_backend", "vigeo", "--geo_vigeo_weights", str(args.vigeo_weights), "--geo_vigeo_mode", "chunk",
            "--geo_vigeo_scale_mode", "auto", "--geo_vigeo_scale_value", "0.0", "--geo_vigeo_depth_median_target", "5",
            "--geo_vigeo_anchor_windows", "4", "--geo_vigeo_cache_keep_frames", "6", "--geo_vigeo_intr_source", "gt",
            "--geo_da3_render_mode", "backward_zbuf", "--geo_bw_fill_iters", "12", "--geo_warp_stage0_only",
            "--image_noise_sigma_min", "0.0", "--image_noise_sigma_max", "0.0", "--video_noise_sigma_min", "0.0",
            "--video_noise_sigma_max", "0.0", "--joystick_hud", "off"]


def call_arguments(*, prompt, sections, generator, c2ws, ks=None, **extra):
    """The keyword arguments of the published recipe's pipeline call; frames are collected by the overrides."""
    return dict(prompt=prompt, height=HEIGHT, width=WIDTH, num_frames=WINDOW * sections, num_inference_steps=3, guidance_scale=1.0,
                image_noise_sigma_min=0.0, image_noise_sigma_max=0.0, video_noise_sigma_min=0.0, video_noise_sigma_max=0.0,
                use_dynamic_shifting=True, time_shift_type="exponential", is_keep_x0=True, history_sizes=[16, 2, 1],
                is_enable_stage2=True, stage2_num_stages=3, stage2_num_inference_steps_list=[1, 1, 1],
                vae_decode_type="persistent", stream_output=True, return_dict=True, generator=generator, use_kv_cache=False,
                use_dmd=False, is_amplify_first_chunk=False, chunk_prompts={},
                attention_kwargs={"stage2_warp_compression_mode": "fixed_mem", "history_visible_token_threshold": 0.1},
                use_geometric_state=True, geo_disable_prev_short=False, geo_lora_path=None, geo_score="v1", geo_nearby_k=0,
                geo_select_k=5, geo_top_k=5, geo_bank_max=None, geo_init_k=10, geo_score_kwargs={}, lingbot_c2ws=c2ws,
                lingbot_Ks=lingbot_ks() if ks is None else ks, event_chunks=[], event_prompt="", **extra)


def cast_parameters(model, dtype, keep_fp32=()):
    """The dtypes a model has on its device, still on the meta device; keep_fp32 follows diffusers' from_pretrained rule."""
    for module_name, module in model.named_modules():
        for name, param in list(module.named_parameters(recurse=False)):
            qualified = f"{module_name}.{name}" if module_name else name
            wanted = torch.float32 if any(m in qualified.split(".") for m in keep_fp32) else dtype
            module._parameters[name] = torch.nn.Parameter(param.to(wanted), requires_grad=False)
    return model


def load_straight(model, directory, device, dtype, keep_fp32=()):
    """Weights of a model built on the meta device, read from its safetensors files straight to the device, one tensor at a time."""
    from safetensors import safe_open
    cast_parameters(model, dtype, keep_fp32)
    model.to_empty(device=device)
    params = dict(model.named_parameters())
    seen = set()
    with torch.no_grad():
        for path in sorted(Path(directory).glob("*.safetensors")):
            with safe_open(str(path), "pt", device="cpu") as f:
                for key in f.keys():
                    if key not in params:
                        raise KeyError(f"{path}: unexpected tensor {key}")
                    tensor = f.get_tensor(key)
                    if tensor.shape != params[key].shape:
                        raise ValueError(f"{key}: {tuple(tensor.shape)} != {tuple(params[key].shape)}")
                    params[key].copy_(tensor.to(params[key].dtype))
                    seen.add(key)
    missing = set(params) - seen
    if missing:
        raise KeyError(f"{directory} lacks {sorted(missing)[:5]} ({len(missing)} tensors)")
    return model.eval().requires_grad_(False)


def meta_transformer(path):
    """The DiT's class from its config, without tensor data."""
    from evoke.modules.transformer_evoke import EvokeTransformer3DModel
    with torch.device("meta"):
        return EvokeTransformer3DModel.from_config(EvokeTransformer3DModel.load_config(str(path)))


def load_transformer(path, device):
    """The DiT as from_pretrained(torch_dtype=bf16) leaves it; the RoPE base frequencies are non-persistent buffers, built again."""
    from evoke.modules.transformer_evoke import EvokeRotaryPosEmbed, EvokeTransformer3DModel
    model = meta_transformer(path)
    load_straight(model, path, device, torch.bfloat16, EvokeTransformer3DModel._keep_in_fp32_modules)
    model.rope = EvokeRotaryPosEmbed(rope_dim=tuple(model.config.rope_dim), theta=model.config.rope_theta).to(device)
    return model


def meta_vae(path):
    """The VAE's class from its config, without tensor data."""
    from diffusers import AutoencoderKLWan
    with torch.device("meta"):
        return AutoencoderKLWan.from_config(AutoencoderKLWan.load_config(str(path)))


def load_vae(path, device):
    """The VAE in float32, read straight to the device."""
    return load_straight(meta_vae(path), path, device, torch.float32)


def vigeo_state(checkpoint):
    """The parameter names of ViGeo.from_pretrained for what vigeo.pt holds."""
    return {k.removeprefix("model."): v for k, v in checkpoint.get("state_dict", checkpoint.get("model", checkpoint)).items()}


def load_vigeo(estimator):
    """The depth estimator's lazy load without its CPU detour: the state dict mapped to the device, the model assigned from it."""
    from evoke.modules.geometric_state.vigeo_cloud import check_assets
    check_assets(estimator.src, estimator.weights)
    if str(estimator.src) not in sys.path:
        sys.path.insert(0, str(estimator.src))
    from vigeo import ViGeo
    state = vigeo_state(torch.load(estimator.weights / "vigeo.pt", map_location=estimator.device, mmap=True, weights_only=True))
    with torch.device("meta"):
        model = ViGeo(encoder="vitg", with_mask_head=any(k.startswith("mask_head.") for k in state))
    model.load_state_dict(state, strict=True, assign=True)
    model.pretrained.last_scores = torch.zeros_like(model.pretrained.last_scores, device=estimator.device)
    estimator._model = model.eval()


def watch_estimator(estimator):
    """The point cloud's ingest swallows every exception of the depth backend (a CUDA OOM would end as a black warp): failures are kept for
    TextAndWeights to raise, and only the backend's own zero-baseline skip of a still camera is legal (counted)."""
    if not hasattr(estimator, "depth_window") or hasattr(estimator, "ingest_failures"):
        return
    original, original_single = estimator.depth_window, getattr(estimator, "depth_single", None)
    estimator.ingest_failures, estimator.ingest_skips = [], 0

    def depth_window(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except Exception as exc:
            if isinstance(exc, RuntimeError) and "no usable window (zero baseline" in str(exc):
                estimator.ingest_skips += 1
            else:
                estimator.ingest_failures.append(exc)
            raise

    def depth_single(*args, **kwargs):
        try:
            return original_single(*args, **kwargs)
        except Exception as exc:
            estimator.ingest_failures.append(exc)
            raise
    estimator.depth_window = depth_window
    if callable(original_single):
        estimator.depth_single = depth_single


def encode_texts(base_ckpt, prompts, device):
    """{prompt: [tokens, 4096] bf16 on the CPU} up to the padding, through the pipeline's own T5 encoding; the encoder is released afterwards."""
    from transformers import AutoTokenizer, UMT5EncoderModel
    _, module = import_evoke()
    encoder = UMT5EncoderModel.from_pretrained(str(Path(base_ckpt) / "text_encoder"), torch_dtype=torch.bfloat16,
                                               device_map={"": "cuda:0" if device.type == "cuda" else str(device)}).eval()
    owner = SimpleNamespace(tokenizer=AutoTokenizer.from_pretrained(str(Path(base_ckpt) / "tokenizer")), text_encoder=encoder)
    out = {}
    with torch.no_grad():
        for prompt in dict.fromkeys(prompts):
            embeds, mask = module.EvokePipeline._get_t5_prompt_embeds(owner, prompt=prompt, max_sequence_length=TEXT_LEN,
                                                                      device=device, dtype=torch.bfloat16)
            out[prompt] = embeds[0, :int(mask[0].sum())].cpu()
    del encoder, owner
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def pad_text(tokens, device):
    """[1, 512, D], zero-padded as the pipeline pads."""
    return torch.cat([tokens, tokens.new_zeros(TEXT_LEN - tokens.shape[0], tokens.shape[1])])[None].to(device)


class PreparedFrames:
    """The pipeline's video_processor for tensors already in [-1, 1] at the model's size (the pipeline's own processor rescales a frame
    without a negative value, which a bright wall filling the view can be); PIL images go through it."""

    def __init__(self, processor):
        self.processor = processor

    def preprocess(self, image, height=None, width=None):
        return image if isinstance(image, torch.Tensor) else self.processor.preprocess(image, height=height, width=width)

    def preprocess_video(self, video, height=None, width=None):
        return video.permute(1, 0, 2, 3)[None]

    def __getattr__(self, name):
        return getattr(self.processor, name)


class StopAfterProbe(Exception):
    """The last block a head reads has run: the rest of the replay's forward is never used."""


class Correction(Exception):
    """The head's choice is not the command: the chunk starts again from step 0 with the chosen poses."""


class TextAndWeights:
    """What replay and generation change alike in the pipeline: the text encoded once, given reference latents, no CPU detours, and depth
    backend failures raised."""
    text_contexts = {}
    session = None
    depth_estimator = None

    def to(self, *args, **kwargs):
        return self

    def encode_prompt(self, prompt=None, device=None, **kwargs):
        return pad_text(self.text_contexts[prompt], device), None, None, None

    def prepare_video_latents(self, video, latents_mean, latents_std, latent_window_size, dtype=None, device=None, generator=None,
                              latents=None):
        if latents is None:
            return super().prepare_video_latents(video, latents_mean, latents_std, latent_window_size, dtype=dtype, device=device,
                                                 generator=generator)
        latents = latents.to(device=device, dtype=dtype)
        return latents[:, :, :1], latents

    def _geo_da3_setup(self, geo_state, *args, **kwargs):
        super()._geo_da3_setup(geo_state, *args, **kwargs)
        estimator = self.depth_estimator = geo_state["da3_est"]
        if getattr(estimator, "_model", "ready") is None:
            gc.collect()                 # the replaced estimator and its watch reference each other: its memory is freed before the new one is built
            torch.cuda.empty_cache()
            load_vigeo(estimator)
        watch_estimator(estimator)
        if hasattr(estimator, "ingest_failures"):              # the estimator outlives a video
            estimator.ingest_failures, estimator.ingest_skips = [], 0

    def _geo_da3_ingest(self, geo_state, frames_local, global_pix_start, c2w_full, device):
        super()._geo_da3_ingest(geo_state, frames_local, global_pix_start, c2w_full, device)
        failures = getattr(geo_state.get("da3_est"), "ingest_failures", None)
        if failures:
            raise RuntimeError(f"the depth backend failed on the window from frame {global_pix_start}") from failures.pop()

    def _geo_render_chunk_da3(self, **kwargs):
        out = super()._geo_render_chunk_da3(**kwargs)
        failures = getattr(kwargs["geo_state"].get("da3_est"), "ingest_failures", None)
        if failures:
            raise RuntimeError(f"the depth backend failed on the warp of chunk {kwargs['chunk_idx']}") from failures.pop()
        return out

    def sample_block_noise(self, batch_size, channel, num_frames, height, width, patch_size=(1, 2, 2), device=None, generator=None):
        """The pyramid's block noise as published; where its float32 Cholesky fails (an eigenvalue of 1e-8), nothing is drawn yet and the
        same factor is computed in float64."""
        try:
            return super().sample_block_noise(batch_size, channel, num_frames, height, width, patch_size, device, generator)
        except torch.linalg.LinAlgError:
            pass
        generator = generator[0] if isinstance(generator, list) else generator
        _, ph, pw = patch_size
        size = ph * pw
        gamma = self.scheduler.config.gamma
        cov = torch.eye(size, dtype=torch.float64) * (1 + gamma) - torch.ones(size, size, dtype=torch.float64) * gamma + torch.eye(size, dtype=torch.float64) * 1e-8
        factor = torch.linalg.cholesky(cov).float().to(device)
        z = torch.randn(batch_size * channel * num_frames * (height // ph) * (width // pw), size, generator=generator, device=generator.device).to(device=device)
        noise = (z @ factor.T).view(batch_size, channel, num_frames, height // ph, width // pw, ph, pw)
        return noise.permute(0, 1, 2, 3, 5, 4, 6).reshape(batch_size, channel, num_frames, height, width)


class ReplayMixin(TextAndWeights):
    """The chunk loop with real latents and frames in place of the model's: pipeline chunk k is row chunk k + 1."""

    def _geo_render_chunk(self, **kwargs):
        s = self.session
        k = kwargs["chunk_idx"]
        s.begin_chunk(k)
        if not s.wanted(k):
            return s.no_warp()
        if not kwargs["geo_state"]["da3_bank"].frames:
            # a camera standing still from the first frame leaves only the backend's legal zero baseline: a blank warp, as published
            if not getattr(kwargs["geo_state"].get("da3_est"), "ingest_skips", 0):
                raise RuntimeError(f"row chunk {k + 1}: a probe on an empty cloud (a black warp, no camera signal), and no window was skipped")
            s.blank_warps += 1
        return super()._geo_render_chunk(**{**kwargs, "camera_poses_pix_window": s.window(k)})

    def _geo_encode_warp_to_latents(self, **kwargs):
        s = self.session
        if not s.wanted(self._evoke_progress_chunk_index):
            return s.no_warp_latents()
        with s.vae_lock:                 # vae.encode clears the state of every other VAE call
            return super()._geo_encode_warp_to_latents(**kwargs)

    def stage2_sample(self, latents=None, **kwargs):
        s = self.session
        k = self._evoke_progress_chunk_index
        if s.wanted(k):
            s.tap.begin(GRID)
            try:
                super().stage2_sample(latents=latents, **kwargs)
            except StopAfterProbe:
                pass
            s.observe_state(k, (getattr(self, "_geo_last_render_stats", None) or {}).get("cov"))     # outside the except: its frames hold the forward's activations
        return s.real_latents(k)

    def _decode_chunk_persistent_cache(self, **kwargs):
        return self.session.real_frames(self._evoke_progress_chunk_index)


class ReplaySession:
    """One video's replay: what ReplayMixin reads. k is the pipeline's chunk; row k + 1 holds its labels and poses."""

    def __init__(self, rows, inputs, observe, *, windows, probed, transformer, layers, seed, generator, device, vae_lock, first_probe):
        self.rows, self.inputs, self.observe = rows, inputs, observe
        self.windows, self.probed = windows, probed
        self.seed, self.generator, self.device, self.vae_lock = seed, generator, device, vae_lock
        self.first_probe, self.block_count = first_probe, len(transformer.blocks)
        self.tap = TokenTap(transformer.blocks, layers, extract=noise_tokens)

        def stop(module, inputs_, output):
            raise StopAfterProbe

        def forward_begins(module, args, kwargs):
            self.forward_seen = kwargs["timestep"], kwargs["is_first_denoising_step"]
        self.stop_hook = transformer.blocks[max(layers)].register_forward_hook(stop)    # after the tap's hook on that block
        self.forward_hook = transformer.register_forward_pre_hook(forward_begins, with_kwargs=True)
        self.forward_seen = None
        self.states, self.seconds, self.began, self.blank_warps = 0, [], 0.0, 0

    def close(self):
        self.stop_hook.remove()
        self.forward_hook.remove()
        self.tap.close()

    def wanted(self, k):
        return k + 1 in self.probed

    def begin_chunk(self, k):
        self.began = time.perf_counter()
        self.generator.manual_seed(chunk_seed(self.seed, k))
        os.environ["EVOKE_WARP_SEED"] = str(chunk_seed(self.seed, k))

    def window(self, k):
        return torch.as_tensor(self.windows[k + 1], dtype=torch.float32, device=self.device)

    def no_warp(self):
        return (torch.zeros(1, 3, WINDOW, HEIGHT, WIDTH, device=self.device),
                torch.zeros(1, 1, WINDOW, HEIGHT, WIDTH, device=self.device))

    def no_warp_latents(self):
        return torch.zeros(1, 16, CHUNK, LAT_H, LAT_W, device=self.device)

    def observe_state(self, k, warp_coverage):
        row = self.rows[k + 1]
        hidden = self.tap.finish()
        t, first_step = self.forward_seen
        self.first_probe(row["chunk"], t, hidden, blocks=self.block_count, first_denoising_step=bool(first_step), warp_coverage=warp_coverage)
        camera = plucker_field(np.asarray(row["candidate_poses"])[:, 1:], KP, HEIGHT, WIDTH, GRID[1:], device=self.device)
        self.observe(hidden, camera, torch.tensor(row["interval_ends"], device=self.device), row)
        self.states += 1
        self.seconds.append(time.perf_counter() - self.began)

    def real_latents(self, k):
        return self.inputs.clean[:, :, (k + 1) * CHUNK:(k + 2) * CHUNK].to(self.device)

    def real_frames(self, k):
        """[1, 3, 36, 384, 640] on the CPU: the frames 36(k+1)-3 .. 36(k+1)+32."""
        first = (k + 1) * CHUNK_FRAMES - 3
        return self.inputs.frames[:, first:first + CHUNK_FRAMES][None]


@dataclass
class VideoInputs:
    """Prepared inputs of one video; nothing here is a DiT state."""
    clean: torch.Tensor                  # [1, 16, L, 48, 80] latents of the real frames
    frames: torch.Tensor                 # [3, 1 + 4 (L - 1), 384, 640] on the CPU


def pipeline_class(mixin, base):
    """The pipeline class with a mixin's overrides in front."""
    return type(f"{mixin.__name__}{base.__name__}", (mixin, base), {})


def guard_vae(vae, lock):
    """The pipeline's closing vae.clear_cache() under the (reentrant) lock, so that it cannot wipe the state of an encoding in the prefetch thread."""
    original = vae.clear_cache

    def clear_cache():
        with lock:
            original()
    vae.clear_cache = clear_cache


def build_pipeline(args, cls, device):
    """An instance of cls with the published recipe, built by the repository's own build_pipe with its from_pretrained calls answered by
    what load_straight read."""
    infer_single, module = import_evoke()
    from diffusers import AutoencoderKLWan
    from evoke.modules.transformer_evoke import EvokeTransformer3DModel
    transformer = load_transformer(Path(args.transformer_ckpt) / "transformer", device)
    vae = load_vae(Path(args.base_ckpt) / "vae", device)

    def assemble(_path, *, transformer, vae, scheduler, torch_dtype):
        return cls(tokenizer=None, text_encoder=None, vae=vae, scheduler=scheduler, transformer=transformer)
    with (mock.patch.object(EvokeTransformer3DModel, "from_pretrained", return_value=transformer),
          mock.patch.object(AutoencoderKLWan, "from_pretrained", return_value=vae),
          mock.patch.object(module.EvokePipeline, "from_pretrained", side_effect=assemble)):
        pipe = infer_single.build_pipe(infer_single.parse_args(recipe_argv(args)))[0]
    infer_single._PIPE_CACHE.clear()     # one reference to the DiT: the adapter's
    pipe.video_processor = PreparedFrames(pipe.video_processor)
    return pipe


class EvokeTurboAdapter(Adapter):
    """Evoke-Turbo replay (step-0 tap) and generation; no commit pass, so no prediction head."""
    generator_id = "evoke_turbo"
    field = FieldConfig(token_dim=5120, camera_dim=6, frames=CHUNK, height=GRID[1], width=GRID[2], camera_tokens="field",
                        camera_hidden=128)
    model_commit = "1d271be45128bbda4263e9e3c7dfc8f6ba260492"
    weights = ("base_ckpt", "transformer_ckpt", "vigeo_weights")
    step0_t = 998.0                      # sigma 0.999 shifted by the launcher's dynamic shifting to 0.999422, t = 998.85 truncated

    def weight_files(self, args):
        args.base_ckpt = self.path("weights") / BASE
        args.transformer_ckpt = self.path("weights") / TRANSFORMER
        args.vigeo_weights = self.path("weights") / VIGEO

    def run_config(self):
        return dict(RUN_CONFIG)

    def head_must_match(self, args):
        return {name: RUN_CONFIG[name] for name in HEAD_MUST_MATCH}

    def load(self, args):
        self.args = args
        _, module = import_evoke(self.path("code"))
        self.weight_files(args)
        self.pipe, self.device = build_pipeline(args, pipeline_class(ReplayMixin, module.EvokePipeline), GPU), GPU
        self.vae_lock = threading.RLock()
        guard_vae(self.pipe.vae, self.vae_lock)
        self.contexts, self.sources = {}, {}
        self.pipe.text_contexts = self.contexts
        self.prefetcher = Prefetcher(self.prepare_video, workers=1 if args.prefetch else 0, depth=max(args.prefetch, 1))

    def prepare_text(self, prompts):
        """Every prompt encoded once with T5 on the GPU; the encoder is released afterwards."""
        with self.held_gpu():
            self.contexts.update(encode_texts(self.args.base_ckpt, prompts, self.device))

    def announce(self, rows_of_next_videos):
        keys = []
        for rows in rows_of_next_videos:
            check_pixels(rows[0])
            self.sources[rows[0]["pair_id"]] = rows[0]
            keys.append(rows[0]["pair_id"])
        self.prefetcher.announce(keys)

    def video_seed(self, pair_id):
        return chunk_seed(self.args.seed, pair_id)

    @torch.no_grad()
    def prepare_video(self, pair_id):
        """Decode, resize and VAE-encode one video; the frames stay on the CPU."""
        rgb = read_video(resolve_data(self.sources[pair_id]["source_video"]))
        latents = (len(rgb) - 1) // STRIDE + 1
        latents -= latents % CHUNK
        frames = to_model_frames(rgb[:(latents - 1) * STRIDE + 1])
        generator = torch.Generator(device=self.device).manual_seed(self.video_seed(pair_id))
        with self.vae_lock:
            clean = encode_real(self.pipe.vae, frames, generator=generator, device=self.device)
        return VideoInputs(clean=clean, frames=frames)

    def replay(self, rows, observe, *, layers, seed, probe=None):
        row = rows[0]
        check_pixels(row)
        self.sources[row["pair_id"]] = row
        if not any(is_decision(r) and (probe is None or bool(probe(r))) for r in rows):
            self.prefetcher.discard(row["pair_id"])
            return 0, [0.0]
        inputs = self.prefetcher.get(row["pair_id"])
        return self.run_replay(rows, inputs, observe, layers=layers, seed=seed, probe=probe)

    def run_replay(self, rows, inputs, observe, *, layers, seed, probe=None):
        """One replay on prepared inputs: at a probed chunk the warp at the command's poses and step 0 up to the last block read; every chunk
        commits the real latents and frames with the real poses."""
        rows = sorted(rows, key=lambda r: r["chunk"])
        probed = {r["chunk"] for r in rows if is_decision(r) and (probe is None or bool(probe(r)))}
        if not probed:
            return 0, [0.0]
        if inputs.clean.shape[2] < (max(probed) + 1) * CHUNK:
            raise ValueError(f"{rows[0]['pair_id']}: {inputs.clean.shape[2]} latents do not reach row chunk {max(probed)}")
        known = chained_poses(rows)
        track = video_track(rows, known)
        windows = {k: intent_window(rows[k], known[rows[k]["chunk_bounds"][0]]) for k in probed}
        c2ws = torch.as_tensor(track, dtype=torch.float32, device=self.device)[None]
        generator = torch.Generator(device=self.device).manual_seed(seed)
        session = ReplaySession(rows, inputs, observe, windows=windows, probed=probed, transformer=self.pipe.transformer, layers=layers,
                                seed=seed, generator=generator, device=self.device, vae_lock=self.vae_lock, first_probe=self.first_probe)
        placeholder = torch.zeros(1, 16, 1, LAT_H, LAT_W, device=self.device)    # never read in video-to-video mode
        self.pipe.session = session
        try:
            with self.held_gpu():
                self.pipe(**call_arguments(
                    prompt=rows[0]["instruction"], sections=max(probed), generator=generator, c2ws=c2ws,
                    image=inputs.frames[:, :1].permute(1, 0, 2, 3), image_latents=placeholder, fake_image_latents=placeholder,
                    add_noise_to_image_latents=False, video=inputs.frames[:, :WINDOW].permute(1, 0, 2, 3),
                    video_latents=inputs.clean[:, :, :CHUNK], add_noise_to_video_latents=False))
        finally:
            session.close()
            self.pipe.session = None
            with self.vae_lock:
                self.pipe.vae.clear_cache()
            self.release_gpu()
        skips = getattr(self.pipe.depth_estimator, "ingest_skips", 0)
        if skips:
            print(json.dumps({"video": rows[0]["pair_id"], "zero_baseline_windows_skipped": skips, "states_on_a_blank_warp": session.blank_warps}), flush=True)
        return session.states, session.seconds or [0.0]

    def load_generation(self, args):
        self.args = args
        _, module = import_evoke(self.path("code"))
        self.weight_files(args)
        return build_pipeline(args, pipeline_class(GenerationMixin, module.EvokePipeline), GPU)

    def prepare_generation_text(self, pipe, packages):
        """All prompts encoded once, then T5 is gone: the pipeline never holds it."""
        prompts = [json.loads((Path(p) / "prompts.json").read_text())["per_model"]["evoke_turbo"] for p in packages]
        with self.held_gpu():
            pipe.text_contexts = encode_texts(self.args.base_ckpt, prompts, pipe.device)

    def generate_case(self, pipe, pkg, out, **kwargs):
        with self.held_gpu():                                    # the pipeline empties the cache after every chunk
            return generate_case(pipe, pkg, out, **kwargs)


class GenerationMixin(TextAndWeights):
    """The published chunk loop but for the per-chunk seeding, with a head deciding after step 0 when the session has one, and the decoded
    frames collected on the CPU."""

    def _geo_render_chunk(self, **kwargs):
        window = self.session.window(kwargs["chunk_idx"])
        if window is not None:           # the training's window, on the track the pipeline reads, corrections included
            kwargs = {**kwargs, "camera_poses_pix_window": window}
        self.session.begin_chunk(kwargs)
        return super()._geo_render_chunk(**kwargs)

    def _geo_encode_warp_to_latents(self, **kwargs):
        self.session.encode_args = kwargs
        return super()._geo_encode_warp_to_latents(**kwargs)

    def stage2_sample(self, latents=None, **kwargs):
        s = self.session
        if s.current is None:
            out = super().stage2_sample(latents=latents, **kwargs)
            s.end_chunk()
            return out
        rng = kwargs["generator"].get_state()  # the noise is drawn; stage 0 draws nothing, stage 1 starts with its block noise
        s.start_decision(self.transformer)
        corrected = False
        try:
            out = super().stage2_sample(latents=latents, **{**kwargs, "callback_on_step_end": s.after_step,
                                                            "callback_on_step_end_tensor_inputs": ["latents"]})
        except Correction:
            corrected = True
        s.close_decision()
        if corrected:
            out = super().stage2_sample(latents=latents, **self.correct(kwargs, rng))
        s.end_chunk(applied=corrected)
        return out

    def correct(self, kwargs, rng):
        """The chosen poses in the track and in the chunk's warp, the warp encoded with the random numbers of the first encoding."""
        s = self.session
        window = s.apply_choice()
        generator = kwargs["generator"]
        generator.set_state(s.rng_begin)
        budget = getattr(self, "_geo_chunk0_target_disparity_px", None)
        if s.chunk == 0 and s.ref == 0 and budget is not None:
            # image to video, chunk 0: the parallax is solved to a pixel budget from the target poses; it scales with the candidate's length
            self._geo_chunk0_target_disparity_px = budget * RHO[s.choice]
        try:
            warp, visible = super()._geo_render_chunk(**{**s.render_args, "camera_poses_pix_window": window})
        finally:
            if budget is not None:
                self._geo_chunk0_target_disparity_px = budget
        warp_latents = super()._geo_encode_warp_to_latents(**{**s.encode_args, "warp_video": warp, "visibility_mask_pix": visible})
        generator.set_state(rng)
        short = kwargs["latents_history_short"]            # [prefix | warp | prev_short]
        return {**kwargs, "latents_history_short": torch.cat([short[:, :, :1], warp_latents, short[:, :, 1 + CHUNK:]], dim=2),
                "history_visible_mask_short": self._geo_build_short_visibility_mask(
                    visibility_mask_pix=visible, num_lat_per_chunk=CHUNK, has_prev_short=True, H_lat=LAT_H, W_lat=LAT_W,
                    device=kwargs["device"], vae_t_stride=STRIDE)}

    def _decode_chunk_persistent_cache(self, **kwargs):
        out = super()._decode_chunk_persistent_cache(**kwargs)
        self.session.decoded.append(out.cpu())
        return out


def window_lead(ref):
    """Frames from the pipeline's cursor ref + 36 k to the first pose of chunk k's warp: with history (ref > 0) the warp sits at the frames its
    latents end on, as in training (3); image to video keeps the cursor (0)."""
    return STRIDE - 1 if ref else 0


def chunk_frames(k, ref, *, lead):
    """(anchor, first window frame) of pipeline chunk k in the pose track. lead 0: the window starts at the cursor and the anchor is the
    pipeline's own (the last pose of the previous window, or the start image). lead 3: the anchor is the newest frame before the chunk."""
    start = ref + CHUNK_FRAMES * k
    if lead:
        return start - 1, start + lead
    return (start - STRIDE if k else max(ref - 1, 0)), start


def chunk_length(track, k, ref, *, lead):
    """Ground distance (m) from the anchor to the chunk's last pose."""
    anchor, start = chunk_frames(k, ref, lead=lead)
    return float(np.linalg.norm(track[start + WINDOW - 1, :2, 3] - track[anchor, :2, 3]))


class CaseSession:
    """One route in the chunk loop (GenerationMixin). The track is the command after the published smoothing and cap; a correction rewrites
    it and the tensor the pipeline reads. With a scorer stage 1 runs while the head scores and a correction starts the chunk again."""

    def __init__(self, pipe, head, *, track, c2ws, ref, seed, generator, selection, w, observe_only, scorer, rays_k=KP):
        self.pipe, self.head, self.track, self.c2ws, self.ref = pipe, head, track, c2ws, ref
        self.rays_k = rays_k
        self.lead = window_lead(ref)
        self.seed, self.generator, self.selection, self.w = seed, generator, selection, w
        self.observe_only = observe_only
        self.records, self.decoded = [], []
        self.chunk, self.current, self.result, self.future, self.abort_hook = None, None, None, None, None
        self.record, self.began, self.choice, self.render_args, self.encode_args, self.rng_begin = None, 0.0, None, None, None, None
        self.tap, self.scorer = None, None
        if head is not None:
            if getattr(head.config, "command_input", False):
                raise ValueError("a prediction head: Evoke has no commit pass, so it would read step-0 tokens it was never trained on")
            if (head.config.frames, head.config.height, head.config.width) != GRID:
                raise ValueError(f"the head's grid {(head.config.frames, head.config.height, head.config.width)} is not {GRID}")
            self.tap = TokenTap(pipe.transformer.blocks, head.config.layers, extract=noise_tokens)
            if scorer:
                self.scorer = AsyncScorer(self.score)

    def close(self):
        if self.tap is not None:
            self.tap.close()
        if self.scorer is not None:
            self.scorer.close()

    def decides(self):
        return self.head is not None

    def frames(self, k):
        return chunk_frames(k, self.ref, lead=self.lead)

    def window(self, k):
        """[33, 4, 4] chunk k's warp poses on the track the pipeline reads, or None where the pipeline's own window is the one (lead 0)."""
        if not self.lead:
            return None
        _, first = self.frames(k)
        window = self.c2ws[0, first:first + WINDOW]
        if window.shape[0] != WINDOW:
            raise ValueError(f"chunk {k}: the pose track ends at {self.c2ws.shape[1]}, before the window's last frame {first + WINDOW - 1}")
        return window

    def begin_chunk(self, render_args):
        k = render_args["chunk_idx"]
        self.chunk, self.render_args, self.began = k, render_args, time.perf_counter()
        self.generator.manual_seed(chunk_seed(self.seed, k))
        os.environ["EVOKE_WARP_SEED"] = str(chunk_seed(self.seed, k))
        self.rng_begin = self.generator.get_state()
        self.current = self.candidates(k) if self.decides() else None
        anchor, start = self.frames(k)
        self.record = {"chunk": k, "decided": self.current is not None, "intent_length_m": round(chunk_length(self.track, k, self.ref, lead=self.lead), 4),
                       "on_rows_time_axis": start - anchor == STRIDE, "zero_baseline_skips_before": zero_baseline_skips(self.pipe)}
        if self.current is not None:
            keep = torch.as_tensor(self.current["world"][0], dtype=self.c2ws.dtype, device=self.c2ws.device)
            check_keep(float((keep - self.c2ws[0, torch.as_tensor(self.current["frames"], device=self.c2ws.device)]).abs().max()), k, self.record)
        self.result, self.future, self.choice = None, None, None

    def candidates(self, k):
        """The 34 candidates on the pose track, laid out as in the training rows: every frame of the window for the warp, relative to the
        anchor at each latent's end for the head. None for an in-place turn."""
        anchor, start = self.frames(k)
        frames = np.arange(anchor, start + WINDOW)
        if chunk_length(self.track, k, self.ref, lead=self.lead) <= 1e-3:
            return None
        world = candidate_world_poses(self.track[frames], path_length(self.track, frames), first=False)
        ends = start - anchor + np.arange(CHUNK) * STRIDE
        return {"frames": frames, "world": world, "offset": start - anchor, "relative": np.linalg.inv(self.track[anchor]) @ world[:, ends],
                "ends": torch.tensor(ends / FPS, dtype=torch.float32, device=self.c2ws.device),
                "decoded_end": self.ref + CHUNK_FRAMES * (k + 1)}

    def score(self, hidden, current):
        """The head's choice for a chunk, on the main thread or the scorer's."""
        mask = torch.ones(1, len(NAMES), dtype=torch.bool, device=hidden.device)
        camera = plucker_field(current["relative"], self.rays_k, HEIGHT, WIDTH, GRID[1:], device=hidden.device)
        fit = intent_fit(DELTA, RHO, w=self.w)[None].to(hidden.device)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            out = self.head(hidden[None], camera[None], current["ends"][None], mask)
            a = torch.sigmoid(out["intent_logits"]) if "intent_logits" in out else fit
            sel = choose(out["safe_logits"], a, mask, temperature=self.selection.get("temperature", 1.0),
                         keep_margin=self.selection.get("keep_margin", 0.0), rule=self.selection.get("rule", "product"),
                         tau=self.selection.get("tau", 0.5))
        return {"choice": int(sel["choice"][0]), "p": sel["P"][0].float().cpu(), "r": sel["R"][0].float().cpu(), "a": a[0].float().cpu(),
                "clearance": out["clearance"][0].float().cpu()}

    def wants_change(self, result):
        return result["choice"] != 0 and not self.observe_only

    def start_decision(self, transformer):
        self.transformer = transformer
        self.tap.begin(GRID)

    def after_step(self, pipe, i, t, tensors):
        """The pipeline's step-end callback: i = 0 and 1 are the ends of pyramid stages 0 and 1."""
        if i == 0:
            hidden = self.tap.finish()
            if self.scorer is None:
                self.result = self.score(hidden, self.current)
                if self.wants_change(self.result):
                    raise Correction
            else:
                self.future = self.scorer.submit(hidden, self.current)
                blocks = self.transformer.blocks
                self.abort_hook = blocks[len(blocks) // 2].register_forward_hook(self.abort)
        elif i == 1 and self.future is not None:
            self.abort_hook.remove()
            self.abort_hook = None
            self.result = self.future.result()
            if self.wants_change(self.result):
                raise Correction
        return {}

    def abort(self, *_):
        if self.future.done() and self.wants_change(self.future.result()):
            raise Correction

    def close_decision(self):
        if self.abort_hook is not None:
            self.abort_hook.remove()
            self.abort_hook = None
        if self.result is None:
            self.result = self.future.result()
        self.tap.cancel()
        self.log(self.result)

    def log(self, result):
        self.choice = result["choice"]
        p, r = result["p"], result["r"]
        top = r.topk(5).indices.tolist()
        self.record.update(choice=NAMES[self.choice], p_keep=round(float(p[0]), 4), p_choice=round(float(p[self.choice]), 4),
                           r_keep=round(float(r[0]), 4), r_choice=round(float(r[self.choice]), 4),
                           clearance_keep=round(float(result["clearance"][0]), 3),
                           top=[[NAMES[i], round(float(p[i]), 3), round(float(r[i]), 3)] for i in top],
                           p_all=[round(float(x), 4) for x in p], r_all=[round(float(x), 4) for x in r],
                           a_all=[round(float(x), 4) for x in result["a"]])
        stats = getattr(self.pipe, "_geo_last_render_stats", None) or {}
        self.record.update(coverage_command=stats.get("cov"))

    def apply_choice(self):
        """[33, 4, 4]: the chosen window. The track takes the candidate's poses and every later pose moves with the chunk's new end."""
        current = self.current
        frames, new = current["frames"], current["world"][self.choice]
        last = frames[-1]
        move = new[-1] @ np.linalg.inv(self.track[last])
        self.track[frames] = new
        self.track[last + 1:] = move @ self.track[last + 1:]
        # lead 0: the frames between this window and the next are decoded and ingested too; they continue the chosen speed
        tail = np.arange(last + 1, min(current["decoded_end"], len(self.track)))
        self.track[tail, :3, 3] = new[-1, :3, 3] + (tail - last)[:, None] * (new[-1, :3, 3] - new[-2, :3, 3])
        self.c2ws[0, frames[0]:] = torch.as_tensor(self.track[frames[0]:], dtype=self.c2ws.dtype, device=self.c2ws.device)
        return torch.as_tensor(new[current["offset"]:], dtype=torch.float32, device=self.c2ws.device)

    def end_chunk(self, applied=False):
        anchor, start = self.frames(self.chunk)
        stats = getattr(self.pipe, "_geo_last_render_stats", None) or {}
        self.record.update(applied=applied, seconds=round(time.perf_counter() - self.began, 3), pool=stats.get("pool"), coverage=stats.get("cov"))
        self.record["executed"] = (np.linalg.inv(self.track[anchor]) @ self.track[start + np.arange(CHUNK) * STRIDE]).round(5).tolist()
        self.records.append(self.record)


class DepthCaseSession(CaseSession):
    """The external depth method: no head. Before each chunk with a commanded move DA3 scores the candidates on the frames decoded so far and
    the collision head's decision rule keeps the command or stops; a stop goes the head's way (a correction after step 0)."""

    def __init__(self, pipe, *, geo, **kw):
        super().__init__(pipe, None, **kw)
        self.geo = geo

    def decides(self):
        return True

    def begin_chunk(self, render_args):
        super().begin_chunk(render_args)
        if self.current is not None:
            self.result = self.depth_decision()

    def depth_decision(self):
        from nopeworld.inference.methods.external_depth import chunk_len16, clock16, disps_in_camera, geo_choose, to_rgb
        k = self.chunk
        anchor, _ = self.frames(k)
        decoded = torch.cat(self.decoded, dim=2)[0] if self.decoded else None
        n = 0 if decoded is None else int(decoded.shape[1])
        end16, index_of = clock16(n, FPS, first_is_handoff=self.ref == 0)
        length = min(chunk_len16(int(self.decoded[-1].shape[2]), FPS), end16) if self.decoded else 0
        self.geo.size = [WIDTH, HEIGHT] if decoded is None else [int(decoded.shape[3]), int(decoded.shape[2])]
        disps = disps_in_camera(self.track[anchor], self.current["world"][:, -1])
        began = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()           # room for DA3
        reply = self.geo.scores(k, end16, length, disps, lambda offs: to_rgb(decoded[:, [index_of(o) for o in offs]]))
        result = geo_choose(reply, self.selection)
        self.record.update(depth_seconds=round(time.perf_counter() - began, 2), decoded_end16=end16, no_scale=result is None,
                           h_all=reply.get("h"), in_view=reply.get("in_view"),
                           depth={key: v for key, v in reply.items() if key not in ("h", "P", "A", "in_view", "slide")})
        return result

    def start_decision(self, transformer):
        self.transformer = transformer

    def after_step(self, pipe, i, t, tensors):
        if i == 0 and self.result is not None and self.wants_change(self.result):
            raise Correction
        return {}

    def close_decision(self):
        if self.result is None:                # no scale: the command
            self.choice = 0
            self.record.update(choice=NAMES[0])
        else:
            self.log(self.result)


def zero_baseline_skips(pipe):
    """The ingest windows the depth backend skipped in this video for a camera that did not move."""
    return int(getattr(getattr(pipe, "depth_estimator", None), "ingest_skips", 0) or 0)


def published_track(track, ref):
    """The published command line's smoothing, then its per-chunk cap; with the training's window (history) the first block is 36 frames too."""
    import_evoke()
    from evoke.utils.trajectory_capping import clamp_target_per_chunk_motion, smooth_c2w_trajectory
    track, _ = smooth_c2w_trajectory(track, ref_pix=ref, win=POSE_SMOOTH_WIN, verbose=False)
    first = CHUNK_FRAMES if window_lead(ref) else WINDOW
    track, _, _ = clamp_target_per_chunk_motion(c2ws=track, ref_pix=ref, frames_per_chunk=CHUNK_FRAMES, first_chunk_frames=first,
                                                max_deg=MAX_DEG_PER_CHUNK, max_trans=MAX_TRANS_PER_CHUNK, verbose=False)
    return np.array(track, np.float64)


def case_inputs(pkg, meta, history_chunks, chunks):
    """(pose track [ref + 36 chunks, 4, 4], history frames [3, ref, 384, 640] or None, ref). Without history the clip starts at the hand-off
    frame; with k history chunks the reference is the 33 + 36 (k - 1) frames of a 24 fps package that end at the hand-off frame. The poses
    continue the last commanded step past the package's end; the origin is the hand-off camera."""
    kh = meta["handoff_frame"]
    fps = meta.get("fps", 16)
    poses = np.load(pkg / "poses.npy").astype(np.float64)
    ref = WINDOW + CHUNK_FRAMES * (history_chunks - 1) if history_chunks else 0
    if ref and fps != FPS:
        raise ValueError(f"{pkg.name}: history.mp4 is {fps} fps and the clock is {FPS} fps: use the 24 fps package")
    last_ref = max(ref - 1, 0)
    source = np.round(kh + (np.arange(ref) - last_ref) * fps / FPS).astype(int)
    if len(source) and source[0] < 0:
        raise HistoryDoesNotFit(f"{pkg.name}: hand-off frame {kh} leaves no room for {history_chunks} history chunks of {ref} frames at {FPS} fps")
    track = resample_poses(poses, (np.arange(len(poses)) - kh) / fps, (np.arange(ref + CHUNK_FRAMES * chunks) - last_ref) / FPS)
    if ref:
        track[:ref] = poses[source]
    track[:, :3, 3] -= track[last_ref, :3, 3]
    video = to_model_frames(read_video(pkg / "history.mp4")[source]) if ref else None
    return track, video, ref


def package_length(meta):
    """(frames kept at 24 fps, chunks) for the package's generate_frames at its fps: 5 s -> (121, 4), 8 s -> (193, 6). Chunk 0 is WINDOW
    frames, every further chunk CHUNK_FRAMES."""
    generate, fps = int(meta["generate_frames"]), meta.get("fps", 16)
    if generate * FPS % fps:
        raise ValueError(f"generate_frames {generate} at {fps} fps is no whole number of frames at {FPS} fps")
    frames = generate * FPS // fps + 1
    return frames, math.ceil((frames - WINDOW) / CHUNK_FRAMES) + 1


def generate_case(pipe, pkg, out, *, track, head=None, payload=None, observe_only=False, scorer=False, history_chunks=0, seed=None,
                  depth=None):
    """One route: track A (the generator alone) or B (a head, or the external depth method with `depth`, deciding each chunk). The video has
    the hand-off frame first (the reconstruction without history, the real frame with it), then the frames after it. Returns (log, video);
    writes <out>/<route>.mp4 and <route>.decisions.json. executed_controls are the commanded poses after the corrections."""
    from PIL import Image
    pkg = Path(pkg)
    case = route_id(pkg)
    meta = json.loads((pkg / "meta.json").read_text())
    prompt = json.loads((pkg / "prompts.json").read_text())["per_model"]["evoke_turbo"]
    seed = seed if seed is not None else int(hashlib.sha256(case.encode()).hexdigest()[:8], 16)
    frames, chunks = package_length(meta)
    poses, video, ref = case_inputs(pkg, meta, history_chunks, chunks)
    package_end = (len(np.load(pkg / "poses.npy", mmap_mode="r")) - 1 - meta["handoff_frame"]) / meta.get("fps", 16)
    past_end = (np.arange(len(poses)) - max(ref - 1, 0)) / FPS > package_end + 1e-9
    poses = published_track(poses, ref)
    ks, rays_k, intrinsics = package_intrinsics(pkg)
    device = pipe.device
    c2ws = torch.as_tensor(poses, dtype=torch.float32, device=device)[None]
    generator = torch.Generator(device=device).manual_seed(seed)
    common = dict(track=poses, c2ws=c2ws, ref=ref, seed=seed, generator=generator, selection=dict((payload or {}).get("selection") or {}),
                  w=(payload or {}).get("w", 0.5), observe_only=observe_only, rays_k=rays_k)
    if depth is not None:
        if track != "B" or head is not None or scorer:
            raise ValueError("external depth: track B without a head and without an asynchronous scorer")
        geo = depth.geo(case, crop_box(*Image.open(pkg / "handoff.png").size), i2v=not ref)
        session = DepthCaseSession(pipe, geo=geo, scorer=False, **common)
    else:
        geo = None
        session = CaseSession(pipe, head if track == "B" else None, scorer=scorer, **common)
    if ref:
        placeholder = torch.zeros(1, 16, 1, LAT_H, LAT_W, device=device)        # never read in video-to-video mode
        condition = dict(image=video[:, :1].permute(1, 0, 2, 3), image_latents=placeholder, fake_image_latents=placeholder,
                         video=video.permute(1, 0, 2, 3))
    else:
        condition = dict(image=crop_image(Image.open(pkg / "handoff.png").convert("RGB")))
    if hasattr(pipe, "_geo_vsnoise_cfg"):
        pipe._geo_vsnoise_cfg["vigeo_scale_mode"] = "anchor" if ref else "depth_median"
    pipe.session = session
    began = time.perf_counter()
    try:
        pipe(**call_arguments(prompt=prompt, sections=chunks, generator=generator, c2ws=c2ws, ks=ks, **condition))
    finally:
        pipe.session = None
        session.close()
    seconds = time.perf_counter() - began
    decoded = torch.cat(session.decoded, dim=2)[0]
    kept = torch.cat([video[:, ref - 1:ref], decoded[:, :frames - 1]], dim=1) if ref else decoded[:, :frames]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    write_video(out / f"{case}.mp4", kept, FPS)
    source_size = Image.open(pkg / "handoff.png").size
    info = {"case": case, "track": track, "seed": seed, "seconds": round(seconds, 1), "frames": int(kept.shape[1]), "fps": FPS,
            "size": list(kept.shape[2:]), "crop": crop_box(*source_size), "prompt_sha256": prompt_sha256(prompt),
            "protocol": "v2v" if ref else "i2v", "history_chunks": history_chunks, "package_fps": meta.get("fps", 16),
            "intrinsics": {"read": intrinsics, "model_K": [round(float(v), 4) for v in ks], "rays_K": [round(float(v), 4) for v in rays_k]},
            "warp_lead": session.lead,
            "poses_past_package_end": int(past_end.sum()), "kept_frames_past_package_end": int(past_end[max(ref - 1, 0):max(ref - 1, 0) + frames].sum()),
            "zero_baseline_windows_skipped": zero_baseline_skips(pipe),
            "observe_only": observe_only, "async": bool(scorer), "selection": session.selection,
            "interventions": sum(r.get("applied", False) for r in session.records),
            "executed_controls": [r["executed"] for r in session.records],
            "chunks": [{k: v for k, v in r.items() if k != "executed"} for r in session.records]}
    if geo is not None:
        info.update(external_depth=True, depth_mode=geo.mode, depth_package=relative_to_roots(geo.package), depth_answers=geo.asked)
    (out / f"{case}.decisions.json").write_text(json.dumps(info, indent=1))
    return info, kept
