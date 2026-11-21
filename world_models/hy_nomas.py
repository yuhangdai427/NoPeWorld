"""Hy-Nomas (distilled DiT in bf16): teacher-forced replay and generation with a head deciding each chunk.

The model: 54 double-stream blocks of width 2048; DCAE-V latents of 4 frames (latent i = frames 4i..4i+3, 32x smaller in space);
1280 x 720 frames centre-cropped to 1280 x 704, 22 x 40 = 880 tokens per latent. One persistent session holds the history in the KV
cache: a start written once without denoising, then one call per segment, chunk by chunk (4 latents = 16 frames). Per chunk: the
denoising forwards (the first at t = 1000) and one cache write of the chunk's latents noised to sigma .206897. The camera enters after
every block through action_modules[L] (a cross-attention to one 64-d token per latent, built once per call from its per-frame poses),
which is where the heads read. Replay runs a training video in pieces of 7 decision chunks, each in a fresh world started from the 4
chunks before it, with one denoising step per chunk.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time

import numpy as np
import torch

from nopeworld.head.candidates import DELTA, NAMES, RHO, W_LATERAL, candidate_world_poses, intent_fit
from nopeworld.head.choose import choose
from nopeworld.head.labels import is_decision
from nopeworld.head.model import FieldConfig
from nopeworld.paths import resolve_data
from nopeworld.utils.rays import plucker_field, scale_intrinsics_cover
from nopeworld.utils.runtime import Prefetcher, TokenTap, read_video, write_video
from .base import Adapter, prompt_sha256, relative_to_roots

SOURCE_WIDTH, SOURCE_HEIGHT, SOURCE_HFOV = 1280, 720, 90.0          # the Unreal camera of the dataset and the routes
HEIGHT, WIDTH = 704, 1280                                             # 720 rows centre-cropped to a multiple of 32
K_MODEL = scale_intrinsics_cover(SOURCE_WIDTH, SOURCE_HEIGHT, SOURCE_HFOV, HEIGHT, WIDTH)
FPS, STRIDE, CHUNK = 24, 4, 4
GRID = (CHUNK, HEIGHT // 32, WIDTH // 32)                             # (4, 22, 40)
TOKEN_DIM, RAY_DIM, CAM_TOKEN_DIM = 2048, 6, 64
CAMERA_DIM = RAY_DIM + CAM_TOKEN_DIM
STEPS = 4                                                             # the distilled model's denoising steps
STEP0_T = 1000.0
CONTINUE = json.dumps({"segments": [{}]})                             # a camera-only segment
MIN_INTENT_M = 1e-3                                                   # an in-place turn has nothing to correct
PIECE = 7                                # decision chunks per replay piece (28 latents)
START_LATENTS = 16                       # a start: at most the 4 chunks (64 frames) before the first chunk of a call
IMAGE_START_FRAMES = 128                 # an image start: the picture repeated over 8 whole chunks
CONTEXT_FRAMES = 0                       # a chunk's DCAE-V window: its own 16 frames
# The released heads' run_config names the latent and splice rules with these ids: a head is accepted only under the same values.
LATENT_RULE = f"causal_chunk_v1:context{CONTEXT_FRAMES}"
SPLICE_RULE = "repropagate_v1"


@dataclass(frozen=True)
class Feeding:
    """How the camera reaches the model. metric_speed None: the repository's normalisation (translations over the call's 99th-percentile
    step); else metric_speed model units per metre. camera_shift: a call's camera rows are its frames from the fourth on."""
    metric_speed: float | None = None
    camera_shift: bool = False

    @property
    def translation_scale(self):
        return self.metric_speed if self.metric_speed is not None else 1.0

    def switches(self):
        """What a head and a run record of how the model is fed (the keys of the heads' run_config)."""
        return {"translation_scale": self.translation_scale, "metric_speed": self.metric_speed is not None, "cam_shift3": self.camera_shift,
                "latent_rule": LATENT_RULE, "start_latents": START_LATENTS, "image_start_frames": IMAGE_START_FRAMES, "splice": SPLICE_RULE,
                "start_camera": True}


def feeding_of(args):
    return Feeding(metric_speed=getattr(args, "metric_speed", None), camera_shift=bool(getattr(args, "camera_shift", False)))


# --- the repository ------------------------------------------------------------------------------------------------------------

def repo_config(code, weights, text_encoder, *, steps, feeding, rollout="chunk_ar", seed=42):
    """The repository's Hydra config: inference preset, distilled DiT in bf16, decoder PixelUMM_FT, DCAE-V for starts, one GPU with flex
    attention."""
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    weights = Path(weights)
    over = ["+experiment=inference", f"dit_weight={weights / 'DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors'}",
            f"model.vae.checkpoint={weights / 'VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors'}",
            f"++model.vae.encoder_checkpoint={weights / 'VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt'}",
            f"model.text_encoder.model_name={text_encoder}", "++model.dit.seg_ar_attention_backend=flex", "model.vae.parallel_mode=none",
            "parallelism.sp_size=1", "parallelism.tp_size=1", "inference.overlay_keys=false", f"inference.rollout_type={rollout}",
            f"inference.inference_chunk_size={CHUNK if rollout == 'chunk_ar' else 0}", f"inference.infer_steps_video={steps}",
            f"inference.video_size=[720,{WIDTH}]", f"inference.save_path={tempfile.mkdtemp(prefix='hy_nomas_')}", f"inference.seed={seed}"]
    argv, sys.argv = sys.argv, [sys.argv[0], *over]
    try:
        from configs.config import parse_args
        cfg = parse_args(mode="inference")
    finally:
        sys.argv = argv
    if rollout == "chunk_ar" and not getattr(cfg, "inference_chunk_size", 0):
        cfg.inference_chunk_size = CHUNK                                # the pipeline reads it at the top level
    if feeding.metric_speed is not None:    # read by the pipeline (inference.* first) and by ModelCamera alike
        cfg.inference.normalize_displacement = False
        cfg.inference.normalize_displacement_per_video = False
        cfg.inference.trajectory_translation_scale = feeding.translation_scale
    return cfg


def camera_flags(cfg):
    """The pipeline's _cam_flag: inference.*, then the root."""
    def flag(name):
        v = getattr(cfg.inference, name, None)
        return bool(v) if v is not None else bool(getattr(cfg, name, False))
    return {k: flag(k) for k in ("normalize_relative_to_first_frame", "normalize_framewise", "normalize_displacement",
                                 "normalize_displacement_per_video")}


def json_prompt(text, perspective="First-person view"):
    """The first segment of a world as whole-item JSON, the shape the repository's prompt shaping produces."""
    return json.dumps({"Style": "Photorealistic", "perspective_view": perspective, "segments": [{"visual_description_long": text}]})


def reset_session(p):
    """Starts a new world: resets the pipeline's session state (KV cache, counters, interactive caches) in place, as the repository's demo does."""
    if hasattr(p, "_finish_interactive_decode_phase"):
        p._finish_interactive_decode_phase()
    kv = getattr(p, "kv_cache", None)
    if kv is not None and hasattr(kv, "reset") and hasattr(kv, "cache_masks"):
        kv.reset()
    else:
        p.kv_cache = None
    kv = getattr(p, "kv_cache", None)
    p.cache_mask_list = kv.cache_masks if kv is not None and hasattr(kv, "cache_masks") else None
    p.cache_valid_len, p.global_frame_counter = 0, 0
    if hasattr(p, "segment_token_lengths"):
        p.segment_token_lengths.clear()
    p._interactive_cached_action_tails, p._interactive_cached_seg_video_ranges = [], []
    p._interactive_segment_count, p._interactive_segment_cache_tokens, p._interactive_segment_latent_lengths = 0, [], []
    for f in ("_reset_interactive_noise_state", "_reset_interactive_pixelumm_decode_state"):
        if hasattr(p, f):
            getattr(p, f)()


# --- frames and poses ----------------------------------------------------------------------------------------------------------

def to_frame(img):
    """[H, W, 3] uint8 1280 x 720 -> [704, 1280, 3]: the centre rows; other sizes are scaled to width 1280 first."""
    import cv2
    s = WIDTH / img.shape[1]
    rows = round(img.shape[0] * s)
    if rows < HEIGHT:
        raise ValueError(f"a {img.shape[1]} x {img.shape[0]} picture is too short for {WIDTH} x {HEIGHT} without padding")
    if s != 1:
        img = cv2.resize(img, (WIDTH, rows), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    top = (rows - HEIGHT) // 2
    return img[top:top + HEIGHT]


def crop_box(width=SOURCE_WIDTH, height=SOURCE_HEIGHT):
    s = WIDTH / width
    top = (round(height * s) - HEIGHT) // 2
    return [0.0, top / s, float(width), (top + HEIGHT) / s]


def resample(poses, fps_in, times):
    """Camera-to-world poses sampled at fps_in -> at the given times (s): rotations slerped, positions linear."""
    from scipy.spatial.transform import Rotation, Slerp
    poses = np.asarray(poses, np.float64)
    src = np.arange(len(poses)) / fps_in
    if times[0] < src[0] - 1e-9 or times[-1] > src[-1] + 1e-9:
        raise ValueError(f"times {times[0]:.3f}-{times[-1]:.3f} s outside the poses' {src[-1]:.3f} s")
    out = np.repeat(np.eye(4)[None], len(times), 0)
    out[:, :3, :3] = Slerp(src, Rotation.from_matrix(poses[:, :3, :3]))(np.clip(times, src[0], src[-1])).as_matrix()
    out[:, :3, 3] = np.stack([np.interp(times, src, poses[:, k, 3]) for k in range(3)], 1)
    return out


def chunk_frames_from_latents(state, latent_poses):
    """A chunk's 16 per-frame poses from its state pose (frame 16k - 1) and the poses at its latents' last frames (16k + 3 + 4i)."""
    from scipy.spatial.transform import Rotation, Slerp
    keys = np.concatenate([np.asarray(state, np.float64)[None], np.asarray(latent_poses, np.float64)])
    key_frames = np.array([-1, 3, 7, 11, 15], np.float64)
    t = np.arange(16, dtype=np.float64)
    out = np.repeat(np.eye(4)[None], 16, 0)
    out[:, :3, :3] = Slerp(key_frames, Rotation.from_matrix(keys[:, :3, :3]))(t).as_matrix()
    out[:, :3, 3] = np.stack([np.interp(t, key_frames, keys[:, k, 3]) for k in range(3)], 1)
    return out


def splice(path, chunk, frames):
    """A call's per-frame poses with chunk `chunk`'s 16 frames replaced by `frames` and the later frames re-propagated from the new end
    by the path's own steps: what the model would be given if the chunk were executed so."""
    path = np.asarray(path, np.float64)
    out = path.copy()
    s, e = 16 * chunk, 16 * (chunk + 1)
    out[s:e] = frames
    for i in range(e, len(out)):
        out[i] = out[i - 1] @ np.linalg.inv(path[i - 1]) @ path[i]
    return out


def pieces(chunks):
    """Replay pieces [(first chunk, last chunk)] of a video of `chunks` chunks: chunk 0 is only ever history."""
    return [(c, min(c + PIECE - 1, chunks - 1)) for c in range(1, chunks, PIECE)]


def causal_windows(chunks, first=0, context=CONTEXT_FRAMES):
    """[(a, b)] frame windows of DCAE-V encodes: chunk j (frames first + 16 j .. first + 16 j + 15) from max(0, its start - context)."""
    return [(max(0, first + 16 * j - context), first + 16 * (j + 1)) for j in range(chunks)]


@torch.no_grad()
def causal_chunk_latents(encoder, pixels, *, first=0, device=None, lock=None):
    """[1, 3, T, 704, 1280] uint8 frames -> [1, 32, 4 n, 22, 40] bf16, n = (T - first) // 16 chunks from frame `first`: chunk j's 4
    latents are the last 4 of DCAE-V's encode of its own window, so no latent depends on a later frame."""
    out = []
    for a, b in causal_windows((pixels.shape[2] - first) // 16, first):
        if (b - a) % STRIDE:
            raise ValueError(f"a DCAE-V window of {b - a} frames ({a}..{b - 1}): the encoder needs whole latents of {STRIDE}")
        with lock if lock is not None else contextlib.nullcontext():
            clip = pixels[:, :, a:b].to(device or next(encoder.parameters()).device, torch.bfloat16).div_(127.5).sub_(1)
            out.append(encoder.encode(clip).latent_dist.mode().to(torch.bfloat16)[:, :, -CHUNK:])
            del clip
    return torch.cat(out, dim=2)


def video_pixels(source_video):
    """A 24 fps clean.mp4 -> [1, 3, n, 704, 1280] uint8 on the CPU: whole chunks of 16 frames, the model's centre crop."""
    rgb = read_video(source_video)
    if rgb.shape[1:3] != (SOURCE_HEIGHT, SOURCE_WIDTH):
        raise ValueError(f"{source_video}: frames {rgb.shape[1:3]}, the crop assumes {SOURCE_HEIGHT} x {SOURCE_WIDTH}")
    n = len(rgb) // (CHUNK * STRIDE) * CHUNK * STRIDE
    return torch.from_numpy(np.stack([to_frame(f) for f in rgb[:n]])).permute(3, 0, 1, 2)[None]


@dataclass
class VideoInputs:
    """Prepared inputs of one video; nothing here is a DiT state."""
    clean: torch.Tensor                  # [1, 32, 124, 22, 40] bf16 on the GPU: the real frames' DCAE-V latents
    executed: np.ndarray                 # [496, 4, 4] float64: the real camera per frame (OpenCV camera-to-world, metres)
    seconds: float


# --- the camera the model reads --------------------------------------------------------------------------------------------------

class ModelCamera:
    """Per-frame camera-to-world poses of one call -> the model's camera tokens [1, latents, 64], by the repository's own code."""

    def __init__(self, pipe, cfg, work, camera_shift):
        self.pipe, self.cfg, self.flags, self.work = pipe, cfg, camera_flags(cfg), Path(work)
        self.camera_shift = camera_shift
        self.work.mkdir(parents=True, exist_ok=True)

    def rows(self, c2w):
        """A call's per-frame poses (its frames 0..N-1) -> the camera rows the model is given."""
        c2w = np.asarray(c2w)
        return c2w[3:] if self.camera_shift else c2w

    def npz(self, c2w, name):
        """The pipeline's camera file of a call whose frames have the poses c2w."""
        rows = np.asarray(self.rows(c2w), np.float32)
        K = np.repeat(np.array([[0.5, 0, 0.5], [0, 0.5 * WIDTH / HEIGHT, 0.5], [0, 0, 1]], np.float32)[None], len(rows), 0)
        f = self.work / f"{name}.npz"
        np.savez(f, c2ws=rows, intrinsics=K)
        return f

    @torch.no_grad()
    def tokens(self, c2w):
        """The model's camera tokens of a call whose frames have the poses c2w, read as the pipeline reads them."""
        from hymm.pipeline.bidirectional import prepare_camera_inputs
        inf = self.cfg.inference
        f = self.npz(c2w, f"tokens_{threading.get_ident()}")
        n = len(self.rows(c2w))
        cam = prepare_camera_inputs(str(f), n, self.pipe.device, self.pipe.target_dtype, segment_boundaries_pixel=[0, n],
                                    translation_scale=inf.trajectory_translation_scale, rotation_scale=inf.trajectory_rotation_scale, **self.flags)
        with torch.autocast(device_type="cuda", dtype=self.pipe.target_dtype):
            return self.pipe.transformer._preprocess_camera(norm_extrinsics=cam["norm_extrinsics"], norm_intrinsics=cam["norm_intrinsics"],
                                                            action_onehot=cam["action_onehot"], oh=GRID[1], ow=GRID[2],
                                                            img_dtype=self.pipe.target_dtype, device=self.pipe.device)

    def chunk_tokens(self, call_path, chunk, chunk_frames):
        """The model's 4 tokens of `chunk` (within the call) when its 16 frames are `chunk_frames` (spliced into the call's path)."""
        return self.tokens(splice(call_path, chunk, chunk_frames))[:, chunk * CHUNK:(chunk + 1) * CHUNK]


def candidate_camera(camera, poses, *, call_path, chunk, state, device):
    """[N, 5, 4, 4] (state frame and latents, relative to the state frame, metres) -> [N, 4, 880, 6 + 64] float32: the rays and the
    model's own camera token of each candidate."""
    poses = np.asarray(poses, np.float64)
    rays = plucker_field(poses[:, 1:], K_MODEL, HEIGHT, WIDTH, GRID[1:], device=device)
    toks = torch.stack([camera.chunk_tokens(call_path, chunk, chunk_frames_from_latents(state, state @ p[1:]))[0].float() for p in poses])
    return torch.cat([rays, toks[:, :, None].expand(-1, -1, rays.shape[2], -1)], dim=-1)


def after_camera(out):
    return out[0] if isinstance(out, tuple) else out


class StartTap:
    """Wraps transformer.forward_with_cache during a call that writes a start: keeps the tokens of its last whole-chunk cache write at
    action_modules[L], the commit pass the first decided chunk after the start reads."""

    def __init__(self, dit, layers):
        self.dit, self.orig, self.last, self.t = dit, dit.forward_with_cache, None, None
        self.tap = TokenTap(dit.action_modules, layers, extract=after_camera)

    def __enter__(self):
        self.dit.forward_with_cache = self
        return self

    def __exit__(self, *exc):
        self.dit.forward_with_cache = self.orig
        self.tap.close()

    def __call__(self, *args, **kw):
        if not kw.get("update_cache"):
            return self.orig(*args, **kw)
        if kw.get("camera_embedding") is None:
            raise RuntimeError("a start without its camera: no action module runs there, nothing to read at action_modules[L]")
        whole = kw["hidden_states"].shape[2] == CHUNK
        if whole:
            self.tap.begin(GRID)
        out = self.orig(*args, **kw)
        self.last, self.t = (self.tap.finish(), kw["timestep"].flatten()[0]) if whole else (None, None)
        return out


class CameraSwap:
    """Wraps transformer.forward_with_cache during one replay call: step-0 camera override, the tap and observe, teacher forcing.
    on_write(k, kw, write): wraps each cache write (chunk k of the video), for the commit tap."""

    def __init__(self, pipe, *, clean, chunk_offset, on_step0, wanted, on_write=None):
        self.pipe, self.dit = pipe, pipe.transformer
        self.clean, self.offset, self.on_step0, self.wanted, self.on_write = clean, chunk_offset, on_step0, wanted, on_write
        self.chunk, self.orig = -1, self.dit.forward_with_cache

    def __enter__(self):
        self.dit.forward_with_cache = self
        return self

    def __exit__(self, *exc):
        self.dit.forward_with_cache = self.orig

    def __call__(self, *args, **kw):
        if kw.get("update_cache"):
            x = kw["hidden_states"]
            s0 = (self.offset + self.chunk) * CHUNK
            real = self.clean[:, :, s0:s0 + x.shape[2]].to(x.device, torch.float32)
            if real.shape[2] != x.shape[2]:
                raise ValueError(f"cache write of chunk {self.chunk}: {x.shape[2]} latents, the video has {real.shape[2]} there")
            sigma = self.pipe.cache_noise_sigma
            x = x.clone()
            x[:, :real.shape[1]] = ((1 - sigma) * real + sigma * torch.randn_like(real)).to(x.dtype)
            kw["hidden_states"] = x
            if self.on_write is None:
                return self.orig(*args, **kw)
            return self.on_write(self.offset + self.chunk, kw, lambda: self.orig(*args, **kw))
        if getattr(self.dit, "_sparse_step_index", 0) != 0:
            raise RuntimeError("a replay runs one denoising step per chunk")
        self.chunk += 1
        k = self.offset + self.chunk
        if not self.wanted(k):
            return self.orig(*args, **kw)
        kw["camera_embedding"] = self.on_step0("camera", k, kw["camera_embedding"])
        out = self.orig(*args, **kw)
        self.on_step0("observe", k, kw["timestep"])
        return out


# --- the adapter -----------------------------------------------------------------------------------------------------------------

class _Flag(argparse.Action):
    """A feeding flag is also set on the adapter when it is parsed (run_config is read before load)."""

    def __init__(self, *args, switch=False, **kwargs):
        if switch:
            kwargs.update(nargs=0, default=False)
        self.switch = switch
        super().__init__(*args, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        value = True if self.switch else values
        setattr(namespace, self.dest, value)
        self.const.feeding = Feeding(**{**self.const.feeding.__dict__, self.dest: value})


class HYNomasAdapter(Adapter):
    """Hy-Nomas: replay (step 0 and commit pass), generation with a head or the external depth rule."""
    generator_id = "hy_nomas"
    field = FieldConfig(token_dim=TOKEN_DIM, camera_dim=CAMERA_DIM, frames=CHUNK, height=GRID[1], width=GRID[2],
                        camera_tokens="field", camera_hidden=128)
    model_commit = "5ecf99c981432d40a3e6b8788432f007f7acc28b"
    step0_t = STEP0_T
    weights = ()
    supports_commit_state = True
    feeding = Feeding()

    def add_feeding_arguments(self, ap):
        ap.add_argument("--metric-speed", type=float, default=None, action=_Flag, const=self,
                        help="model units per metre of a camera row's step (the displacement normalisation off); unset: the repository's feeding")
        ap.add_argument("--camera-shift", action=_Flag, const=self, switch=True,
                        help="a call's camera rows are its frames from the fourth on, so token j holds latent j's own frames")

    def add_arguments(self, ap):
        self.add_feeding_arguments(ap)

    def run_config(self):
        return {"camera": "rays+token", "window": 0, "tap_after_camera": 1, **self.feeding.switches()}

    def head_must_match(self, args):
        return {"window": 0, **feeding_of(args).switches(), "tap_after_camera": 1}

    def repo_config(self, steps):
        return repo_config(self.path("code"), self.path("weights"), self.path("text_encoder"), steps=steps, feeding=self.feeding)

    def load(self, args):
        self.args = args
        self.feeding = feeding_of(args)
        torch.set_grad_enabled(True)
        if str(self.path("code")) not in sys.path:
            sys.path.insert(0, str(self.path("code")))
        from hymm.models.vae import load_vae
        from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
        self.cfg = self.repo_config(1)
        self.device = torch.device("cuda")
        self.pipe = CPTCausalInferencePipeline(args=self.cfg, device=self.device, dit_model_path=self.cfg.dit_weight, world_mesh=None)
        self.pipe._decode_latents = lambda latents, num_frames, *a, **k: torch.zeros(3, 1, 1, 1)
        self.pipe.vae.to("cpu")                                         # the decoder: a replay never decodes
        torch.cuda.empty_cache()
        self.encoder, _, _, _ = load_vae(vae_path=str(self.cfg.vae.encoder_checkpoint), vae_precision="bf16", device=self.device)
        self.encoder_lock = threading.Lock()
        self.camera_model = ModelCamera(self.pipe, self.cfg, tempfile.mkdtemp(prefix="hy_nomas_camera_"), self.feeding.camera_shift)
        self.prefetcher = Prefetcher(self.prepare_video, workers=1 if args.prefetch else 0, depth=max(args.prefetch, 1))
        self.sources = {}

    def prepare_text(self, prompts):
        """Every prompt a replay gives the pipeline encoded once by its own _encode_single_prompt and kept on the CPU; the pipeline's calls
        are then served from them and the text encoder leaves the GPU."""
        pipe = self.pipe
        texts = [CONTINUE] + [json_prompt(p) for p in dict.fromkeys(prompts)]
        orig, memo = pipe._encode_single_prompt, {}
        keep = lambda out: tuple(None if t is None else t.cpu() for t in out)
        with torch.no_grad():
            for text in texts:
                memo[(text, "video", False)] = keep(orig(text, data_type="video"))
            memo[(self.cfg.inference.neg_prompt or "", "video", True)] = keep(orig(self.cfg.inference.neg_prompt or "", data_type="video", is_uncond=True))
            again = keep(orig(texts[-1], data_type="video"))
        same = all(a is None and b is None or torch.equal(a, b) for a, b in zip(again, memo[(texts[-1], "video", False)]))
        if not same:
            raise RuntimeError("the text encoder gave another result for the same prompt: its states cannot be encoded ahead")

        def served(prompt, data_type="video", is_uncond=False):
            key = (prompt, data_type, bool(is_uncond))
            if key not in memo:
                raise RuntimeError(f"prompt not encoded ahead: {key[1:]} {prompt[:120]!r}")
            return tuple(None if t is None else t.to(pipe.device) for t in memo[key])

        pipe._encode_single_prompt = served
        pipe.text_encoder.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def announce(self, rows_of_next_videos):
        keys = []
        for rows in rows_of_next_videos:
            self.sources[rows[0]["pair_id"]] = rows[0]
            keys.append(rows[0]["pair_id"])
        self.prefetcher.announce(keys)

    @torch.no_grad()
    def prepare_video(self, pair_id):
        """One video's causal latents (a worker thread, ahead of its replay) and its camera per frame (the episode's render/poses.npy)."""
        began = time.perf_counter()
        row = self.sources[pair_id]
        if row["pixel_fps"] != FPS:
            raise ValueError(f"{pair_id}: {row['pixel_fps']} fps frames, the model's clock is {FPS} fps (render24)")
        source = resolve_data(row["source_video"])
        clean = causal_chunk_latents(self.encoder, video_pixels(source), device=self.device, lock=self.encoder_lock)
        n = clean.shape[2] * STRIDE
        if tuple(clean.shape) != (1, 32, n // STRIDE, GRID[1], GRID[2]) or n % 16:
            raise ValueError(f"{pair_id}: latents {tuple(clean.shape)}")
        executed = resample(np.load(source.parent.parent / "render/poses.npy"), 16, np.arange(n) / FPS)
        return VideoInputs(clean=clean, executed=executed, seconds=time.perf_counter() - began)

    def check_rows(self, pair_id, by_chunk, inputs):
        """The rows' executed poses agree with the video's own camera (within 2 cm and 1 degree)."""
        chunks = inputs.clean.shape[2] // CHUNK
        if sorted(by_chunk) != list(range(chunks)):
            raise ValueError(f"{pair_id}: the frames give {chunks} chunks, the rows have {sorted(by_chunk)}")
        for k in (1, chunks // 2, chunks - 1):
            row = by_chunk[k]
            state = inputs.executed[16 * k - 1]
            got = np.linalg.inv(state) @ inputs.executed[[16 * k + 3, 16 * k + 7, 16 * k + 11, 16 * k + 15]]
            want = np.asarray(row["executed_poses"], np.float64)[1:]
            dt = np.abs(got[:, :3, 3] - want[:, :3, 3]).max()
            dr = np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", got[:, :3, :3], want[:, :3, :3]) - 1) / 2, -1, 1))).max()
            if dt > 0.02 or dr > 1.0:
                raise ValueError(f"{pair_id} chunk {k}: the rows' executed poses differ from render/poses.npy by {dt:.3f} m, {dr:.2f} deg")

    def replay(self, rows, observe, *, layers, seed, probe=None):
        pair_id = rows[0]["pair_id"]
        self.sources[pair_id] = rows[0]
        wanted_row = lambda r: is_decision(r) and (probe is None or bool(probe(r)))
        if not any(wanted_row(r) for r in rows):
            self.prefetcher.discard(pair_id)
            return 0, [0.0]
        inputs = self.prefetcher.get(pair_id)
        self.prepared(inputs)
        by_chunk = {r["chunk"]: r for r in rows}
        self.check_rows(pair_id, by_chunk, inputs)
        chunks = inputs.clean.shape[2] // CHUNK
        pipe, dit = self.pipe, self.pipe.transformer
        tap = TokenTap(dit.action_modules, layers, extract=after_camera)
        commit_tap = TokenTap(dit.action_modules, layers, extract=after_camera) if self.commit_state else None
        commits = {}                                                     # chunk -> the tokens of the previous chunk's cache write
        states, seconds, began = 0, [], [0.0]
        piece = {}                                                       # the current call's path and chunks

        def on_write(k, kw, write):
            """Chunk k's cache write: the commit pass chunk k + 1 is decided from, read when k + 1 is a state of this piece."""
            reading = commit_tap is not None and k + 1 <= piece["c1"] and kw["hidden_states"].shape[2] == CHUNK and wanted_row(by_chunk[k + 1])
            if reading:
                commit_tap.begin(GRID)
            out = write()
            if reading:
                commits[k + 1] = commit_tap.finish()
                self.first_commit(k + 1, kw["timestep"].flatten()[0], commits[k + 1], blocks=len(dit.double_blocks))
            return out

        def on_step0(what, k, value):
            nonlocal states
            row, c0, path = by_chunk[k], piece["c0"], piece["path"]
            state = inputs.executed[16 * k - 1]                          # the frame before the chunk
            if what == "camera":                                         # step 0 sees the command (candidate 0)
                began[0] = time.perf_counter()
                intent = chunk_frames_from_latents(state, state @ np.asarray(row["candidate_poses"][0], np.float64)[1:])
                tok = self.camera_model.chunk_tokens(path, k - c0, intent).to(value.dtype)
                tap.begin(GRID)
                return tok.expand(value.shape[0], -1, -1)
            hidden = tap.finish()
            self.first_probe(k, value.flatten()[0], hidden, blocks=len(dit.double_blocks))
            camera = candidate_camera(self.camera_model, row["candidate_poses"], call_path=path, chunk=k - c0, state=state, device=self.device)
            extra = {"commit_hidden": commits.pop(k, None)} if commit_tap is not None else {}
            with torch.enable_grad(), torch.autocast(device_type="cuda", enabled=False):
                observe(hidden, camera, torch.tensor(row["interval_ends"], device=self.device), row, **extra)
            del hidden, camera, extra
            states += 1
            seconds.append(time.perf_counter() - began[0])

        inf = self.cfg.inference
        common = dict(height=720, width=WIDTH, guidance_scale=inf.cfg_scale_video, camera_guidance_scale=inf.camera_cfg_scale, num_inference_steps=1,
                      flow_shift=inf.infer_flow_shift_video, seed=seed, negative_prompt=inf.neg_prompt, preserve_kv_cache=True,
                      trajectory_translation_scale=inf.trajectory_translation_scale, trajectory_rotation_scale=inf.trajectory_rotation_scale)
        perspective = "First-person view" if rows[0].get("view", "FP") == "FP" else "Third-person view"
        try:
            for c0, c1 in pieces(chunks):
                if not any(wanted_row(by_chunk[k]) for k in range(c0, c1 + 1)):
                    continue
                # a fresh world: the chunks before the piece as its start, with their camera; then the piece teacher-forced
                reset_session(pipe)
                s = max(0, 4 * c0 - START_LATENTS)
                piece.update(c0=c0, c1=c1, path=inputs.executed[16 * c0:16 * (c1 + 1)])
                commits.clear()
                with torch.no_grad():
                    start_tap = StartTap(dit, layers) if commit_tap is not None and wanted_row(by_chunk[c0]) else contextlib.nullcontext()
                    start_cam = self.camera_model.npz(inputs.executed[4 * s:16 * c0], f"start_{os.getpid()}")
                    with start_tap:
                        pipe.inference(prompts=[json_prompt(rows[0]["instruction"], perspective)], num_frames=4 * (4 * c0 - s),
                                       segment_frame_boundaries=[0, 4 * (4 * c0 - s)], camera_trajectory=str(start_cam), segment_has_camera=[True],
                                       reference_latents=inputs.clean[:, :, s:4 * c0].contiguous(), **common)
                    if isinstance(start_tap, StartTap):
                        if start_tap.last is None:
                            raise RuntimeError(f"{pair_id} piece {c0}: the start's last cache write is not a whole chunk")
                        commits[c0] = start_tap.last
                        self.first_commit(c0, start_tap.t, start_tap.last, source="start", blocks=len(dit.double_blocks))
                    cam = self.camera_model.npz(piece["path"], f"call_{os.getpid()}")
                    n = 16 * (c1 + 1 - c0)
                    with CameraSwap(pipe, clean=inputs.clean, chunk_offset=c0, on_step0=on_step0, wanted=lambda k: wanted_row(by_chunk[k]),
                                    on_write=on_write if commit_tap is not None else None):
                        pipe.inference(prompts=[CONTINUE], num_frames=n, segment_frame_boundaries=[0, n], camera_trajectory=str(cam),
                                       segment_has_camera=[True], **common)
        finally:
            tap.close()
            if commit_tap is not None:
                commit_tap.close()
            commits.clear()
            reset_session(pipe)
            self.release_gpu()
        return states, seconds or [0.0]

    def first_commit(self, chunk, t, hidden, *, source="call", **extra):
        """The first_probe log line for the commit pass, once per source (a call's own cache write, or the start's last)."""
        printed = self.__dict__.setdefault("first_commit_printed", set())
        if source not in printed:
            printed.add(source)
            print(json.dumps({"first_commit": {"source": source, "chunk": int(chunk), "t": float(t), "hidden_shape": list(hidden.shape),
                                               "hidden_dtype": str(hidden.dtype), **extra}}), flush=True)

    # --- generation interface ----------------------------------------------------------------------------------------------------

    def add_generation_arguments(self, ap):
        self.add_feeding_arguments(ap)

    def load_generation(self, args):
        self.args = getattr(self, "args", None) or args
        self.feeding = feeding_of(args)
        if str(self.path("code")) not in sys.path:
            sys.path.insert(0, str(self.path("code")))
        from hymm.models.vae import load_vae
        from hymm.pipeline.causal_ar import CPTCausalInferencePipeline
        torch.set_grad_enabled(False)
        cfg = self.repo_config(STEPS)
        device = torch.device("cuda")
        pipe = CPTCausalInferencePipeline(args=cfg, device=device, dit_model_path=cfg.dit_weight, world_mesh=None)
        encoder, _, _, _ = load_vae(vae_path=str(cfg.vae.encoder_checkpoint), vae_precision="bf16", device=device)
        camera = ModelCamera(pipe, cfg, tempfile.mkdtemp(prefix="hy_nomas_generation_"), self.feeding.camera_shift)
        return Generation(pipe=pipe, cfg=cfg, encoder=encoder, camera=camera, feeding=self.feeding)

    def prepare_generation_text(self, pipe, packages):
        """The pipeline encodes each call's text itself."""

    def generate_case(self, pipe, package, out, *, track, head=None, payload=None, observe_only=False, scorer=False, history_chunks=0,
                      seed=None, depth=None):
        if scorer:
            raise ValueError("Hy-Nomas: the head is scored inline (no asynchronous scorer)")
        if depth is not None and (track != "B" or head is not None or observe_only):
            raise ValueError("external depth: track B without a head, no observe-only")
        payload = payload or {}
        if head is not None and track == "B":
            trained = payload.get("run_config") or {}
            differ = {k: {"trained": trained[k], "now": v} for k, v in pipe.feeding.switches().items() if k in trained and trained[k] != v}
            if differ:
                raise ValueError(f"the head was trained under another camera feeding: {differ}")
        with self.held_gpu():
            return generate_case(self, pipe, package, out, track=track, head=head if track == "B" else None, payload=payload,
                                 observe_only=observe_only, seed=seed, depth=depth)


# --- generation ------------------------------------------------------------------------------------------------------------------

@dataclass
class Generation:
    pipe: object
    cfg: object
    encoder: object
    camera: ModelCamera
    feeding: Feeding

    @property
    def device(self):
        return self.pipe.device


class FramePath:
    """The camera per generated frame of one case in the package's world (OpenCV camera-to-world, metres): frame 0 the hand-off frame,
    then the generated frames. A correction replaces its chunk's 16 frames; the later commands stay relative steps."""

    def __init__(self, c2w):
        self.command = np.array(c2w, np.float64)
        self.c2w = self.command.copy()

    def call_path(self):
        return self.c2w[1:]

    def chunk(self, chunk):
        """[5, 4, 4] the chunk's state frame (the frame before it) and its latents' last frames."""
        s = 16 * chunk
        return self.c2w[[s, s + 4, s + 8, s + 12, s + 16]]

    def apply(self, chunk, latent_poses):
        s = 16 * chunk
        self.c2w[s + 1:s + 17] = chunk_frames_from_latents(self.c2w[s], latent_poses)
        for i in range(s + 17, len(self.c2w)):
            self.c2w[i] = self.c2w[i - 1] @ np.linalg.inv(self.command[i - 1]) @ self.command[i]


def path_candidates(path, chunk):
    """[34, 5, 4, 4] candidates of a generated chunk in the package's world (KEEP = the command); None for an in-place turn."""
    poses = path.chunk(chunk)
    if np.linalg.norm(poses[-1, :2, 3] - poses[0, :2, 3]) <= MIN_INTENT_M:
        return None
    steps = np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1)
    return candidate_world_poses(poses, np.concatenate([[0.0], np.cumsum(steps)]), first=False)


class HeadSwap:
    """Wraps transformer.forward_with_cache during the generated segment: each chunk's camera tokens come from the FramePath, and at a
    decided chunk's step 0 a collision head reads the tokens after the camera; a correction shifts the path and step 0 runs again on
    the same input. The later steps and the cache write read the chunk's final tokens.
    A prediction head reads the previous chunk's cache write (the first generated chunk: the start's last write) and decides before
    step 0; nothing is rolled back. The 2-latent tail is never decided."""

    def __init__(self, ad, gen, path, head, *, selection, observe_only):
        self.ad, self.gen, self.path, self.head = ad, gen, path, head
        self.dit = gen.pipe.transformer
        self.selection, self.observe_only = dict(selection or {}), observe_only
        self.fit = intent_fit(DELTA, RHO, w=W_LATERAL)
        self.chunk, self.tokens, self.records, self.orig = -1, None, [], self.dit.forward_with_cache
        self.tap = None
        self.prediction = head is not None and bool(head.config.command_input)
        self.commit = None                       # prediction: the tokens of this call's last cache write of a whole chunk
        if head is not None:
            cfg = head.config
            if (cfg.frames, cfg.height, cfg.width) != tuple(GRID):
                raise ValueError(f"the head's grid {(cfg.frames, cfg.height, cfg.width)} is not the model's {tuple(GRID)}")
            if cfg.camera_dim != CAMERA_DIM:
                raise ValueError(f"the head's camera width {cfg.camera_dim} is not the rays and token's {CAMERA_DIM}")
            self.tap = TokenTap(self.dit.action_modules, cfg.layers, extract=after_camera)

    def __enter__(self):
        self.dit.forward_with_cache = self
        return self

    def __exit__(self, *exc):
        self.dit.forward_with_cache = self.orig
        if self.tap is not None:
            self.tap.close()

    def chunk_tokens(self, chunk, n_latents):
        return self.gen.camera.tokens(self.path.call_path())[:, 4 * chunk:4 * chunk + n_latents]

    def candidates(self, chunk, n_latents):
        if self.head is None or n_latents != CHUNK:
            return None
        return path_candidates(self.path, chunk)

    def score(self, hidden, relative, chunk):
        device = hidden.device
        state = self.path.chunk(chunk)[0]
        camera = candidate_camera(self.gen.camera, relative, call_path=self.path.call_path(), chunk=chunk, state=state, device=self.gen.pipe.device)
        ends = torch.tensor([(i + 1) * STRIDE / FPS for i in range(CHUNK)], device=device)
        mask = torch.ones(1, len(DELTA), dtype=torch.bool, device=device)
        s = self.selection
        with torch.autocast(device_type=device.type, enabled=False):
            out = self.head(hidden[None], camera[None], ends[None], mask)
            a = torch.sigmoid(out["intent_logits"]) if "intent_logits" in out else self.fit[None].to(device)
            sel = choose(out["safe_logits"], a, mask, temperature=s.get("temperature", 1.0), keep_margin=s.get("keep_margin", 0.0),
                         rule=s.get("rule", "product"), tau=s.get("tau", 0.5))
        return {"choice": int(sel["choice"][0]), "p": sel["P"][0].float().cpu(), "r": sel["R"][0].float().cpu(), "a": a[0].float().cpu(),
                "clearance": out["clearance"][0].float().cpu()}

    def __call__(self, *args, **kw):
        emb = kw.get("camera_embedding")
        if kw.get("update_cache"):
            if self.tokens is not None and emb is not None:
                kw["camera_embedding"] = self.tokens.to(emb.dtype).expand(emb.shape[0], -1, -1)
            if not self.prediction:
                return self.orig(*args, **kw)
            reading = kw["hidden_states"].shape[2] == CHUNK
            if reading:
                self.tap.begin(GRID)
            out = self.orig(*args, **kw)
            self.commit = self.tap.finish() if reading else None
            if reading:
                self.ad.first_commit(self.chunk + 1, kw["timestep"].flatten()[0], self.commit, blocks=len(self.dit.double_blocks))
            return out
        if getattr(self.dit, "_sparse_step_index", 0) != 0:
            if self.tokens is not None and emb is not None:
                kw["camera_embedding"] = self.tokens.to(emb.dtype).expand(emb.shape[0], -1, -1)
            return self.orig(*args, **kw)
        self.chunk += 1
        c, n = self.chunk, emb.shape[1] if emb is not None else 0
        began = time.perf_counter()
        record = {"chunk": c, "latents": n}
        self.tokens = self.chunk_tokens(c, n)
        record["keep_rebuild_error"] = round(float((self.tokens.float() - emb[:1].float()).abs().max()), 6)
        candidates = self.candidates(c, n)
        commit, self.commit = self.commit, None
        if self.prediction:
            record["mode"] = "prediction"
            if commit is None:                                           # no whole-chunk write before it: not decided
                candidates = None
        record["decided"] = candidates is not None
        kw["camera_embedding"] = self.tokens.to(emb.dtype).expand(emb.shape[0], -1, -1)
        if candidates is not None and self.prediction:
            return self.predict(args, kw, record, c, n, candidates, commit, emb, began)
        if candidates is None:
            out = self.orig(*args, **kw)
            record.update(applied=False, seconds=round(time.perf_counter() - began, 3))
            self.records.append(record)
            return out
        relative = np.linalg.inv(candidates[:, :1]) @ candidates
        self.tap.begin(GRID)
        out = self.orig(*args, **kw)
        applied = False
        hidden = self.tap.finish()
        self.ad.first_probe(c, kw["timestep"].flatten()[0], hidden, blocks=len(self.dit.double_blocks))
        result = self.score(hidden, relative, c)
        choice = result["choice"]
        self.log_choice(record, result, choice)
        if choice != 0 and not self.observe_only:
            self.path.apply(c, candidates[choice, 1:])
            self.tokens = self.chunk_tokens(c, n)
            kw["camera_embedding"] = self.tokens.to(emb.dtype).expand(emb.shape[0], -1, -1)
            out = self.orig(*args, **kw)                                 # step 0 again on the same input, the corrected control
            applied = True
        record.update(applied=applied, seconds=round(time.perf_counter() - began, 3))
        self.records.append(record)
        return out

    def log_choice(self, record, result, choice):
        p, r = result["p"], result["r"]
        top = r.topk(5).indices.tolist()
        record.update(choice=NAMES[choice], p_keep=round(float(p[0]), 4), p_choice=round(float(p[choice]), 4), r_keep=round(float(r[0]), 4),
                      r_choice=round(float(r[choice]), 4), clearance_keep=round(float(result["clearance"][0]), 3),
                      top=[[NAMES[i], round(float(p[i]), 3), round(float(r[i]), 3)] for i in top], p_all=[round(float(x), 4) for x in p],
                      r_all=[round(float(x), 4) for x in r], a_all=[round(float(x), 4) for x in result["a"]])

    def predict(self, args, kw, record, c, n, candidates, commit, emb, began):
        """A prediction head's chunk: its choice on the previous chunk's cache write, then step 0 once on the chosen control."""
        result = self.score(commit, np.linalg.inv(candidates[:, :1]) @ candidates, c)
        choice = result["choice"]
        self.log_choice(record, result, choice)
        applied = choice != 0 and not self.observe_only
        if applied:
            self.path.apply(c, candidates[choice, 1:])
            self.tokens = self.chunk_tokens(c, n)
            kw["camera_embedding"] = self.tokens.to(emb.dtype).expand(emb.shape[0], -1, -1)
        out = self.orig(*args, **kw)
        record.update(applied=applied, seconds=round(time.perf_counter() - began, 3))
        self.records.append(record)
        return out


_EMPTY_CACHE = torch.cuda.empty_cache   # the real empty_cache: held_gpu() patches torch.cuda.empty_cache away, and the DA3 process needs the memory


def call_latents():
    """The generated call's latents so far (`output_latents` of the pipeline's _process_single_prompt, filled chunk by chunk before the
    chunk's cache write), read from its frame while forward_with_cache runs inside it."""
    f = sys._getframe(1)
    while f is not None and f.f_code.co_name != "_process_single_prompt":
        f = f.f_back
    latents = None if f is None else f.f_locals.get("output_latents")
    if latents is None:
        raise RuntimeError("external depth: the pipeline's output_latents are not reachable from forward_with_cache")
    return latents


class DepthSwap(HeadSwap):
    """The external depth method: no head. Before a decided chunk's step 0 (4 latents with a commanded move, chunk 0 too) the generated
    chunks so far are decoded with the pipeline's own decode and the DA3 process scores the 34 candidates on them; the decision rule
    keeps the command or stops; step 0 and the later steps then run once on the chosen control. Without scale the chunk keeps the
    command."""

    def __init__(self, ad, gen, path, *, geo, selection):
        super().__init__(ad, gen, path, None, selection=selection, observe_only=False)
        self.geo = geo

    def candidates(self, chunk, n_latents):
        if n_latents != CHUNK:
            return None
        return path_candidates(self.path, chunk)

    def decide(self, chunk, candidates, record):
        from nopeworld.inference.methods.external_depth import chunk_len16, clock16, geo_choose, to_rgb
        began = time.perf_counter()
        video = None
        if chunk > 0:                                                    # chunks 0..chunk-1 of this call: frames 1..16 chunk after the hand-off
            _EMPTY_CACHE()
            lat = call_latents()[:, :, :CHUNK * chunk]
            video = self.gen.pipe._decode_latents(lat, STRIDE * lat.shape[2])[0].float() * 2 - 1      # [3, T, H, W] in [-1, 1]
        decode_s = time.perf_counter() - began
        end16, index_of = clock16(0 if video is None else int(video.shape[1]), FPS, first_is_handoff=False)
        length = min(chunk_len16(CHUNK * STRIDE, FPS), end16)
        self.geo.size = [WIDTH, HEIGHT] if video is None else [int(video.shape[3]), int(video.shape[2])]
        relative = np.linalg.inv(candidates[:, :1]) @ candidates
        disps = relative[:, -1, :3, 3]                                   # the chunk's end in the camera of its state frame (metres)
        _EMPTY_CACHE()                                                   # room for DA3
        reply = self.geo.scores(chunk, end16, length, disps, lambda offs: to_rgb(video[:, [index_of(o) for o in offs]]))
        del video
        result = geo_choose(reply, self.selection)
        record.update(depth_seconds=round(time.perf_counter() - began, 2), decode_seconds=round(decode_s, 2), decoded_end16=end16,
                      no_scale=result is None, h_all=reply.get("h"), in_view=reply.get("in_view"),
                      depth={key: v for key, v in reply.items() if key not in ("h", "P", "A", "in_view", "slide")})
        if result is None:
            record.update(choice=NAMES[0])
            return 0
        choice, p, r = result["choice"], result["p"], result["r"]
        record.update(choice=NAMES[choice], p_keep=round(float(p[0]), 4), p_choice=round(float(p[choice]), 4), r_keep=round(float(r[0]), 4),
                      r_choice=round(float(r[choice]), 4), clearance_keep=round(float(result["clearance"][0]), 3),
                      p_all=[round(float(x), 4) for x in p], r_all=[round(float(x), 4) for x in r], a_all=[round(float(x), 4) for x in result["a"]])
        return choice

    def __call__(self, *args, **kw):
        if kw.get("update_cache") or getattr(self.dit, "_sparse_step_index", 0) != 0:
            return super().__call__(*args, **kw)
        emb = kw.get("camera_embedding")
        self.chunk += 1
        c, n = self.chunk, emb.shape[1] if emb is not None else 0
        began = time.perf_counter()
        record = {"chunk": c, "latents": n}
        self.tokens = self.chunk_tokens(c, n)
        record["keep_rebuild_error"] = round(float((self.tokens.float() - emb[:1].float()).abs().max()), 6)
        candidates = self.candidates(c, n)
        record["decided"] = candidates is not None
        applied = False
        if candidates is not None:
            choice = self.decide(c, candidates, record)
            if choice != 0:
                self.path.apply(c, candidates[choice, 1:])
                self.tokens = self.chunk_tokens(c, n)
                applied = True
        kw["camera_embedding"] = self.tokens.to(emb.dtype).expand(emb.shape[0], -1, -1)
        out = self.orig(*args, **kw)
        record.update(applied=applied, seconds=round(time.perf_counter() - began, 3))
        self.records.append(record)
        return out


def gen_frames(meta):
    """Frames generated after the hand-off: the package's own length on the 24 fps clock, whole latents (5 s: 120, 8 s: 192)."""
    fps = int(meta["fps"])
    after = int(meta["generate_frames"]) if "generate_frames" in meta else int(meta["frames"]) - 1 - int(meta.get("handoff_frame", 0))
    return round(after * FPS / fps) // STRIDE * STRIDE


def gen_plan(pkg):
    """(meta, the hand-off frame and the frames after it at 24 fps [n + 1, 4, 4], start kind, prompt, perspective, n)."""
    pkg = Path(pkg)
    meta = json.loads((pkg / "meta.json").read_text())
    poses = np.load(pkg / "poses.npy").astype(np.float64)
    fps, h = int(meta["fps"]), int(meta.get("handoff_frame", 0))
    n = gen_frames(meta)
    times = h / fps + np.arange(n + 1) / FPS
    c2w = resample(poses, fps, times) if fps != FPS else poses[h:h + n + 1]
    if len(c2w) != n + 1:
        raise ValueError(f"{meta['case']}: {len(c2w)} poses from the hand-off on, {n + 1} needed")
    start = "history" if (pkg / "history.mp4").is_file() and fps == FPS else "image"
    p = json.loads((pkg / "prompts.json").read_text()) if (pkg / "prompts.json").is_file() else {}
    prompt = (p.get("prompt") if isinstance(p, dict) else None) or (pkg / "prompt.txt").read_text().strip()
    third = meta.get("group") == "third_person"
    return meta, c2w, start, prompt.strip(), "Third-person view" if third else "First-person view", n


def history_start(n, name=""):
    """(cut, first) of a history of n frames: its first `cut` frames are dropped and its chunks start at frame `first` of the rest, so that
    the last min(4, n // 16) whole chunks end on the hand-off frame and every window has whole latents of 4 frames."""
    chunks = min(START_LATENTS // CHUNK, n // 16)
    if chunks < 1:
        raise ValueError(f"{name}: {n} history frames, at least 16 needed")
    first = n - 16 * chunks
    cut = first % STRIDE
    return cut, first - cut


def start_poses(pkg, meta, start, n):
    """The camera of a start's n frames: a history start's own poses ending on the hand-off frame; an image start the hand-off pose held."""
    poses = np.load(Path(pkg) / "poses.npy").astype(np.float64)
    h = int(meta.get("handoff_frame", 0))
    if start == "history":
        if h + 1 < n:
            raise ValueError(f"{meta['case']}: {h + 1} history poses, the start has {n} frames")
        return poses[h + 1 - n:h + 1]
    return np.repeat(poses[h][None], n, 0)


@torch.no_grad()
def encode_start(gen, pkg, start, case):
    """The start's latents in whole chunks (causal_chunk_latents): a history start's last chunks, at most START_LATENTS / 4, the hand-off
    frame last; an image start the picture repeated IMAGE_START_FRAMES times. Returns (latents, crop box)."""
    import cv2
    pkg = Path(pkg)
    if start == "history":
        frames = [to_frame(f) for f in read_video(pkg / "history.mp4")]
        cut, first = history_start(len(frames), case)
        px, size = np.stack(frames[cut:]), (SOURCE_WIDTH, SOURCE_HEIGHT)
    else:
        img = cv2.cvtColor(cv2.imread(str(pkg / "handoff.png")), cv2.COLOR_BGR2RGB)
        px, first, size = np.repeat(to_frame(img)[None], IMAGE_START_FRAMES, 0), 0, (img.shape[1], img.shape[0])
    lat = causal_chunk_latents(gen.encoder, torch.from_numpy(px).permute(3, 0, 1, 2)[None], first=first, device=gen.pipe.device)
    torch.cuda.empty_cache()
    return lat.contiguous(), crop_box(*size)


def case_seed(case):
    return int(hashlib.sha256(case.encode()).hexdigest()[:8], 16)


def generate_case(ad, gen, pkg, out, *, track, head, payload, observe_only, seed, depth=None):
    """One route or case: track A (the generator as it is) or B (a head deciding each chunk of 4 latents, or with `depth` the external
    depth rule). The video: the start's last decoded frame (the hand-off frame's reconstruction), then the generated frames, at 24 fps."""
    pkg, out = Path(pkg), Path(out)
    meta, c2w, start, prompt, perspective, n_gen = gen_plan(pkg)
    case = meta["case"]
    seed = seed if seed is not None else case_seed(case)
    pipe, inf = gen.pipe, gen.cfg.inference
    lat, crop = encode_start(gen, pkg, start, case)
    path = FramePath(c2w)
    common = dict(height=720, width=WIDTH, guidance_scale=inf.cfg_scale_video, camera_guidance_scale=inf.camera_cfg_scale, num_inference_steps=STEPS,
                  flow_shift=inf.infer_flow_shift_video, seed=seed, negative_prompt=inf.neg_prompt, preserve_kv_cache=True,
                  trajectory_translation_scale=inf.trajectory_translation_scale, trajectory_rotation_scale=inf.trajectory_rotation_scale)
    if depth is not None:      # history start: DA3 reads the real history; image start: metres from the floor
        geo = depth.geo(case, crop, i2v=start == "image")
        swap = DepthSwap(ad, gen, path, geo=geo, selection=payload.get("selection"))
    else:
        geo = None
        swap = HeadSwap(ad, gen, path, head, selection=payload.get("selection"), observe_only=observe_only)
    began = time.perf_counter()
    try:
        reset_session(pipe)
        n_ref = lat.shape[2] * 4
        start_tap = StartTap(pipe.transformer, head.config.layers) if swap.prediction else contextlib.nullcontext()
        start_cam = gen.camera.npz(start_poses(pkg, meta, start, n_ref), f"start_{case}")
        with start_tap:                                                  # a prediction head decides the first chunk from the start's last write
            first = pipe.inference(prompts=[json_prompt(prompt, perspective)], num_frames=n_ref, segment_frame_boundaries=[0, n_ref],
                                   camera_trajectory=str(start_cam), segment_has_camera=[True], reference_latents=lat, **common)
        if isinstance(start_tap, StartTap):
            swap.commit = start_tap.last
            if start_tap.last is not None:
                ad.first_commit(0, start_tap.t, start_tap.last, source="start", blocks=len(pipe.transformer.double_blocks))
        cam = gen.camera.npz(path.call_path(), f"gen_{case}")
        with swap:
            video = pipe.inference(prompts=[CONTINUE], num_frames=n_gen, segment_frame_boundaries=[0, n_gen], camera_trajectory=str(cam),
                                   segment_has_camera=[True], **common)
    finally:
        reset_session(pipe)
        torch.cuda.empty_cache()
    seconds = time.perf_counter() - began
    kept = torch.cat([first[0][:, -1:], video[0]], dim=1).float() * 2 - 1          # [C, 1 + n, H, W] in [-1, 1]
    out.mkdir(parents=True, exist_ok=True)
    write_video(out / f"{case}.mp4", kept, FPS)
    bad = [r for r in swap.records if r.get("keep_rebuild_error", 0) > 1e-2 and r["chunk"] == 0]
    info = {"case": case, "track": track, "seed": seed, "seconds": round(seconds, 1), "frames": int(kept.shape[1]), "fps": FPS,
            "size": list(kept.shape[2:]), "crop": crop, "prompt_sha256": prompt_sha256(json_prompt(prompt, perspective)), "start": start,
            "start_frames": int(lat.shape[2]) * 4, "selection": swap.selection, "history_chunks": 0, "observe_only": observe_only,
            "steps": STEPS, "interventions": sum(r.get("applied", False) for r in swap.records),
            "head_mode": None if head is None else "prediction" if swap.prediction else "collision", **gen.feeding.switches(),
            "keep_rebuild_error_chunk0": bad[0]["keep_rebuild_error"] if bad else None,
            "executed_controls": {"c2w_metres": path.c2w.tolist()}, "chunks": swap.records}
    if geo is not None:
        info.update(external_depth=True, depth_mode=geo.mode, depth_package=relative_to_roots(geo.package), depth_answers=geo.asked)
    (out / f"{case}.decisions.json").write_text(json.dumps(info, indent=1))
    return info, kept
