"""HY-WorldPlay (HY-World 1.5, ar_distilled_action_model): teacher-forced replay and generation with a head deciding each chunk.

The model has no rolling memory: its history is the latent tensor itself. At the start of every chunk up to 20 clean latents (all of
them up to chunk 5; from chunk 6 on the last 12, the first chunk's 4 and one chunk of 4 retrieved by field-of-view overlap with the
chunk's poses) go through all 54 blocks at t = 14 and their K/V replace the whole table (the pipeline's `_ar_rollout_inner`); the text
K/V (t = 0) are written once per video. The denoising forwards only read the table, so step 0 can run twice on it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from nopeworld.head.candidates import DELTA, NAMES, RHO, W_LATERAL, candidate_world_poses, intent_fit, resample_poses
from nopeworld.head.choose import choose
from nopeworld.head.labels import is_decision
from nopeworld.head.model import FieldConfig
from nopeworld.paths import resolve_data
from nopeworld.utils.rays import plucker_field, scale_intrinsics_cover
from nopeworld.utils.runtime import Prefetcher, TokenTap, read_video, write_video
from .base import Adapter, HistoryDoesNotFit, check_keep, prompt_sha256, relative_to_roots

HEIGHT, WIDTH = 480, 832
SOURCE_WIDTH, SOURCE_HEIGHT, SOURCE_HFOV = 1280, 720, 90.0        # the Unreal camera of the dataset and the routes
K_MODEL = scale_intrinsics_cover(SOURCE_WIDTH, SOURCE_HEIGHT, SOURCE_HFOV, HEIGHT, WIDTH)
CHUNK = 4
STRIDE = 4                               # video frames per latent; latent i >= 1 ends at frame 4 i, latent 0 is frame 0
FPS = 24
GRID = (CHUNK, HEIGHT // 16, WIDTH // 16)
TOKEN_DIM = 2048
RAY_DIM = 6
CAMERA_DIM = RAY_DIM + TOKEN_DIM         # Plucker rays and the frozen label embedding
STEPS = 4
STEP0_T = 1000.0                         # sigma = 1, pure noise (shift 5, `_ar_rollout_inner` timesteps[0])
CONTEXT_T = 14.0                         # stabilization_level - 1: clean latents
MOVE_NORM = 1e-4                         # generate.py `move_norm_valid`
MIN_INTENT_M = 1e-3                      # an in-place turn has nothing to correct
FRAME_WORKERS = 8
QUOTED = re.compile(r"\"(.*?)\"|“(.*?)”")     # the pipeline's `_extract_glyph_texts`


def first_output(output):
    """A block returns (tokens, K/V dict); the tokens of a denoising forward are the chunk's alone."""
    return output[0]


def last_chunk_tokens(output):
    """The tokens of a K/V write are those of all its latents in order, [1, N x 1560, D]: the last 4 latents' are the previous chunk's."""
    return output[0][:, -GRID[0] * GRID[1] * GRID[2]:]


def open_commit_tap(tap, selected, start):
    """Before the K/V write of the chunk at latent `start`: the write must end with the previous chunk's latents."""
    if selected[-CHUNK:] != list(range(start - CHUNK, start)):
        raise ValueError(f"the K/V write of the chunk at latent {start} ends with latents {selected[-CHUNK:]}, not the chunk before it")
    tap.begin(GRID)


def import_hy(code):
    """The repository's pipeline module with its infer state and a single-process parallel state set up by hand (no torchrun)."""
    sys.dont_write_bytecode = True
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import hyvideo.commons.infer_state as infer_state
    import hyvideo.commons.parallel_states as parallel_states
    if getattr(parallel_states, "__parallel_dims") is None:
        setattr(parallel_states, "__parallel_dims", SimpleNamespace(sp_enabled=False, sp=1, sp_rank=0))
    if getattr(infer_state, "__infer_state") is None:
        setattr(infer_state, "__infer_state", infer_state.InferState(sage_blocks_range=[]))
    from hyvideo.pipelines import worldplay_video_pipeline as module
    return module


# --- the model's control ---------------------------------------------------------------------------------------------------

_LABEL = np.zeros((2, 2, 2, 2), dtype=np.int64)
for _key, _value in {(1, 0, 0, 0): 1, (0, 1, 0, 0): 2, (0, 0, 1, 0): 3, (0, 0, 0, 1): 4, (1, 0, 1, 0): 5, (1, 0, 0, 1): 6,
                     (0, 1, 1, 0): 7, (0, 1, 0, 1): 8}.items():
    _LABEL[_key] = _value


def action_labels(c2w, threshold=MOVE_NORM):
    """[..., T, 4, 4] camera-to-world -> [..., T] the discrete label of every pose (9 x translation + rotation, the first one 0), as the
    repository's `pose_to_input` computes it. OpenCV camera: +z forward, +x right."""
    from scipy.spatial.transform import Rotation
    c2w = np.asarray(c2w, np.float64)
    rel = np.linalg.inv(c2w[..., :-1, :, :]) @ c2w[..., 1:, :, :]
    move = rel[..., :3, 3]
    norm = np.linalg.norm(move, axis=-1, keepdims=True)
    moving = norm[..., 0] > threshold
    angle = np.arccos(np.clip(move / np.where(norm > threshold, norm, 1.0), -1.0, 1.0)) * (180.0 / np.pi)
    forward, backward = moving & (angle[..., 2] < 60), moving & (angle[..., 2] > 120)
    right, left = moving & (angle[..., 0] < 60), moving & (angle[..., 0] > 120)
    euler = Rotation.from_matrix(rel[..., :3, :3].reshape(-1, 3, 3)).as_euler("xyz", degrees=True)
    euler = euler.reshape(*rel.shape[:-2], 3)
    turn_right, turn_left = euler[..., 1] > 5e-2, euler[..., 1] < -5e-2
    up, down = euler[..., 0] > 5e-2, euler[..., 0] < -5e-2
    labels = _LABEL[forward * 1, backward * 1, right * 1, left * 1] * 9 + _LABEL[turn_right * 1, turn_left * 1, up * 1, down * 1]
    return np.concatenate([np.zeros_like(labels[..., :1]), labels], axis=-1)


def scaled(poses, scale):
    """A copy with the translations in the model's unit."""
    poses = np.array(poses, np.float64)
    poses[..., :3, 3] *= scale
    return poses


def executed_path(rows, scale):
    """[T, 4, 4] camera-to-world of all latents of a real video in the model's unit, from the rows of all its chunks in order."""
    path, state = [], np.eye(4)
    for row in rows:
        path.append(state @ scaled(row["executed_poses"], scale)[1:])
        state = path[-1][-1]
    return np.concatenate(path)


def model_intrinsics(latents, k=K_MODEL):
    """[1, T, 3, 3] float64: fx / (2 cx), fy / (2 cy) and the principal point 0.5 per latent."""
    fx, fy, cx, cy = k
    K = torch.tensor([[fx / (2 * cx), 0.0, 0.5], [0.0, fy / (2 * cy), 0.5], [0.0, 0.0, 1.0]], dtype=torch.float64)
    return K.expand(1, latents, 3, 3)


def package_k(pkg, size):
    """(fx, fy, cx, cy) in pixels of the 480 x 832 model frame for a package's start picture of `size` (width, height), from the package's
    own intrinsics; K_MODEL itself when they agree within 1e-3 px (every Task 2 route).
      intrinsics_native.npy: K of the package's own picture.
      intrinsics.npy alone (Task 3): K of a (2 cx) x (2 cy) picture that is the start picture covered and centre-cropped to that size.
    A K that changes along the case, or a stretched K (fx != fy) without a native file, is refused."""
    width, height = size
    native = Path(pkg) / "intrinsics_native.npy"
    path = native if native.exists() else Path(pkg) / "intrinsics.npy"
    ks = np.load(path).astype(np.float64).reshape(-1, 4)
    if not np.allclose(ks, ks[0], atol=1e-4):
        raise ValueError(f"{path}: the intrinsics change along the case")
    fx, fy, cx, cy = ks[0]
    if native.exists():
        k_w, k_h = width, height
        if abs(2 * cx - width) > 1.0 or abs(2 * cy - height) > 1.0:
            raise ValueError(f"{path}: principal point ({cx}, {cy}) is not the centre of the {width} x {height} start picture")
    else:
        k_w, k_h = int(round(2 * cx)), int(round(2 * cy))
        if abs(fx - fy) > 0.01 * fx:
            raise ValueError(f"{path}: fx {fx} != fy {fy}: the K of a stretched picture, which cover-and-crop cannot map")
    s_k = max(k_w / width, k_h / height)
    s_m = max(WIDTH / width, HEIGHT / height)
    k = (fx / s_k * s_m, fy / s_k * s_m, WIDTH / 2.0, HEIGHT / 2.0)
    return K_MODEL if np.allclose(k, K_MODEL, atol=1e-3) else tuple(float(v) for v in k)


def candidate_camera(dit, poses, *, scale, device, dtype):
    """[N, 5, 4, 4] state frame and latents relative to the state frame, metres -> [N, 4, 1560, 6 + 2048] float32: the rays (metres,
    K_MODEL) and the frozen action_in embedding of the label the model would get."""
    poses = np.asarray(poses, np.float64)
    rays = plucker_field(poses[:, 1:], K_MODEL, HEIGHT, WIDTH, GRID[1:], device=device)
    labels = action_labels(scaled(poses, scale))[:, 1:]
    n, frames, tokens = rays.shape[:3]
    with torch.autocast(device_type="cuda", dtype=dtype):
        embedded = dit.action_in(torch.as_tensor(labels.reshape(-1), device=device).to(dtype))
    return torch.cat([rays, embedded.float().reshape(n, frames, 1, -1).expand(-1, -1, tokens, -1)], dim=-1)


# --- frames and latents -----------------------------------------------------------------------------------------------------

def crop_box(width=None, height=None):
    """[left, top, right, bottom] of the part of a width x height source picture that reaches the model, in its pixels (the cover is
    rounded to whole pixels before the centre crop, so the box has fractions)."""
    width = SOURCE_WIDTH if width is None else width
    height = SOURCE_HEIGHT if height is None else height
    scale = max(WIDTH / width, HEIGHT / height)
    cover_w, cover_h = int(round(width * scale)), int(round(height * scale))
    left, top = int(round((cover_w - WIDTH) / 2.0)), int(round((cover_h - HEIGHT) / 2.0))
    return [left * width / cover_w, top * height / cover_h, (left + WIDTH) * width / cover_w, (top + HEIGHT) * height / cover_h]


def check_start_image(case, size):
    """The model's K assumes a 16:9 start picture (within 0.2 %)."""
    width, height = size
    if abs(width * SOURCE_HEIGHT - height * SOURCE_WIDTH) > 0.002 * height * SOURCE_WIDTH:
        raise ValueError(f"{case}: start image {width} x {height} is not 16:9, the model's K assumes the {SOURCE_WIDTH} x {SOURCE_HEIGHT} picture's shape")


def to_model_frames(rgb):
    """[T, H, W, 3] uint8 -> [1, 3, T, 480, 832] in [-1, 1] on the CPU, frame by frame as the pipeline treats its reference image:
    PIL Lanczos to cover, centre crop."""
    from PIL import Image
    from torchvision import transforms
    height, width = rgb.shape[1:3]
    scale = max(WIDTH / width, HEIGHT / height)
    transform = transforms.Compose([transforms.Resize((int(round(height * scale)), int(round(width * scale))),
                                                      interpolation=transforms.InterpolationMode.LANCZOS),
                                    transforms.CenterCrop((HEIGHT, WIDTH)), transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    out = torch.empty(1, 3, len(rgb), HEIGHT, WIDTH)

    def fill(i):
        out[0, :, i] = transform(Image.fromarray(rgb[i]))
    with ThreadPoolExecutor(FRAME_WORKERS) as pool:
        list(pool.map(fill, range(len(rgb))))
    return out


def vae_encode(vae, pixels):
    """[1, 3, T, H, W] -> [1, 32, L, H / 16, W / 16] fp16: posterior mode x scaling factor under fp16 autocast. The VAE keeps its
    causal cache on itself: one caller at a time."""
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        latents = vae.encode(pixels).latent_dist.mode()
        latents.mul_(vae.config.scaling_factor)
    return latents


def build_cond(image_latent, latents):
    """[1, 33, T, H, W]: the start image's latent and the mask channel at latent 0, zeros elsewhere (fp32, as the pipeline's)."""
    cond = image_latent.repeat(1, 1, latents, 1, 1)
    cond[:, :, 1:] = 0.0
    mask = torch.zeros(1, 1, latents, *image_latent.shape[3:], device=image_latent.device)
    mask[:, :, 0] = 1.0
    return torch.cat([cond, mask], dim=1)


def vision_image(image):
    """The picture SigLIP sees: the start image cropped to its 480p bucket (480 x 848 for 16:9)."""
    from hyvideo.utils.data_utils import generate_crop_size_list, get_closest_ratio, resize_and_center_crop
    buckets = generate_crop_size_list(640, 16)
    ratios = np.array([round(float(h) / float(w), 5) for h, w in buckets])
    (height, width), _ = get_closest_ratio(image.shape[0], image.shape[1], ratios, buckets)
    return resize_and_center_crop(image, target_width=width, target_height=height)


def load_first_frame(png, first):
    """The replay's start image, uint8 [H, W, 3]: the lossless first frame when the video has one."""
    if Path(png).exists():
        from PIL import Image
        return np.array(Image.open(png).convert("RGB"))
    return first


# --- the forwards, as `_ar_rollout_inner` calls them ------------------------------------------------------------------------

def no_glyphs(device):
    """byT5's input for a prompt without quoted text: zeros and an empty mask."""
    return {"byt5_text_states": torch.zeros((1, 256, 1472), device=device),
            "byt5_text_mask": torch.zeros((1, 256), device=device, dtype=torch.int64)}


def text_kv(dit, embeds, mask, vision, glyphs, *, device, dtype):
    """K/V table with the prompt's and the start image's K/V (t = 0), once per video; the vision entries stay empty until the first write."""
    kv = [{"k_vision": None, "v_vision": None, "k_txt": None, "v_txt": None} for _ in dit.double_blocks]
    return dit(bi_inference=False, ar_txt_inference=True, ar_vision_inference=False,
               timestep_txt=torch.tensor([0]).to(device).to(dtype), text_states=embeds, encoder_attention_mask=mask,
               vision_states=vision, mask_type="i2v", extra_kwargs=glyphs, kv_cache=kv, cache_txt=True)


def write_context(dit, kv, latents, cond, viewmats, Ks, action, selected, *, device, dtype):
    """Replaces the vision K/V with those of the selected clean latents; the previous entries are dropped first (the forward never reads them)."""
    for entry in kv:
        entry["k_vision"] = entry["v_vision"] = None
    if not selected:
        return kv
    x = torch.cat([latents[:, :, selected], cond[:, :, selected]], dim=1)
    return dit(bi_inference=False, ar_txt_inference=False, ar_vision_inference=True, hidden_states=x,
               timestep=torch.full((len(selected),), CONTEXT_T, device=device, dtype=torch.float32), timestep_r=None,
               mask_type="i2v", return_dict=False, viewmats=viewmats[:, selected].to(device).to(dtype),
               Ks=Ks[:, selected].to(device).to(dtype), action=action[:, selected].to(device).to(dtype), kv_cache=kv,
               cache_vision=True, rope_temporal_size=x.shape[2], start_rope_start_idx=0)


def denoise_forward(dit, latent, cond, viewmats, Ks, action, kv, t, context, *, device, dtype):
    """One forward on the chunk's latents (reads the table only); RoPE's time index is the position after the `context` latents."""
    x = torch.cat([latent, cond], dim=1)
    return dit(bi_inference=False, ar_txt_inference=False, ar_vision_inference=True, hidden_states=x,
               timestep=torch.full((x.shape[2],), t, device=device, dtype=torch.float32), timestep_r=None, mask_type="i2v",
               return_dict=False, viewmats=viewmats.to(device).to(dtype), Ks=Ks.to(device).to(dtype),
               action=action.to(device).to(dtype), kv_cache=kv, cache_vision=False, rope_temporal_size=x.shape[2] + context,
               start_rope_start_idx=context)[0]


def retrieve(w2c, start, points, device):
    """Indices of the context latents of the chunk starting at latent `start` (the repository's retrieval on its device)."""
    from hyvideo.utils.retrieval_context import select_aligned_memory_frames
    return sorted(set(select_aligned_memory_frames(w2c, start, memory_frames=20, temporal_context_size=12, pred_latent_size=4,
                                                   points_local=points, device=device)))


@contextlib.contextmanager
def global_cpu_seed(seed):
    """The global CPU generator at this seed inside, as it was afterwards (the GPU's global generator is not touched)."""
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        yield


def fixed_points(seed, device):
    """The 50 000 sample points of the retrieval's overlap, fixed per video (drawn from the global CPU generator, as the pipeline)."""
    from hyvideo.utils.retrieval_context import generate_points_in_sphere
    with global_cpu_seed(seed):
        return generate_points_in_sphere(50000, 8.0).to(device)


# --- loading: straight to the GPU, the numbers of the repository's loaders ----------------------------------------------------

def copy_weights(model, path, dtype):
    """Every tensor of a safetensors file into a model already on its device, one at a time; the conversion runs on the CPU."""
    from safetensors import safe_open
    params = dict(model.named_parameters())
    seen = set()
    with safe_open(str(path), "pt", device="cpu") as f, torch.no_grad():
        for key in f.keys():
            if key not in params:
                raise KeyError(f"{path}: unexpected tensor {key}")
            tensor = f.get_tensor(key)
            if tensor.shape != params[key].shape:
                raise ValueError(f"{key}: {tuple(tensor.shape)} != {tuple(params[key].shape)}")
            params[key].copy_(tensor.to(dtype=dtype))
            seen.add(key)
    missing = set(params) - seen
    if missing:
        raise KeyError(f"{path} lacks {sorted(missing)[:5]} ({len(missing)} tensors)")


def load_dit(runtime, ckpt, device, dtype):
    """The action DiT in bf16, built on the meta device and filled from the checkpoint (the model has no buffers)."""
    from hyvideo.models.transformers.worldplay_1_5_transformer import HunyuanVideo_1_5_DiffusionTransformer
    config = HunyuanVideo_1_5_DiffusionTransformer.load_config(str(Path(runtime) / "transformer/480p_i2v"))
    with torch.device("meta"):
        model = HunyuanVideo_1_5_DiffusionTransformer.from_config(config)
        model.add_action_parameters()
    model.to(dtype=dtype)
    model.to_empty(device=device)
    copy_weights(model, ckpt, dtype)
    return model.eval().requires_grad_(False)


def load_vae(runtime, device):
    """The VAE in fp32 (fp16 autocast at use), straight to the GPU."""
    from hyvideo.models.autoencoders.hunyuanvideo_15_vae_w_cache import AutoencoderKLConv3D
    config = AutoencoderKLConv3D.load_config(str(Path(runtime) / "vae"))
    with torch.device("meta"):
        model = AutoencoderKLConv3D.from_config(config)
    model.to_empty(device=device)
    copy_weights(model, Path(runtime) / "vae/diffusion_pytorch_model.safetensors", torch.float32)
    return model.eval().requires_grad_(False)


def convert_through_cpu(module, dtype, device):
    """module.to(dtype), one tensor at a time through the CPU, as the repository's loaders convert (file bf16 -> fp32 -> dtype)."""
    with torch.no_grad():
        for tensor in itertools.chain(module.parameters(), module.buffers()):
            if tensor.is_floating_point():
                tensor.data = tensor.data.cpu().float().to(dtype).to(device)
    return module


def one_device(device):
    """from_pretrained's device_map for one device."""
    device = torch.device(device)
    return {"": "cpu" if device.type == "cpu" else f"{device.type}:{device.index or 0}"}


def load_llm(runtime, device):
    """The repository's TextEncoder (Qwen2.5-VL's language model, fp16) with its weights going straight to the device."""
    import hyvideo.models.text_encoders as module
    from transformers import AutoModel

    def straight(text_encoder_type, text_encoder_precision=None, text_encoder_path=None, logger=None, device=None):
        model = AutoModel.from_pretrained(text_encoder_path, low_cpu_mem_usage=True, torch_dtype=torch.bfloat16,
                                          device_map=one_device(device))
        if hasattr(model, "language_model"):
            model = model.language_model
        model.final_layer_norm = model.norm
        convert_through_cpu(model, module.PRECISION_TO_TYPE[text_encoder_precision], device)
        return model.requires_grad_(False), text_encoder_path
    original, module.load_text_encoder = module.load_text_encoder, straight
    try:
        return module.TextEncoder(text_encoder_type="llm", tokenizer_type="llm", max_length=1000,
                                  text_encoder_precision="fp16", text_encoder_path=str(Path(runtime) / "text_encoder/llm"),
                                  prompt_template=module.PROMPT_TEMPLATE["li-dit-encode-image-json"],
                                  prompt_template_video=module.PROMPT_TEMPLATE["li-dit-encode-video-json"],
                                  hidden_state_skip_layer=2, apply_final_norm=False, reproduce=False, device=device)
    finally:
        module.load_text_encoder = original


def load_siglip(runtime, device):
    """The repository's VisionEncoder (SigLIP, fp16), loaded the same way."""
    import hyvideo.models.vision_encoder as module
    from transformers import SiglipVisionModel

    def straight(vision_encoder_type, vision_encoder_precision=None, vision_encoder_path=None, logger=None, device=None):
        model = SiglipVisionModel.from_pretrained(vision_encoder_path, subfolder="image_encoder", low_cpu_mem_usage=True,
                                                  torch_dtype=torch.bfloat16, device_map=one_device(device))
        convert_through_cpu(model, module.PRECISION_TO_TYPE[vision_encoder_precision], device)
        return model.requires_grad_(False), vision_encoder_path
    original, module.load_vision_encoder = module.load_vision_encoder, straight
    try:
        return module.VisionEncoder(vision_encoder_type="siglip", vision_encoder_precision="fp16",
                                    vision_encoder_path=str(Path(runtime) / "vision_encoder/siglip"), device=device)
    finally:
        module.load_vision_encoder = original


def encode_prompts(encoder, prompts, device):
    """prompt -> {"embeds" [1, 1000, 3584] fp16, "mask" [1, 1000]} on the CPU, the pipeline's `encode_prompt` without guidance. A
    prompt with quoted text would need byT5, which is not loaded: such a prompt is refused."""
    out = {}
    with torch.no_grad():
        for prompt in dict.fromkeys(prompts):
            if QUOTED.search(prompt):
                raise ValueError(f"a prompt with quoted text needs byT5, which this adapter does not load: {prompt[:80]!r}")
            tokens = encoder.text2tokens(prompt, data_type="video", max_length=encoder.max_length)
            encoded = encoder.encode(tokens, data_type="video", device=device)
            out[prompt] = {"embeds": encoded.hidden_state.to(dtype=encoder.dtype).cpu(),
                           "mask": encoded.attention_mask.cpu()}
    return out


def text_file(cache, prompt):
    return Path(cache) / f"{hashlib.sha256(prompt.encode()).hexdigest()}.pt"


def read_text_cache(cache, prompts):
    """(prompt -> what encode_prompts gives, read from the cache; the prompts it does not have)."""
    def read(prompt):
        path = text_file(cache, prompt)
        if not path.exists():
            return prompt, None
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["prompt"] != prompt:
            raise ValueError(f"{path} holds another prompt")
        return prompt, {"embeds": saved["embeds"], "mask": saved["mask"]}
    with ThreadPoolExecutor(16) as pool:
        read_all = list(pool.map(read, dict.fromkeys(prompts)))
    return {p: text for p, text in read_all if text is not None}, [p for p, text in read_all if text is None]


def write_text_cache(cache, encoded):
    Path(cache).mkdir(parents=True, exist_ok=True)
    for prompt, text in encoded.items():
        path = text_file(cache, prompt)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save({"prompt": prompt, "embeds": text["embeds"], "mask": text["mask"]}, tmp)
        os.replace(tmp, path)


def encoded_prompts(cache, prompts, make_encoder, device):
    """What encode_prompts gives for prompts; with a cache directory the LLM is loaded only for prompts the cache does not have."""
    found, missing = read_text_cache(cache, prompts) if cache else ({}, list(dict.fromkeys(prompts)))
    if missing:
        encoder = make_encoder()
        new = encode_prompts(encoder, missing, device)
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        if cache:
            write_text_cache(cache, new)
        found.update(new)
    return found


# --- training replay ---------------------------------------------------------------------------------------------------------

@dataclass
class VideoInputs:
    """Prepared inputs of one video; nothing here is a DiT state."""
    clean: torch.Tensor                  # [1, 32, T, 30, 52] bf16, the real frames
    cond: torch.Tensor                   # [1, 33, T, 30, 52] fp32, latent 0 holds the start image and the mask
    vision: torch.Tensor                 # [1, 729, 1152] bf16, SigLIP of the start image
    seconds: float


class _Flag(argparse.Action):
    """A flag that changes the hidden states is also set on the adapter when it is parsed (run_config is read before load)."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        self.const.set_flag(self.dest, values)


class HYWorldPlayAdapter(Adapter):
    """HY-WorldPlay: replay (step 0 and commit pass), generation with a head or the external depth rule."""
    generator_id = "hy_worldplay"
    field = FieldConfig(token_dim=TOKEN_DIM, camera_dim=CAMERA_DIM, frames=CHUNK, height=GRID[1], width=GRID[2],
                        camera_tokens="field", camera_hidden=128)
    translation_scale = 0.4
    model_commit = "1588e1336e842b03b0a7860c654ebd7c46bb065e"
    step0_t = STEP0_T
    weights = ("model_path", "action_ckpt")
    supports_commit_state = True

    def set_flag(self, name, value):
        setattr(self, name, value)

    def add_model_arguments(self, ap, *, flag=True):
        extra = {"action": _Flag, "const": self} if flag else {}
        ap.add_argument("--model-path", type=Path, default=self.path("weights") / "runtime", help="VAE, LLM, SigLIP, scheduler, DiT config")
        ap.add_argument("--action-ckpt", type=Path, default=self.path("weights") / "ar_distilled_action_model/model.safetensors")
        ap.add_argument("--translation-scale", type=float, default=type(self).translation_scale, **extra,
                        help="metres -> the model's translation unit")
        ap.add_argument("--text-cache", type=Path, help="text embeddings on disk: the LLM is loaded only for a prompt it does not have")

    def add_arguments(self, ap):
        self.add_model_arguments(ap)

    def run_config(self):
        return {"translation_scale": self.translation_scale, "camera": "rays+label", "allow_retime": False}

    def head_must_match(self, args):
        return {"translation_scale": args.translation_scale, "allow_retime": False}

    def load(self, args):
        self.args = args
        self.set_flag("translation_scale", args.translation_scale)
        import_hy(self.path("code"))
        self.device = torch.device("cuda")
        self.target_dtype = torch.bfloat16
        self.transformer = load_dit(args.model_path, args.action_ckpt, self.device, self.target_dtype)
        self.vae = load_vae(args.model_path, self.device)
        self.vae_lock = threading.Lock()
        self.vision_encoder = load_siglip(args.model_path, self.device)
        self.make_llm = lambda: load_llm(args.model_path, self.device)
        self.contexts = {}
        self.prefetcher = Prefetcher(self.prepare_video, workers=1 if args.prefetch else 0, depth=max(args.prefetch, 1))
        self.sources = {}

    def prepare_text(self, prompts):
        """Every prompt encoded once with the repository's LLM, the results on the CPU; the encoder goes afterwards."""
        self.contexts.update(encoded_prompts(getattr(self.args, "text_cache", None), prompts, self.make_llm, self.device))

    def announce(self, rows_of_next_videos):
        keys = []
        for rows in rows_of_next_videos:
            self.check_pixels(rows[0])
            self.sources[rows[0]["pair_id"]] = rows[0]
            keys.append(rows[0]["pair_id"])
        self.prefetcher.announce(keys)

    def check_pixels(self, row):
        if row["pixel_fps"] != FPS:
            raise ValueError(f"{row['pair_id']}: {row['pixel_fps']} fps frames, the model's clock is {FPS} fps (render24)")

    @torch.no_grad()
    def prepare_video(self, pair_id):
        """Decode, resize and VAE-encode one video (a worker thread, ahead of its replay)."""
        began = time.perf_counter()
        row = self.sources[pair_id]
        source = resolve_data(row["source_video"])
        rgb = read_video(source)
        if rgb.shape[1:3] != (SOURCE_HEIGHT, SOURCE_WIDTH):
            raise ValueError(f"{pair_id}: frames {rgb.shape[1:3]}, the model's K assumes {SOURCE_HEIGHT} x {SOURCE_WIDTH}")
        latents = (len(rgb) - 1) // STRIDE + 1
        latents -= latents % CHUNK
        frames = rgb[:(latents - 1) * STRIDE + 1]
        pixels = to_model_frames(frames)
        start = load_first_frame(source.parent / "first_frame.png", frames[0])
        with self.vae_lock:
            clean = vae_encode(self.vae, pixels.to(self.device))
            image = vae_encode(self.vae, to_model_frames(start[None]).to(self.device))
        vision = self.vision_encoder.encode_images(vision_image(start)).last_hidden_state
        return VideoInputs(clean=clean.to(self.target_dtype), cond=build_cond(image, latents),
                           vision=vision.to(self.device, self.target_dtype), seconds=time.perf_counter() - began)

    def encode_candidates(self, poses):
        return candidate_camera(self.transformer, poses, scale=self.translation_scale, device=self.device, dtype=self.target_dtype)

    def replay(self, rows, observe, *, layers, seed, probe=None):
        pair_id = rows[0]["pair_id"]
        self.check_pixels(rows[0])
        self.sources[pair_id] = rows[0]
        wanted = lambda r: is_decision(r) and (probe is None or bool(probe(r)))
        if not any(wanted(r) for r in rows):
            self.prefetcher.discard(pair_id)
            return 0, [0.0]
        inputs = self.prefetcher.get(pair_id)
        self.prepared(inputs)
        by_chunk = {r["chunk"]: r for r in rows}
        latents = inputs.clean.shape[2]
        chunks = latents // CHUNK
        if sorted(by_chunk) != list(range(chunks)):
            raise ValueError(f"{pair_id}: the frames give {chunks} chunks, the rows have {sorted(by_chunk)}")
        probed = [k for k in range(chunks) if wanted(by_chunk[k])]
        device, dtype = self.device, self.target_dtype
        path = executed_path([by_chunk[k] for k in range(chunks)], self.translation_scale)
        w2c = np.linalg.inv(path)
        viewmats, action = torch.as_tensor(w2c)[None], torch.as_tensor(action_labels(path))[None]
        Ks = model_intrinsics(latents)
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(inputs.clean.shape, generator=generator, device=device, dtype=dtype)
        points = fixed_points(seed, device)
        text = self.contexts[rows[0]["instruction"]]
        dit, blocks = self.transformer, self.transformer.double_blocks
        tap = TokenTap(blocks, layers, extract=first_output)
        commit_tap = TokenTap(blocks, layers, extract=last_chunk_tokens) if self.commit_state else None
        dit.double_blocks = blocks[:max(layers) + 1]       # later blocks cannot change an earlier block's output or K/V
        given = {}
        reader = dit.register_forward_pre_hook(lambda module, args, kwargs: given.update(t=kwargs.get("timestep")), with_kwargs=True)
        # autocast around the DiT's forwards only, never around observe: a cached cast of a trainable head weight would outlive its update
        amp = lambda: torch.autocast(device_type="cuda", dtype=dtype)
        kv, states, seconds = None, 0, []
        try:
            with torch.no_grad():
                with amp():
                    kv = text_kv(dit, text["embeds"].to(device), text["mask"].to(device), inputs.vision,
                                 no_glyphs(device), device=device, dtype=dtype)
                for k in probed:
                    row = by_chunk[k]
                    began = time.perf_counter()
                    s, e = k * CHUNK, (k + 1) * CHUNK
                    state = path[s - 1] if k else np.eye(4)
                    intent = state @ scaled(row["candidate_poses"][0], self.translation_scale)[1:]
                    intent_w2c = np.linalg.inv(intent)
                    selected = retrieve(np.concatenate([w2c[:s], intent_w2c]), s, points, device) if k else []
                    labels = action_labels(np.concatenate([state[None], intent]))[1:]
                    reading = self.commit_state and k > 0          # chunk 0 has no write, so no state for a commit head
                    with amp():
                        if reading:
                            open_commit_tap(commit_tap, selected, s)
                        kv = write_context(dit, kv, inputs.clean, inputs.cond, viewmats, Ks, action, selected, device=device,
                                           dtype=dtype)
                        commit_hidden = commit_tap.finish() if reading else None
                        if reading:
                            self.first_commit(k, given["t"][0], commit_hidden, latents_in_pass=len(selected))
                        tap.begin(GRID)
                        denoise_forward(dit, noise[:, :, s:e], inputs.cond[:, :, s:e], torch.as_tensor(intent_w2c)[None],
                                        Ks[:, s:e], torch.as_tensor(labels)[None], kv, STEP0_T, len(selected), device=device,
                                        dtype=dtype)
                    hidden = tap.finish()
                    self.first_probe(k, given["t"][0], hidden, blocks=len(blocks))
                    camera = self.encode_candidates(row["candidate_poses"])
                    extra = {"commit_hidden": commit_hidden} if self.commit_state else {}
                    observe(hidden, camera, torch.tensor(row["interval_ends"], device=device), row, **extra)
                    del hidden, camera, commit_hidden, extra
                    states += 1
                    seconds.append(time.perf_counter() - began)
        finally:
            reader.remove()
            tap.close()
            if commit_tap is not None:
                commit_tap.close()
            dit.double_blocks = blocks
            del kv
            self.release_gpu()
        return states, seconds or [0.0]

    def first_commit(self, chunk, t, hidden, **extra):
        """The first_probe log line for the commit pass."""
        if not getattr(self, "first_commit_printed", False):
            self.first_commit_printed = True
            print(json.dumps({"first_commit": {"chunk": int(chunk), "t": float(t), "hidden_shape": list(hidden.shape), "hidden_dtype": str(hidden.dtype), **extra}}), flush=True)

    # --- generation interface --------------------------------------------------------------------------------------------------

    def add_generation_arguments(self, ap):
        self.add_model_arguments(ap, flag=False)

    def load_generation(self, args):
        self.args = getattr(self, "args", None) or args
        return load_generation(args, self.path("code"))

    def prepare_generation_text(self, pipe, packages):
        """All prompts encoded once, then the LLM leaves the GPU."""
        prompts = [json.loads((Path(p) / "prompts.json").read_text())["per_model"]["hy_worldplay"] for p in packages]
        if pipe.options.text_cache:
            release_llm(pipe, encoded_prompts(pipe.options.text_cache, prompts, lambda: load_llm(pipe.options.model_path, pipe.execution_device), pipe.execution_device))
        else:
            release_llm(pipe, encode_prompts(pipe.text_encoder, prompts, pipe.execution_device))

    def generate_case(self, pipe, package, out, *, track, head=None, payload=None, observe_only=False, scorer=False, history_chunks=0,
                      seed=None, depth=None):
        if scorer:
            raise ValueError("HY-WorldPlay: the head is scored inline (no asynchronous scorer)")
        with self.held_gpu():
            return generate_case(pipe, package, out, track=track, head=head, payload=payload, observe_only=observe_only,
                                 history_chunks=history_chunks, seed=seed, depth=depth)


# --- generation --------------------------------------------------------------------------------------------------------------

class CommandedPath:
    """The camera per latent of one clip in the package's world (OpenCV camera-to-world, metres): what is commanded and, after
    corrections, what is asked of the model. A correction replaces its chunk; the later commands stay relative steps."""

    def __init__(self, c2w, scale):
        self.command = np.array(c2w, np.float64)
        self.c2w = self.command.copy()
        self.scale = scale

    def model_c2w(self):
        """In the model's unit, the world being the camera of latent 0."""
        return scaled(np.linalg.inv(self.c2w[0]) @ self.c2w, self.scale)

    def w2c(self):
        return np.linalg.inv(self.model_c2w())

    def labels(self):
        return action_labels(self.model_c2w())

    def chunk(self, chunk):
        """[5, 4, 4] the chunk's state frame and latents; chunk 0's state frame is its first latent."""
        s, e = chunk * CHUNK, (chunk + 1) * CHUNK
        return np.concatenate([self.c2w[max(s - 1, 0)][None], self.c2w[s:e]])

    def apply(self, chunk, poses):
        s, e = chunk * CHUNK, (chunk + 1) * CHUNK
        self.c2w[s:e] = poses
        for i in range(e, len(self.c2w)):
            self.c2w[i] = self.c2w[i - 1] @ np.linalg.inv(self.command[i - 1]) @ self.command[i]


def chunk_candidates(path, chunk):
    """[34, 5, 4, 4] camera-to-world candidates of a chunk in the package's world, metres (KEEP = the command); None for an in-place turn."""
    poses = path.chunk(chunk)
    if np.linalg.norm(poses[-1, :2, 3] - poses[0, :2, 3]) <= MIN_INTENT_M:
        return None
    steps = np.linalg.norm(np.diff(poses[:, :2, 3], axis=0), axis=1)
    return candidate_world_poses(poses, np.concatenate([[0.0], np.cumsum(steps)]), first=chunk == 0)


class CalibratedRollout:
    """`_ar_rollout_inner` with a head deciding each chunk; without a head it is the pipeline's own loop (the previous K/V dropped before each write).

    Collision head: step 0 (t = 1000) runs with the commanded control and its tokens go to the head with the candidates. If the choice
    is not KEEP, the chunk's and the later poses and labels are replaced and step 0 runs again on the same noise; from chunk 6 on the
    retrieval runs again and the K/V are rewritten only if the retrieved set changed.
    Prediction head: decides before step 0 on the tokens of the chunk's K/V write; a correction only changes the control. Chunk 0 has no
    write before it and keeps the command.
    history: chunks before history_chunks take the real latents without any forward."""

    def __init__(self, pipe, head, *, path, w=W_LATERAL, selection=None, observe_only=False, history_latents=None, history_chunks=0):
        self.pipe, self.head, self.path = pipe, head, path
        self.selection, self.observe_only = dict(selection or {}), observe_only
        self.fit = intent_fit(DELTA, RHO, w=w)
        self.history_latents, self.history_chunks = history_latents, history_chunks
        self.records, self.tap = [], None
        self.prediction = head is not None and bool(head.config.command_input)
        self.early = self.prediction             # decides before step 0
        if head is not None:
            cfg = head.config
            if (cfg.frames, cfg.height, cfg.width) != tuple(GRID):
                raise ValueError(f"the head's grid {(cfg.frames, cfg.height, cfg.width)} is not the model's {tuple(GRID)}")
            if cfg.camera_dim != CAMERA_DIM:
                raise ValueError(f"the head's camera width {cfg.camera_dim} is not the rays and label's {CAMERA_DIM}")
            self.tap = TokenTap(pipe.transformer.double_blocks, cfg.layers, extract=last_chunk_tokens if self.prediction else first_output)

    def close(self):
        if self.tap is not None:
            self.tap.close()

    def candidates(self, chunk):
        """None for a chunk that is not decided: a history chunk, an in-place turn, or chunk 0 of a prediction head."""
        if self.head is None or chunk < self.history_chunks or (self.prediction and chunk == 0):
            return None
        return chunk_candidates(self.path, chunk)

    def score(self, hidden, relative, ends):
        """The head's choice for a chunk."""
        device = hidden.device
        camera = candidate_camera(self.pipe.transformer, relative, scale=self.path.scale, device=device, dtype=self.pipe.target_dtype)
        mask = torch.ones(1, len(DELTA), dtype=torch.bool, device=device)
        s = self.selection
        with torch.autocast(device_type=device.type, enabled=False):
            out = self.head(hidden[None], camera[None], ends[None], mask)
            a = torch.sigmoid(out["intent_logits"]) if "intent_logits" in out else self.fit[None].to(device)
            sel = choose(out["safe_logits"], a, mask, temperature=s.get("temperature", 1.0),
                         keep_margin=s.get("keep_margin", 0.0), rule=s.get("rule", "product"), tau=s.get("tau", 0.5))
        return {"choice": int(sel["choice"][0]), "p": sel["P"][0].float().cpu(), "r": sel["R"][0].float().cpu(),
                "a": a[0].float().cpu(), "clearance": out["clearance"][0].float().cpu()}

    def early_result(self, commit, relative, ends, **_):
        """A decision before step 0: the prediction head's, on the commit pass's tokens."""
        return self.score(commit, relative, ends)

    def apply(self, result, chunk, record, candidates):
        """Log the decision; True when it changed the chunk's poses."""
        choice = result["choice"]
        p, r = result["p"], result["r"]
        top = r.topk(5).indices.tolist()
        record.update(choice=NAMES[choice], p_keep=round(float(p[0]), 4), p_choice=round(float(p[choice]), 4),
                      r_keep=round(float(r[0]), 4), r_choice=round(float(r[choice]), 4),
                      clearance_keep=round(float(result["clearance"][0]), 3),
                      top=[[NAMES[i], round(float(p[i]), 3), round(float(r[i]), 3)] for i in top],
                      p_all=[round(float(x), 4) for x in p], r_all=[round(float(x), 4) for x in r],
                      a_all=[round(float(x), 4) for x in result["a"]])
        if choice == 0 or self.observe_only:
            return False
        self.path.apply(chunk, candidates[choice, 1:])
        return True

    def run(self, latents, timesteps, prompt_embeds, prompt_mask, vision_states, cond_latents, task_type, extra_kwargs, viewmats,
            Ks, action, device):
        pipe, dit = self.pipe, self.pipe.transformer
        if pipe.do_classifier_free_guidance:
            raise ValueError("the distilled model runs without guidance (few_step)")
        dtype = pipe.target_dtype
        amp = lambda: torch.autocast(device_type="cuda", dtype=dtype, enabled=pipe.autocast_enabled)
        glyphs = {key: extra_kwargs[key][0, None] for key in ("byt5_text_states", "byt5_text_mask")}
        with amp():
            kv = text_kv(dit, prompt_embeds[0, None], prompt_mask[0, None], vision_states[0, None], glyphs, device=device,
                         dtype=latents.dtype)
        n = pipe.chunk_latent_frames
        selected = []

        def write(chosen):
            with amp():
                return write_context(dit, kv, latents, cond_latents, viewmats, Ks, action, chosen, device=device,
                                     dtype=dtype)

        def pick(s):
            return retrieve(viewmats[0].cpu().numpy(), s, pipe.points_local, device)

        for chunk in range(pipe.chunk_num):
            s, e = chunk * n, (chunk + 1) * n
            if chunk < self.history_chunks:
                latents[:, :, s:e] = self.history_latents[:, :, s:e].to(latents.device, latents.dtype)
                continue
            began = time.perf_counter()
            candidates = self.candidates(chunk)
            commit = None
            if chunk > 0:
                selected = pick(s)
                reading = self.prediction and candidates is not None
                if reading:
                    open_commit_tap(self.tap, selected, s)
                kv = write(selected)
                if reading:
                    commit = self.tap.finish()
                pipe.scheduler.set_timesteps(pipe.num_inference_steps, device=device)

            def forward(t):
                with amp():
                    return denoise_forward(dit, latents[:, :, s:e], cond_latents[:, :, s:e], viewmats[:, s:e], Ks[:, s:e],
                                           action[:, s:e], kv, t, len(selected), device=device, dtype=dtype)

            def advance(velocity, t):
                x = pipe.scheduler.step(velocity, t, latents[:, :, s:e], return_dict=False)[0]
                latents[:, :, s:e] = x[:, :, -n:]

            record = {"chunk": chunk, "decided": candidates is not None}
            applied, rest = False, list(timesteps)
            if candidates is not None:
                keep = np.linalg.inv(scaled(np.linalg.inv(self.path.c2w[0]) @ candidates[0, 1:], self.path.scale))
                check_keep(float((torch.as_tensor(keep, dtype=viewmats.dtype, device=viewmats.device) - viewmats[0, s:e]).abs().max()), chunk, record)
                if not self.early:
                    self.tap.begin(GRID)
                    output = forward(rest[0])
                    hidden = self.tap.finish()
                relative = np.linalg.inv(candidates[:, :1]) @ candidates
                ends = torch.tensor([(i + (0 if chunk == 0 else 1)) * STRIDE / FPS for i in range(n)], device=device)

                def corrected():
                    """The control after a correction, and the K/V if the retrieval changed."""
                    nonlocal kv, selected
                    viewmats[0] = torch.as_tensor(self.path.w2c(), dtype=viewmats.dtype)
                    action[0] = torch.as_tensor(self.path.labels(), dtype=action.dtype)
                    chosen = pick(s) if chunk > 0 else []
                    record["retrieval_changed"] = chosen != selected
                    if chosen != selected:
                        selected = chosen
                        kv = write(selected)

                if self.early:
                    result, commit = self.early_result(commit, relative, ends, chunk=chunk, latents=latents, start=s, record=record), None
                    if result is not None and self.apply(result, chunk, record, candidates):
                        applied = True
                        corrected()
                else:
                    if self.apply(self.score(hidden, relative, ends), chunk, record, candidates):
                        applied = True
                        corrected()
                        output = forward(rest[0])
                    advance(output, rest[0])
                    rest = rest[1:]
            for t in rest:
                advance(forward(t), t)
            record.update(applied=applied, seconds=round(time.perf_counter() - began, 3))
            self.records.append(record)
        return latents


class DepthRollout(CalibratedRollout):
    """The external depth method: no head. Before each chunk's step 0 the committed latents are decoded and the DA3 process scores
    the 34 candidates on the frames after the hand-off; the decision rule keeps the command or stops; all 4 steps then run on the
    chosen control. Chunk 0 of an image-to-video case is decided too. Without scale the chunk keeps the command."""

    def __init__(self, pipe, *, geo, decode, keep_from, **kw):
        super().__init__(pipe, None, **kw)
        self.geo, self.decode, self.keep_from = geo, decode, keep_from
        self.early = True

    def candidates(self, chunk):
        if chunk < self.history_chunks:
            return None
        return chunk_candidates(self.path, chunk)

    def early_result(self, commit, relative, ends, *, chunk, latents, start, record):
        from nopeworld.inference.methods.external_depth import chunk_len16, clock16, geo_choose, to_rgb
        began = time.perf_counter()
        frames_so_far = 1 + STRIDE * (start - 1) if start else 0          # latent 0 is frame 0, latent i ends at frame 4 i
        video = None
        if frames_so_far - 1 > self.keep_from:
            torch.cuda.empty_cache()
            video = self.decode(latents[:, :, :start])[:, self.keep_from:]           # the hand-off's reconstruction first
        decode_s = time.perf_counter() - began
        n = 1 if video is None else int(video.shape[1])
        end16, index_of = clock16(n, FPS, first_is_handoff=True)
        length = min(chunk_len16(CHUNK * STRIDE, FPS), end16)
        self.geo.size = [WIDTH, HEIGHT] if video is None else [int(video.shape[3]), int(video.shape[2])]
        disps = np.asarray(relative)[:, -1, :3, 3]                       # the chunk's end in the camera of its state frame (metres)
        torch.cuda.empty_cache()                                         # room for DA3
        reply = self.geo.scores(chunk, end16, length, disps, lambda offs: to_rgb(video[:, [index_of(o) for o in offs]]))
        result = geo_choose(reply, self.selection)
        record.update(depth_seconds=round(time.perf_counter() - began, 2), decode_seconds=round(decode_s, 2), decoded_end16=end16,
                      no_scale=result is None, h_all=reply.get("h"), in_view=reply.get("in_view"),
                      depth={key: v for key, v in reply.items() if key not in ("h", "P", "A", "in_view", "slide")})
        if result is None:
            record.update(choice=NAMES[0])
        return result


def decode_latents(pipe, latents):
    """[1, C, t, h, w] latents -> [3, T, H, W] in [-1, 1]: the pipeline's own decode after the rollout."""
    vae = pipe.vae
    shift = getattr(vae.config, "shift_factor", None)
    latents = latents / vae.config.scaling_factor + (shift if shift else 0.0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=pipe.vae_dtype, enabled=pipe.vae_autocast_enabled):
        frames = vae.decode(latents, return_dict=False, generator=None)[0]
    return frames[0].float().clamp(-1, 1)


def pipeline_class(module):
    """The repository's pipeline with `_ar_rollout_inner` handed to a CalibratedRollout when one is set."""

    class Pipeline(module.HunyuanVideo_1_5_Pipeline):
        rollout = None
        options = None

        @property
        def device(self):
            return self.execution_device

        def _ar_rollout_inner(self, *args):
            if self.rollout is None:
                return super()._ar_rollout_inner(*args)
            return self.rollout.run(*args)

    return Pipeline


def release_llm(pipe, encoded):
    """From here on the pipeline takes its prompts from `encoded`; the LLM leaves the GPU."""
    def encode_prompt(prompt, device, *args, **kwargs):
        text = encoded[prompt]
        return text["embeds"].to(device), None, text["mask"].to(device), None
    pipe.encode_prompt = encode_prompt
    pipe.text_encoder = None
    gc.collect()
    torch.cuda.empty_cache()


class NoEncoder(torch.nn.Module):
    """What the pipeline's constructor reads of its text encoder when the prompts come from a text cache."""
    max_length = 1000


def load_generation(args, code):
    """The repository's pipeline with everything resident on the GPU; byT5 is not loaded (encode_prompts refuses quoted text)."""
    module = import_hy(code)
    device = torch.device("cuda")
    runtime = Path(args.model_path)
    text_cache = getattr(args, "text_cache", None)
    pipe = pipeline_class(module)(
        vae=load_vae(runtime, device), text_encoder=NoEncoder() if text_cache else load_llm(runtime, device),
        transformer=load_dit(runtime, args.action_ckpt, device, torch.bfloat16),
        scheduler=module.FlowMatchDiscreteScheduler.from_pretrained(str(runtime / "scheduler")), text_encoder_2=None,
        progress_bar_config=None, byt5_model=None, byt5_tokenizer=None, byt5_max_length=256, prompt_format=None,
        execution_device="cuda", vision_encoder=load_siglip(runtime, device), enable_offloading=False,
        **module.PIPELINE_CONFIGS["480p_i2v"])
    pipe.options = SimpleNamespace(translation_scale=args.translation_scale, text_cache=text_cache, model_path=runtime)
    return pipe


def case_plan(pkg, meta, history_chunks):
    """The clip of a case on the model's 24 fps clock: latent i at t = i / 6 s, the package's poses (camera-to-world) resampled.
    With history the clip starts so that the hand-off frame ends chunk history_chunks - 1; latent 0 is the world's origin."""
    poses = np.load(Path(pkg) / "poses.npy").astype(np.float64)
    handoff = meta["handoff_frame"]
    rate = FPS / meta["fps"]                                      # model frames per package frame
    generated = int(round(meta["generate_frames"] * rate))
    before = 16 * history_chunks - 4 if history_chunks else 0     # model frames before the hand-off frame
    if handoff - before / rate < 0:
        raise HistoryDoesNotFit(f"{meta['case']}: hand-off frame {handoff} leaves no room for {history_chunks} history chunks")
    latents = (before + generated) // STRIDE + 1
    latents += -latents % CHUNK
    times = handoff + (np.arange(latents) * STRIDE - before) / rate
    history_frames = np.round(handoff + (np.arange(before + 1) - before) / rate).astype(int)
    return dict(c2w=resample_poses(poses, np.arange(len(poses), dtype=np.float64), times), latents=latents,
                frames=(latents - 1) * STRIDE + 1, keep_from=before, kept=generated + 1, fps=FPS, history_frames=history_frames)


def case_seed(case):
    return int(hashlib.sha256(case.encode()).hexdigest()[:8], 16)


def generate_case(pipe, pkg, out, *, track, head=None, payload=None, observe_only=False, history_chunks=0, seed=None, depth=None):
    """One route or case: track A (the generator as it is) or B (a head, or the external depth rule with `depth`). The video is the
    hand-off frame's reconstruction and the frames after it; <out>/<case>.mp4 and <case>.decisions.json. The seed fixes the noise and
    the retrieval's points for every method."""
    from PIL import Image
    pkg, out = Path(pkg), Path(out)
    meta = json.loads((pkg / "meta.json").read_text())
    case = meta["case"]
    prompt = json.loads((pkg / "prompts.json").read_text())["per_model"]["hy_worldplay"]
    seed = seed if seed is not None else case_seed(case)
    if history_chunks and meta["fps"] != FPS:
        raise ValueError(f"{case}: {meta['fps']} fps history at the model's {FPS} fps clock: use the route's 24 fps package")
    plan = case_plan(pkg, meta, history_chunks)
    image, history_latents = Image.open(pkg / "handoff.png").convert("RGB"), None
    if history_chunks:
        video = read_video(pkg / "history.mp4")[plan["history_frames"]]
        image = Image.fromarray(video[0])
        with torch.no_grad():
            history_latents = vae_encode(pipe.vae, to_model_frames(video).to(pipe.execution_device))
    check_start_image(case, image.size)
    k_case = package_k(pkg, image.size)          # the model's Ks; the head's rays stay at K_MODEL
    path = CommandedPath(plan["c2w"], pipe.options.translation_scale)
    viewmats, action = torch.as_tensor(path.w2c())[None], torch.as_tensor(path.labels())[None]
    payload = payload or {}
    common = dict(path=path, w=payload.get("w", W_LATERAL), selection=payload.get("selection"), observe_only=observe_only,
                  history_latents=history_latents, history_chunks=history_chunks)
    if depth is not None:
        if track != "B" or head is not None:
            raise ValueError("external depth: track B without a head")
        geo = depth.geo(case, crop_box(*image.size), i2v=not history_chunks)
        rollout = DepthRollout(pipe, geo=geo, decode=lambda latents: decode_latents(pipe, latents), keep_from=plan["keep_from"], **common)
    else:
        geo = None
        rollout = CalibratedRollout(pipe, head if track == "B" else None, **common)
    generator = torch.Generator(device=pipe.execution_device).manual_seed(seed)
    pipe.rollout = rollout
    began = time.perf_counter()
    try:
        with global_cpu_seed(seed):
            output = pipe(prompt=prompt, aspect_ratio="16:9", video_length=plan["frames"], prompt_rewrite=False,
                          num_inference_steps=STEPS, enable_sr=False, generator=generator, seed=seed, reference_image=image,
                          output_type="pt", viewmats=viewmats, Ks=model_intrinsics(plan["latents"], k_case), action=action,
                          few_step=True, chunk_latent_frames=CHUNK, model_type="ar", user_height=HEIGHT, user_width=WIDTH)
    finally:
        pipe.rollout = None
        rollout.close()
        gc.collect()
        torch.cuda.empty_cache()
    seconds = time.perf_counter() - began
    video = output.videos[0] * 2 - 1
    kept = video[:, plan["keep_from"]:plan["keep_from"] + plan["kept"]]
    out.mkdir(parents=True, exist_ok=True)
    write_video(out / f"{case}.mp4", kept, plan["fps"])
    info = {"case": case, "track": track, "seed": seed, "seconds": round(seconds, 1), "frames": int(kept.shape[1]),
            "fps": plan["fps"], "size": list(kept.shape[2:]), "crop": crop_box(*image.size), "prompt_sha256": prompt_sha256(prompt),
            "selection": rollout.selection, "history_chunks": history_chunks, "observe_only": observe_only,
            "translation_scale": path.scale, "k_model": [round(float(v), 4) for v in k_case], "k_head": [round(float(v), 4) for v in K_MODEL],
            "interventions": sum(r.get("applied", False) for r in rollout.records),
            "kv_rewrites": sum(r.get("retrieval_changed", False) for r in rollout.records),
            "generator_state_sha256": hashlib.sha256(generator.get_state().cpu().numpy().tobytes()).hexdigest(),
            "executed_controls": {"c2w_metres": path.c2w.tolist(), "labels": path.labels().tolist()},
            "chunks": rollout.records}
    if geo is not None:
        info.update(external_depth=True, depth_mode=geo.mode, depth_package=relative_to_roots(geo.package), depth_answers=geo.asked)
    (out / f"{case}.decisions.json").write_text(json.dumps(info, indent=1))
    return info, kept
