"""LingBot-World 2.0: teacher-forced replay (training, Task 1) and closed-loop generation (Task 2, Task 3).

The chunk loop is LingBot-World's causal_fast sampler (wan/image2video.py) with a session called at fixed points: before each
chunk, after its step-0 forward (the head's state; a correction re-runs that forward only), around the commit pass (the
prediction head's state) and in place of a real chunk's denoising (teacher forcing). LingBot-World's own code is imported from
paths.code (its `wan` package); nothing of it is copied here except that loop. Replay and generation use the same sampler
settings (a 464 x 832 video, 29 x 52 tokens per latent).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nopeworld.head.candidates import DELTA, NAMES, RHO, W_LATERAL, candidate_world_poses, intent_fit, path_length
from nopeworld.head.choose import choose
from nopeworld.head.labels import is_decision
from nopeworld.head.model import FieldConfig
from nopeworld.paths import resolve_data
from nopeworld.utils.runtime import Prefetcher, write_video
from .base import Adapter, HistoryDoesNotFit, check_keep, prompt_sha256

FPS = 16
CHUNK = 3                                    # latents per chunk
CHUNK_FRAMES = 12                            # video frames per chunk
TIMESTEPS_INDEX = [0, 179, 358, 679]
SHIFT = 5.0
MAX_AREA = 480 * 832
LOCAL_ATTN_SIZE = 18
TEXT_LENGTH = 512
FRAMES = 81                                  # a Task 2 video: the hand-off frame's reconstruction + 80 generated frames
HISTORY_TAIL = 84                            # history protocol: clip frames after the hand-off (a whole number of chunks)
I2V_DECIDE_FROM = 1                          # without history the first decided chunk (chunk 0 has only the start image)
MIN_INTENT_M = 1e-3                          # a chunk without a commanded ground move is not decided
MODEL_COMMIT = "a3bd3c96416c77b5a5431a92c14da3c7b30d4cc7"
CAMERA_WEIGHTS = ("patch_embedding_wancamctrl.weight", "patch_embedding_wancamctrl.bias",
                  "c2ws_hidden_states_layer1.weight", "c2ws_hidden_states_layer1.bias",
                  "c2ws_hidden_states_layer2.weight", "c2ws_hidden_states_layer2.bias")


class ReplayDone(Exception):
    """The last decision state of a replay is read; the rest of the video is not needed."""


# --- inputs ---------------------------------------------------------------------------------------------------------------

def case_seed(case):
    return int(hashlib.sha256(case.encode("utf-8")).hexdigest()[:8], 16)


def chunk_frames(k):
    """Frames whose poses bound chunk k's three latents."""
    return [0, 0, 4, 8] if k == 0 else [12 * k - 4, 12 * k, 12 * k + 4, 12 * k + 8]


def chunk_candidates(poses, plen, *, first):
    """[N, 3, 4, 4] relative camera controls (metres) of one chunk's candidates; KEEP (index 0) is the command, chunk 0's first
    control the identity. poses: [4, 4, 4] commanded camera-to-world poses at chunk_frames(k); plen: their path lengths."""
    out = []
    for T in candidate_world_poses(poses, plen, first=first):
        c = [np.linalg.inv(T[j]) @ T[j + 1] for j in range(3)]
        if first:
            c[0] = np.eye(4)
        out.append(c)
    return np.array(out)


def candidate_disps(poses):
    """[34, 3] each candidate's ground displacement over a chunk in the camera at its start (OpenCV camera, world z up)."""
    poses = np.asarray(poses, np.float64)
    d_u = poses[-1, :2, 3] - poses[0, :2, 3]
    out = []
    for dl, r in zip(DELTA, RHO):
        a = np.deg2rad(dl)
        d = r * np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) @ d_u
        out.append(poses[0, :3, :3].T @ np.array([d[0], d[1], 0.0]))
    return np.array(out)


def fit_history(handoff_frame, history_chunks):
    """The history chunks a route holds, at most history_chunks."""
    fitted = min(history_chunks, (handoff_frame + 4) // CHUNK_FRAMES)
    if fitted < 1:
        raise HistoryDoesNotFit(f"hand-off frame {handoff_frame} leaves no room for a history chunk")
    return fitted


def read_video(path, limit=None):
    """uint8 [T, H, W, 3] RGB frames, decoded with OpenCV."""
    import cv2
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while limit is None or len(frames) < limit:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"could not decode video: {path}")
    return np.stack(frames)


def history_inputs(pkg, history_chunks):
    """History protocol: the clip starts where the hand-off frame ends chunk history_chunks - 1 and runs 84 frames past it (the last
    commanded step is continued). Returns (clip poses, clip intrinsics, history frames, first clip frame)."""
    meta = json.loads((pkg / "meta.json").read_text())
    kh = meta["handoff_frame"]
    start = kh + 1 - (CHUNK_FRAMES * history_chunks - 3)
    if start < 0:
        raise HistoryDoesNotFit(f"{pkg}: hand-off frame {kh} leaves no room for {history_chunks} history chunks")
    poses = np.load(pkg / "poses.npy").astype(np.float64)
    ks = np.load(pkg / "intrinsics.npy")
    end = kh + 1 + HISTORY_TAIL
    while len(poses) < end:
        poses = np.concatenate([poses, (poses[-1] @ np.linalg.inv(poses[-2]) @ poses[-1])[None]])
        ks = np.concatenate([ks, ks[-1:]])
    history = read_video(pkg / "history.mp4")[start:kh + 1]
    if len(history) != kh + 1 - start:
        raise ValueError(f"{pkg}: history.mp4 has too few frames")
    return poses[start:end], np.ascontiguousarray(ks[start:end]), history, start


def case_inputs(pkg, frames=FRAMES):
    """Without history: the commanded camera from the hand-off frame on (poses, intrinsics)."""
    meta = json.loads((pkg / "meta.json").read_text())
    kh = meta["handoff_frame"]
    poses = np.load(pkg / "poses.npy")[kh:kh + frames]
    ks = np.load(pkg / "intrinsics.npy")[kh:kh + frames]
    if len(poses) != frames:
        raise ValueError(f"{pkg}: {len(poses)} commanded poses after the hand-off, need {frames}")
    return np.ascontiguousarray(poses), np.ascontiguousarray(ks)


# --- the generator --------------------------------------------------------------------------------------------------------

def import_lingbot(code):
    """LingBot-World's `wan` package from the repository at `code`."""
    code = str(code)
    if code not in sys.path:
        sys.path.insert(0, code)
    from wan.configs import WAN_CONFIGS
    from wan.image2video import WanI2VCausal
    return WAN_CONFIGS, WanI2VCausal


def create_pipeline(code, weights, *, local_attn_size=LOCAL_ATTN_SIZE):
    """The causal_fast pipeline on one GPU: DiT in bf16, T5 kept on the CPU (prompts are encoded once on the GPU, TextContexts)."""
    configs, pipeline = import_lingbot(code)
    sink_size = min(6, max(0, local_attn_size - 3))
    pipe = pipeline(configs["i2v-A14B"], str(weights), t5_cpu=True, convert_model_dtype=True,
                    local_attn_size=local_attn_size, sink_size=sink_size)
    pipe.model.eval().requires_grad_(False)
    return pipe


def pack_camera(controls, intrinsics, height, width):
    """The native camera tensor [B, 384, F, H/8, W/8] of normalized relative controls [B, F, 4, 4] (or [F, 4, 4])."""
    from einops import rearrange
    from wan.utils.cam_utils import get_plucker_embeddings
    if controls.ndim == 3:
        controls = controls[None]
    if controls.ndim != 4 or controls.shape[-2:] != (4, 4):
        raise ValueError("controls must be [B,F,4,4]")
    b, frames = controls.shape[:2]
    if height % 16 or width % 16:
        raise ValueError("video dimensions must be multiples of 16")
    if intrinsics.ndim == 1:
        intrinsics = intrinsics[None, None].expand(b, frames, -1)
    elif intrinsics.ndim == 2:
        intrinsics = intrinsics[None].expand(b, -1, -1)
    if intrinsics.shape != (b, frames, 4):
        raise ValueError("intrinsics must be [4], [F,4] or [B,F,4]")
    if not torch.isfinite(controls).all() or not torch.isfinite(intrinsics).all():
        raise ValueError("nonfinite camera input")
    if (intrinsics[..., :2] <= 0).any():
        raise ValueError("camera focal lengths must be positive")
    fields = []
    for sample, ks in zip(controls.float(), intrinsics.float()):
        if torch.equal(ks, ks[:1].expand_as(ks)):
            fields.append(get_plucker_embeddings(sample, ks, height, width))
        else:                                # the native helper reads Ks[0]: varying intrinsics frame by frame
            fields.append(torch.cat([get_plucker_embeddings(pose[None], k[None], height, width)
                                     for pose, k in zip(sample, ks)]))
    rays = torch.stack(fields)
    return rearrange(rays, "b f (h p) (w q) c -> b (c p q) f h w", p=8, q=8)


class CameraFieldEncoder(nn.Module):
    """LingBot's frozen camera adapter per token: Plucker field -> 2 x 2 patch embedding -> residual MLP, [N, F, H*W, 5120]."""

    def __init__(self, weights):
        super().__init__()
        for name in CAMERA_WEIGHTS:
            self.register_buffer(name.replace(".", "__"), weights[name].detach().clone())

    def weight(self, name):
        return getattr(self, name.replace(".", "__"))

    @classmethod
    def from_model(cls, model):
        return cls({name: value for name, value in model.state_dict().items() if name in CAMERA_WEIGHTS})

    @torch.no_grad()
    def forward(self, controls, intrinsics, height, width, *, batch_size=4):
        from einops import rearrange
        if controls.ndim != 4:
            raise ValueError("encode controls as [N,F,4,4]")
        outputs = []
        for start in range(0, len(controls), batch_size):
            local_k = intrinsics[start:start + batch_size] if intrinsics.ndim == 3 else intrinsics
            packed = pack_camera(controls[start:start + batch_size], local_k, height, width)
            patches = rearrange(packed, "b c f (h p) (w q) -> b (f h w) (c p q)", p=2, q=2)
            dtype = self.weight(CAMERA_WEIGHTS[0]).dtype
            patch = F.linear(patches.to(dtype), self.weight(CAMERA_WEIGHTS[0]), self.weight(CAMERA_WEIGHTS[1]))
            middle = F.silu(F.linear(patch, self.weight(CAMERA_WEIGHTS[2]), self.weight(CAMERA_WEIGHTS[3])))
            embedded = patch + F.linear(middle, self.weight(CAMERA_WEIGHTS[4]), self.weight(CAMERA_WEIGHTS[5]))
            outputs.append(rearrange(embedded, "b (f t) c -> b f t c", f=controls.shape[1]))
        return torch.cat(outputs)


class HiddenTap:
    """Outputs of the tapped DiT blocks during one open window: [L, F, H, W, D] of the chunk's tokens."""

    def __init__(self, model, layers):
        self.layers = tuple(layers)
        self.active = False
        self.values = {}
        self.handles = [model.blocks[layer].register_forward_hook(self._hook(layer)) for layer in self.layers]

    def _hook(self, layer):
        def record(module, inputs, output):
            if not self.active:
                return
            if not isinstance(output, torch.Tensor) or output.shape[0] != 1:
                raise ValueError("the tap reads single-stream, unsharded DiT output")
            frames, height, width = self.grid
            tokens = frames * height * width
            if output.shape[1] < tokens:
                raise ValueError("tap grid exceeds the DiT token sequence")
            self.values[layer] = output[0, :tokens].detach().reshape(frames, height, width, output.shape[-1]).clone()
        return record

    def begin(self, grid):
        if self.active:
            raise RuntimeError("tap window already open")
        self.grid = tuple(grid)
        self.values = {}
        self.active = True

    def finish(self):
        self.active = False
        if set(self.values) != set(self.layers):
            raise RuntimeError("missing tapped blocks")
        result = torch.stack([self.values[layer] for layer in self.layers])
        self.values = {}
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.active = False
        self.values = {}


class TextContexts:
    """Every prompt of a run encoded once with T5 on the GPU, kept on the CPU; context(prompt) hands one to the chunk loop."""

    def __init__(self, pipe, prompts):
        self.pipe, self.contexts = pipe, {}
        encoder = pipe.text_encoder
        encoder.model.to(pipe.device)
        try:
            with torch.no_grad():
                for prompt in dict.fromkeys(prompts):
                    self.contexts[prompt_sha256(prompt)] = [t.cpu() for t in encoder([prompt], pipe.device)]
        finally:
            encoder.model.cpu()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def __len__(self):
        return len(self.contexts)

    def context(self, prompt):
        """This prompt's context on the GPU."""
        return [t.to(self.pipe.device) for t in self.contexts[prompt_sha256(prompt)]]


def prompt_context(pipe, contexts, prompt):
    """The prompt's T5 context on the GPU: from the run's TextContexts, else encoded as the pipeline does (T5 on the CPU)."""
    if contexts is not None and prompt_sha256(prompt) in contexts.contexts:
        return contexts.context(prompt)
    return [t.to(pipe.device) for t in pipe.text_encoder([prompt], torch.device("cpu"))]


class StreamDecoder:
    """The VAE decode of one committed chunk at a time, its causal feature cache carried between calls (the full decode's steps)."""

    def __init__(self, vae):
        self.vae, self.model = vae, vae.model
        self.feat = [None] * sum(type(m).__name__ == "CausalConv3d" for m in self.model.decoder.modules())

    @torch.no_grad()
    def push(self, z):
        """z [C, t, h, w], the next latents -> [3, frames, H, W] in [-1, 1]."""
        m, scale = self.model, self.vae.scale
        with torch.amp.autocast("cuda", dtype=self.vae.dtype):
            z = z.unsqueeze(0)
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, m.z_dim, 1, 1, 1) + scale[0].view(1, m.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]
            x = m.conv2(z)
            outs = [m.decoder(x[:, :, i:i + 1], feat_cache=self.feat, feat_idx=[0]) for i in range(x.shape[2])]
            return torch.cat(outs, 2).float().clamp_(-1, 1).squeeze(0)


def run_chunks(pipe, session, context, image, c2ws, intrinsics, *, frame_num, seed, chunk_size=CHUNK, shift=SHIFT,
               max_area=MAX_AREA, timesteps_index=TIMESTEPS_INDEX):
    """WanI2VCausal._generate_causal_fast (one stream, no offload) with the session's hooks. c2ws [N, 4, 4] OpenCV camera-to-world
    poses and intrinsics [N, 4] of the clip; context: the prompt's T5 context on the GPU. Returns the decoded video [3, T, H, W]
    in [-1, 1], or None for an offline session (a replay).

    This function follows LingBot-World's own sampling loop and is distributed under LingBot-World's license, not this repository's."""
    import torchvision.transforms.functional as TF
    from contextlib import contextmanager

    from einops import rearrange
    from wan.utils.cam_utils import compute_relative_poses, get_Ks_transformed, get_plucker_embeddings, interpolate_camera_poses
    if pipe.infer_mode != "causal_fast" or pipe.sp_size != 1:
        raise ValueError("the chunk loop runs single-stream causal_fast")
    len_c2ws = ((len(c2ws) - 1) // 4) * 4 + 1
    frame_num = ((frame_num - 1) // 4) * 4 + 1
    frame_num = min(frame_num, len_c2ws)
    c2ws = c2ws[:frame_num]
    if ((frame_num - 1) // 4 + 1) % chunk_size:
        raise ValueError("a clip needs complete latent chunks: 12 n - 3 frames for chunk_size 3")

    img = TF.to_tensor(image).sub_(0.5).div_(0.5).to(pipe.device)
    frames = frame_num
    h, w = img.shape[1:]
    aspect_ratio = h / w
    lat_h = round(np.sqrt(max_area * aspect_ratio) // pipe.vae_stride[1] // pipe.patch_size[1] * pipe.patch_size[1])
    lat_w = round(np.sqrt(max_area / aspect_ratio) // pipe.vae_stride[2] // pipe.patch_size[2] * pipe.patch_size[2])
    h = lat_h * pipe.vae_stride[1]
    w = lat_w * pipe.vae_stride[2]
    lat_f = (frames - 1) // pipe.vae_stride[0] + 1
    lat_f = int(lat_f - (lat_f % chunk_size))
    frames = (lat_f - 1) * 4 + 1
    max_seq_len = chunk_size * lat_h * lat_w // (pipe.patch_size[1] * pipe.patch_size[2])
    max_seq_len = int(math.ceil(max_seq_len / pipe.sp_size)) * pipe.sp_size
    pipe._cross_attn_initialized = False

    seed_g = torch.Generator(device=pipe.device)
    seed_g.manual_seed(seed)
    noise = torch.randn(16, lat_f, lat_h, lat_w, dtype=torch.float32, generator=seed_g, device=pipe.device)
    msk = torch.ones(1, frames, lat_h, lat_w, device=pipe.device)
    msk[:, 1:] = 0
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    pipe.scheduler.set_timesteps(pipe.num_train_timesteps, shift=shift)
    timesteps = pipe.scheduler.timesteps[timesteps_index]

    Ks = torch.from_numpy(intrinsics).float()
    Ks = get_Ks_transformed(Ks, height_org=480, width_org=832, height_resize=h, width_resize=w, height_final=h, width_final=w)
    Ks = Ks[0]
    len_c2ws = len(c2ws)
    len_c2ws_ = int((len_c2ws - 1) // 4) + 1
    len_c2ws_ = int(len_c2ws_ - (len_c2ws_ % chunk_size))
    c2ws_infer = interpolate_camera_poses(src_indices=np.linspace(0, len_c2ws - 1, len_c2ws), src_rot_mat=c2ws[:, :3, :3],
                                          src_trans_vec=c2ws[:, :3, 3], tgt_indices=np.linspace(0, len_c2ws - 1, len_c2ws_))
    raw_controls = compute_relative_poses(c2ws_infer, framewise=True, normalize_trans=False)
    c2ws_infer = session.normalize_controls(raw_controls)
    Ks = Ks.repeat(len(c2ws_infer), 1)
    c2ws_infer = c2ws_infer.to(pipe.device)
    Ks = Ks.to(pipe.device)
    plucker = get_plucker_embeddings(c2ws_infer, Ks, h, w)
    plucker = rearrange(plucker, "f (h c1) (w c2) c -> (f h w) (c c1 c2)", c1=int(h // lat_h), c2=int(w // lat_w))
    plucker = plucker[None, ...]
    plucker = rearrange(plucker, "b (f h w) c -> b c f h w", f=lat_f, h=lat_h, w=lat_w).to(pipe.param_dtype)

    y = pipe.vae.encode([
        torch.concat([F.interpolate(img[None].cpu(), size=(h, w), mode="bicubic").transpose(0, 1),
                      torch.zeros(3, frames - 1, h, w)], dim=1).to(pipe.device)])[0]
    y = torch.concat([msk, y])

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_model = getattr(pipe.model, "no_sync", noop_no_sync)
    model_args = pipe.model.config
    transformer_dtype = pipe.pipe_dtype
    frame_seqlen = int(noise.shape[-2] * noise.shape[-1] // 4)
    kv_size = frame_seqlen * pipe.local_attn_size if pipe.local_attn_size > -1 else frame_seqlen * lat_f
    head_dim = model_args.dim // model_args.num_heads
    local_num_heads = model_args.num_heads // pipe.sp_size
    self_kv_cache = pipe._initialize_self_kv_cache(num_layers=model_args.num_layers, shape=[1, kv_size, local_num_heads, head_dim],
                                                   dtype=transformer_dtype, device=pipe.device)
    cross_kv_cache = pipe._initialize_crossattn_cache(num_layers=model_args.num_layers,
                                                      shape=[1, TEXT_LENGTH, model_args.num_heads, head_dim],
                                                      dtype=transformer_dtype, device=pipe.device)
    session.prepare(controls=c2ws_infer, intrinsics=Ks, height=h, width=w, chunk_size=chunk_size, timesteps=timesteps,
                    shift=shift, seed=seed, frame_count=frames)
    with torch.amp.autocast("cuda", dtype=pipe.param_dtype), torch.no_grad(), no_sync_model():
        latents_chunk = noise.split(chunk_size, dim=1)
        condition_chunk = y.split(chunk_size, dim=1)
        plucker_chunk = plucker.split(chunk_size, dim=2)
        pred_latent_chunks = []
        for chunk_id in range(len(latents_chunk)):
            current_latent = latents_chunk[chunk_id]
            kwargs = {"context": [context[0]], "seq_len": max_seq_len, "y": [condition_chunk[chunk_id]],
                      "dit_cond_dict": {"c2ws_plucker_emb": plucker_chunk[chunk_id].chunk(1, dim=0)},
                      "kv_cache": self_kv_cache, "crossattn_cache": cross_kv_cache,
                      "current_start": chunk_id * chunk_size * frame_seqlen, "max_attention_size": kv_size,
                      "frame_seqlen": frame_seqlen}
            session.begin_chunk(chunk_id, current_latent, kwargs, seed_g)
            # a skipped chunk commits its real clean latent directly; the commit overwrites the K/V slots a step-0 forward fills,
            # but the first denoising call of a clip fills the text cross-attention cache, so chunk 0 is never skipped
            skip = session.skip_probe(chunk_id)
            if skip:
                if not pipe._cross_attn_initialized:
                    raise RuntimeError("the first chunk's denoising forward cannot be skipped")
                x0 = session.clean_chunk(chunk_id, pipe.device)
            for timestep_idx in range(0 if skip else len(timesteps)):
                timestep = torch.stack([timesteps[timestep_idx]]).to(pipe.device)
                noise_pred = pipe.model(x=[current_latent.to(pipe.device)], t=timestep,
                                        cross_attn_first_call=not pipe._cross_attn_initialized, **kwargs)[0]
                pipe._cross_attn_initialized = True
                if timestep_idx == 0:
                    noise_pred, teacher_latent = session.after_step0(noise_pred, current_latent, timestep, kwargs)
                    if teacher_latent is not None:
                        x0 = teacher_latent
                        break
                x0 = pipe._convert_flow_pred_to_x0(flow_pred=noise_pred, xt=current_latent, timestep=timesteps[timestep_idx],
                                                   scheduler=pipe.scheduler)
                if timestep_idx < len(timesteps) - 1:
                    current_latent = pipe.scheduler.add_noise(
                        x0, torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype), timesteps[timestep_idx + 1])
                else:
                    break
            pred_latent_chunks.append(x0)
            timestep = torch.stack([timesteps[-1] * 0.0]).to(pipe.device)
            session.before_commit(kwargs)
            pipe.model(x=[x0], t=timestep, cross_attn_first_call=False, **kwargs)
            session.after_commit()
        if session.offline:
            return None
        return pipe.vae.decode([torch.cat(pred_latent_chunks, dim=1)])[0]


# --- teacher-forced replay ------------------------------------------------------------------------------------------------

class ReplaySession:
    """One video teacher-forced: every chunk's step-0 forward runs with the row's command (the heads' state) and then the real
    clean latent is committed; steps 1-3 never run. Chunks without a probed state skip the step-0 forward. The commit pass is tapped
    when a commit head is in the run; its state waits for the next chunk."""

    offline = True

    def __init__(self, pipe, rows, video, observe, *, layers, camera, probe=None, commit_state=False, clock=None):
        self.pipe, self.video, self.layers = pipe, video, tuple(layers)
        self.probe = probe
        self.rows = {row["chunk"]: row for row in rows}
        self.tap = HiddenTap(pipe.model, self.layers)
        self.camera = camera
        self.scale = rows[0]["normalization_scale"]
        self.observe, self.states, self.timings = observe, 0, []
        self.commit_state, self.commit_hidden = commit_state, None
        self.mark = clock.mark if clock is not None else (lambda part: None)
        self.last_wanted = max((c for c, row in self.rows.items() if self.wanted(row)), default=0)

    def normalize_controls(self, raw):
        """Translations over the rows' normalization_scale (the clip's largest commanded step), so the replay's controls are the rows' own."""
        result = raw.clone()
        result[:, :3, 3] /= self.scale
        return result

    def wanted(self, row):
        return row is not None and is_decision(row) and (self.probe is None or self.probe(row))

    @torch.no_grad()
    def prepare(self, *, controls, intrinsics, height, width, chunk_size, timesteps, shift, seed, frame_count):
        self.mark("other")
        latents = chunk_size * (self.last_wanted + 1)          # the VAE is causal: a shorter clip's latents are the longer one's
        controls, intrinsics, frame_count = controls[:latents], intrinsics[:latents], (latents - 1) * 4 + 1
        self.controls, self.intrinsics = controls, intrinsics
        self.height, self.width, self.frames = height, width, chunk_size
        self.grid = (chunk_size, height // 16, width // 16)
        if frame_count > len(self.video):
            raise ValueError("source video shorter than the clip")
        rgb = torch.from_numpy(self.video[:frame_count]).permute(0, 3, 1, 2).to(self.pipe.device).float()
        rgb = F.interpolate(rgb / 127.5 - 1, size=(height, width), mode="bicubic", align_corners=False)
        self.clean = self.pipe.vae.encode([rgb.permute(1, 0, 2, 3)])[0].detach().cpu()
        if self.clean.shape[1] != len(controls):
            raise ValueError("VAE/control temporal alignment mismatch")
        self.video = None
        self.mark("vae_encode")

    def skip_probe(self, chunk):
        return self.probe is not None and chunk != 0 and not self.wanted(self.rows.get(chunk))

    def begin_chunk(self, chunk, latent, kwargs, generator):
        self.mark("other")
        self.chunk, self.started, self.actual_condition = chunk, time.perf_counter(), kwargs["dit_cond_dict"]
        if self.skip_probe(chunk):
            return
        row = self.rows.get(chunk)
        self.tap.begin(self.grid)
        if row is not None:
            controls = torch.tensor(row["intent_controls"], device=latent.device, dtype=torch.float32)
            ks = self.intrinsics[chunk * self.frames:(chunk + 1) * self.frames]
            packed = pack_camera(controls, ks, self.height, self.width).to(self.pipe.param_dtype)
            kwargs["dit_cond_dict"] = {"c2ws_plucker_emb": packed.chunk(1, dim=0)}

    def clean_chunk(self, chunk, device):
        return self.clean[:, chunk * self.frames:(chunk + 1) * self.frames].to(device)

    def candidate_cameras(self, row, device):
        ks = self.intrinsics[self.chunk * self.frames:(self.chunk + 1) * self.frames]
        controls = torch.tensor(row["candidate_controls"], device=device, dtype=torch.float32)
        cameras = self.camera(controls, ks, self.height, self.width)
        if not torch.isfinite(cameras).all():
            raise FloatingPointError("invalid camera embedding")
        self.mark("camera_field")
        return cameras

    @torch.no_grad()
    def after_step0(self, noise_pred, latent, timestep, kwargs):
        self.mark("step0_forward")
        hidden = self.tap.finish()
        if not torch.isfinite(hidden).all() or hidden.float().std() <= 1e-8:
            raise FloatingPointError("invalid DiT hidden state")
        row = self.rows.get(self.chunk)
        if self.wanted(row):
            cameras = self.candidate_cameras(row, latent.device)
            extra = {"commit_hidden": self.commit_hidden} if self.commit_state else {}
            self.observe(hidden, cameras, torch.tensor(row["interval_ends"], device=latent.device), row, **extra)
            self.states += 1
        if self.chunk >= self.last_wanted:
            raise ReplayDone
        return noise_pred, self.clean_chunk(self.chunk, latent.device)

    def before_commit(self, kwargs):
        self.mark("other")
        kwargs["dit_cond_dict"] = self.actual_condition
        if self.commit_state:
            self.tap.begin(self.grid)

    def after_commit(self):
        if self.commit_state:
            self.commit_hidden = self.tap.finish()
        self.mark("commit_forward")
        self.timings.append(time.perf_counter() - self.started)

    def close(self):
        self.tap.close()
        self.commit_hidden = None


# --- generation -----------------------------------------------------------------------------------------------------------

class RolloutSession:
    """One route or case. Without a head and without depth (track A) nothing is decided. A step-0 head decides after the chunk's
    step-0 forward and a correction re-runs it; a commit head decides before step 0 on the previous chunk's commit pass; the
    external depth method decides before step 0 on DA3's scores of the frames decoded so far (only KEEP and STOP). With history the
    first history_chunks chunks are the route's real frames, teacher-forced."""

    offline = False

    def __init__(self, pipe, poses, *, head=None, geo=None, history=None, history_chunks=0, selection=None, w=W_LATERAL,
                 observe_only=False, decide_from=0):
        self.pipe, self.head, self.geo = pipe, head, geo
        self.poses, self.records = np.asarray(poses, np.float64), []
        self.history, self.history_chunks = history, history_chunks
        self.observe_only, self.decide_from = observe_only, decide_from
        self.selection = dict(selection or {"temperature": 1.0, "keep_margin": 0.0})
        self.commit_state, self.commit_hidden, self.tap = False, None, None
        if head is not None:
            self.commit_state = bool(head.config.command_input)
            self.tap = HiddenTap(pipe.model, head.config.layers)
            self.camera = CameraFieldEncoder.from_model(pipe.model).to(pipe.device)
            self.fit = intent_fit(DELTA, RHO, w=w)
        if geo is not None:
            self.armed, self.committed, self.decoded = False, None, {}
            self.decoder = StreamDecoder(pipe.vae)
            self.hook = pipe.model.register_forward_pre_hook(self.take, with_kwargs=True)
            self.decode_seconds = self.check_seconds = 0.0

    def normalize_controls(self, raw):
        """As compute_relative_poses: translations over the clip's largest one."""
        self.scale = raw[:, :3, 3].norm(dim=-1).max()
        result = raw.clone()
        if self.scale > 0:
            result[:, :3, 3] /= self.scale
        return result

    def prepare(self, *, controls, intrinsics, height, width, chunk_size, timesteps, shift, seed, frame_count):
        if self.head is not None:
            cfg = self.head.config
            if (chunk_size, height // 16, width // 16) != (cfg.frames, cfg.height, cfg.width):
                raise ValueError("runtime grid / chunk size does not match the head")
        if frame_count != len(self.poses):
            raise ValueError(f"{frame_count} frames generated, {len(self.poses)} commanded poses")
        self.controls, self.intrinsics = controls, intrinsics
        self.height, self.width, self.frames = height, width, chunk_size
        self.grid = (chunk_size, height // 16, width // 16)
        if self.history is not None:
            rgb = torch.from_numpy(np.stack(self.history)).permute(0, 3, 1, 2).to(self.pipe.device).float()
            rgb = F.interpolate(rgb / 127.5 - 1, size=(height, width), mode="bicubic", align_corners=False)
            self.clean = self.pipe.vae.encode([rgb.permute(1, 0, 2, 3)])[0]
            if self.clean.shape[1] != self.history_chunks * chunk_size:
                raise ValueError(f"{len(self.history)} history frames gave {self.clean.shape[1]} latents, "
                                 f"need {self.history_chunks * chunk_size}")
            self.history = None

    def skip_probe(self, chunk):
        return 0 < chunk < self.history_chunks

    def clean_chunk(self, chunk, device):
        return self.clean[:, chunk * self.frames:(chunk + 1) * self.frames].to(device)

    def begin_chunk(self, chunk, latent, kwargs, generator):
        self.chunk, self.started = chunk, time.perf_counter()
        if chunk < self.history_chunks:
            return
        if self.geo is not None:
            with torch.no_grad():
                self.decide_depth(latent.device, kwargs)
        elif self.head is not None:
            if self.commit_state:
                with torch.no_grad():
                    self.decide(self.commit_hidden, latent.device, kwargs)
            else:
                self.tap.begin(self.grid)

    @torch.no_grad()
    def after_step0(self, noise_pred, latent, timestep, kwargs):
        if self.chunk < self.history_chunks:
            return noise_pred, self.clean_chunk(self.chunk, latent.device)
        if self.head is None or self.commit_state:
            return noise_pred, None
        if self.decide(self.tap.finish(), latent.device, kwargs):
            noise_pred = self.pipe.model(x=[latent], t=timestep, cross_attn_first_call=False, **kwargs)[0]
        self.records[-1]["seconds"] = round(time.perf_counter() - self.started, 3)
        return noise_pred, None

    def open_record(self, k):
        """Chunk k's record and commanded poses at chunk_frames(k); decided only with a commanded ground move from decide_from on."""
        p = self.poses[chunk_frames(k)]
        length = float(np.linalg.norm(p[-1, :2, 3] - p[0, :2, 3]))
        record = {"chunk": k, "intent_length_m": round(length, 4)}
        self.records.append(record)
        if length <= MIN_INTENT_M or k < self.decide_from:
            record.update(decided=False, choice="KEEP", applied=False)
            return record, None
        return record, p

    def candidates(self, k, p, device, record):
        """The candidates' controls normalized as the command (KEEP is the command itself), the commanded slice, its intrinsics."""
        n, fs = self.frames, chunk_frames(k)
        commanded = self.controls[k * n:(k + 1) * n]
        controls = torch.as_tensor(chunk_candidates(p, path_length(self.poses, fs), first=k == 0), dtype=torch.float32, device=device)
        controls[:, :, :3, 3] /= self.scale
        check_keep(float((controls[0] - commanded.float()).abs().max()), k, record)
        controls[0] = commanded
        return controls, commanded, self.intrinsics[k * n:(k + 1) * n]

    def apply(self, choice, controls, commanded, ks, kwargs, record):
        applied = choice != 0 and not self.observe_only
        if applied:
            packed = pack_camera(controls[choice], ks, self.height, self.width).to(self.pipe.param_dtype)
            kwargs["dit_cond_dict"] = {"c2ws_plucker_emb": packed.chunk(1, dim=0)}
            commanded.copy_(controls[choice].to(commanded.dtype))
        record.update(applied=applied, seconds=round(time.perf_counter() - self.started, 3))
        return applied

    def decide(self, hidden, device, kwargs):
        """The head's decision for this chunk; True when it changed the chunk's camera."""
        k, n = self.chunk, self.frames
        record, p = self.open_record(k)
        if p is None:
            return False
        if hidden is None:                   # a commit head at chunk 0: no pass before it
            record.update(decided=False, choice="KEEP", applied=False)
            return False
        controls, commanded, ks = self.candidates(k, p, device, record)
        camera = self.camera(controls, ks, self.height, self.width)
        ends = (torch.arange(1, n + 1, device=device).float() - (1.0 if k == 0 else 0.0)) * (4 / FPS)
        mask = torch.ones(1, len(NAMES), dtype=torch.bool, device=device)
        sel = self.selection
        with torch.autocast(device_type=device.type, enabled=False):
            out = self.head(hidden[None], camera[None], ends[None], mask)
            a = torch.sigmoid(out["intent_logits"]) if "intent_logits" in out else self.fit[None].to(device)
            chosen = choose(out["safe_logits"], a, mask, temperature=sel["temperature"], keep_margin=sel["keep_margin"],
                            rule=sel.get("rule", "product"), tau=sel.get("tau", 0.5))
        p_safe, r = chosen["P"][0].float().cpu(), chosen["R"][0].float().cpu()
        choice = int(chosen["choice"][0])
        top = r.topk(5).indices.tolist()
        record.update(decided=True, choice=NAMES[choice], p_keep=round(float(p_safe[0]), 4), p_choice=round(float(p_safe[choice]), 4),
                      r_keep=round(float(r[0]), 4), r_choice=round(float(r[choice]), 4),
                      top=[[NAMES[i], round(float(p_safe[i]), 3), round(float(r[i]), 3)] for i in top],
                      clearance_keep=round(float(out["clearance"][0, 0]), 3),
                      p_all=[round(float(x), 4) for x in p_safe], r_all=[round(float(x), 4) for x in r],
                      a_all=[round(float(x), 4) for x in a[0].float().cpu()])
        return self.apply(choice, controls, commanded, ks, kwargs, record)

    def take(self, module, args, kwargs):
        if self.armed:
            self.armed = False
            self.committed = kwargs["x"][0].detach()

    def before_commit(self, kwargs):
        if self.geo is not None:
            self.armed, self.committed = True, None
        if self.commit_state:
            self.tap.begin(self.grid)

    def after_commit(self):
        if self.commit_state:
            self.commit_hidden = self.tap.finish()
        if self.geo is not None:
            self.decode_committed()

    def decode_committed(self):
        from nopeworld.inference.methods.external_depth import lingbot_offsets, to_rgb
        if self.committed is None:
            raise RuntimeError(f"chunk {self.chunk}: the commit's latent was not seen")
        began = time.perf_counter()
        pictures = to_rgb(self.decoder.push(self.committed))
        first, last = lingbot_offsets(self.chunk, self.history_chunks)
        if len(pictures) != last - first + 1:
            raise RuntimeError(f"chunk {self.chunk}: {len(pictures)} frames decoded for clip offsets {first}..{last}")
        for i, o in enumerate(range(first, last + 1)):
            if o >= 1:
                self.decoded[o] = pictures[i]
        self.committed = None
        self.geo.size = [int(pictures.shape[2]), int(pictures.shape[1])]
        self.decode_seconds += time.perf_counter() - began

    def geometry(self, k, poses):
        """DA3's scores of chunk k's candidates from the frames decoded before it (none before the first generated chunk)."""
        from nopeworld.inference.methods.external_depth import lingbot_offsets
        began = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()                         # room for DA3
        if k - 1 >= self.history_chunks:
            first, last = lingbot_offsets(k - 1, self.history_chunks)
            decoded_end, chunk_len = last, last - max(first - 1, 0)
        else:
            decoded_end, chunk_len = 0, CHUNK_FRAMES
        reply = self.geo.scores(k - self.history_chunks, decoded_end, chunk_len, candidate_disps(poses),
                                lambda offsets: np.stack([self.decoded[o] for o in offsets]))
        self.check_seconds += time.perf_counter() - began
        return reply

    def decide_depth(self, device, kwargs):
        """The external depth method's decision for this chunk: the labels' P and A on DA3's geometry in the collision head's decision rule."""
        from nopeworld.inference.methods.external_depth import geo_choose
        k = self.chunk
        record, p = self.open_record(k)
        if p is None:
            return False
        controls, commanded, ks = self.candidates(k, p, device, record)
        reply = self.geometry(k, p)
        geo = {key: v for key, v in reply.items() if key not in ("h", "P", "A", "in_view")}
        chosen = geo_choose(reply, self.selection, device)
        if chosen is None:
            record.update(decided=True, choice="KEEP", applied=False, no_scale=True, geo=geo,
                          seconds=round(time.perf_counter() - self.started, 3))
            return False
        p_t, r = chosen["p"], chosen["r"]
        choice = chosen["choice"]
        top = r.topk(5).indices.tolist()
        record.update(decided=True, choice=NAMES[choice], p_keep=round(float(p_t[0]), 4), p_choice=round(float(p_t[choice]), 4),
                      r_keep=round(float(r[0]), 4), r_choice=round(float(r[choice]), 4),
                      top=[[NAMES[i], round(float(p_t[i]), 3), round(float(r[i]), 3)] for i in top], h_all=reply["h"], p_all=reply["P"],
                      a_all=reply["A"], in_view=reply["in_view"], choice_in_view=bool(reply["in_view"][choice]), no_scale=False, geo=geo,
                      stop_only=True)
        return self.apply(choice, controls, commanded, ks, kwargs, record)

    def close(self):
        if self.tap is not None:
            self.tap.close()
        if self.geo is not None:
            self.hook.remove()
            self.decoder = self.committed = None


def generate_case(pipe, pkg, out, *, track, head=None, payload=None, observe_only=False, scorer=False, history_chunks=0, seed=None,
                  depth=None):
    """One route (history_chunks > 0: the route's history teacher-forced, fitted to what it holds) or one image-to-video case
    (history_chunks = 0: the package's own length, decided from chunk 1). Track A is the generator as it is; track B has a head or,
    with `depth`, the external depth rule. Writes <out>/<case>.mp4 (the hand-off frame's reconstruction, then the generated frames)
    and, for track B, <case>.decisions.json. Returns (log, video)."""
    from PIL import Image

    pkg, out = Path(pkg), Path(out)
    if track not in ("A", "B"):
        raise ValueError(f"unknown track {track!r}")
    if depth is not None and (track != "B" or head is not None):
        raise ValueError("external depth: track B without a head")
    if track == "B" and head is None and depth is None:
        raise ValueError("track B needs a head or the external depth method")
    meta = json.loads((pkg / "meta.json").read_text())
    case = meta["case"]
    payload = payload or {}
    selection = dict(payload.get("selection") or {"temperature": 1.0, "keep_margin": 0.0})
    seed = case_seed(case) if seed is None else int(seed)
    prompt = (pkg / "prompt.txt").read_text().strip()
    if history_chunks:
        history_chunks = fit_history(meta["handoff_frame"], history_chunks)
        poses, intrinsics, history, _ = history_inputs(pkg, history_chunks)
        image = Image.fromarray(history[0])
        frames, keep, decide_from = len(poses), slice(CHUNK_FRAMES * history_chunks - 4, CHUNK_FRAMES * history_chunks - 4 + FRAMES), 0
    else:
        frames = int(meta.get("frames", FRAMES))
        if (frames + 3) % CHUNK_FRAMES:
            raise ValueError(f"{case}: {frames} frames; a clip has 12 m - 3")
        poses, intrinsics = case_inputs(pkg, frames)
        history, image, keep = None, Image.open(pkg / "handoff.png").convert("RGB"), slice(None)
        decide_from = I2V_DECIDE_FROM if track == "B" else 0
    geo = depth.geo(case, None, i2v=not history_chunks) if depth is not None else None
    session = RolloutSession(pipe, poses, head=head if track == "B" else None, geo=geo, history=history, history_chunks=history_chunks,
                             selection=selection, w=payload.get("w", W_LATERAL), observe_only=observe_only, decide_from=decide_from)
    context = prompt_context(pipe, getattr(pipe, "text_contexts", None), prompt)
    began = time.perf_counter()
    try:
        video = run_chunks(pipe, session, context, image, poses, intrinsics, frame_num=frames, seed=seed)
    finally:
        session.close()
    seconds = time.perf_counter() - began
    decoded = int(video.shape[1])
    video = video[:, keep]
    out.mkdir(parents=True, exist_ok=True)
    write_video(out / f"{case}.mp4", video, FPS)
    info = {"case": case, "track": track, "seed": seed, "seconds": round(seconds, 1), "frames": int(video.shape[1]),
            "decoded_frames": decoded, "size": list(video.shape[2:]), "history_chunks": history_chunks, "fps": FPS, "crop": None,
            "prompt_sha256": prompt_sha256(prompt), "policy": (payload.get("head") or {}).get("name") if head is not None else None}
    if track == "B":
        decided = [r for r in session.records if r["decided"]]
        info.update(observe_only=observe_only, selection=selection, chunks=session.records,
                    interventions=sum(bool(r.get("applied")) for r in session.records),
                    would_intervene=sum(r["choice"] != "KEEP" for r in decided), decided=len(decided),
                    executed_controls=session.controls.detach().cpu().tolist())
        if decide_from:
            info.update(decide_from=decide_from)
        if geo is not None:
            info.update(external_depth=True, depth_stop=dict(
                asked=geo.asked, decided=len(decided), no_scale=sum(bool(r.get("no_scale")) for r in decided),
                chosen_out_of_view=sum(r.get("choice_in_view") is False for r in decided),
                changed_out_of_view=sum(r.get("choice_in_view") is False and r["choice"] != "KEEP" for r in decided),
                seconds_decode=round(session.decode_seconds, 2), seconds_checks=round(session.check_seconds, 2)))
        (out / f"{case}.decisions.json").write_text(json.dumps(info, indent=1))
    return info, video


# --- adapter --------------------------------------------------------------------------------------------------------------

class LingBotWorld2Adapter(Adapter):
    """LingBot-World 2.0 behind the trainer's and the rollout's interface."""

    generator_id = "lingbot_world_2"
    field = FieldConfig(layers=(30, 38, 39), token_dim=5120, camera_dim=5120, frames=CHUNK, height=29, width=52)
    supports_commit_state = True
    model_commit = MODEL_COMMIT
    weights = ("checkpoint",)

    def add_arguments(self, ap):
        ap.add_argument("--checkpoint", type=Path, default=self.path("weights"), help="the LingBot-World 2.0 checkpoint folder")

    def load(self, args):
        self.args = args
        self.pipe = create_pipeline(self.path("code"), args.checkpoint)
        self.camera = CameraFieldEncoder.from_model(self.pipe.model).to(self.pipe.device)
        prefetch = getattr(args, "prefetch", 1)
        self.prefetcher = Prefetcher(lambda source: read_video(resolve_data(source)), workers=1 if prefetch else 0, depth=max(prefetch, 1))
        self.text = None

    def prepare_text(self, prompts):
        self.text = TextContexts(self.pipe, prompts)

    def announce(self, rows_of_next_videos):
        self.prefetcher.announce([rows[0]["source_video"] for rows in rows_of_next_videos])

    def replay(self, rows, observe, *, layers, seed, probe=None):
        """One teacher-forced replay of a video; returns (decision states read, seconds per chunk)."""
        from PIL import Image
        row = rows[0]
        source = resolve_data(row["source_video"])
        video = self.prefetcher.get(row["source_video"])
        self.mark("read_video")
        session = ReplaySession(self.pipe, rows, video, observe, layers=layers, camera=self.camera, probe=probe,
                                commit_state=self.commit_state, clock=self.clock)
        try:
            png = source.parent / "first_frame.png"
            start = Image.open(png).convert("RGB") if png.exists() else Image.fromarray(video[0])
            directory = resolve_data(row["directory"])
            c2ws, ks = np.load(directory / "poses.npy"), np.load(directory / "intrinsics.npy")
            context = prompt_context(self.pipe, self.text, row["instruction"])
            try:
                run_chunks(self.pipe, session, context, start, c2ws, ks, frame_num=len(video), chunk_size=row["frames"], seed=seed)
            except ReplayDone:
                pass
            self.mark("other")
            return session.states, list(session.timings)
        finally:
            session.close()
            del session
            self.release_gpu()

    def add_generation_arguments(self, ap):
        ap.add_argument("--checkpoint", type=Path, default=self.path("weights"), help="the LingBot-World 2.0 checkpoint folder")

    def load_generation(self, args):
        pipe = create_pipeline(self.path("code"), args.checkpoint)
        pipe.text_contexts = None
        return pipe

    def prepare_generation_text(self, pipe, packages):
        """All prompts encoded once on the GPU, then T5 goes back to the CPU."""
        pipe.text_contexts = TextContexts(pipe, [(Path(p) / "prompt.txt").read_text().strip() for p in packages])

    def generate_case(self, pipe, package, out, *, track, head=None, payload=None, observe_only=False, scorer=False,
                      history_chunks=0, seed=None, depth=None):
        return generate_case(pipe, package, out, track=track, head=head, payload=payload, observe_only=observe_only,
                             history_chunks=history_chunks, seed=seed, depth=depth)
