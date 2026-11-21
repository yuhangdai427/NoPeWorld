"""WorldPlay2 (WorldPlay2-Fast, Wan2.2-I2V-A14B experts): teacher-forced replay on the high-noise expert, and generation with a head deciding each chunk.

The fixed PDD schedule runs high (t = 1000), high (961.5), low (900), low (732.6) per chunk, so step 0 belongs to the high-noise expert and
reads its own K/V cache (kv_high), built by the expert's memory compressor and blocks from the clean history at t = 0: replay loads the
high-noise expert only, generation needs both. A candidate reaches a head as the expert's own action embedding (frozen `_encode_action`),
one 5120-d vector per latent. In replay the clean chunks since the last decision state are committed in one pass (`compress`, then
`ar_prefill`): the cache update is chunk-causal, so one pass equals one per chunk up to rounding.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import gc
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch
from torch.nn import functional as F

from nopeworld.head.labels import freeze_points, is_decision
from nopeworld.head.model import FieldConfig
from nopeworld.paths import resolve_data
from nopeworld.utils.runtime import Prefetcher, TokenTap, read_video, write_video
from .base import Adapter, HistoryDoesNotFit, check_keep, prompt_sha256

DEVICE = torch.device("cuda")
HEIGHT, WIDTH = 448, 832
FPS = 16
LAT_H, LAT_W = HEIGHT // 8, WIDTH // 8
CHUNK = 4
GRID = (CHUNK, LAT_H // 2, LAT_W // 2)
TOKENS = GRID[0] * GRID[1] * GRID[2]
SINK, TEMPORAL = 1, 1
Y_CHECKS, Y_EVERY = 3, 25                # the first conditions and then every Y_EVERY-th are also encoded in full and compared bit for bit
Y_SETTLE = 30                            # latents of [image, grey frames] after which the VAE's output is constant (receptive field 28 latents)
UNUSED_KEYS = ("head.head.weight", "head.head.bias")
HELD_ROW = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]                # a latent that stands still: no turn, no key (first-person column 4)
ATTENTION_BACKEND = "flash"
COMMIT_KEY = "intent_actions"                            # the rows' keys a replay commits for the history: the commanded ones
RUN_CONFIG = {"lr_source": "decode", "y_tile": True, "expert": "high", "attention_backend": ATTENTION_BACKEND, "commit_actions": "intent"}
PACKAGE_FOLDERS = ("fps16", "fps24", "default", "lingbot_world_2")      # dataset sub-folders named after the clock or the generator


def route_id(pkg):
    """The route or case id of a package folder: its own name, or its parent's for the dataset's fps16/, fps24/ and default/ folders."""
    pkg = Path(pkg)
    return pkg.parent.name if pkg.name in PACKAGE_FOLDERS else pkg.name


def case_seed(case):
    return int(hashlib.sha256(case.encode()).hexdigest()[:8], 16)


def perspective_of(package_command):
    """Column 4 of every row of a route: 0.0 for "perspective": "tps" (third person), 1.0 otherwise."""
    perspective = package_command.get("perspective", "fps")
    if perspective not in ("tps", "fps"):
        raise ValueError(f"command.json perspective must be 'tps' or 'fps', not {perspective!r}")
    return 0.0 if perspective == "tps" else 1.0


def held_row(perspective=1.0):
    """HELD_ROW in a route's perspective."""
    return HELD_ROW[:4] + [float(perspective)] + HELD_ROW[5:]


COMPILE_HELP = "use the generator's compiled kernels (about as fast as eager on one GPU, and not bit-reproducible between runs)"


def compiled_kernels(enabled):
    """WorldPlay2 compiles its kernels with max-autotune, whose choices vary between processes: without `--compile` they run eagerly,
    so a route gives the same frames on every run. torch.compile reads this switch when the kernels are first built."""
    if enabled:
        os.environ.pop("TORCHDYNAMO_DISABLE", None)
    else:
        os.environ["TORCHDYNAMO_DISABLE"] = "1"


def import_worldplay2(code):
    """The generator's repository on sys.path; returns its I2V config and pipeline module."""
    sys.dont_write_bytecode = True
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path("~/.cache/rank_00/torchinductor").expanduser()))
    os.environ.setdefault("TRITON_CACHE_DIR", str(Path("~/.cache/rank_00/triton").expanduser()))
    from configs import I2V_A14B_CONFIG
    from worldplay2 import pipeline as module
    return I2V_A14B_CONFIG, module


def crop_box(width, height):
    """[left, top, right, bottom] of the picture that reaches the generator: the centre crop to its 13:7."""
    keep = int(round(width * HEIGHT / WIDTH))
    top = (height - keep) // 2
    return [0, top, width, top + keep]


def to_model_frames(rgb, device, *, video=True):
    """[T, H, W, 3] uint8 -> [3, T, 448, 832] in [-1, 1]: centre crop, then bicubic. Video frames are antialiased and clamped; the start image
    (video=False) is resized as the pipeline treats it."""
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).to(device).float() / 127.5 - 1
    h, w = x.shape[2:]
    _, top, _, bottom = crop_box(w, h)
    x = x[:, :, top:bottom]
    x = F.interpolate(x, size=(HEIGHT, WIDTH), mode="bicubic", align_corners=False, antialias=video)
    return (x.clamp(-1, 1) if video else x).permute(1, 0, 2, 3)


def crop_image(image):
    """The PIL start image with the same centre crop (the pipeline resizes it to 448 x 832)."""
    return image.crop(tuple(crop_box(*image.size)))


def lr_frames(chunk_frames):
    """[3, T, H, W] -> every other frame at a quarter of the size (`_decode_chunk_and_lr_encode`)."""
    x = chunk_frames[:, ::2].permute(1, 0, 2, 3)
    x = F.interpolate(x, size=(HEIGHT // 4, WIDTH // 4), mode="bilinear", align_corners=False)
    return x.permute(1, 0, 2, 3)


def commit_grid():
    """The commit pass's tokens a prediction head reads: one latent x rows x columns."""
    return (TEMPORAL, GRID[1], GRID[2])


def anchor_tokens(output):
    """A prefill block's output -> the anchor latent's tokens, the last ones in a first prefill and in an update alike."""
    frames, height, width = commit_grid()
    return output[:, -frames * height * width:]


class Unkept(dict):
    """A layer's K/V entry during a frozen pass that keeps nothing: it reads as the real entry and what the pass writes goes nowhere (a block only
    replaces its entry's tensors, as its last act on them)."""

    def __setitem__(self, key, value):
        pass


def kept_entries(kv):
    """The K/V of a frozen pass that step 0 reads afterwards: each layer's entry as a dict of its own holding the real tensors; the pass replaces
    the copies' tensors and the real entries keep theirs."""
    return [dict(entry) for entry in kv]


@torch.no_grad()
def encode_with_freezes(vae, video, points):
    """video [3, T, H, W] encoded as Wan2_1_VAE.encode does, and at each chunk in points also the continuation in which the last frame so far is
    held for one chunk: a branch of the causal encoder's cache, CHUNK of its steps. Returns (latents [16, lat_f, h, w], {chunk: [16, CHUNK, h, w]})."""
    model, x, points = vae.model, video[None], set(points)
    if 0 in points:
        raise ValueError("chunk 0 has no frame before it to hold")

    def step(frames):
        model._enc_conv_idx = [0]
        return model.encoder(frames, feat_cache=model._enc_feat_map, feat_idx=model._enc_conv_idx)

    def latents_of(out):
        mu, _ = model.conv1(out).chunk(2, dim=1)
        return ((mu - vae.scale[0].view(1, model.z_dim, 1, 1, 1)) * vae.scale[1].view(1, model.z_dim, 1, 1, 1)).float().squeeze(0)
    with torch.amp.autocast("cuda", dtype=vae.dtype):
        model.clear_encode_cache()
        out, frozen = [], {}
        for i in range(1 + (x.shape[2] - 1) // 4):
            if i % CHUNK == 0 and i // CHUNK in points:
                saved = list(model._enc_feat_map)
                held = x[:, :, 4 * (i - 1)].unsqueeze(2).repeat(1, 1, 4, 1, 1)
                frozen[i // CHUNK] = latents_of(torch.cat([step(held) for _ in range(CHUNK)], 2))
                model._enc_feat_map = saved
            out.append(step(x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1):1 + 4 * i]))
        latents = latents_of(torch.cat(out, 2))
        model.clear_encode_cache()
    return latents, frozen


@dataclass
class VideoInputs:
    """Prepared inputs of one video; nothing here is a DiT state."""
    clean: torch.Tensor                  # [16, T, 56, 104] latents of the real frames
    lr: torch.Tensor                     # [16, T / 2, 14, 26] low-resolution memory latents
    hr_y: torch.Tensor                   # [20, T, 56, 104]
    lr_y: torch.Tensor                   # [20, T / 2, 14, 26]
    seconds: float
    frozen: dict = field(default_factory=dict)   # {chunk: (latents [16, 4, 56, 104], low-resolution latents [16, 2, 14, 26])} on the CPU


class WorldPlay2Adapter(Adapter):
    """WorldPlay2-Fast: step-0 and commit taps of the high-noise expert, candidates as its action embeddings, generation with both experts."""
    generator_id = "worldplay2"
    field = FieldConfig(token_dim=5120, camera_dim=5120, frames=CHUNK, height=GRID[1], width=GRID[2], camera_tokens="frame")
    model_commit = "c5d83e32099116ff3a1437a8a05c764d579f704b"
    control_file = "command.json"
    weights = ("ckpt_dir", "high_ckpt", "low_ckpt")
    step0_t = 1000.0
    supports_commit_state = True
    supports_frozen_states = True
    supports_frozen_step0 = True         # step-0 heads with `frozen`: step 0 of the next chunk on the frozen pass's K/V
    frozen_step0 = False                 # the trainer sets it when a step-0 head of the run has `frozen`
    y_tile_conditions, y_tile_mismatch = 0, None

    def commit_field(self, field):
        """The commit pass's grid: one latent; a candidate keeps its action vectors of 4 latents, each reading that latent's tokens."""
        return replace(field, frames=TEMPORAL, camera_frames=CHUNK)

    def weight_paths(self):
        weights = self.path("weights")
        return {"ckpt_dir": weights / "Wan2.2-I2V-A14B",
                "high_ckpt": weights / "WorldPlay2-Fast/high_noise_model/diffusion_pytorch_model.safetensors",
                "low_ckpt": weights / "WorldPlay2-Fast/low_noise_model/diffusion_pytorch_model.safetensors"}

    def add_arguments(self, ap):
        paths = self.weight_paths()
        ap.add_argument("--ckpt-dir", type=Path, default=paths["ckpt_dir"], help="T5, VAE and expert configs")
        ap.add_argument("--high-ckpt", type=Path, default=paths["high_ckpt"])
        ap.add_argument("--compile", action="store_true", help=COMPILE_HELP)

    def run_config(self):
        return dict(RUN_CONFIG)

    def run_notes(self):
        return {"compiled": compiled_regions(), "y_tile_mismatch": self.y_tile_mismatch}

    def head_must_match(self, args):
        return {"attention_backend": ATTENTION_BACKEND}

    def load(self, args):
        self.args = args
        compiled_kernels(getattr(args, "compile", False))
        i2v, module = import_worldplay2(self.path("code"))
        self.i2v, self.module = i2v, module
        self.device = DEVICE
        from worldplay2.pdd import FixedPDD4Scheduler
        from worldplay2.text.encoder import T5EncoderModel
        from worldplay2.vae.wan21 import Wan2_1_VAE
        self.scheduler = FixedPDD4Scheduler()
        self.t0 = self.scheduler.timesteps[0:1].to(self.device)
        self.param_dtype = i2v.param_dtype
        self.vae = Wan2_1_VAE(vae_pth=str(args.ckpt_dir / i2v.vae_checkpoint), device=self.device)
        self.vae_lock = threading.Lock()
        self.make_t5 = lambda: T5EncoderModel(text_len=i2v.text_len, dtype=i2v.t5_dtype, device=self.device,
                                              checkpoint_path=str(args.ckpt_dir / i2v.t5_checkpoint),
                                              tokenizer_path=str(args.ckpt_dir / i2v.t5_tokenizer))
        self.model = None                # after the text: T5 and the expert never share the GPU
        self.contexts = {}
        prefetch = getattr(args, "prefetch", 1)
        self.prefetcher = Prefetcher(self.prepare_video, workers=1 if prefetch else 0, depth=max(prefetch, 1))
        self.sources = {}

    def prepare_text(self, prompts):
        """Every prompt encoded once with T5, then the encoder leaves the GPU and the expert is loaded."""
        prompts = list(dict.fromkeys(prompts))
        encoder = self.make_t5()
        with torch.no_grad():
            for p in prompts:
                self.contexts[p] = encoder([p], self.device)[0].cpu()
        del encoder
        gc.collect()
        torch.cuda.empty_cache()
        if self.model is None:
            self.model = load_expert(self.module, self.args.ckpt_dir, self.args.high_ckpt, "high", self.i2v.param_dtype, self.device)

    def source_of(self, rows, frozen):
        """What prepare_video reads of a video: its first row and, when its replay reads frozen states, the chunks to encode frozen too."""
        return {**rows[0], "freeze_at": freeze_points(rows)} if frozen else rows[0]

    def announce(self, rows_of_next_videos):
        keys = []
        for rows in rows_of_next_videos:
            self.sources[rows[0]["pair_id"]] = self.source_of(rows, self.frozen_states)
            keys.append(rows[0]["pair_id"])
        self.prefetcher.announce(keys)

    @torch.no_grad()
    def prepare_video(self, pair_id):
        """Decode, resize and VAE-encode one video. The streaming encoder's chunks stay in one call under the lock, so its cache belongs to this video."""
        began = time.perf_counter()
        row = self.sources[pair_id]
        source = resolve_data(row["source_video"])
        rgb = read_video(source)
        png = source.parent / "first_frame.png"
        latents = (len(rgb) - 1) // 4 + 1
        latents -= latents % CHUNK
        frames = rgb[:(latents - 1) * 4 + 1]
        with self.vae_lock:
            pixels = to_model_frames(frames, self.device)
            clean, frozen = encode_with_freezes(self.vae, pixels, row["freeze_at"]) if row.get("freeze_at") else (self.vae.encode([pixels])[0], {})
            lr = []
            for k in range(latents // CHUNK):
                if k in frozen:
                    frozen[k] = (frozen[k].cpu(), self.frozen_lr(frozen[k]).cpu())
                decoded = self.vae.decode_chunk([clean[:, k * CHUNK:(k + 1) * CHUNK]], is_first_chunk=k == 0)[0]
                lr.append(self.vae.encode_chunk([lr_frames(decoded)], is_first_chunk=k == 0)[0])
            lr = torch.cat(lr, dim=1)
            self.vae.model.clear_decode_cache()
            start = load_start(png, frames[0], self.device)
            hr_y, lr_y = self.conditions(start, latents)
            self.vae.model.clear_cache()
        return VideoInputs(clean=clean, lr=lr, hr_y=hr_y, lr_y=lr_y, seconds=time.perf_counter() - began, frozen=frozen)

    def frozen_lr(self, latents):
        """The low-resolution latents of a frozen chunk: decoded and encoded as a branch of both streaming caches, which are restored after it."""
        model = self.vae.model
        decoder, encoder = list(model._feat_map), list(model._enc_feat_map)
        decoded = self.vae.decode_chunk([latents], is_first_chunk=False)[0]
        lr = self.vae.encode_chunk([lr_frames(decoded)], is_first_chunk=False)[0]
        model._feat_map, model._enc_feat_map = decoder, encoder
        return lr

    def conditions(self, start, latents):
        """The i2v conditioning, tiled after Y_SETTLE latents; the first Y_CHECKS and every Y_EVERY-th are also built in full and compared bit for
        bit, and after a difference the full build is used for the rest of the process."""
        if self.y_tile_mismatch is not None or latents <= Y_SETTLE:
            return self.build_y(start, latents)
        self.y_tile_conditions += 1
        tiled = self.build_y(start, latents, tile=True)
        if self.y_tile_conditions > Y_CHECKS and self.y_tile_conditions % Y_EVERY:
            return tiled
        full = self.build_y(start, latents)
        if not all(torch.equal(a, b) for a, b in zip(tiled, full)):
            self.y_tile_mismatch = {"condition": self.y_tile_conditions}
            print(json.dumps({"y_tile_mismatch": self.y_tile_mismatch}), flush=True)
        return full

    def build_y(self, start, latents, *, tile=False):
        """The i2v conditioning volumes (the pipeline's `_build_y`, full and low resolution); with tile only Y_SETTLE latents are encoded and the
        last one is repeated."""
        small = F.interpolate(start[None].cpu(), size=(HEIGHT // 4, WIDTH // 4), mode="bilinear")[0].to(self.device)
        out = []
        for image, t_lat, h, w in ((start, latents, HEIGHT, WIDTH), (small, latents // 2, HEIGHT // 4, WIDTH // 4)):
            head = min(Y_SETTLE, t_lat) if tile else t_lat
            y = self.module._build_y(self.vae, image, t_lat=head, h_pix=h, w_pix=w, device=self.device, dtype=self.param_dtype)
            if head < t_lat:
                y = torch.cat([y, y[:, -1:].expand(-1, t_lat - head, -1, -1)], dim=1)
                y = fix_mask(y, t_lat)
            out.append(y)
        return out

    def add_generation_arguments(self, ap):
        paths = self.weight_paths()
        ap.add_argument("--ckpt-dir", type=Path, default=paths["ckpt_dir"], help="T5, VAE and expert configs")
        ap.add_argument("--high-ckpt", type=Path, default=paths["high_ckpt"])
        ap.add_argument("--low-ckpt", type=Path, default=paths["low_ckpt"])
        ap.add_argument("--compile", action="store_true", help=COMPILE_HELP)

    def load_generation(self, args):
        compiled_kernels(getattr(args, "compile", False))
        return load_generation(args, self.path("code"))

    def prepare_generation_text(self, pipe, packages):
        """All prompts encoded once, then T5 leaves the GPU."""
        prompts = [json.loads((Path(p) / "prompts.json").read_text())["per_model"]["worldplay2"] for p in packages]
        contexts = {}
        with torch.no_grad():
            for p in dict.fromkeys(prompts):
                contexts[p] = pipe.text_encoder([p], pipe.device)[0].cpu()
        pipe.text_encoder = lambda texts, device: [contexts[t].to(device) for t in texts]
        gc.collect()
        torch.cuda.empty_cache()

    def generate_case(self, pipe, package, out, *, depth=None, **kwargs):
        if depth is not None:
            from .worldplay2_depth import generate_case as depth_case
            return depth_case(pipe, package, out, depth=depth, **kwargs)
        return generate_case(pipe, package, out, **kwargs)

    def encode_candidates(self, actions):
        """[N, 4, 6] action rows -> [N, 4, 5120] with the expert's frozen action encoder."""
        rows = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        n = rows.shape[0]
        return self.model._encode_action(rows.reshape(-1, 6)).reshape(n, CHUNK, -1)

    def commit(self, state, inputs, upto, context, actions, tap=None):
        """The K/V cache up to the first `upto` latents, all real: compress + prefill. tap: open during the prefill; state["commit_hidden"] then
        holds this pass."""
        cache_y = inputs.hr_y[:, :upto]
        hr_input = torch.cat([inputs.clean[:, :upto], cache_y], dim=0).to(self.param_dtype)[None]
        lr_input = torch.cat([inputs.lr[:, :upto // 2], inputs.lr_y[:, :upto // 2]], dim=0).to(self.param_dtype)[None]
        temporal = torch.cat([inputs.clean[:, upto - TEMPORAL:upto], inputs.hr_y[:, upto - TEMPORAL:upto]],
                             dim=0).to(self.param_dtype)[None]
        memory = self.model(forward_mode="compress", hr_latents_input=hr_input, lr_latents_input=lr_input,
                            lr_action=actions[::2][:upto // 2],
                            memory_t=torch.zeros(1, device=self.device, dtype=self.param_dtype),
                            hr_temporal_latents_input=temporal, hr_temporal_action=actions[upto - TEMPORAL:upto],
                            sink_action=actions[:SINK])
        if tap is not None:
            tap.begin(commit_grid())
        state["kv"] = self.model(forward_mode="ar_prefill", context=[context], memory_dict=memory, kv_cache=state["kv"])
        if tap is not None:
            state["commit_hidden"] = tap.finish()
        # the denoising forward reads only the counts and the RoPE shift; the tokens would hold about 1 GB by the end of a video
        state["memory"] = {k: memory[k] for k in ("rope_shift_idx", "sink_token_count", "cmp_token_count",
                                                  "cmp_chunk_token_count", "tmp_token_count")}
        state["committed"] = upto

    def read_frozen(self, k, row, state, inputs, context, executed, tap, frozen, step0=None):
        """The state after one frozen chunk at chunk k, handed to frozen() outside the autocast. From wherever the K/V stands; nothing of the pass is
        kept. step0: (the step-0 tap, the video's noise) when step-0 heads with `frozen` read it too; frozen() then also gets hidden=."""
        s, committed = k * CHUNK, state["committed"]
        hidden = None
        with torch.amp.autocast("cuda", dtype=self.param_dtype), torch.no_grad():
            latents, lr = inputs.frozen[k]
            longer = replace(inputs, clean=torch.cat([inputs.clean[:, :s], latents.to(self.device)], dim=1), lr=torch.cat([inputs.lr[:, :s // 2], lr.to(self.device)], dim=1))
            actions = torch.cat([executed[:s], torch.tensor([held_row(executed[0, 4])] * CHUNK, dtype=executed.dtype, device=self.device)])
            branch = {"kv": kept_entries(state["kv"]) if step0 is not None else [Unkept(entry) for entry in state["kv"]], "memory": None, "committed": committed, "commit_hidden": None}
            self.commit(branch, longer, s + CHUNK, context, actions, tap=tap)
            if step0 is not None:
                hidden = self.frozen_step0_read(k, row, branch, inputs, context, *step0)
            camera = self.encode_candidates(row["candidate_actions"])
        commit_hidden, branch = branch["commit_hidden"], None          # the copies' K/V is freed before the heads read the state
        frozen(camera, torch.tensor(row["interval_ends"], device=self.device), row, commit_hidden=commit_hidden, **({} if hidden is None else {"hidden": hidden}))

    def frozen_step0_read(self, k, row, branch, inputs, context, tap, noise):
        """[L, 4, 28, 52, D]: step 0 at chunk k + 1's position on the frozen pass's K/V and memory, with row k's command and the noise and
        conditioning of that position. Writes nothing."""
        s = (k + 1) * CHUNK
        tap.begin(GRID)
        self.model([noise[:, s:s + CHUNK]], t=self.t0, context=[context], seq_len=TOKENS, y=[inputs.hr_y[:, s:s + CHUNK]],
                   hr_action=torch.tensor(row["intent_actions"], dtype=torch.float32, device=self.device),
                   forward_mode="pdd_denoise", pdd_block_index=0, kv_cache=branch["kv"], ar_chunk_start=s,
                   memory_dict=branch["memory"])
        return tap.finish()

    def replay(self, rows, observe, *, layers, seed, probe=None, frozen=None):
        """One teacher-forced replay; observe(hidden, camera, interval_ends, row[, commit_hidden=]) per probed state. frozen(camera, interval_ends,
        row, *, commit_hidden[, hidden]) or None: also the state after one frozen chunk at each of labels.freeze_points(rows), after that chunk's
        own read; it does not count in the states returned."""
        pair_id = rows[0]["pair_id"]
        self.sources[pair_id] = self.source_of(rows, frozen is not None)
        points = () if frozen is None else self.sources[pair_id]["freeze_at"]
        wanted = lambda r: r is not None and is_decision(r) and (probe is None or bool(probe(r)))
        if not points and not any(wanted(r) for r in rows):
            self.prefetcher.discard(pair_id)
            return 0, [0.0]
        inputs = self.prefetcher.get(pair_id)
        by_chunk = {r["chunk"]: r for r in rows}
        chunks = inputs.clean.shape[1] // CHUNK
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noise = torch.randn(16, inputs.clean.shape[1], LAT_H, LAT_W, dtype=torch.float32, generator=generator,
                            device=self.device)
        context = self.contexts[rows[0]["instruction"]].to(self.device)
        executed = torch.tensor([row for k in range(chunks) for row in by_chunk[k][COMMIT_KEY]],
                                dtype=torch.float32, device=self.device)
        tap = TokenTap(self.model.blocks, layers)
        pass_tap = TokenTap(self.model.blocks, layers, extract=anchor_tokens) if self.commit_state or points else None
        commit_tap = pass_tap if self.commit_state else None
        step0 = (tap, noise) if frozen is not None and self.frozen_step0 else None
        state = {"kv": self.model.init_kv_cache(), "memory": None, "committed": 0, "commit_hidden": None}
        self.model.clear_conditioning_cache()
        states, seconds = 0, []
        amp = lambda: torch.amp.autocast("cuda", dtype=self.param_dtype)
        try:
            for k in range(chunks):
                row = by_chunk.get(k)
                if not wanted(row):
                    if k in points:
                        self.read_frozen(k, row, state, inputs, context, executed, pass_tap, frozen, step0)
                    continue
                began = time.perf_counter()
                s, e = k * CHUNK, (k + 1) * CHUNK
                with amp(), torch.no_grad():
                    if s > state["committed"]:           # the pass ends with the chunk before this one, whatever it skipped
                        self.commit(state, inputs, s, context, executed, tap=commit_tap)
                    tap.begin(GRID)
                    self.model([noise[:, s:e]], t=self.t0, context=[context], seq_len=TOKENS, y=[inputs.hr_y[:, s:e]],
                               hr_action=torch.tensor(row["intent_actions"], dtype=torch.float32, device=self.device),
                               forward_mode="pdd_denoise", pdd_block_index=0, kv_cache=state["kv"], ar_chunk_start=s,
                               memory_dict=state["memory"])
                    hidden = tap.finish()
                    self.first_probe(k, self.t0[0], hidden, blocks=len(self.model.blocks), expert="high", compiled=compiled_regions())
                    camera = self.encode_candidates(row["candidate_actions"])
                # outside the autocast: inside one, autocast would keep serving its first bf16 copy of the head's weights after optimizer steps
                extra = {"commit_hidden": state["commit_hidden"]} if self.commit_state else {}
                observe(hidden, camera, torch.tensor(row["interval_ends"], device=self.device), row, **extra)
                states += 1
                seconds.append(time.perf_counter() - began)
                if k in points:
                    self.read_frozen(k, row, state, inputs, context, executed, pass_tap, frozen, step0)
        finally:
            tap.close()
            if pass_tap is not None:
                pass_tap.close()
            self.model.clear_conditioning_cache()
            del state
            self.release_gpu()
        return states, seconds or [0.0]


def compiled_regions():
    """Which of the generator's kernels ran compiled so far (one that fails to compile falls back to eager). None without the repository."""
    kernels = sys.modules.get("worldplay2.models.kernels")
    return None if kernels is None else {"qk_rope": kernels.get_qk_rope_compile_stats(), **kernels.get_compiled_region_stats()}


def fix_mask(y, t_lat):
    """The 4 mask channels of a tiled conditioning: only the first latent is known (`_build_i2v_mask_4ch`)."""
    y[:4, 1:] = 0
    return y


def load_start(png, first, device):
    """The replay's start image as [3, 448, 832] in [-1, 1]: the lossless first frame when the video has one."""
    if png.exists():
        from PIL import Image
        first = np.array(Image.open(png).convert("RGB"))
    return to_model_frames(first[None], device, video=False)[:, 0]


def parameter_name(key):
    """The expert's parameter a key of its file fills (the files' names carry the wrapper's prefix)."""
    return key[len("model."):] if key.startswith("model.") else key


def load_expert(module, ckpt_dir, ckpt, expert, dtype, device):
    """One expert in the state WorldPlay2Pipeline builds (compiled kernels, fused q/k/v, BF16), its weights read from the safetensors file straight
    to the GPU (the pipeline's own path first initialises it in float32 on the CPU)."""
    from safetensors import safe_open

    from worldplay2.models.attention import resolve_attention_backend
    from worldplay2.models.model import WorldPlay2Model, rope_params
    config = WorldPlay2Model.load_config(str(ckpt_dir / f"{expert}_noise_model"))
    with torch.device("meta"):
        model = WorldPlay2Model.from_config(config)
        model.add_embedding_action_parameters()
        model.add_memory_compress_model()
        model.head.make_fixed_pdd(num_blocks=2)
    model.to(dtype=dtype)
    model.to_empty(device=device)
    params = dict(model.named_parameters())
    seen = set()
    with safe_open(str(ckpt), "pt", device="cpu") as f, torch.no_grad():
        for key in f.keys():
            name = parameter_name(key)
            if name in UNUSED_KEYS:
                continue
            if name not in params:
                raise KeyError(f"{ckpt}: unexpected tensor {key}")
            tensor = f.get_tensor(key)
            if tensor.shape != params[name].shape:
                raise ValueError(f"{key}: {tuple(tensor.shape)} != {tuple(params[name].shape)}")
            params[name].copy_(tensor.to(dtype=dtype))
            seen.add(name)
    missing = set(params) - seen
    if missing:
        raise KeyError(f"{ckpt} lacks {sorted(missing)[:5]} ({len(missing)} tensors)")
    d = model.dim // model.num_heads                 # not a buffer of the model
    model.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)),
                             rope_params(1024, 2 * (d // 6))], dim=1).to(device)
    model.fuse_self_attention_qkv()
    model.eval().requires_grad_(False)
    model.set_sequence_parallel(False)
    model.set_self_attention_backend(resolve_attention_backend(ATTENTION_BACKEND))
    model.set_compiled_qk_rope(True)
    model.set_conditioning_cache(True)
    model.set_compiled_regions(block_glue=True, ffn=True)
    logging.info("%s expert on %s", expert, device)
    return model


class Restart(Exception):
    """A correction arrived while the next denoising block ran on the uncorrected action."""


class CalibratedChunks:
    """The fixed PDD loop of one chunk with a head deciding (WorldPlay2Pipeline._run_pdd_chunk).

    Collision head: step 0 (high expert, t = 1000) runs with the commanded rows and its tapped blocks go to the head with the candidates' action
    embeddings. If the choice is not KEEP, the chosen rows replace the command in place (`hr_action` is the tensor the memory reads later) and step 0
    runs again; it only reads kv_high, so only the random generator is restored. With a scorer the next block runs while the head scores, and a
    correction rolls back to the state after step 0; the video is bit for bit that of the waiting path.

    Prediction head (head.config.command_input): decides before step 0 on the high expert's commit pass of the previous chunk; a correction goes into
    the rows before any forward reads them. Chunk 0 without history has no pass before it and keeps the command.

    history: chunks before this index return the real latents without denoising; the pipeline's memory path commits them.
    """

    def __init__(self, pipe, head, *, candidates, names, w, selection, observe_only=False, scorer=False, history_latents=None, history_chunks=0):
        from nopeworld.head.candidates import intent_fit
        self.pipe, self.head = pipe, head
        self.candidates, self.names = candidates, names
        self.selection, self.observe_only = selection, observe_only
        self.w, self.intent_fit = w, intent_fit
        self.history_latents, self.history_chunks = history_latents, history_chunks
        self.records, self.tap = [], None
        self.scorer = None
        self.predicts, self.commit_hidden = False, None
        if head is not None:
            self.predicts = head.config.command_input
            if self.predicts and scorer:
                raise ValueError("a prediction head decides before step 0: no forward can run while it scores")
            self.tap = TokenTap(pipe.high_noise_model.blocks, head.config.layers, extract=anchor_tokens if self.predicts else None)
            if scorer:
                from nopeworld.utils.runtime import AsyncScorer
                self.scorer = AsyncScorer(self.score)

    def close(self):
        if self.tap is not None:
            self.tap.close()
        if self.scorer is not None:
            self.scorer.close()
        self.commit_hidden = None

    def before_commit(self):
        """The high expert's prefill of the chunk just generated or teacher-forced comes next."""
        if self.predicts:
            self.tap.begin(commit_grid())

    def after_commit(self):
        if self.predicts:
            self.commit_hidden = self.tap.finish()

    def encode(self, rows):
        model = self.pipe.high_noise_model
        with torch.amp.autocast("cuda", dtype=self.pipe.param_dtype):
            return model._encode_action(rows.reshape(-1, 6)).reshape(rows.shape[0], CHUNK, -1)

    def score(self, hidden, chunk, ends):
        """The head's choice for a chunk, on the main thread or the scorer's."""
        from nopeworld.head.choose import choose
        cand = self.candidates(chunk)
        rows = torch.tensor(cand["rows"], dtype=torch.float32, device=hidden.device)
        mask = torch.tensor([cand["mask"]], dtype=torch.bool, device=hidden.device)
        camera = self.encode(rows)
        fit = self.intent_fit(cand["delta"], cand["rho"], w=self.w)[None].to(hidden.device)
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            out = self.head(hidden[None], camera[None], ends[None], mask)
            a = torch.sigmoid(out["intent_logits"]) if "intent_logits" in out else fit
            sel = choose(out["safe_logits"], a, mask, temperature=self.selection.get("temperature", 1.0),
                         keep_margin=self.selection.get("keep_margin", 0.0), rule=self.selection.get("rule", "product"),
                         tau=self.selection.get("tau", 0.5))
        p, r = sel["P"][0].float().cpu(), sel["R"][0].float().cpu()
        return {"choice": int(sel["choice"][0]), "p": p, "r": r, "rows": rows, "a": a[0].float().cpu(), "clearance": out["clearance"][0].float().cpu()}

    def run(self, *, latent, y_chunk, hr_action_chunk, context, memory_dict_high, memory_dict_low, kv_high, kv_low,
            chunk_start, seed_g):
        chunk = chunk_start // CHUNK
        if chunk < self.history_chunks:
            return self.history_latents[:, chunk_start:chunk_start + CHUNK].to(latent.device, latent.dtype)
        decides = self.head is not None and self.candidates(chunk) is not None and (not self.predicts or self.commit_hidden is not None)
        record = {"chunk": chunk, "decided": decides, "choice": "KEEP"}
        if decides:
            keep = torch.tensor(self.candidates(chunk)["rows"][0], dtype=torch.float32, device=hr_action_chunk.device)
            check_keep(float((keep - hr_action_chunk.float()).abs().max()), chunk, record)
        began = time.perf_counter()
        pipe, sched = self.pipe, self.pipe.pdd_scheduler
        ends = torch.tensor([(i + (0 if chunk == 0 else 1)) * 0.25 for i in range(CHUNK)], device=latent.device)
        reads_step0 = decides and not self.predicts
        if reads_step0:
            self.tap.begin(GRID)

        def forward(block, x):
            model = pipe._prepare_named_expert(block.expert, pipe.offload_model)
            kv, memory = (kv_high, memory_dict_high) if block.expert == "high" else (kv_low, memory_dict_low)
            return model([x.to(pipe.device)], t=sched.timesteps[block.start:block.start + 1].to(pipe.device),
                         context=[context[0]], seq_len=pipe.seq_len_chunk, y=[y_chunk], hr_action=hr_action_chunk,
                         forward_mode="pdd_denoise", pdd_block_index=block.local_index, kv_cache=kv,
                         ar_chunk_start=chunk_start, memory_dict=memory)[0]

        def advance(block, x, output):
            displacement, endpoint = output
            sigma_endpoint = sched.sigmas[block.end - 1].to(device=x.device, dtype=torch.float32)
            x0 = (x.float() + displacement.float() - sigma_endpoint * endpoint.float()).to(x.dtype)
            sigma_end = sched.sigmas[block.end].to(device=x.device, dtype=torch.float32)
            if block.end == sched.num_intervals:
                return x0
            noise = torch.randn(x0.shape, generator=seed_g, device=x0.device, dtype=x0.dtype)
            return ((1.0 - sigma_end) * x0.float() + sigma_end * noise.float()).to(x0.dtype)

        blocks = sched.ordered_blocks
        start = latent
        applied = False
        if decides and self.predicts:
            applied = self.apply(self.score(self.commit_hidden, chunk, ends), chunk, record, hr_action_chunk)
        output = forward(blocks[0], start)
        if reads_step0:
            hidden = self.tap.finish()
            rng = seed_g.get_state()
            if self.scorer is not None:
                future = self.scorer.submit(hidden, chunk, ends)
                x = advance(blocks[0], start, output)

                def abort(*_):
                    if future.done() and self.wants_change(future.result()):
                        raise Restart()
                hook = pipe.high_noise_model.blocks[len(pipe.high_noise_model.blocks) // 2].register_forward_hook(abort)
                try:
                    output1 = forward(blocks[1], x)
                except Restart:
                    output1 = None
                finally:
                    hook.remove()
                result = future.result()
                if self.apply(result, chunk, record, hr_action_chunk):
                    applied = True
                    seed_g.set_state(rng)
                    output = forward(blocks[0], start)
                    x = advance(blocks[0], start, output)
                    output1 = forward(blocks[1], x)
                x = advance(blocks[1], x, output1)
                rest = blocks[2:]
            else:
                result = self.score(hidden, chunk, ends)
                if self.apply(result, chunk, record, hr_action_chunk):
                    applied = True
                    output = forward(blocks[0], start)
                x = advance(blocks[0], start, output)
                rest = blocks[1:]
        else:
            x = advance(blocks[0], start, output)
            rest = blocks[1:]
        for block in rest:
            x = advance(block, x, forward(block, x))
        record.update(applied=applied, seconds=round(time.perf_counter() - began, 3))
        self.records.append(record)
        return x

    def wants_change(self, result):
        return result["choice"] != 0 and not self.observe_only

    def apply(self, result, chunk, record, hr_action_chunk):
        """Log the decision; True when the action rows changed (a head that read step 0 then runs it again)."""
        choice = result["choice"]
        p, r = result["p"], result["r"]
        names = self.names
        top = r.topk(5).indices.tolist()
        record.update(choice=names[choice], p_keep=round(float(p[0]), 4), p_choice=round(float(p[choice]), 4),
                      r_keep=round(float(r[0]), 4), r_choice=round(float(r[choice]), 4),
                      clearance_keep=round(float(result["clearance"][0]), 3),
                      top=[[names[i], round(float(p[i]), 3), round(float(r[i]), 3)] for i in top],
                      p_all=[round(float(x), 4) for x in p], r_all=[round(float(x), 4) for x in r],
                      a_all=[round(float(x), 4) for x in result["a"]])
        if choice == 0 or self.observe_only:
            return False
        hr_action_chunk.copy_(result["rows"][choice].to(hr_action_chunk.dtype))
        return True


def output_frames(chunk, keep_from, frames):
    """[first, last] of the written video that a chunk's latents decode to: chunk 0 frames 0..12, chunk k 16k - 3 .. 16k + 12, shifted by keep_from
    (the hand-off frame, the last frame of the history)."""
    first, last = (0, 12) if chunk == 0 else (4 * CHUNK * chunk - 3, 4 * CHUNK * chunk + 12)
    return [max(first - keep_from, 0), min(last - keep_from, frames - 1)]


def commanded_rows(package_command, frames):
    """The action rows of a route's command: per latent the keys and yaw of its frames. frames: the count the pipeline generates (1 + 4 (L - 1));
    the first row is zero but for the perspective; a row covers frames 4i - 3 .. 4i."""
    from nopeworld.head.candidates import KEY_SIGNS, action_rows
    per_frame = {p["f"]: p for p in package_command["per_frame"]}
    last = max(per_frame)
    latents = (frames - 1) // 4 + 1
    keys, yaws = [(0, 0)], [0.0]
    for i in range(1, latents):
        p = per_frame.get(min(4 * i, last))
        keys.append(KEY_SIGNS["" if p["keys"] == "none" else p["keys"]])
        yaws.append(float(p["yaw_rate_dps"]) * 4 / 16)
    return action_rows(keys, yaws, perspective_of(package_command)), keys, yaws


def package_history_rows(pkg, start, kh, history_chunks, perspective=1.0):
    """Action rows of the history frames from the true poses (poses.npy, 16 fps, OpenCV camera-to-world, world z up): the executed motion,
    quantised as in the training rows."""
    from nopeworld.head.candidates import action_rows, body_to_world
    poses = np.load(Path(pkg) / "poses.npy").astype(np.float64)
    fs = [start] + [start + 4 * i for i in range(1, 4 * history_chunks)]
    keys, yaws = [(0, 0)], [0.0]
    heading = lambda T: float(np.degrees(np.arctan2(-T[1, 2], T[0, 2])))
    steps = [np.linalg.norm(poses[b, :2, 3] - poses[a, :2, 3]) for a, b in zip(fs[:-1], fs[1:])]
    step = max(max(steps), 1e-6)
    for a, b in zip(fs[:-1], fs[1:]):
        ya, yb = heading(poses[a]), heading(poses[b])
        dyaw = (yb - ya + 180) % 360 - 180
        d = poses[b, :2, 3] - poses[a, :2, 3]
        body = np.linalg.solve(body_to_world(ya + 0.5 * dyaw), np.array([d[0], -d[1]]))
        keys.append(tuple(int(np.sign(c)) if abs(c) >= 0.25 * step else 0 for c in body))
        yaws.append(dyaw)
    return action_rows(keys, yaws, perspective), fs


def package_frames(meta):
    """(frames generated, frames kept) for a package's generate_frames: 80 -> (93, 81), 128 -> (141, 129); whole chunks of latents, decoded to
    1 + 4 (L - 1) frames with the hand-off frame's reconstruction first."""
    generate = int(meta["generate_frames"])
    need = -(-generate // 4) + 1
    latents = -(-need // CHUNK) * CHUNK
    return 1 + 4 * (latents - 1), generate + 1


def plan_case(pkg, history_chunks=0, seed=None):
    """Everything a route's generation needs that runs no model: its id, seed, perspective, length, action rows (history rows from the true poses,
    then the command) and where the written video starts. HistoryDoesNotFit when the hand-off frame leaves no room for history_chunks."""
    pkg = Path(pkg)
    case = route_id(pkg)
    meta = json.loads((pkg / "meta.json").read_text())
    if meta.get("case", case) != case:
        raise ValueError(f"{pkg}: the package is route {meta['case']}, its folder says {case}")
    command = json.loads((pkg / "command.json").read_text())
    seed = seed if seed is not None else case_seed(case)
    perspective = perspective_of(command)
    frames, keep = package_frames(meta)
    rows, _, _ = commanded_rows(command, frames)
    plan = dict(case=case, seed=seed, meta=meta, command=command, perspective=perspective, keep=keep, history_start=None, handoff_frame=None,
                history_chunks=0, keep_from=0)
    if history_chunks:
        kh = meta["handoff_frame"]
        start = kh - (16 * history_chunks - 4)            # the hand-off frame ends the last history chunk
        if start < 0:
            raise HistoryDoesNotFit(f"{case}: hand-off frame {kh} leaves no room for {history_chunks} history chunks")
        hist_rows, _ = package_history_rows(pkg, start, kh, history_chunks, perspective)
        rows = hist_rows + rows[1:]
        plan.update(history_start=start, handoff_frame=kh, history_chunks=history_chunks, keep_from=16 * history_chunks - 4)
    latents = len(rows) - len(rows) % CHUNK
    plan["rows"] = rows[:latents]
    return plan


def chunk_candidates(rows, chunk, history_chunks, perspective):
    """A chunk's candidates compiled from the commanded rows; None for a history chunk or one without a key pressed (nothing to correct)."""
    from nopeworld.head.candidates import runtime_key_candidates
    if chunk < history_chunks:
        return None
    part = rows[chunk * CHUNK:(chunk + 1) * CHUNK]
    command_keys = [(int(r[2]), int(r[3])) for r in part]
    if chunk == 0:
        command_keys[0] = (0, 0)
    if all(k == (0, 0) for k in command_keys):
        return None
    return runtime_key_candidates(command_keys, [r[1] for r in part], first=chunk == 0, perspective=perspective)


def load_generation(args, code):
    """WorldPlay2Pipeline with both experts resident, built without its CPU detour (load_expert) and with the chunk loop handed to a
    CalibratedChunks when one is set; the rest of generate_chunked is the pipeline's own."""
    config, module = import_worldplay2(code)
    from worldplay2.pdd import FixedPDD4Scheduler
    from worldplay2.text.encoder import T5EncoderModel
    from worldplay2.vae.wan21 import Wan2_1_VAE

    class Pipeline(module.WorldPlay2Pipeline):
        chunks = None

        def __init__(self):                     # the attributes generate_chunked and _prepare_chunk_memory read
            self.inference_mode, self.attention_backend = module.InferenceMode.PDD, ATTENTION_BACKEND
            self.device, self.config, self.rank, self.init_on_cpu = DEVICE, config, 0, False
            self.denoise_forward_mode, self.pdd_scheduler = "pdd_denoise", FixedPDD4Scheduler()
            self.use_sp, self.sp_size = False, 1
            self.num_train_timesteps, self.boundary = config.num_train_timesteps, config.boundary
            self.param_dtype, self.vae_stride, self.patch_size = config.param_dtype, config.vae_stride, config.patch_size
            self.sample_neg_prompt = config.sample_neg_prompt
            self.text_encoder = T5EncoderModel(text_len=config.text_len, dtype=config.t5_dtype, device=self.device,
                                               checkpoint_path=str(args.ckpt_dir / config.t5_checkpoint),
                                               tokenizer_path=str(args.ckpt_dir / config.t5_tokenizer))
            self.vae = Wan2_1_VAE(vae_pth=str(args.ckpt_dir / config.vae_checkpoint), device=self.device)
            self.high_noise_model = load_expert(module, args.ckpt_dir, args.high_ckpt, "high", config.param_dtype, self.device)
            self.low_noise_model = load_expert(module, args.ckpt_dir, args.low_ckpt, "low", config.param_dtype, self.device)

        def _run_pdd_chunk(self, **kwargs):
            if self.chunks is None:
                return super()._run_pdd_chunk(**kwargs)
            return self.chunks.run(**kwargs)

        def _prefill_kv_cache(self, model, *args, **kwargs):
            """The same call; around the high expert's, a prediction head's tap is open (the low expert's blocks are other modules)."""
            chunks = self.chunks if model is self.high_noise_model and isinstance(self.chunks, CalibratedChunks) else None
            if chunks is not None:
                chunks.before_commit()
            caches = super()._prefill_kv_cache(model, *args, **kwargs)
            if chunks is not None:
                chunks.after_commit()
            return caches

    return Pipeline()


def generate_case(pipe, pkg, out, *, track, head=None, payload=None, observe_only=False, scorer=False, history_chunks=0, seed=None,
                  chunks_class=None):
    """One route: track A (the generator as it is) or B (a head deciding each chunk). The video starts with the hand-off frame's reconstruction;
    with history_chunks the history's frames are cut off. Writes <out>/<case>.mp4 and <case>.decisions.json; returns (log, video).
    chunks_class: a CalibratedChunks subclass that decides instead of the head (the external depth method)."""
    from PIL import Image

    from nopeworld.head.candidates import NAMES
    pkg, out = Path(pkg), Path(out)
    if (payload or {}).get("head", {}).get("encoding"):
        raise ValueError("this adapter reads heads on the action embedding only")
    plan = plan_case(pkg, history_chunks, seed)
    case, seed, command, perspective, keep = plan["case"], plan["seed"], plan["command"], plan["perspective"], plan["keep"]
    prompt = json.loads((pkg / "prompts.json").read_text())["per_model"]["worldplay2"]
    rows, history_chunks_used, keep_from = plan["rows"], plan["history_chunks"], plan["keep_from"]
    history_latents = None
    handoff = Image.open(pkg / "handoff.png").convert("RGB")
    image = crop_image(handoff)
    if history_chunks_used:
        start, kh = plan["history_start"], plan["handoff_frame"]
        video = read_video(pkg / "history.mp4")[start:kh + 1]
        if len(video) != kh + 1 - start:
            raise ValueError(f"{case}: history.mp4 has too few frames")
        image = crop_image(Image.fromarray(video[0]))
        with torch.no_grad():
            history_latents = pipe.vae.encode([to_model_frames(video, pipe.device)])[0]
    # on the GPU already: generate_chunked keeps this tensor, the one corrections are written to and the log reads
    hr_action = torch.tensor(rows, dtype=torch.float32, device=pipe.device)
    total = hr_action.shape[0]
    candidates = lambda chunk: chunk_candidates(rows, chunk, history_chunks_used, perspective)
    selection = dict(payload.get("selection") or {}) if payload else {}
    chunks = (chunks_class or CalibratedChunks)(pipe, head if track == "B" else None, candidates=candidates, names=NAMES,
                                                w=(payload or {}).get("w", 0.5), selection=selection, observe_only=observe_only,
                                                scorer=scorer, history_latents=history_latents, history_chunks=history_chunks_used)
    pipe.chunks = chunks
    began = time.perf_counter()
    try:
        video = pipe.generate_chunked(prompts={"p": prompt}, chunk_prompts=tuple(["p"] * (total // CHUNK)), img=image,
                                      hr_action=hr_action, total_latent_frames=total, chunk_length=CHUNK, shift=7.0,
                                      sampling_steps=4, guide_scale=1.0, seed=seed, offload_model=False,
                                      sink_size=SINK, temporal_size=TEMPORAL, height=HEIGHT, width=WIDTH)
    finally:
        pipe.chunks = None
        chunks.close()
    seconds = time.perf_counter() - began
    kept = video[:, keep_from:keep_from + keep]
    if kept.shape[1] != keep:
        raise ValueError(f"{case}: {kept.shape[1]} frames decoded after the hand-off, the package asks for {keep}")
    out.mkdir(parents=True, exist_ok=True)
    write_video(out / f"{case}.mp4", kept, FPS)
    for record in chunks.records:
        cand = candidates(record["chunk"])
        record["intent_length_m"] = round(cand["length"] * command["speed_mps"] * 4 / FPS, 4) if cand is not None else 0.0
        record["frames"] = output_frames(record["chunk"], keep_from, int(kept.shape[1]))
        if cand is not None:    # a candidate whose keys compile to an earlier one's is masked
            record.update(candidate_mask=cand["mask"], candidate_delta=cand["delta"], candidate_rho=cand["rho"])
    decided = [r for r in chunks.records if r["decided"]]
    info = {"case": case, "track": track, "seed": seed, "seconds": round(seconds, 1), "frames": int(kept.shape[1]), "decoded_frames": int(video.shape[1]),
            "frames_per_chunk": 4 * CHUNK, "command_speed_mps": command["speed_mps"], "fps": FPS,
            "size": list(kept.shape[2:]), "crop": crop_box(*handoff.size), "prompt_sha256": prompt_sha256(prompt),
            "selection": selection, "history_chunks": history_chunks, "observe_only": observe_only,
            "async": bool(scorer), "interventions": sum(r.get("applied", False) for r in chunks.records),
            "would_intervene": sum(r["choice"] != "KEEP" for r in decided), "decided": len(decided), "kernels_disabled": (compiled_regions() or {}).get("disabled_after_failure", []),
            "executed_actions": hr_action.tolist(), "chunks": chunks.records}
    (out / f"{case}.decisions.json").write_text(json.dumps(info, indent=1))
    return info, kept
