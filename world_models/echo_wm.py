"""Echo-WM: teacher-forced replay of training videos (step-0 and commit taps) and generation with a head deciding each chunk."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

from nopeworld.head.candidates import DELTA, NAMES, RHO, candidate_world_poses, intent_fit, path_length, resample_poses
from nopeworld.head.choose import choose
from nopeworld.head.labels import is_decision
from nopeworld.head.model import FieldConfig
from nopeworld.paths import resolve_data
from nopeworld.utils.rays import plucker_field
from nopeworld.utils.runtime import AsyncScorer, Prefetcher, TokenTap, read_video, write_video
from .base import Adapter, HistoryDoesNotFit, check_keep, prompt_sha256, relative_to_roots

HEIGHT, WIDTH = 704, 1280
SOURCE_HEIGHT = 720
CROP = (SOURCE_HEIGHT - HEIGHT) // 2     # the model's resize_and_center_crop of a 1280 x 720 picture
CROP_BOX = [0, CROP, WIDTH, CROP + HEIGHT]
FPS, STRIDE, CHUNK = 24, 8, 3
GRID = (CHUNK, HEIGHT // 32, WIDTH // 32)
RAY_DIM, UCPE_DIM = 6, 12
CAMERA_DIM = RAY_DIM + UCPE_DIM
SINK_LATENTS = 7                         # latents 0 .. 6 stay in the caches for good, the rest is a FIFO
MAX_HISTORY_CHUNKS = (SINK_LATENTS - 1) // CHUNK
K_PIX = (640.0, 640.0, 640.0, 352.0)     # the 1280 x 720 FOV-90 camera after the crop
ENDS = [1 / 3, 2 / 3, 1.0]
TRANSLATION_CALIBRATION = 30.0
TRANSLATION_SCALE = 1.0
START_CRF = 33                           # H.264 round trip of the start frame, the model's default
SILENCE_CHUNKS = 40
SPLIT_ABOVE = 2 ** 27                    # numbers in one Conv3d call before it goes in pieces along time
PREFETCH_GIB = 101.0                     # free GPU memory a background whole-video encode needs next to the replay
CHECKPOINT, GEMMA = "echo-wm-flash.safetensors", "gemma-3"
INTRINSICS = ("constant", "package")
PACKAGE_FOLDERS = ("fps16", "fps24", "default", "lingbot_world_2")      # dataset sub-folders named after the clock or the generator
RUN_CONFIG = {"audio": "silence", "translation_scale": TRANSLATION_SCALE, "start_crf": START_CRF, "vae_tile": False, "allow_retime": False,
              "fp8": False, "cut_step0": False, "fast_kv": False, "camera": "rays+ucpe"}
HEAD_MUST_MATCH = ("audio", "translation_scale", "start_crf", "allow_retime", "fp8", "camera")


def import_echo(code=None):
    """The repository's packages on the path (read-only); returns ltx_causal.rollout."""
    sys.dont_write_bytecode = True
    if code is not None:
        for package in ("ltx-core/src", "ltx-causal/src", "ltx-pipelines/src", "."):
            path = str((Path(code) / package).resolve())
            if path not in sys.path:
                sys.path.insert(0, path)
    from ltx_causal import rollout
    return rollout


def route_id(package) -> str:
    """The route or case id of a package folder (<route>/fps24, <case>/default, or a folder named after the route)."""
    package = Path(package)
    return package.parent.name if package.name in PACKAGE_FOLDERS else package.name


def video_tokens(output):
    """A block returns (video args, audio args); x holds the current block's video tokens."""
    return output[0].x


@dataclass(frozen=True)
class Layout:
    """The model's token layout."""
    height: int = GRID[1]
    width: int = GRID[2]
    video_channels: int = 128
    audio_channels: int = 8
    audio_bins: int = 16

    @property
    def grid(self):
        return (CHUNK, self.height, self.width)

    @property
    def tokens(self):
        return self.height * self.width


class Trajectory:
    """The camera path as the model reads it: one camera-to-world pose per latent relative to the first frame, translations scaled by
    translation_scale / 30. `viewmats` [1, F, 4, 4] float32 is the tensor every layer reads; `path` holds the same poses in float64."""

    def __init__(self, path, translation_scale, device, viewmats=None):
        self.scale = translation_scale / TRANSLATION_CALIBRATION
        self.path = np.array(path, np.float64)
        self.viewmats = (torch.tensor(self.path.astype(np.float32), device=device)[None] if viewmats is None else viewmats)

    @classmethod
    def at_rest(cls, latents, translation_scale, device):
        return cls(np.tile(np.eye(4), (latents, 1, 1)), translation_scale, device)

    @classmethod
    def commanded(cls, poses, translation_scale, device):
        """[F, 4, 4] camera-to-world poses in metres, in any world."""
        poses = np.asarray(poses, np.float64)
        path = np.linalg.inv(poses[0]) @ poses
        path[..., :3, 3] *= translation_scale / TRANSLATION_CALIBRATION
        return cls(path, translation_scale, device)

    def native(self, relative):
        """Poses in metres relative to some camera, in the model's numbers."""
        out = np.array(relative, np.float64)
        out[..., :3, 3] *= self.scale
        return out

    def write(self, start, end, relative):
        """Latents start .. end - 1 become the pose before them composed with relative [end - start, 4, 4] (metres)."""
        self.path[start:end] = self.path[start - 1] @ self.native(relative)
        self._publish(start, end)

    def correct(self, start, end, relative):
        """write, with the poses after the chunk carried along so that they keep their local motion."""
        old_end = self.path[end - 1].copy()
        self.write(start, end, relative)
        self.path[end:] = self.path[end - 1] @ np.linalg.inv(old_end) @ self.path[end:]
        self._publish(end, len(self.path))

    def _publish(self, start, end):
        self.viewmats[:, start:end] = torch.tensor(self.path[start:end].astype(np.float32), device=self.viewmats.device)


def intrinsics(latents, device, k=K_PIX):
    """[1, F, 3, 3] float32 in pixels of the 1280 x 704 picture: the model's ucpe_Ks."""
    fx, fy, cx, cy = k
    K = torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=torch.float32, device=device)
    return K[None, None].expand(1, latents, -1, -1).contiguous()


def package_k(pkg, size):
    """(fx, fy, cx, cy) in pixels of the model's 1280 x 704 frame for a package whose start picture is `size` (width, height); K_PIX itself
    when they agree within 1e-3 px. intrinsics_native.npy is the K of the start picture; without it, intrinsics.npy is the K of the start
    picture covered and centre-cropped to (2 cx) x (2 cy). The principal point is the frame's centre."""
    width, height = size
    native = Path(pkg) / "intrinsics_native.npy"
    path = native if native.exists() else Path(pkg) / "intrinsics.npy"
    if not path.exists():
        raise ValueError(f"{pkg}: no intrinsics_native.npy or intrinsics.npy, --intrinsics package has no K to read")
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
    return K_PIX if np.allclose(k, K_PIX, atol=1e-3) else tuple(float(v) for v in k)


def ucpe_numbers(state, chunk):
    """[..., F, 12] float32: the top three rows of V_i V_s^-1 of the model's pose numbers, a chunk's latents i against the state latent s
    (state [4, 4], chunk [..., F, 4, 4]); on the CPU so that replay and generation compute it alike."""
    state = torch.as_tensor(state, dtype=torch.float32).cpu()
    chunk = torch.as_tensor(chunk, dtype=torch.float32).cpu()
    return (chunk @ torch.linalg.inv(state))[..., :3, :].flatten(-2)


def candidate_ucpe(trajectory, start, relative):
    """[N, F, 12]: each candidate's ucpe_numbers at the state before latent `start`; candidate 0 is what the camera tensor holds now."""
    end = start + np.asarray(relative).shape[1]
    chunk = torch.from_numpy(np.stack([trajectory.path[start - 1] @ trajectory.native(r) for r in relative]).astype(np.float32))
    chunk[0] = trajectory.viewmats[0, start:end].cpu()
    return ucpe_numbers(trajectory.viewmats[0, start - 1], chunk)


def candidate_camera(relative, layout, device, ucpe):
    """A chunk's candidates for the head, [N, 3, H * W, 18]: the Plucker ray field of their poses relative to the state camera (metres, at
    K_PIX) and each latent's 12 pose numbers on each of its tokens."""
    rays = plucker_field(relative, K_PIX, HEIGHT, WIDTH, (layout.height, layout.width), device=device)
    ucpe = ucpe.to(device=rays.device, dtype=rays.dtype)
    return torch.cat([rays, ucpe[:, :, None, :].expand(-1, -1, rays.shape[2], -1)], dim=-1)


def to_model_frames(rgb, device, dtype):
    """[T, 720, 1280, 3] uint8 -> [1, 3, T, 704, 1280] in [-1, 1], 8 rows off top and bottom."""
    if tuple(rgb.shape[1:3]) != (SOURCE_HEIGHT, WIDTH):
        raise ValueError(f"expected {WIDTH} x {SOURCE_HEIGHT} frames, got {tuple(rgb.shape)}")
    x = torch.from_numpy(rgb[:, CROP:CROP + HEIGHT]).permute(3, 0, 1, 2)[None]
    return (x.to(device=device, dtype=torch.float32) / 127.5 - 1.0).to(dtype=dtype)


def conv3d_in_pieces(conv, x, weight, bias, above):
    """conv's own forward, in pieces along time when its input or output holds more than `above` numbers."""
    plain = torch.nn.Conv3d._conv_forward
    k, s, d = conv.kernel_size[0], conv.stride[0], conv.dilation[0]
    out_t = (x.shape[2] - 1 - (k - 1) * d) // s + 1
    spatial = [(size + 2 * p - 1 - (kk - 1) * dd) // ss + 1 for size, p, kk, dd, ss in zip(x.shape[3:], conv.padding[1:], conv.kernel_size[1:], conv.dilation[1:], conv.stride[1:])]
    numbers = max(x.numel(), x.shape[0] * conv.out_channels * out_t * spatial[0] * spatial[1])
    if numbers <= above:
        return plain(conv, x, weight, bias)
    if conv.padding[0] != 0:
        raise ValueError(f"a Conv3d with padding in time cannot go in pieces: {conv}")
    step = math.ceil(out_t / math.ceil(numbers / above))
    return torch.cat([plain(conv, x[:, :, a * s:(min(a + step, out_t) - 1) * s + (k - 1) * d + 1], weight, bias) for a in range(0, out_t, step)], dim=2)


def split_convs_in_time(encoder, above=SPLIT_ABOVE):
    """Every Conv3d of the encoder in pieces along time: bf16 Conv3d runs without cuDNN here, through a path whose buffer grows with the
    whole video."""
    encoder.split_above = above
    for conv in encoder.modules():
        if isinstance(conv, torch.nn.Conv3d):
            conv._conv_forward = lambda x, weight, bias, conv=conv: conv3d_in_pieces(conv, x, weight, bias, encoder.split_above)


def encode_frames(encoder, rgb, device, dtype):
    """[1, tokens, 128] of real frames, encoded whole."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    return VideoLatentPatchifier(1).patchify(encoder(to_model_frames(rgb, device, dtype)))


def encode_start(encoder, rgb, crf, device, dtype):
    """[1, 880, 128] of a start frame through the model's own image path."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_pipelines.utils.media_io import normalize_latent, preprocess, resize_and_center_crop
    image = torch.tensor(preprocess(image=rgb, crf=crf), dtype=torch.float32, device=device)
    image = normalize_latent(resize_and_center_crop(image, HEIGHT, WIDTH), device, dtype)
    return VideoLatentPatchifier(1).patchify(encoder(image))


def decode_latents(decoder, tokens, layout):
    """[1, L tokens, 128] -> [3, T, H, W] in [-1, 1] with the pipeline's default tiling; shifted by half a level so that write_video's
    truncation gives back the decoder's uint8 values."""
    from ltx_core.components.patchifiers import VideoLatentPatchifier
    from ltx_core.model.video_vae import decode_video
    from ltx_core.model.video_vae.tiling import TilingConfig
    from ltx_core.types import VideoLatentShape
    latents = tokens.shape[1] // layout.tokens
    shape = VideoLatentShape(1, layout.video_channels, latents, layout.height, layout.width)
    latent = VideoLatentPatchifier(1).unpatchify(tokens, output_shape=shape)
    frames = torch.cat([part.cpu() for part in decode_video(latent, decoder, tiling_config=TilingConfig.default())])
    return (frames.permute(3, 0, 1, 2).float() + 0.5) / 127.5 - 1


def crop_box(width, height):
    """[left, top, right, bottom] of a width x height start picture that reaches the model, in its pixels (cover, then centre crop)."""
    scale = max(HEIGHT / height, WIDTH / width)
    new_h, new_w = math.ceil(height * scale), math.ceil(width * scale)
    top, left = (new_h - HEIGHT) // 2, (new_w - WIDTH) // 2
    sx, sy = width / new_w, height / new_h
    return [left * sx, top * sy, (left + WIDTH) * sx, (top + HEIGHT) * sy]


def read_image(path):
    """An RGB picture as uint8 [H, W, 3]."""
    from PIL import Image
    return np.array(Image.open(path))[..., :3]


def encode_silence(ledger, device, chunks=SILENCE_CHUNKS):
    """[1, 2 + 25 chunks, 128]: the model's audio encoder on the digital silence of a clip of that many chunks; shorter clips take its head."""
    from ltx_core.components.patchifiers import AudioPatchifier
    from ltx_core.model.audio_vae import encode_audio
    from ltx_core.types import Audio
    encoder = ledger.audio_encoder()
    samples = round(encoder.sample_rate * (chunks * FPS + 1) / FPS)
    waveform = torch.zeros(1, 2, samples, dtype=torch.float32, device=device)
    with torch.no_grad():
        tokens = AudioPatchifier(1).patchify(encode_audio(Audio(waveform=waveform, sampling_rate=encoder.sample_rate), encoder))
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    if tokens.shape[1] != 2 + 25 * chunks:
        raise ValueError(f"{samples} samples of silence gave {tokens.shape[1]} audio latents, the layout has {2 + 25 * chunks}")
    return tokens


@torch.no_grad()
def encode_texts(ledger, prompts, contexts):
    """Each prompt once through Gemma and the embeddings processor, results on the CPU; both are released afterwards."""
    prompts = [p for p in dict.fromkeys(prompts) if p not in contexts]
    if not prompts:
        return
    encoder = ledger.text_encoder()
    processor = ledger.gemma_embeddings_processor()
    for prompt in prompts:
        hidden, mask = encoder.encode(prompt)
        out = processor.process_hidden_states(hidden, mask)
        if out.audio_encoding is None:
            raise ValueError("the causal AV checkpoint must provide audio text embeddings")
        contexts[prompt] = (out.video_encoding.cpu(), out.audio_encoding.cpu(), out.attention_mask.cpu())
    del encoder, processor
    gc.collect()
    torch.cuda.empty_cache()


def latent_states(layout, latents, device, dtype):
    """Empty video and audio states as the pipeline makes them: zero clean latents and the token positions."""
    repo = import_echo()
    from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
    from ltx_core.tools import AudioLatentTools, VideoLatentTools
    from ltx_core.types import AudioLatentShape, VideoLatentShape
    shape = VideoLatentShape(1, layout.video_channels, latents, layout.height, layout.width)
    video = VideoLatentTools(VideoLatentPatchifier(1), shape, float(FPS)).create_initial_state(device, dtype)
    audio_shape = AudioLatentShape(1, layout.audio_channels, repo.causal_audio_frames(latents, CHUNK), layout.audio_bins)
    audio = AudioLatentTools(AudioPatchifier(1), audio_shape).create_initial_state(device, dtype)
    return video, audio


def open_forward(wrapper, layout, latents, context, trajectory, dtype, k=K_PIX):
    """One video's `_BlockForward` (caches, text cache, camera, all new) and its empty clean latents. k: the intrinsics of the model's camera."""
    repo = import_echo()
    device = trajectory.viewmats.device
    video, sound = latent_states(layout, latents, device, dtype)
    action = {"ucpe_viewmats": trajectory.viewmats, "ucpe_Ks": intrinsics(latents, device, k)}
    forward = repo._BlockForward.create(wrapper=wrapper, clean_video=video.clean_latent, clean_audio=sound.clean_latent,
                                        video_positions=video.positions, audio_positions=sound.positions,
                                        video_context=context[0], audio_context=context[1], context_mask=context[2],
                                        action_cond=action)
    if not all(layer["video_ucpe"]["full_ucpe_viewmats"] is trajectory.viewmats for layer in forward.caches):
        raise RuntimeError("the layers do not read the trajectory's tensor: writing a pose would change nothing")
    return forward, video.clean_latent, sound.clean_latent


def sink_buffers(clean, clean_audio, first, ppf, generator):
    """The rollout's buffers: latent 0 written into the empty clean latents (the pipeline's memory layout) and the noise of the seed."""
    clean[:, :ppf] = first
    return import_echo()._RolloutBuffers.create(clean, clean_audio, ppf, generator)


def chunk_latents(real, block, ppf):
    """A block's real latents for a commit, contiguous as the x0 the model commits itself (bf16 products round differently otherwise)."""
    return real[:, block[0] * ppf:block[1] * ppf].contiguous()


def build_model(checkpoint, gemma_root):
    """The DiT read straight to the GPU, resident for the whole run. Returns (ledger, velocity model, wrapper)."""
    import_echo()
    from helpers.action_condition import action_config
    from ltx_causal import CausalCacheConfig, CausalModelWrapper
    from ltx_pipelines.utils.model_ledger import ModelLedger
    device = torch.device("cuda")
    ledger = ModelLedger(dtype=torch.bfloat16, device=device, checkpoint_path=str(checkpoint), gemma_root_path=str(gemma_root), quantization=None)
    model = ledger.transformer(action_config=action_config(WIDTH, HEIGHT)).velocity_model.requires_grad_(False)
    wrapper = CausalModelWrapper(model, patches_per_frame=GRID[1] * GRID[2], cache=CausalCacheConfig())
    return ledger, model, wrapper


def check_pixels(row):
    """The grid is 24 fps: a row on frames of another rate would move faster per frame than its poses say."""
    if row.get("pixel_fps", FPS) != FPS:
        raise ValueError(f"{row['pair_id']}: {row['source_video']} holds {row['pixel_fps']} fps frames and Echo's clock is {FPS} fps")


@dataclass
class VideoInputs:
    """Prepared inputs of one video: [1, 880 L, 128] tokens of the real frames, latent 0 through the image path."""
    real: torch.Tensor


class EchoWMAdapter(Adapter):
    """Echo-WM replay (step 0 and commit taps) and generation."""
    generator_id = "echo_wm"
    field = FieldConfig(token_dim=4096, camera_dim=CAMERA_DIM, frames=CHUNK, height=GRID[1], width=GRID[2], camera_tokens="field",
                        camera_hidden=128)
    layout = Layout()
    model_commit = "a08f1274573d7cd6ec66719f204951f4227a5173"
    weights = ("checkpoint", "gemma_root")
    step0_t = 1.0
    supports_commit_state = True        # the forward at sigma 0 that writes a chunk's clean latents to the caches

    def weight_files(self, args):
        args.checkpoint = self.path("weights") / CHECKPOINT
        args.gemma_root = self.path("weights") / GEMMA

    def run_config(self):
        return dict(RUN_CONFIG)

    def head_must_match(self, args):
        return {name: RUN_CONFIG[name] for name in HEAD_MUST_MATCH}

    def load(self, args):
        self.args = args
        import_echo(self.path("code"))
        self.weight_files(args)
        if args.prefetch and torch.cuda.is_available() and torch.cuda.mem_get_info()[0] / 2 ** 30 < PREFETCH_GIB:
            raise ValueError(f"--prefetch {args.prefetch} encodes the next video next to the replay's caches: {PREFETCH_GIB} GiB free on the GPU needed, "
                             f"{torch.cuda.mem_get_info()[0] / 2 ** 30:.1f} there; use --prefetch 0")
        self.device, self.dtype = torch.device("cuda"), torch.bfloat16
        self.ledger, self.model, self.wrapper = build_model(args.checkpoint, args.gemma_root)
        self.sigmas = import_echo().resolve_causal_sigmas()
        self.vae = self.ledger.video_encoder()
        split_convs_in_time(self.vae)
        self.vae_lock = threading.Lock()
        self.silence = encode_silence(self.ledger, self.device)
        self.contexts = {}
        self.prefetcher = Prefetcher(self.prepare_video, workers=1 if args.prefetch else 0, depth=max(args.prefetch, 1))
        self.sources = {}

    def prepare_text(self, prompts):
        encode_texts(self.ledger, prompts, self.contexts)

    def announce(self, rows_of_next_videos):
        keys = []
        for rows in rows_of_next_videos:
            check_pixels(rows[0])
            self.sources[rows[0]["pair_id"]] = rows[0]
            keys.append(rows[0]["pair_id"])
        self.prefetcher.announce(keys)

    @torch.no_grad()
    def prepare_video(self, pair_id):
        """Decode, crop and VAE-encode one video."""
        row = self.sources[pair_id]
        source = resolve_data(row["source_video"])
        rgb = read_video(source)
        latents = 1 + (len(rgb) - 1) // STRIDE // CHUNK * CHUNK
        rgb = rgb[:(latents - 1) * STRIDE + 1]
        png = source.parent / "first_frame.png"
        first = read_image(png) if png.exists() else rgb[0]
        with self.vae_lock:
            if not self.args.prefetch and torch.cuda.is_available():
                torch.cuda.empty_cache()        # the encode's large requests do not fit in the freed caches of the video before
            real = encode_frames(self.vae, rgb, self.device, self.dtype)
            real[:, :self.layout.tokens] = encode_start(self.vae, first, START_CRF, self.device, self.dtype)
        return VideoInputs(real=real)

    def replay(self, rows, observe, *, layers, seed, probe=None):
        """Teacher-forced replay: at a probed state the command's poses and one forward at sigma 1 on fresh noise, then the executed poses
        and the real latents committed. With a commit head in the run the commit before a probed state is tapped too."""
        check_pixels(rows[0])
        pair_id = rows[0]["pair_id"]
        self.sources[pair_id] = rows[0]
        by_chunk = {r["chunk"]: r for r in rows}
        wanted = lambda r: is_decision(r) and (probe is None or bool(probe(r)))
        probed = [k for k, r in by_chunk.items() if wanted(r)]
        if not probed:
            self.prefetcher.discard(pair_id)
            return 0, [0.0]
        inputs = self.prefetcher.get(pair_id)
        repo = import_echo()
        layout, device, dtype = self.layout, self.device, self.dtype
        ppf = layout.tokens
        latents = inputs.real.shape[1] // ppf
        blocks, audio_blocks = repo.causal_video_blocks(latents), repo.causal_audio_blocks(latents)
        last = max(probed)
        if last >= len(blocks):
            raise ValueError(f"{pair_id}: a state at chunk {last}, the video has {len(blocks)} blocks: frames and rows do not match")
        if any(k not in by_chunk for k in range(1, last + 1)):
            raise ValueError(f"{pair_id}: rows are missing, every chunk up to {last} has to be committed")
        for k in range(1, last + 1):
            row, first = by_chunk[k], blocks[k][0]
            if "chunk_bounds" in row and (row.get("stride", STRIDE) != STRIDE or list(row["chunk_bounds"]) != [STRIDE * (first - 1 + i) for i in range(CHUNK + 1)]):
                raise ValueError(f"{pair_id} chunk {k}: the row's frames {row['chunk_bounds']} (stride {row.get('stride')}) are not block {blocks[k]}'s")
        sigma = self.sigmas[0]
        generator = torch.Generator(device=device).manual_seed(seed)
        context = tuple(x.to(device) for x in self.contexts[rows[0]["instruction"]])
        trajectory = Trajectory.at_rest(latents, TRANSLATION_SCALE, device)
        tap = TokenTap(self.model.transformer_blocks, layers, extract=video_tokens)
        commit_tap = TokenTap(self.model.transformer_blocks, layers, extract=video_tokens) if self.commit_state else None
        states, seconds, commit_hidden = 0, [], None
        try:
            with torch.no_grad():
                forward, clean, clean_audio = open_forward(self.wrapper, layout, latents, context, trajectory, dtype)
                buffers = sink_buffers(clean, clean_audio, inputs.real[:, :ppf], ppf, generator)
                forward(buffers.clean_image, blocks[0], 0.0, *self.committed_audio(audio_blocks[0]))
                for k in range(1, last + 1):
                    row = by_chunk[k]
                    began = time.perf_counter()
                    start, end = blocks[k]
                    if wanted(row):
                        poses = np.asarray(row["candidate_poses"], np.float64)
                        trajectory.write(start, end, poses[0, 1:])          # KEEP: the command
                        tap.begin(layout.grid)
                        forward(buffers.initial_video[:, start * ppf:end * ppf], blocks[k], sigma,
                                *self.noisy_audio(buffers, audio_blocks[k], sigma))
                        hidden = tap.finish()
                        self.first_probe(k, sigma, hidden, blocks=len(self.model.transformer_blocks))
                        camera = candidate_camera(poses[:, 1:], layout, device, candidate_ucpe(trajectory, start, poses[:, 1:]))
                        observe(hidden, camera, torch.tensor(row["interval_ends"], device=device), row, **({"commit_hidden": commit_hidden} if self.commit_state else {}))
                        states += 1
                    commit_hidden = None
                    if k < last:
                        trajectory.write(start, end, np.asarray(row["executed_poses"], np.float64)[1:])
                        read = commit_tap is not None and wanted(by_chunk[k + 1])
                        if read:
                            commit_tap.begin(layout.grid)
                        forward(chunk_latents(inputs.real, blocks[k], ppf), blocks[k], 0.0, *self.committed_audio(audio_blocks[k]))
                        if read:
                            commit_hidden = commit_tap.finish()
                    if wanted(row):
                        seconds.append(time.perf_counter() - began)
        finally:
            tap.close()
            if commit_tap is not None:
                commit_tap.close()
            forward = buffers = commit_hidden = None
            self.release_gpu()
        return states, seconds or [0.0]

    def committed_audio(self, block):
        """The fixed silent audio latent of a block, at sigma 0."""
        return (self.silence[:, block[0]:block[1]], block, 0.0)

    def noisy_audio(self, buffers, block, sigma):
        """The rollout's audio noise of a block, at the video's sigma."""
        return (buffers.initial_audio[:, block[0]:block[1]], block, sigma)

    def add_generation_arguments(self, ap):
        ap.add_argument("--intrinsics", choices=INTRINSICS, default="package",
                        help="the model's camera: K_PIX for every package (constant), or the package's own K (package; K_PIX on every Task 2 route); "
                             "the head's rays stay at K_PIX")

    def load_generation(self, args):
        self.args = args
        import_echo(self.path("code"))
        self.weight_files(args)
        return EchoPipe(args)

    def prepare_generation_text(self, pipe, packages):
        prompts = [json.loads((Path(p) / "prompts.json").read_text())["per_model"]["echo_wm"] for p in packages]
        encode_texts(pipe.ledger, prompts, pipe.contexts)

    def generate_case(self, pipe, pkg, out, **kwargs):
        with self.held_gpu():
            return generate_case(pipe, pkg, out, **kwargs)


class Restart(Exception):
    """A correction arrived while the next denoising step was running on the uncorrected poses."""


class CalibratedForward:
    """What the repository's block loops call as `forward` (the signature of `_BlockForward`), with a head deciding each chunk.

    Collision head: the step-0 forward runs with the commanded poses and its tokens go to the head; if the choice is not KEEP, the chosen
    poses are written into the camera tensor (later poses carried along) and the forward runs again with the same inputs. With a scorer the
    loop goes on to step 1 while the head scores, and a correction rolls back to the state after step 0, generator included.
    Prediction head: it reads the commit of the chunk before and decides before step 0, so nothing runs twice."""

    def __init__(self, inner, blocks, *, trajectory, generator, sigmas, layout, head=None, candidates=None, w=0.5, selection=None,
                 observe_only=False, scorer=False):
        self.inner, self.wrapper = inner, inner.wrapper
        self.blocks = list(blocks)
        self.chunk_of = {block: j for j, block in enumerate(blocks)}
        self.trajectory, self.generator, self.sigmas, self.layout = trajectory, generator, sigmas, layout
        self.head, self.candidates, self.w = head, candidates, w
        self.selection, self.observe_only = selection or {}, observe_only
        self.records, self.record, self.pending, self.tap, self.scorer = [], None, None, None, None
        self.prediction, self.commit_hidden = False, None
        if head is not None:
            if head.config.camera_dim != CAMERA_DIM:
                raise ValueError(f"the head reads a candidate of {head.config.camera_dim} numbers per token, this adapter gives {CAMERA_DIM}")
            self.tap = TokenTap(inner.wrapper.model.transformer_blocks, head.config.layers, extract=video_tokens)
            self.prediction = head.config.command_input
            if scorer:
                self.scorer = AsyncScorer(self.score)

    def close(self):
        if self.tap is not None:
            self.tap.close()
            self.tap = None
        if self.scorer is not None:
            self.scorer.close()
            self.scorer = None
        self.commit_hidden = None

    def __call__(self, video_latent, video_block, video_sigma, audio_latent=None, audio_block=(0, 0), audio_sigma=None):
        call = (video_latent, video_block, video_sigma, audio_latent, audio_block, audio_sigma)
        if video_sigma == 0.0:                     # a commit or the sink
            output = self.commit(call) if self.prediction and video_block[0] > 0 else self.inner(*call)
            if self.record is not None:
                self.record["seconds"] = round(time.perf_counter() - self.record["seconds"], 3)
                self.records.append(self.record)
                self.record = None
            return output
        if video_sigma == self.sigmas[0]:
            chunk = self.chunk_of[video_block]
            decides = self.head is not None and self.candidates(chunk) is not None and (not self.prediction or self.commit_hidden is not None)
            self.record = {"chunk": chunk, "decided": decides, "choice": "KEEP", "applied": False, "seconds": time.perf_counter()}
            if decides and self.prediction:
                self.before_step0(call[1], chunk)
                return self.inner(*call)
            return self.first_step(call, chunk) if decides else self.inner(*call)
        if self.pending is not None:
            return self.second_step(call)
        return self.inner(*call)

    def check_command(self, block, chunk):
        start, end = block
        path = self.trajectory
        check_keep(float(np.abs(path.path[start - 1] @ path.native(self.candidates(chunk)["relative"][0, 1:]) - path.path[start:end]).max()), chunk, self.record)

    def commit(self, call):
        """A chunk's commit with the tap open, for the next chunk's decision."""
        self.commit_hidden = None
        self.tap.begin(self.layout.grid)
        output = self.inner(*call)
        self.commit_hidden = self.tap.finish()
        return output

    def before_step0(self, block, chunk):
        self.check_command(block, chunk)
        hidden, self.commit_hidden = self.commit_hidden, None
        self.apply(self.score(hidden, chunk), block)

    def first_step(self, call, chunk):
        self.check_command(call[1], chunk)
        self.tap.begin(self.layout.grid)
        output = self.inner(*call)
        hidden = self.tap.finish()
        if self.scorer is not None:
            self.pending = (self.scorer.submit(hidden, chunk), call, self.generator.get_state())
            return output
        if self.apply(self.score(hidden, chunk), call[1]):
            output = self.inner(*call)
        return output

    def second_step(self, call):
        future, first_call, rng = self.pending
        self.pending = None
        blocks = self.wrapper.model.transformer_blocks

        def abort(*_):
            if future.done() and self.changes(future.result()):
                raise Restart()
        hook = blocks[len(blocks) // 2].register_forward_hook(abort)
        try:
            output = self.inner(*call)
        except Restart:
            output = None
        finally:
            hook.remove()
        if self.apply(future.result(), first_call[1]):
            advance = import_echo()._advance_sample
            self.generator.set_state(rng)
            denoised_video, denoised_audio = self.inner(*first_call)
            video = advance(denoised_video, call[2], self.generator)            # the order of the repository's denoising step
            audio = None if denoised_audio is None else advance(denoised_audio, call[2], self.generator)
            output = self.inner(video, call[1], call[2], audio, call[4], call[5])
        return output

    def changes(self, result):
        return result["choice"] != 0 and not self.observe_only

    def score(self, hidden, chunk):
        """The head's choice for a chunk, on the main thread or the scorer's."""
        cand = self.candidates(chunk)
        device = hidden.device
        ucpe = candidate_ucpe(self.trajectory, self.blocks[chunk][0], cand["relative"][:, 1:])
        camera = candidate_camera(cand["relative"][:, 1:], self.layout, device, ucpe)
        mask = torch.ones(1, len(cand["relative"]), dtype=torch.bool, device=device)
        fit = intent_fit(cand["delta"], cand["rho"], w=self.w)[None].to(device)
        ends = torch.tensor(ENDS, device=device)
        with torch.autocast(device_type=device.type, enabled=False):
            out = self.head(hidden[None], camera[None], ends[None], mask)
            a = torch.sigmoid(out["intent_logits"]) if "intent_logits" in out else fit
            sel = choose(out["safe_logits"], a, mask, temperature=self.selection.get("temperature", 1.0),
                         keep_margin=self.selection.get("keep_margin", 0.0), rule=self.selection.get("rule", "product"),
                         tau=self.selection.get("tau", 0.5))
        return {"choice": int(sel["choice"][0]), "p": sel["P"][0].float().cpu(), "r": sel["R"][0].float().cpu(),
                "a": a[0].float().cpu(), "clearance": out["clearance"][0].float().cpu()}

    def apply(self, result, block):
        """Log the decision; True when the camera changed (then a step 0 that has run must run again)."""
        chunk = self.chunk_of[block]
        choice = result["choice"]
        p, r = result["p"], result["r"]
        top = r.topk(5).indices.tolist()
        self.record.update(choice=NAMES[choice], p_keep=round(float(p[0]), 4), p_choice=round(float(p[choice]), 4),
                           r_keep=round(float(r[0]), 4), r_choice=round(float(r[choice]), 4),
                           clearance_keep=round(float(result["clearance"][0]), 3),
                           top=[[NAMES[i], round(float(p[i]), 3), round(float(r[i]), 3)] for i in top],
                           p_all=[round(float(x), 4) for x in p], r_all=[round(float(x), 4) for x in r],
                           a_all=[round(float(x), 4) for x in result["a"]])
        if not self.changes(result):
            return False
        self.trajectory.correct(*block, self.candidates(chunk)["relative"][choice, 1:])
        self.record["applied"] = True
        return True


class DepthForward(CalibratedForward):
    """The external depth method: before a chunk's first forward, the committed latents so far are decoded and DA3 scores the candidates on
    the frames after the hand-off; the collision head's decision rule keeps the command or stops. Without scale the chunk keeps the command."""

    def __init__(self, inner, blocks, *, geo, decode, buffers, keep_from, **kw):
        super().__init__(inner, blocks, head=None, **kw)
        self.geo, self.decode, self.buffers, self.keep_from = geo, decode, buffers, keep_from

    def __call__(self, video_latent, video_block, video_sigma, audio_latent=None, audio_block=(0, 0), audio_sigma=None):
        call = (video_latent, video_block, video_sigma, audio_latent, audio_block, audio_sigma)
        if video_sigma != self.sigmas[0] or video_sigma == 0.0:
            return super().__call__(*call)
        chunk = self.chunk_of[video_block]
        decides = self.candidates(chunk) is not None
        self.record = {"chunk": chunk, "decided": decides, "choice": "KEEP", "applied": False, "seconds": time.perf_counter()}
        if decides:
            self.check_command(video_block, chunk)
            result = self.depth_decision(video_block, chunk)
            if result is not None:
                self.apply(result, video_block)
        return self.inner(*call)

    def depth_decision(self, block, chunk):
        from nopeworld.inference.methods.external_depth import chunk_len16, clock16, geo_choose, to_rgb
        began = time.perf_counter()
        start = block[0]                                                  # latents 0 .. start - 1 are committed
        frames_so_far = 1 + STRIDE * (start - 1)
        video = None
        if frames_so_far - 1 > self.keep_from:
            ppf = self.wrapper.patches_per_frame
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            video = self.decode(self.buffers.video_output[:, :start * ppf])[:, self.keep_from:]      # the hand-off's reconstruction first
        decode_s = time.perf_counter() - began
        n = 1 if video is None else int(video.shape[1])
        end16, index_of = clock16(n, FPS, first_is_handoff=True)
        length = min(chunk_len16(FPS, FPS), end16)                       # a chunk is FPS frames (1 s)
        self.geo.size = [WIDTH, HEIGHT] if video is None else [int(video.shape[3]), int(video.shape[2])]
        disps = np.asarray(self.candidates(chunk)["relative"])[:, -1, :3, 3]        # the chunk's end in the camera at its start (metres)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reply = self.geo.scores(chunk, end16, length, disps, lambda offs: to_rgb(video[:, [index_of(o) for o in offs]]))
        result = geo_choose(reply, self.selection)
        self.record.update(depth_seconds=round(time.perf_counter() - began, 2), decode_seconds=round(decode_s, 2), decoded_end16=end16,
                           no_scale=result is None, h_all=reply.get("h"), in_view=reply.get("in_view"),
                           depth={key: v for key, v in reply.items() if key not in ("h", "P", "A", "in_view", "slide")})
        return result


def chunk_candidates(poses, chunk):
    """dict(relative [34, 4, 4, 4], delta, rho) of chunk j >= 1 in metres relative to the commanded state camera, or None for an in-place
    turn. poses: [frames, 4, 4] commanded camera-to-world at the model's frames. KEEP is the command."""
    frames = [FPS * (chunk - 1) + STRIDE * i for i in range(CHUNK + 1)]
    state = poses[frames]
    if np.linalg.norm(state[-1, :2, 3] - state[0, :2, 3]) <= 1e-3:
        return None
    world = candidate_world_poses(state, path_length(poses, frames), first=False)
    return dict(relative=np.linalg.inv(state[0]) @ world, delta=DELTA, rho=RHO)


@dataclass
class CasePlan:
    """One route on the model's frames. keep_from, keep: the frames the video keeps."""
    poses: np.ndarray                    # [frames, 4, 4] camera-to-world, metres
    image: np.ndarray                    # the start frame
    history: np.ndarray | None           # the real frames of the history chunks with the start frame, or None
    latents: int
    keep_from: int
    keep: int
    fps: int
    package_fps: int


def case_plan(pkg, history_chunks):
    """The command resampled to 24 fps for whole seconds after the hand-off. With k history chunks the clip starts 24 k frames before the
    hand-off, whose frame then ends chunk k; the history frames are a 24 fps package's, frame for frame."""
    meta = json.loads((pkg / "meta.json").read_text())
    kh, generated, fps = meta["handoff_frame"], meta.get("generate_frames", 80), meta.get("fps", 16)
    poses = np.load(pkg / "poses.npy").astype(np.float64)
    command = poses[kh:kh + generated + 1]
    if len(command) != generated + 1:
        raise ValueError(f"{pkg.name}: {len(command)} commanded poses after the hand-off, need {generated + 1}")
    seconds = generated / fps
    chunks = round(seconds)
    frames = 1 + FPS * chunks
    if not history_chunks:
        return CasePlan(resample_poses(command, np.arange(generated + 1) / fps, np.arange(frames) / FPS),
                        read_image(pkg / "handoff.png"), None, 1 + CHUNK * chunks, 0, frames, FPS, fps)
    if fps != FPS:
        raise ValueError(f"{pkg.name}: history.mp4 is {fps} fps and the clock is {FPS} fps: use the 24 fps package")
    before = FPS * history_chunks
    rate = FPS / fps
    source = np.round(kh + (np.arange(before + 1) - before) / rate).astype(int)
    if source[0] < 0:
        raise HistoryDoesNotFit(f"{pkg.name}: hand-off frame {kh} leaves no room for {history_chunks} history chunks of {FPS} frames at {FPS} fps")
    video = read_video(pkg / "history.mp4")
    if len(video) <= kh:
        raise ValueError(f"{pkg.name}: history.mp4 has too few frames")
    clip = resample_poses(poses, np.arange(len(poses), dtype=np.float64), kh + (np.arange(before + frames) - before) / rate)
    return CasePlan(clip, video[source[0]], video[source], 1 + CHUNK * (history_chunks + chunks), before, frames, FPS, fps)


class EchoPipe:
    """What generation needs, resident on the GPU: DiT, video encoder and decoder. Gemma is loaded only to encode the prompts (encode_texts)."""

    def __init__(self, args):
        self.args = args
        self.device, self.dtype = torch.device("cuda"), torch.bfloat16
        self.ledger, self.model, self.wrapper = build_model(args.checkpoint, args.gemma_root)
        self.encoder = self.ledger.video_encoder()
        split_convs_in_time(self.encoder)
        self.decoder = self.ledger.video_decoder()
        self.sigmas = import_echo().resolve_causal_sigmas()
        self.contexts = {}
        self.silence = None
        self.intrinsics = args.intrinsics
        self.layout = Layout()

    def silence_latent(self):
        """Committed with the real history chunks, as in the replay; generated chunks have the model's own audio."""
        if self.silence is None:
            self.silence = encode_silence(self.ledger, self.device)
        return self.silence


def teacher_force(forward, buffers, real, blocks, audio_blocks, silence):
    """The real history as in the replay: the sink and the history chunks committed with silent audio."""
    ppf = forward.wrapper.patches_per_frame
    for block, audio_block in zip(blocks, audio_blocks):
        start, end = block
        clean = buffers.clean_image if start == 0 else chunk_latents(real, block, ppf)
        buffers.video_output[:, start * ppf:end * ppf] = clean
        forward(clean, block, 0.0, silence[:, audio_block[0]:audio_block[1]], audio_block, 0.0)


def run_blocks(pipe, forward, buffers, blocks, audio_blocks, generator, real=None, history_chunks=0):
    """The whole rollout on `forward`; without history the repository's two loops as its own rollout runs them."""
    repo = import_echo()
    if history_chunks:
        teacher_force(forward, buffers, real, blocks[:history_chunks + 1], audio_blocks[:history_chunks + 1], pipe.silence_latent())
    else:
        repo._generate_audio_prefix(forward, buffers, blocks[0], audio_blocks[0], pipe.sigmas, generator)
    repo._generate_av_blocks(forward, buffers, blocks[history_chunks:], audio_blocks[history_chunks:], pipe.sigmas, generator)


def generate_case(pipe, pkg, out, *, track, head=None, payload=None, observe_only=False, scorer=False, history_chunks=0, seed=None,
                  depth=None):
    """One route: track A (the generator alone) or B (a head, or the external depth method with `depth`, deciding each chunk). The video
    has the hand-off frame's reconstruction first, then the frames after it. Returns (log, video [3, T, H, W]); writes <out>/<route>.mp4
    and <route>.decisions.json."""
    if scorer and track == "B" and head is not None and head.config.command_input:
        raise ValueError("an asynchronous scorer does nothing for a prediction head: it decides before the chunk's first forward")
    if history_chunks > MAX_HISTORY_CHUNKS:
        raise ValueError(f"{history_chunks} history chunks: the sink holds {MAX_HISTORY_CHUNKS} after the start frame")
    pkg, out = Path(pkg), Path(out)
    case = route_id(pkg)
    prompt = json.loads((pkg / "prompts.json").read_text())["per_model"]["echo_wm"]
    seed = seed if seed is not None else int(hashlib.sha256(case.encode()).hexdigest()[:8], 16)
    plan = case_plan(pkg, history_chunks)
    repo = import_echo()
    layout, device, dtype = pipe.layout, pipe.device, pipe.dtype
    ppf = layout.tokens
    blocks, audio_blocks = repo.causal_video_blocks(plan.latents), repo.causal_audio_blocks(plan.latents)
    mode = pipe.intrinsics
    if mode not in INTRINSICS:
        raise ValueError(f"--intrinsics {mode}: one of {INTRINSICS}")
    k_case = package_k(pkg, (plan.image.shape[1], plan.image.shape[0])) if mode == "package" else K_PIX
    context = tuple(x.to(device) for x in pipe.contexts[prompt])
    trajectory = Trajectory.commanded(plan.poses[::STRIDE], TRANSLATION_SCALE, device)
    memo = {}

    def candidates(chunk):
        if chunk <= history_chunks:
            return None
        if chunk not in memo:
            memo[chunk] = chunk_candidates(plan.poses, chunk)
        return memo[chunk]
    selection = dict(payload.get("selection") or {}) if payload else {}
    began = time.perf_counter()
    with torch.no_grad():
        real = None if plan.history is None else encode_frames(pipe.encoder, plan.history, device, dtype)
        first = encode_start(pipe.encoder, plan.image, START_CRF, device, dtype)
        inner, clean, clean_audio = open_forward(pipe.wrapper, layout, plan.latents, context, trajectory, dtype, k=k_case)
        generator = torch.Generator(device=device).manual_seed(seed)
        buffers = sink_buffers(clean, clean_audio, first, ppf, generator)
        common = dict(trajectory=trajectory, generator=generator, sigmas=pipe.sigmas, layout=layout, candidates=candidates,
                      w=(payload or {}).get("w", 0.5), selection=selection, observe_only=observe_only)
        if depth is not None:
            if track != "B" or head is not None or scorer:
                raise ValueError("external depth: track B without a head and without an asynchronous scorer")
            geo = depth.geo(case, crop_box(plan.image.shape[1], plan.image.shape[0]), i2v=not history_chunks)
            forward = DepthForward(inner, blocks, geo=geo, decode=lambda tokens: decode_latents(pipe.decoder, tokens, layout), buffers=buffers,
                                   keep_from=plan.keep_from, scorer=False, **common)
        else:
            geo = None
            forward = CalibratedForward(inner, blocks, head=head if track == "B" else None, scorer=scorer, **common)
        try:
            run_blocks(pipe, forward, buffers, blocks, audio_blocks, generator, real=real, history_chunks=history_chunks)
        finally:
            forward.close()
        tokens = buffers.video_output
        records = forward.records
        for record in records:
            ends = plan.poses[[FPS * (record["chunk"] - 1), FPS * record["chunk"]]]
            record["intent_length_m"] = round(float(np.linalg.norm(ends[1, :2, 3] - ends[0, :2, 3])), 4)
        inner = forward = buffers = None            # the caches are freed before the decoder runs
        torch.cuda.empty_cache()
        video = decode_latents(pipe.decoder, tokens, layout)
    seconds = time.perf_counter() - began
    kept = video[:, plan.keep_from:plan.keep_from + plan.keep]
    out.mkdir(parents=True, exist_ok=True)
    write_video(out / f"{case}.mp4", kept, plan.fps)
    decided = [r for r in records if r["decided"]]
    info = {"case": case, "track": track, "seed": seed, "seconds": round(seconds, 1), "frames": int(kept.shape[1]), "decoded_frames": int(video.shape[1]),
            "size": list(kept.shape[2:]), "crop": crop_box(plan.image.shape[1], plan.image.shape[0]), "prompt_sha256": prompt_sha256(prompt), "selection": selection,
            "fps": plan.fps, "package_fps": plan.package_fps, "history_chunks": history_chunks, "observe_only": observe_only, "async": bool(scorer),
            "intrinsics": mode, "k_model": [round(float(v), 4) for v in k_case],
            "interventions": sum(r.get("applied", False) for r in records),
            "would_intervene": sum(r["choice"] != "KEEP" for r in decided), "decided": len(decided),
            "executed_controls": trajectory.path.tolist(), "chunks": records}
    if geo is not None:
        info.update(external_depth=True, depth_mode=geo.mode, depth_package=relative_to_roots(geo.package), depth_answers=geo.asked)
    (out / f"{case}.decisions.json").write_text(json.dumps(info, indent=1))
    return info, kept
