"""The safety head: per-candidate safety logit, swept clearance and intent score read from full-resolution DiT tokens.

    hidden   [B, L, F, H, W, token_dim]   outputs of the tapped blocks for the chunk's F latents
    camera   "field" [B, N, F, H*W, camera_dim] (one vector per token) or "frame" [B, N, F, camera_dim] (one per latent)
    ends     [B, F]                        end time of each latent within the chunk (s)

Every state token and camera token goes through its own learned linear map to width_hidden; a candidate's camera is added to the
state tokens at the same (frame, row, column), and one learned query reads them with two cross-attention blocks. Nothing is pooled.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CrossAttentionBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.state_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * width, width))

    def forward(self, query, state):
        attended, weights = self.attention(self.query_norm(query), self.state_norm(state), self.state_norm(state),
                                           need_weights=True, average_attn_weights=True)
        query = query + attended
        query = query + self.ffn(self.ffn_norm(query))
        return query, weights


@dataclass(frozen=True)
class FieldConfig:
    layers: tuple[int, ...] = (30, 38)
    token_dim: int = 5120
    camera_dim: int = 5120
    frames: int = 3
    height: int = 29
    width: int = 52
    width_hidden: int = 128
    attention_heads: int = 4
    blocks: int = 2
    dropout: float = 0.1
    intent_head: bool = False
    query_mode: str = "shared"
    camera_tokens: str = "field"     # field: [B, N, F, H*W, C] | frame: [B, N, F, C]
    camera_frames: int = 0           # > 0: the camera's number of latents when it differs from the state's (a commit pass holds one latent)
    command_input: bool = False      # prediction head: the command's camera is mixed into every candidate's
    camera_hidden: int = 0           # > 0: a small MLP instead of one linear map for low-dimensional geometric cameras

    def to_dict(self) -> dict:
        return asdict(self)


class SafetyPolicy(nn.Module):
    recompute_blocks = True          # False keeps the blocks' activations for the backward pass: the same numbers, more memory

    def __init__(self, config: FieldConfig = FieldConfig()):
        super().__init__()
        self.config = config
        d = config.width_hidden
        if config.camera_tokens not in ("field", "frame"):
            raise ValueError(f"unknown camera_tokens {config.camera_tokens!r}")
        if config.query_mode != "shared":
            raise ValueError(f"unknown query_mode {config.query_mode!r}")
        self.state_projection = nn.Linear(config.token_dim, d)
        self.state_norm = nn.LayerNorm(d)
        self.camera_projection = (nn.Linear(config.camera_dim, d) if not config.camera_hidden else
                                  nn.Sequential(nn.Linear(config.camera_dim, config.camera_hidden), nn.GELU(),
                                                nn.Linear(config.camera_hidden, d)))
        self.camera_norm = nn.LayerNorm(d)
        self.layer_position = nn.Parameter(torch.empty(len(config.layers), d))
        self.frame_position = nn.Parameter(torch.empty(config.frames, d))
        self.row_position = nn.Parameter(torch.empty(config.height, d))
        self.col_position = nn.Parameter(torch.empty(config.width, d))
        self.time_features = nn.Linear(2, d)
        self.query = nn.Parameter(torch.empty(1, 1, d))
        for p in (self.layer_position, self.frame_position, self.row_position, self.col_position, self.query):
            nn.init.trunc_normal_(p, std=0.02)
        self.blocks = nn.ModuleList(CrossAttentionBlock(d, config.attention_heads, config.dropout)
                                    for _ in range(config.blocks))
        self.output = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 3 if config.intent_head else 2))
        if config.command_input:
            self.command_mix = nn.Linear(d, d)

    @staticmethod
    def _project(linear, x):
        """bf16 matmul on the wide inputs, fp32 afterwards; no autocast cache, so trained weights are never read stale."""
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, cache_enabled=False):
            return linear(x).float()

    def _check(self, hidden, camera):
        cfg = self.config
        tokens = cfg.height * cfg.width
        if tuple(hidden.shape[1:]) != (len(cfg.layers), cfg.frames, cfg.height, cfg.width, cfg.token_dim):
            raise ValueError(f"hidden must be [B,{len(cfg.layers)},{cfg.frames},{cfg.height},{cfg.width},"
                             f"{cfg.token_dim}], got {tuple(hidden.shape)}")
        frames = cfg.camera_frames or cfg.frames
        expected = (frames, tokens, cfg.camera_dim) if cfg.camera_tokens == "field" else (frames, cfg.camera_dim)
        if tuple(camera.shape[2:]) != expected:
            raise ValueError(f"camera must be [B,N,{','.join(map(str, expected))}], got {tuple(camera.shape)}")

    def _query_and_memory(self, state, cam):
        cfg = self.config
        b, n = cam.shape[:2]
        frames = cfg.camera_frames or cfg.frames
        if cfg.camera_tokens == "field":
            cam = cam.reshape(b, n, 1, frames, cfg.height, cfg.width, -1)
        else:
            cam = cam.reshape(b, n, 1, frames, 1, 1, -1)
        memory = (state[:, None] + cam).reshape(b * n, -1, cfg.width_hidden)
        return self.query.expand(b * n, -1, -1), memory

    def forward(self, hidden, camera, interval_ends, candidate_mask=None):
        cfg = self.config
        b, n = camera.shape[:2]
        self._check(hidden, camera)
        state = self.state_norm(self._project(self.state_projection, hidden))
        state = (state + self.layer_position[None, :, None, None, None]
                 + self.frame_position[None, None, :, None, None]
                 + self.row_position[None, None, None, :, None] + self.col_position[None, None, None, None, :])
        durations = torch.diff(interval_ends, dim=-1, prepend=torch.zeros_like(interval_ends[:, :1]))
        time = self.time_features(torch.stack((interval_ends, durations), dim=-1).float())
        if cfg.camera_hidden:
            with torch.autocast(device_type=camera.device.type, enabled=False):
                cam = self.camera_norm(self.camera_projection(camera.float()))
        else:
            cam = self.camera_norm(self._project(self.camera_projection, camera))
        if cfg.command_input:
            cam = cam + self.command_mix(cam[:, :1])
        cam = cam + (time[:, None, :, None] if cfg.camera_tokens == "field" else time[:, None])
        query, memory = self._query_and_memory(state, cam)
        for block in self.blocks:
            if self.recompute_blocks and self.training and torch.is_grad_enabled():
                query = checkpoint(lambda q, f, blk=block: blk(q, f)[0], query, memory, use_reentrant=False)
            else:
                query, _ = block(query, memory)
        out = self.output(query).reshape(b, n, -1).float()
        safe_logits, clearance = out[..., 0], out[..., 1]
        if candidate_mask is not None:
            safe_logits = safe_logits.masked_fill(~candidate_mask.bool(), -torch.inf)
        result = {"safe_logits": safe_logits, "clearance": clearance}
        if cfg.intent_head:
            result["intent_logits"] = out[..., 2]
        return result


def read_head_file(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def head_generator(payload):
    """The generator a head belongs to: `generator` in a released file, `model` in a training run's file."""
    return payload.get("generator") or payload.get("model")


def check_head_fits(path, generator, flags, aliases=(), needs_selection=False, defaults=None):
    """Before the generator is loaded: a head means something only on its generator and under the flags it was trained with."""
    payload = read_head_file(path)
    owner = head_generator(payload)
    if owner is not None and owner not in (generator, *aliases):
        raise ValueError(f"{path} was trained on {owner}, not {generator}")
    if needs_selection and not payload.get("selection"):
        raise ValueError(f"{path} has no operating point (selection): take the file the selection step wrote")
    trained = {**(defaults or {}), **payload.get("run_config", {})}
    differ = {k: {"trained": trained[k], "now": v} for k, v in flags.items() if k in trained and trained[k] != v}
    if differ:
        raise ValueError(f"{path} was trained under other flags: {differ}")
    return payload


def load_head(path, device):
    payload = read_head_file(path)
    config = dict(payload["config"])
    config["layers"] = tuple(config["layers"])
    head = SafetyPolicy(FieldConfig(**config))
    head.load_state_dict(payload["state_dict"])
    return head.to(device).eval().requires_grad_(False), payload
