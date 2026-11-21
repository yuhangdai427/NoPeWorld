"""NoPeWorld's two modes: the released head of the world model for that mode, with its operating point."""
from __future__ import annotations

from ...head.model import check_head_fits, load_head
from ...paths import weights_path


def head_path(generator_id, mode):
    return weights_path("heads", generator_id, mode, "head.pt")


def load_released_head(adapter, args, mode, device, path=None):
    """(head, payload) after checking that the head belongs to this generator and was trained under the run's flags."""
    path = path or head_path(adapter.generator_id, mode)
    if not path.is_file():
        raise FileNotFoundError(f"no head at {path}: run nopebench-download --what weights, or set NOPE_WEIGHTS")
    check_head_fits(path, adapter.generator_id, adapter.head_must_match(args), needs_selection=True, defaults=adapter.head_flag_defaults())
    return load_head(path, device)
