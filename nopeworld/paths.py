"""The three roots every path is relative to: data, head weights and generator checkpoints."""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ROOTS = {"data": ("NOPE_DATA", "data"), "weights": ("NOPE_WEIGHTS", "weights"), "generators": ("NOPE_GENERATORS", "generators")}


def root(name: str, override: str | os.PathLike | None = None) -> Path:
    """A root from the command line, else its environment variable, else its default folder in the current directory."""
    variable, default = ROOTS[name]
    value = override or os.environ.get(variable) or default
    return Path(value).expanduser()


def set_roots(data=None, weights=None, generators=None) -> None:
    """Command-line roots win over the environment for the rest of the process (child processes inherit them)."""
    for name, value in (("data", data), ("weights", weights), ("generators", generators)):
        if value:
            os.environ[ROOTS[name][0]] = str(Path(value).expanduser())


def data_path(*parts) -> Path:
    return root("data").joinpath(*map(str, parts))


def weights_path(*parts) -> Path:
    return root("weights").joinpath(*map(str, parts))


def generators_path(*parts) -> Path:
    return root("generators").joinpath(*map(str, parts))


def resolve_data(path) -> Path:
    """A path stored in the data (rows, manifests) is relative to the data root; an absolute one is taken as it is."""
    path = Path(path)
    return path if path.is_absolute() else data_path(path)
