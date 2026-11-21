# Installation

## Requirements

- Linux (x86-64), Python 3.10 or newer, a CUDA GPU.
- **A GPU with at least 80 GB of memory is required; a 96 GB GPU is recommended, and WorldPlay2 needs it (peak ≈ 86 GiB, ≈ 89 GiB with external depth), as does training on Echo-WM (peak ≈ 88 GiB).** Each world model runs on a single GPU; the measured peaks are in [WORLD_MODELS.md](WORLD_MODELS.md).
- Disk: 57 to 86 GB of weights per world model, and 15 GB (benchmark) to 180 GB (everything) of data, see [DATA.md](DATA.md).

## The package

```bash
git clone https://github.com/yuhangdai427/NoPeWorld && cd NoPeWorld
pip install -e .              # nopeworld, nopebench, world_models and the console scripts
pip install -e ".[eval]"      # + OpenCV, for the Task 2 evaluator
pip install -e ".[crops]"     # + OpenCV and imageio-ffmpeg 0.6.0, only to re-cut training crops
pip install -e ".[test]"      # + pytest
```

Console scripts (each also exists as a plain file in `scripts/`):

| command | does |
|---|---|
| `nopebench-download` | downloads the heads and the dataset from the Hugging Face Hub |
| `nopeworld-infer` | runs a world model with a method on a task |
| `nopebench-eval` | scores a run on Task 1, 2 or 3 |
| `nopeworld-train` | trains a head and fixes its operating point on validation |

## One environment per world model

The six world models have conflicting requirements (Python 3.10 to 3.12, torch 2.8 to 2.11, transformers 4.51 to 5.3), so each
gets its own environment, built with that model's own install steps. Then install this package into it:

```bash
conda create -n hy_worldplay python=3.10 && conda activate hy_worldplay
# ... the world model's own requirements, see WORLD_MODELS.md
pip install -e /path/to/NoPeWorld
```

The adapters in `world_models/` put each world model's code on `sys.path` themselves; the world-model repositories do not need
to be pip-installed. [WORLD_MODELS.md](WORLD_MODELS.md) lists, per model, the exact commit, weight revision and package
versions we ran.

## Roots

Every path is relative to one of three roots, set on the command line or by environment variable:

| root | flag | variable | default |
|---|---|---|---|
| dataset | `--data-root` | `NOPE_DATA` | `./data` |
| heads | `--weights-root` | `NOPE_WEIGHTS` | `./weights` |
| world-model code and weights | `--generators-root` | `NOPE_GENERATORS` | `./generators` |

## Depth and evaluator models

The `external_depth` method and the Task 2 evaluator use three more models, installed in one further environment:

| model | used by | where | override |
|---|---|---|---|
| DA3 (`DA3-GIANT-1.1`) | `external_depth`; Task 2 with `--recon da3` | `$NOPE_GENERATORS/da3/DA3-GIANT-1.1/` | `NOPE_DA3_WEIGHTS` |
| VGGT-Ω (`vggt_omega_1b_416_reproduce.pt`) | Task 2 evaluator | `$NOPE_GENERATORS/vggt_omega/` | `NOPE_VGGT_OMEGA_WEIGHTS` |
| SAM 2.1 (`sam2.1_hiera_large.pt`) | Task 2 evaluator | `$NOPE_GENERATORS/sam2/` | `NOPE_SAM2_CHECKPOINT` |

The external-depth method runs DA3 in a child process: set `NOPE_DEPTH_PYTHON` to the Python of the environment that has DA3.
Sources, revisions and package versions: [WORLD_MODELS.md](WORLD_MODELS.md#depth-and-evaluator-models).

## Heads and data

```bash
nopebench-download --what weights                                # all heads -> $NOPE_WEIGHTS/heads/<id>/<mode>/
nopebench-download --what weights --generators hy_worldplay      # one world model's heads
nopebench-download --what data --tasks task1 task2 task3         # the benchmark -> $NOPE_DATA
nopebench-download --what data --split train val                 # training and validation data
```

Each archive is checked against its SHA-256 before it is extracted, and deleted afterwards unless `--keep-archives` is given; a
rerun skips archives that are already extracted. Manual download: [DATA.md](DATA.md#manual-download).

## Tests

```bash
python -m pytest tests            # CPU only; tests that need the dataset or the heads skip without them
```
