# Inference

```bash
nopeworld-infer --generator <id> --method <method> --task <task> --out <run folder> [--group <group>] [--cases <id> ...]
```

| option | meaning |
|---|---|
| `--generator` | `lingbot_world_2`, `hy_worldplay`, `echo_wm`, `evoke_turbo`, `worldplay2`, `hy_nomas` |
| `--method` | one of the five methods below; for Task 1 also `nopeworld` (all of the world model's heads in one replay) |
| `--task` | `task1`, `task2`, `task2_open`, `task3` |
| `--group` | Task 3 only: `static_first_person`, `static_third_person`, `dynamic_first_person`, `dynamic_third_person` |
| `--cases`, `--case-file` | route or case ids (default: the task's whole list) |
| `--seeds` | JSON `{route id: seed}`; routes not listed get a seed from the SHA-256 of their id |
| `--data-root`, `--weights-root`, `--generators-root` | the three roots ([INSTALL.md](INSTALL.md#roots)) |

Other flags are passed to the world model's adapter (`world_models/<id>.py`): WorldPlay2 takes `--compile` (below), HY-WorldPlay
`--text-cache <folder>` (precomputed prompt embeddings; without it the text encoder is loaded in-process).

## Methods

| method | what it does |
|---|---|
| `raw` | the world model as released: the commanded motion every chunk |
| `text_prompt` | the route's own prompt plus one sentence naming the obstacle ahead and asking to stop or slide (`text_prompts/` of the dataset) |
| `external_depth` | before each chunk, DA3 depth of the decoded frames, scaled on the true history, sweeps the candidate motions; the command is kept or the camera stops |
| `nopeworld_collision` | the head reads the first denoising step of the current chunk; if it changes the command, step 0 is rerun once with the new motion |
| `nopeworld_prediction` | the head reads the previous chunk's commit pass and decides before the current chunk is denoised; no extra DiT pass |

Every world model supports all five, except Evoke-Turbo, which has no prediction mode (it keeps no memory between chunks).
`external_depth` runs DA3 in a child process: set `NOPE_DEPTH_PYTHON` to the Python of the environment that has DA3.

**The decision.** At every chunk the head scores 34 candidate motions (the input, a stop, shortened steps and turns of up to
90°) with a safety probability and an intent score. Candidates whose safety probability reaches the
threshold τ form the safe set. The input is kept while it is safe and no safe candidate beats its intent score by more than the
keep margin δ; otherwise the safe candidate with the highest intent score runs; with no safe candidate, the camera stops. The
temperature, τ and δ of each head ship inside its `head.pt`.

## Tasks

| task | runs | cases |
|---|---|---|
| `task1` | the heads on every decision state of the true history; no generation (`nopeworld` or one head method) | 100 routes |
| `task2` | 5 s of closed-loop generation from each route's hand-off | 100 obstacle-bound routes |
| `task2_open` | the same on routes with nothing in the way | 50 open routes |
| `task3` | generation from one open-domain image, per `--group` | 53 + 28 static, 30 + 20 moving (first + third person) |

Without `--cases`, a run covers the task's list in `manifest/` of the dataset.

## Outputs

```
<out>/run.json                    generator, method, task, group
<out>/<id>.mp4                    the generated video
<out>/<id>.decisions.json         every decided chunk: the 34 candidates, their scores and the choice
<out>/<id>.meta.json              seed, frames, crop, fps; written last
<out>/generation.jsonl            one line per finished video
```

A rerun skips the videos that are done. A Task 1 run holds `run.json`, `task1_preds.pt` and `task1_report.json` instead (each
state's choice, whether keeping the input was safe, and its regret). Score a run with `nopebench-eval` ([BENCHMARK.md](BENCHMARK.md)).

## WorldPlay2 compiled kernels

WorldPlay2 runs its kernels eagerly by default, so a route gives the same frames on every run. `--compile` uses the
repository's compiled kernels (`torch.compile` with max-autotune) instead: on one GPU it is about as fast, but the autotuner's
choices vary between processes, so two runs of the same route differ slightly and their decisions can change.
`nopeworld-train` takes the same flag.

```bash
nopeworld-infer --generator worldplay2 --method nopeworld_collision --task task2 \
    --out runs/worldplay2/collision --compile
```

## Time per route

Seconds to generate one 5 s Task 2 route with the model loaded, measured on one 96 GB GPU.

| World model | Raw | Text prompt | External depth | Ours (Coll) | Ours (Pred) |
|---|---|---|---|---|---|
| LingBot-World 2.0 <sup>1</sup> | 53.7 | &ndash; | 53.7 + 14.7 | 53.7 + 9.9 | 53.7 + 0.7 |
| HY-WorldPlay | &ndash; | 87.8 | 171.1 | 95.8 | 87.8 |
| Echo-WM | &ndash; | 28.0 | 53.5 | 41.4 | 38.5 |
| Evoke-Turbo | 26.6 | 26.4 | 37.7 | 32.6 | no head |
| WorldPlay2 <sup>2</sup> | &ndash; | 31.3 | 36.2 | 34.5 | 31.2 |
| Hy-Nomas <sup>3</sup> | 50.8 | 56.3 | 180.8 | 51.2 | 47.5 |

<sup>1</sup> Table 5 of the paper: the raw model's time per route, and the mean time each method adds over it on the same route
(49 routes). <sup>2</sup> With compiled kernels (`--compile`); the default eager mode was not slower on a test route.
<sup>3</sup> One route. Other rows: the median over the 100 Task 2 routes of one run per method. &ndash;: not measured.
