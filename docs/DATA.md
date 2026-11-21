# Data

NoPeBench lives in the dataset repository [PeterDAI/NoPeBench](https://huggingface.co/datasets/PeterDAI/NoPeBench): training and
validation data for the heads and the three benchmark tasks. All renders come from UnrealZoo scenes in Unreal Engine, first
person, with camera poses, intrinsics, occupancy maps and per-candidate labels. The generated videos of the paper are not included.

## Archives

Small files are stored as they are; everything else is in uncompressed GNU tar archives whose paths are relative to the data
root. Extracting all of them into one folder gives the full tree. No episode, crop, route or case is split across two archives.

| file | holds | size |
|---|---|---|
| `archives/scenes.tar` | `scenes/`: occupancy maps of all 85 scenes (44 training, 8 validation, 33 benchmark) | 0.11 GB |
| `archives/task1.tar` | `task1/`: decision rows on the 100 routes and the teacher-forced replay videos, per world model | 7.3 GB |
| `archives/task2.tar` | `task2/`: 100 obstacle-bound and 50 open routes: true history at 16 and 24 fps, command, hand-off frame, obstacle mask and depth, Unreal Engine reference execution | 7.3 GB |
| `archives/task3.tar` | `task3/`: start images, 8 s inputs and captions of the 81 static and 50 moving-obstacle cases | 0.56 GB |
| `archives/val.tar` | `val/`: 124 validation videos (93 episodes, 31 near-wall clips) and their rows | 5.5 GB |
| `archives/train-rows.tar` | `train/rows/`: the decision rows the heads trained on and the replay order, per world model | 0.26 GB |
| `archives/train-episodes-*-of-00008.tar` | `train/episodes/`: 729 training videos (509 episodes, 220 near-wall clips) | 35.3 GB |
| `archives/train-crops-*-of-00025.tar` | `train/crops/`: 1,527 later-start crops of the training episodes | 123.6 GB |
| `text_prompts/` | the text-prompt baseline's sentence per route and case | < 1 MB |
| `manifest/` | id lists per split, task and world model; `archives.json`; `sha256.txt` of every extracted file | < 1 MB |

Sizes are extracted sizes. `manifest/archives.json` gives each archive's bytes, SHA-256 and the ids it holds.

## Download

```bash
nopebench-download --what data --tasks task1 task2 task3      # the benchmark, 15 GB
nopebench-download --what data --tasks task2                  # one task
nopebench-download --what data --split train val              # training and validation
nopebench-download --what data                                # everything, 180 GB
```

| option | archives |
|---|---|
| `--tasks task1` | `task1.tar` |
| `--tasks task2`, `--tasks task2_open` | `scenes.tar`, `task2.tar` (the open routes are inside `task2.tar`) |
| `--tasks task3` | `task3.tar` |
| `--split train` | `scenes.tar`, `train-rows.tar`, `train-episodes-*`, `train-crops-*` |
| `--split val` | `scenes.tar`, `val.tar` |
| none of them | every archive |

`README.md`, `LAYOUT.md`, `manifest/` and `text_prompts/` always come along. `--cases <id> ...` keeps, among those, the archives
that hold the given routes, cases or episodes (plus the scene maps and training rows they need); since Tasks 1 to 3 are one
archive each, it mainly narrows the training shards. Each archive is checked against its SHA-256, extracted into `$NOPE_DATA` and
deleted (`--keep-archives` keeps it); a rerun skips what is already extracted.

### Manual download

```bash
huggingface-cli download PeterDAI/NoPeBench --repo-type dataset --local-dir data \
    --include "manifest/*" "text_prompts/*" "archives/scenes.tar" "archives/task*.tar"
cd data
for f in archives/*.tar; do tar -xf "$f"; done
sha256sum -c --ignore-missing manifest/sha256.txt
```

## Layout

```
scenes/<scene_id>/            geometry_gpu.json + geometry/<region>.npz (occupancy, packed bits)
train/episodes/<episode_id>/  render/ (16 fps) and render24/ (24 fps): clean.mp4, first_frame.png, poses.npy, intrinsics.npy, ...
train/crops/<crop_id>/        crop.json, render/, render24/
train/rows/<id>/              rows.jsonl.gz (one decision state per line, labels inside), stream.json (replay order)
val/episodes/, val/rows/<id>/ the same for validation (rows.jsonl.gz, rows_near_wall.jsonl.gz)
task1/rows/<id>.jsonl.gz      decision rows on the 100 routes; task1/replays/<id>/ the replay videos
task2/routes/<route_id>/      route.json, case/, fps16/, fps24/ (generator input), fps16/eval/ (evaluator-only truth), reference/
task2/open_routes/<route_id>/ the same for the 50 open routes
task3/<group>/<case_id>/      case.json, lingbot_world_2/ (full 8 s input), default/ (input of the other world models)
text_prompts/                 task2.json, task3_static.json, task3_dynamic.json, obstacle_names.json
manifest/                     id lists, train_crops.json, archives.json, sha256.txt, files.tsv
```

Ids: `episode_id` = `<scene_id>__fpNNN` (or `__nwK` for a near-wall clip); `route_id` = `<scene_id>__<n|d|v|p>NNN`, or
`<scene_id>__cNN_<kind>` for an open route; `case_id` = `wbench_caseN` or `wbench_dynNNN`.

**Formats.** H.264 MP4 videos (1280×720 renders). `poses.npy`: one camera-to-world 4×4 matrix per frame (metres, float32).
`intrinsics.npy`: one `(fx, fy, cx, cy)` row per frame at the 832×480 model frame (`intrinsics_native.npy`: at 1280×720). Rows are
gzip JSONL. Every path inside a JSON file is relative to the data root. Every file and field is described in
[LAYOUT.md](https://huggingface.co/datasets/PeterDAI/NoPeBench/blob/main/LAYOUT.md).

## Licences

NoPeBench is released for research use. Renders, poses and geometry come from UnrealZoo scenes and follow UnrealZoo's terms and
those of the underlying Unreal Engine assets; no game assets are included. The WBench start images are under the MIT licence.
The moving-obstacle start images were made with an image-generation model from written prompts, and the captions by a
vision-language model. See the [dataset card](https://huggingface.co/datasets/PeterDAI/NoPeBench).
