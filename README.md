<p align="center">
  <img src="assets/logo.svg" width="84" alt="NoPeWorld logo">
</p>

<h1 align="center">Non-penetrating World Models</h1>

<p align="center">
  Official implementation of the paper <b>Non-penetrating World Models</b>:<br>
  <b>NoPeWorld</b>, a small head that stops video world models from walking through walls, and <b>NoPeBench</b>, the benchmark that tests it.
</p>

<p align="center">
  Yuhang Dai &nbsp;·&nbsp; Yinwei Wu &nbsp;·&nbsp; Ziyue Lin &nbsp;·&nbsp; Han Ji &nbsp;·&nbsp; Xingyi Yang<sup>*</sup>
  <br>
  Learning &amp; Generation Lab, The Hong Kong Polytechnic University
  <br>
  <sub><sup>*</sup>Corresponding author</sub>
</p>

<!-- arXiv placeholder: replace ARXIV_ID (badge, link and BibTeX) once the identifier is assigned. -->
<p align="center">
  <a href="https://arxiv.org/abs/ARXIV_ID"><img src="https://img.shields.io/badge/arXiv-ARXIV__ID-b31b1b?logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://nopeworld.github.io"><img src="https://img.shields.io/badge/Project-Page-2e8b4f" alt="Project page"></a>
  <a href="https://github.com/yuhangdai427/NoPeWorld"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github" alt="Code"></a>
  <a href="https://huggingface.co/PeterDAI/NoPeWorld"><img src="https://img.shields.io/badge/Models-NoPeWorld-ffd21e?logo=huggingface&logoColor=black" alt="Models"></a>
  <a href="https://huggingface.co/datasets/PeterDAI/NoPeBench"><img src="https://img.shields.io/badge/Data-NoPeBench-ffd21e?logo=huggingface&logoColor=black" alt="Data"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-3d7dd8" alt="License"></a>
</p>

<p align="center">
  <img src="assets/teaser.gif" width="600" alt="LingBot-World 2.0 with W held at a brick pillar: the original model walks through it, with NoPeWorld it stops at the bricks">
  <br>
  <sub><b>To today's world models, every wall is Platform 9¾.</b> LingBot-World 2.0, <code>W</code> held at the pillar between platforms 9 and 10: the original model walks through the bricks; + NoPeWorld stops at them. Full videos on the <a href="https://nopeworld.github.io">project page</a>.</sub>
</p>

## News

- **2026-11-21**: Code, trained heads for six world models and the NoPeBench dataset released.
- **2026-11-20**: Paper on arXiv.

## Highlights

- **A new problem.** Commanded into a wall, five of six open-source video world models walk right through it.
- **The DiT already knows.** A head of under 2M parameters reads frozen DiT states and replaces unsafe motions.
- **Six world models, two modes.** *Collision* reads the current denoising step, *prediction* the previous chunk.
- **NoPeBench.** Three tasks: true-history decisions, closed-loop generation on Unreal geometry, open-domain images.

<p align="center">
  <img src="assets/method.png" width="100%" alt="NoPeWorld training and inference">
  <br>
  <sub><b>Training:</b> every candidate motion is swept through Unreal Engine's true geometry and labelled; the head learns the labels from the frozen DiT's states. <b>Inference:</b> the raw model enters the wall; with NoPeWorld the head keeps or changes the command, and the camera slides along the wall.</sub>
</p>

## Results

Averaged over the six world models, NoPeWorld (collision mode) cuts the share of Task 2 videos that violate non-penetration from **74%** to **16%**. Per world model (Table 4 of the paper, %):

<table align="center">
  <thead>
    <tr>
      <th rowspan="2" align="left">World model</th>
      <th colspan="3">Task 2 violations &darr;</th>
      <th colspan="3">Overall &uarr;</th>
    </tr>
    <tr>
      <th>Raw</th><th>Ours (Coll)</th><th>Ours (Pred)</th>
      <th>Raw</th><th>Ours (Coll)</th><th>Ours (Pred)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>LingBot-World 2.0</td><td align="center">89</td><td align="center">7</td><td align="center">4</td><td align="center">26.0</td><td align="center">61.6</td><td align="center">66.7</td></tr>
    <tr><td>HY-WorldPlay</td><td align="center">86</td><td align="center">15</td><td align="center">21</td><td align="center">37.9</td><td align="center">54.3</td><td align="center">62.0</td></tr>
    <tr><td>Echo-WM</td><td align="center">86</td><td align="center">3</td><td align="center">5</td><td align="center">31.9</td><td align="center">62.7</td><td align="center">62.1</td></tr>
    <tr><td>Evoke-Turbo</td><td align="center">77</td><td align="center">30</td><td align="center">&ndash;</td><td align="center">34.3</td><td align="center">55.0</td><td align="center">&ndash;</td></tr>
    <tr><td>WorldPlay2</td><td align="center">11</td><td align="center">1</td><td align="center">4</td><td align="center">54.6</td><td align="center">52.1</td><td align="center">50.7</td></tr>
    <tr><td>Hy-Nomas</td><td align="center">95</td><td align="center">40</td><td align="center">45</td><td align="center">17.8</td><td align="center">43.2</td><td align="center">36.8</td></tr>
  </tbody>
</table>

<sub>Overall: mean of the Safe-Intent Score (Task 2) and the share of Task 3 videos that are safe and follow the input. Evoke-Turbo has no prediction mode (it keeps no memory between chunks). WorldPlay2, the only keystroke-driven model, natively retreats from walls rather than penetrating them. The text-prompt and external-depth baselines, SIS, Slide and Task 3 per group are in the paper.</sub>

<p align="center">
  <img src="assets/six_models.jpg" width="100%" alt="Six world models on one NoPeBench route, original and with NoPeWorld, with top views on the true map">
  <br>
  <sub>One NoPeBench route, <code>W</code> + <code>A</code> held into a wall. Original: five models pass through, WorldPlay2 backs away. + NoPeWorld: all six stop at the wall, where Unreal Engine's ground truth slides. Bottom: the paths on the true map.</sub>
</p>

## Installation

You need Linux and Python 3.10 or newer. **A GPU with at least 80 GB of memory is required; a 96 GB GPU is recommended, and WorldPlay2 needs it (peak ≈ 86 GiB, ≈ 89 GiB with external depth), as does training on Echo-WM (peak ≈ 88 GiB).**

```bash
git clone https://github.com/yuhangdai427/NoPeWorld && cd NoPeWorld
pip install -e .            # nopeworld, nopebench, world_models and the console scripts
pip install -e ".[eval]"    # + OpenCV for the Task 2 evaluator
```

The world models are not redistributed. Each runs in its own Python environment from its own official code and weights; install this package into each of them. [docs/INSTALL.md](docs/INSTALL.md) covers the environments, roots and evaluator models; [docs/WORLD_MODELS.md](docs/WORLD_MODELS.md) gives each world model's commit, weight revision, size and memory, and [docs/INFERENCE.md](docs/INFERENCE.md#time-per-route) the seconds per route of every method.

## Quick start

```bash
# 1. Heads and one task of NoPeBench (./weights and ./data by default)
nopebench-download --what weights
nopebench-download --what data --tasks task2

# 2. Generate a Task 2 route: raw model and NoPeWorld in collision mode
nopeworld-infer --generator hy_worldplay --method raw \
    --task task2 --cases Cabin_Lake__d001 --out runs/hy/raw
nopeworld-infer --generator hy_worldplay --method nopeworld_collision \
    --task task2 --cases Cabin_Lake__d001 --out runs/hy/collision

# 3. Evaluate on Unreal Engine geometry (in the evaluator environment: VGGT-Ω, SAM 2)
nopebench-eval --task task2 --run runs/hy/collision --raw-run runs/hy/raw \
    --routes Cabin_Lake__d001

# 4. Train a head (needs --split train val of the dataset)
nopeworld-train --generator hy_worldplay --mode collision --output runs/train/hy_collision
```

Drop `--cases` / `--routes` to run a whole task. Methods: `raw`, `text_prompt`, `external_depth`, `nopeworld_collision`, `nopeworld_prediction` ([docs/INFERENCE.md](docs/INFERENCE.md)). Training and evaluation: [docs/TRAINING.md](docs/TRAINING.md), [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Model zoo

One collision head and one prediction head per world model, each trained on the same 729 Unreal Engine videos and selected on validation videos of that world model. All heads are in [PeterDAI/NoPeWorld](https://huggingface.co/PeterDAI/NoPeWorld).

| World model | Backbone | DiT | Collision head | Prediction head |
|---|---|---|---|---|
| LingBot-World 2.0<br>`lingbot_world_2` | Wan2.2 | 18.5B | [blocks 30, 38, 39](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/lingbot_world_2/collision) · 1.72M | [block 30](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/lingbot_world_2/prediction) · 1.74M |
| HY-WorldPlay<br>`hy_worldplay` | HunyuanVideo-1.5 | 8.6B | [block 27](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/hy_worldplay/collision) · 0.95M | [block 33](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/hy_worldplay/prediction) · 0.97M |
| Echo-WM<br>`echo_wm` | LTX-2.3 | 19.8B | [block 42](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/echo_wm/collision) · 0.95M | [block 30](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/echo_wm/prediction) · 0.97M |
| Evoke-Turbo<br>`evoke_turbo` | Helios | 14.3B | [block 20](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/evoke_turbo/collision) · 1.07M | &ndash; |
| WorldPlay2<br>`worldplay2` | Wan2.2 | 2×17.5B | [block 30](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/worldplay2/collision) · 1.72M | [block 20](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/worldplay2/prediction) · 1.74M |
| Hy-Nomas<br>`hy_nomas` | HunyuanVideo-1.5 | 8.6B | [block 33](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/hy_nomas/collision) · 0.70M | [block 33](https://huggingface.co/PeterDAI/NoPeWorld/tree/main/heads/hy_nomas/prediction) · 0.71M |

Each head cell: the DiT blocks the head reads and the head's parameter count. Each head file carries its operating point (temperature, safety threshold, keep margin).

## NoPeBench

<p align="center">
  <img src="assets/nopebench.jpg" width="100%" alt="NoPeBench: Tasks 1 and 2 on held-out UnrealZoo scenes, Task 3 on open-domain images">
</p>

Every test command drives into a solid obstacle; open-path routes check that a method does not stop when nothing blocks the way. All scenes, assets and images are disjoint from training and validation.

| Task | Tests | Cases | Judged by |
|---|---|---|---|
| **Task&nbsp;1** | the decision on the true history (teacher forcing) | 100 routes, 32 held-out UnrealZoo scenes | Unreal Engine occupancy: missed, false, decision score |
| **Task&nbsp;2** | 5 s of closed-loop generation after a hand-off | the same 100 routes + 50 open routes | VGGT-Ω and SAM 2 on the true geometry: SIS, violations, slide |
| **Task&nbsp;3** | generation from one open-domain image | 81 static (WBench) + 50 moving obstacles, first and third person | VLM judges: violation rate |

Heads are trained on 729 videos of 44 UnrealZoo scenes and selected on 124 videos of 8 more. The dataset is stored as tar archives in [PeterDAI/NoPeBench](https://huggingface.co/datasets/PeterDAI/NoPeBench); `nopebench-download` fetches, checks and extracts the ones you ask for:

```bash
nopebench-download --what data --tasks task1 task2 task3    # the benchmark: 15 GB
nopebench-download --what data --split train                # training: 159 GB, 124 GB of it later-start crops
nopebench-download --what data --split val                  # validation: 5.6 GB
nopebench-download --what data                              # everything: 180 GB
```

Tasks, evaluators and metrics: [docs/BENCHMARK.md](docs/BENCHMARK.md). Archives and file layout: [docs/DATA.md](docs/DATA.md).

## Documentation

- [INSTALL.md](docs/INSTALL.md): environments, roots, evaluator models, tests
- [WORLD_MODELS.md](docs/WORLD_MODELS.md): the six world models: code, weights, environment, GPU memory
- [INFERENCE.md](docs/INFERENCE.md): the five methods, tasks, outputs, time per route
- [TRAINING.md](docs/TRAINING.md): training a head, crops, selection on validation
- [BENCHMARK.md](docs/BENCHMARK.md): tasks, evaluators, metrics, Task 3 VLM judges
- [DATA.md](docs/DATA.md): dataset archives and layout
- [REPRODUCE.md](docs/REPRODUCE.md): reproducing the paper's tables

## Citation

```bibtex
@misc{dai2026nopeworld,
  title         = {Non-penetrating World Models},
  author        = {Yuhang Dai and Yinwei Wu and Ziyue Lin and Han Ji and Xingyi Yang},
  year          = {2026},
  eprint        = {ARXIV_ID},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## License

The code is released under the [Apache-2.0 license](LICENSE). `run_chunks` in `world_models/lingbot_world_2.py` follows LingBot-World's own sampling loop and keeps LingBot-World's license. Each head is used with the world model it reads and follows that model's terms (see the [model card](https://huggingface.co/PeterDAI/NoPeWorld)). NoPeBench is released for research use (see the [dataset card](https://huggingface.co/datasets/PeterDAI/NoPeBench)). The world models, DA3, VGGT-Ω and SAM 2 keep their own licenses.

## Acknowledgements

NoPeWorld builds on the open releases of [LingBot-World 2.0](https://github.com/Robbyant/lingbot-world-v2) (Robbyant), [HY-WorldPlay](https://github.com/Tencent-Hunyuan/HY-WorldPlay) and [Hy-Nomas](https://github.com/tencent-hunyuan/hy-nomas) (Tencent Hunyuan), [Echo-WM](https://github.com/jd-opensource/JoyAI-Echo) (JD Open Source), [Evoke-Turbo](https://github.com/AlayaLab/Evoke) (AlayaLab) and [WorldPlay2](https://github.com/WorldPlay2/WorldPlay2). NoPeBench renders UnrealZoo scenes in Unreal Engine (Epic Games) and uses images from WBench. The baselines and evaluators use [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3), [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) and [SAM 2](https://github.com/facebookresearch/sam2). We thank their authors for making them available, and Microsoft Azure for supporting this work.
