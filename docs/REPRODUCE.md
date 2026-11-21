# Reproducing the paper

The released heads are the paper's heads, and the default seed of each route (from the SHA-256 of its id) is the one the paper's
runs used. Data: `nopebench-download --what data --tasks task1 task2 task3`; heads: `nopebench-download --what weights`.

## Table 3: Task 1

```bash
for G in lingbot_world_2 hy_worldplay echo_wm evoke_turbo worldplay2 hy_nomas; do
  nopeworld-infer --generator $G --method nopeworld --task task1 --out runs/task1/$G
  nopebench-eval  --task task1 --run runs/task1/$G
done
```

`--method nopeworld` runs all of a world model's heads in one replay, so the `common` numbers of `task1_table.json` score both
modes on the decisions both make, as Table 3 does (Evoke-Turbo has its collision head only).

## Table 4: Tasks 2 and 3

For one world model `G` and every method `M` it supports:

```bash
nopeworld-infer --generator $G --method $M --task task2      --out runs/$G/$M/task2
nopeworld-infer --generator $G --method $M --task task2_open --out runs/$G/$M/task2_open
for g in static_first_person static_third_person dynamic_first_person dynamic_third_person; do
  nopeworld-infer --generator $G --method $M --task task3 --group $g --out runs/$G/$M/task3
done

nopebench-eval --task task2 --run runs/$G/$M/task2 --raw-run runs/$G/raw/task2 --open-run runs/$G/$M/task2_open
# Task 3: prepare, judge with both main judges, score (BENCHMARK.md)
```

- Run `raw` first: SIS reads each world model's speed from its raw run.
- The paper's Task 2 numbers use the default VGGT-Ω reconstruction; its Task 3 numbers average Claude Opus 5.5 and GPT-6 Astra.
- The paper's WorldPlay2 runs used compiled kernels; pass `--compile` to match them ([INFERENCE.md](INFERENCE.md#worldplay2-compiled-kernels)).
- GPU time per route and method: [INFERENCE.md](INFERENCE.md#time-per-route).

## Notes

- **Hardware and software.** Videos are reproducible to the bit only on the same GPU type and package versions. The release was
  checked on one 96 GB GPU against the paper's pipeline with the versions in [WORLD_MODELS.md](WORLD_MODELS.md): for the
  NoPeWorld methods, every frame and every decision of a test route matched for each world model.
- **Echo-WM on GPUs with compute capability 10 or higher** needs the attention change given in
  [WORLD_MODELS.md](WORLD_MODELS.md#echo-wm-echo_wm); the paper's runs on such GPUs used it.
- **External depth.** The paper's runs used a build of the `depth_anything_3` package with additional modules; results with the
  official package have not been compared.
- **Task 3 judges** are hosted models; a provider's model update can change their answers. The prompt itself is frozen and
  fingerprinted.
- **Training crops** are byte-identical to the shipped ones only when cut with `imageio-ffmpeg==0.6.0` on x86-64 Linux.
