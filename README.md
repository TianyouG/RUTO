# Code of paper
**On Optimizing Route-Responsive Urban Tree Placement in City-Scale Pedestrian Networks (SIGKDD 2026)**

This repository provides a C++17 command-line tool (`treeopt`) for running the
TreePlacement optimization pipeline on pedestrian networks:
graph I/O, neighbourhood-weight construction (hard/soft kernels), certified
sandwich approximation optimization (with optional proxy baseline / caps), and Monte Carlo
evaluation. Outputs are written as a TSV report plus a mirrored console log.

## Build

Requirements: a C++17 compiler (e.g. `g++` or `clang++`).

```bash
g++ -std=c++17 -O2 -Wall -Wextra main.cpp alg.cpp grid.cpp graphbin.cpp -o treeopt
```

See the full flag list:

```bash
./treeopt --help
```

## Inputs

- `--edges <file>`: edge list text file, one edge per line: `u v`
- `--graph_bin_in <file>`: load a packed CSR graph (skips text parsing)
- `--make_grid rows=<R>,cols=<C>[,diag=0|1][,torus=0|1]`: generate a synthetic grid graph
- `--n <int>`: node count (optional; inferred from `--edges` when omitted)
- `--one_based 0|1`: set to `1` if node IDs start at 1 (default `0`)
- `--undirected 0|1`: treat input as undirected (default `1`)
- Optional subsets:
  - `--cands <file>`: candidate node list (default = all nodes)
  - `--blocked <file>`: blocked node list (excluded nodes)

## Core Options

- Method selection:
  - `--method <name>`: `sandwich` (default), `random_k`, `regular_spacing_grid`, `greedy_original_full`, ...
  - `--margs "k=...,corridor_r=...,..."`: method-specific overrides
  - `--list-methods`: print available methods
- Budget / objective:
  - `--k <int>`: number of facilities to place
  - `--psi <spec>`: node response compression (default `clip:1.0`)
    - `clip:C | expsat:beta | log1p | power:alpha`
  - `--lambda <float>`: shortest-path penalty weight
- Neighbourhood weights (choose one, or load a cache):
  - `--hard_cutoff <D>`: hop-radius hard cutoff
  - `--soft_decay exp:<alpha>:<R>:<eps>` or `gauss:<sigma>`: soft kernel
  - `--save_weights 0|1`, `--load_weights 0|1`, `--weights_bin <file>`: cache control
- Certified optimization / bounds:
  - `--ub_mode CertifiedUnion|CappedUnion|ProxyCorridor|SumDeprecated`
  - `--K_paths_per_agent <int>`: number of paths per agent
  - `--corridor_r <int>`: corridor width used by path factory / certificates
  - `--corridor_band_r <int>`: geometric corridor radius (`-1` disables; `0` centerline only)
  - `--cap_mode Auto|PathlenOnly|TopkLocal|GlobalAll`, `--cap_topk_k <int>`
- Sampling / reproducibility:
  - `--M_agents <int>`: training sample sizes
  - `--M_eval <int>`: evaluation samples (`0` reuses training samples)
  - `--seed <uint64>`: global RNG seed
  - `--graph_bin_out <file>`: export a packed CSR graph for reuse
- Output paths:
  - `--out_dir <dir>`, `--tag <name>`: results go to `<out_dir>/<tag>/`
  - `--paths_mode smart|legacy`: enable/disable automatic path rewrites (default `smart`)

## Outputs

By default results are written under `<out_dir>/<tag>/`:

- `report.tsv`: run metadata + pick/UB/LB statistics
- `stdout.txt`: mirrored console log
- `weights.bin`: optional cached neighbourhood weights

## Experiment Examples

Create a graph (synthetic grid) and write an edge list:

```bash
./treeopt --make_grid rows=64,cols=64 --emit_edges grid64.txt --paths_mode legacy
```

Run the optimizer on this graph:

```bash
./treeopt --edges grid64.txt --method sandwich --ub_mode CertifiedUnion \
  --k 50 --hard_cutoff 2 --psi clip:1.0 --lambda 0.5 \
  --M_agents 100 --K_paths_per_agent 10 --M_eval 100 --reuse_eval_samples 0 \
  --out_dir out --tag grid64 --paths_mode legacy
```
