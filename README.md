# GraphANN — Vamana Index + **AdaptiveVamana**

A clean, modular C++ implementation of the **Vamana graph-based approximate nearest neighbor (ANN) index**, extended into **AdaptiveVamana**: a workload-aware, cluster-routed ANN system.

Implements the core algorithm from [*DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) (NeurIPS 2019).

> ## AdaptiveVamana
> On SIFT1M, vs. the original baseline (96.6% Recall@10, 1510 distance comps, ~708 µs/query):
> - **At matched recall**: −23% distance computations, **−30% latency**, +42% throughput, −22% graph memory.
> - **Tuned for accuracy**: **Recall@10 96.6% → 98.4%**.
> - ~864× fewer candidate evaluations than exhaustive search.
>
> Enhancements (all flag-gated; baseline is fully recoverable):
> **F0** optimized search engine · **F1** K-Means query routing (route-only + IVF) ·
> **F3** adaptive beam width · **F4** edge-usage graph refinement · plus F2/F5/F6 (ablated).
>
> See **`docs/REPORT.md`** (architecture, algorithms, complexity, STAR narrative, resume bullets),
> **`docs/BENCHMARKS.md`** (all tables), **`docs/PROGRESS.md`** (per-feature change log).

### AdaptiveVamana quick start
```bash
# 1. Build the base Vamana graph (optionally edge-refined later)
build_index   --data base.fbin --output index.bin --R 32 --L 75 --alpha 1.2 --gamma 1.5

# 2. Offline K-Means clusters for routing (Feature 1)
build_clusters --data base.fbin --output clusters_64.bin --num-clusters 64

# 3. (Optional) Workload-aware edge refinement on a HELD-OUT set (Feature 4)
refine_index  --index index.bin --data base.fbin --workload learn.fbin \
              --output index_refined.bin --keep-min 20 --min-count 1

# 4. Final system: routing + adaptive beam (Features 1+3 on the refined graph)
search_index  --index index_refined.bin --data base.fbin --queries query.fbin --gt gt.ibin \
              --K 10 --final --clusters clusters_64.bin --clusters-to-search 2 \
              --min-beam 35 --max-beam 90 --patience 10 --epsilon 0.004
```

**Search modes** (`search_index`): default fixed-`L` · `--legacy-search` (original engine) ·
`--adaptive` (F3) · `--clusters … [--route-only | --clusters-to-search N]` (F1) ·
`--hubs H --random-entries R` (F2) · `--final` (routing + adaptive).
**Build options**: `--clusters … --diversity S` (F5) · `--noise RATIO` (F6).

---

## Algorithm Overview

### Build Phase
For each point (in a random order, parallelized with OpenMP):

1. **Greedy Search**: Search the current graph for the point itself, producing a candidate list of size `L`
2. **Robust Prune (α-RNG)**: Prune candidates to at most `R` diverse neighbors using the alpha-RNG rule — a candidate `c` is kept only if `dist(node, c) ≤ α · dist(c, n)` for all already-selected neighbors `n`
3. **Add Edges**: Set forward edges; add backward edges to each neighbor
4. **Degree Check**: If any neighbor's degree exceeds `γR`, prune its neighborhood

Per-node mutexes ensure correctness during parallel construction.

### Search Phase
Greedy beam search starting from a fixed start node, maintaining a candidate set bounded at size `L`. Returns the top-`K` closest points found.

### AdaptiveVamana search variants
The baseline greedy search above is the foundation; AdaptiveVamana layers three flag-gated improvements on top of it, sharing the same graph traversal. All are exposed through `search_index` (see the [AdaptiveVamana quick start](#adaptivevamana-quick-start)):

- **Fast engine (F0, default)** — same traversal as the baseline, but a flat sorted candidate pool + version-stamped visited buffer replace the original `std::set` (O(L²) frontier scan) and the per-query 1 MB `visited` allocation. Recall is bit-identical; only the data structures change. `--legacy-search` restores the original engine.
- **K-Means query routing (F1, `--clusters`)** — instead of always starting from the one fixed node, the query is matched to its nearest cluster medoids, which seed the walk close to the answer (fewer hops, no recall loss in *route-only* mode). `--clusters-to-search N` enables IVF restriction; `--route-only` keeps the full graph searchable.
- **Adaptive beam (F3, `--adaptive`)** — the beam width is chosen per query: the walk stops once the K-th-neighbor distance stops improving (within `--epsilon` for `--patience` expansions), bounded by `--min-beam`/`--max-beam`. Easy queries finish early; hard queries run longer.
- **Final system (`--final`)** — combines routing (F1) + adaptive beam (F3), typically on a graph that has been edge-refined (F4) via `refine_index`.

See `docs/REPORT.md` for the algorithms, complexity analysis, and the ablated features (F2/F5/F6).

### Parameters
| Parameter | Typical Range | Description |
|-----------|--------------|-------------|
| `R` | 32–64 | Max out-degree (graph connectivity) |
| `L` (build) | 75–200 | Search list size during construction (≥ R) |
| `α` (alpha) | 1.0–1.5 | RNG pruning threshold (> 1 keeps long-range edges) |
| `γ` (gamma) | 1.2–1.5 | Degree multiplier triggering neighbor pruning |
| `L` (search) | 10–200 | Search list size at query time (≥ K) |
| `K` | 1–100 | Number of nearest neighbors to return |

---

## Project Structure

```
graphann/
├── CMakeLists.txt              # Build config (C++17, OpenMP, -O3 -march=native)
├── README.md
├── download_sift.py            # Standalone SIFT1M downloader (Python)
├── docs/
│   ├── REPORT.md               # Technical report: architecture, algorithms, complexity, STAR
│   ├── BENCHMARKS.md           # All measured result tables
│   └── PROGRESS.md             # Per-feature change log (rationale + tradeoffs)
├── include/
│   ├── distance.h              # Squared L2 distance function
│   ├── io_utils.h              # fbin/ibin file loaders
│   ├── timer.h                 # Simple chrono-based timer
│   ├── kmeans.h                # K-Means clustering for query routing (Feature 1)
│   └── vamana_index.h          # VamanaIndex class declaration
├── scripts/
│   ├── convert_vecs.py         # Convert fvecs/ivecs → fbin/ibin
│   └── run_sift1m.sh           # One-command SIFT1M download, build & search
└── src/
    ├── distance.cpp            # Distance implementation
    ├── io_utils.cpp            # File I/O implementation
    ├── kmeans.cpp              # K-Means++ init + Lloyd + medoids (Feature 1)
    ├── vamana_index.cpp        # Core: greedy_search, robust_prune, build, search backends
    ├── build_index.cpp         # CLI: build Vamana graph from data
    ├── build_clusters.cpp      # CLI: train offline K-Means clusters (Feature 1)
    ├── refine_index.cpp        # CLI: workload-aware edge-usage refinement (Feature 4)
    └── search_index.cpp        # CLI: search + recall/latency evaluation (all search modes)
```

### Key files to study
- **`src/vamana_index.cpp`** — the core algorithm: `greedy_search()`, `robust_prune()`, `build()`, and the AdaptiveVamana search backends (fast / legacy / adaptive / routed / combined)
- **`include/vamana_index.h`** — data structures (adjacency list graph, per-node locks)
- **`src/kmeans.cpp`** / **`include/kmeans.h`** — K-Means routing (centroids + medoids)
- **`docs/REPORT.md`** — full technical write-up of the AdaptiveVamana enhancements

---

## Build

Requirements: C++17 compiler with OpenMP support (GCC ≥ 7, Clang ≥ 10).

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

This produces four executables:
- `build_index` — build the Vamana graph from a dataset
- `search_index` — search + recall/latency evaluation (all search modes)
- `build_clusters` — train offline K-Means clusters for query routing (Feature 1)
- `refine_index` — workload-aware edge-usage graph refinement (Feature 4)

---

## Quick Start — SIFT1M end-to-end

A single script downloads the [SIFT1M](http://corpus-texmex.irisa.fr/) dataset, converts it to binary format, builds a Vamana index, and runs search with recall evaluation:

```bash
./scripts/run_sift1m.sh
```

This will:
1. Build the project (cmake + make)
2. Download SIFT1M (1M base vectors, 10K queries, ground truth) into `tmp/sift/`
3. Convert `.fvecs`/`.ivecs` files to `.fbin`/`.ibin` format in `tmp/`
4. Build a Vamana index with default parameters (R=32, L=75, α=1.2, γ=1.5)
5. Run search at multiple `L` values and report recall@10, latency, and distance computations

Requires: `curl`, `python3` with `numpy`, and a C++17 compiler with OpenMP.

---

## Usage

### File Formats

**fbin** (float binary): Used for dataset and query vectors.
```
[4 bytes: uint32 npts] [4 bytes: uint32 dims] [npts * dims * 4 bytes: float32 row-major vectors]
```

**ibin** (int binary): Used for ground truth neighbor IDs.
```
[4 bytes: uint32 npts] [4 bytes: uint32 dims] [npts * dims * 4 bytes: uint32 row-major IDs]
```

Standard ANN benchmark datasets (SIFT, GIST, GloVe, etc.) are available in this format from [ANN Benchmarks](http://ann-benchmarks.com/) and [big-ann-benchmarks](https://big-ann-benchmarks.com/).

### Build an Index

```bash
./build_index \
  --data /path/to/base.fbin \
  --output /path/to/index.bin \
  --R 32 --L 75 --alpha 1.2 --gamma 1.5
```

### Search and Evaluate

```bash
./search_index \
  --index /path/to/index.bin \
  --data /path/to/base.fbin \
  --queries /path/to/query.fbin \
  --gt /path/to/gt.ibin \
  --K 10 \
  --L 10,20,30,50,75,100,150,200
```

Output:
```
=== Search Results (K=10) ===
       L     Recall@10   Avg Dist Cmps  Avg Latency (us)  P99 Latency (us)
--------------------------------------------------------------------------
      10         0.5432           320.5             125.3             412.7
      20         0.7891           580.2             198.4             623.1
      50         0.9234          1205.8             385.6            1102.3
     100         0.9812          2280.4             702.1            2015.8
     200         0.9965          4350.2            1305.7            3812.4
```

---

## Performance Notes

- **Parallelism**: OpenMP `parallel for schedule(dynamic)` for both build (point insertion) and search (queries)
- **Memory layout**: Contiguous row-major float arrays, 64-byte aligned for SIMD
- **Vectorization**: `-O3 -march=native` auto-vectorizes the L2 distance loop — no manual intrinsics needed
- **Lock granularity**: Per-node `std::mutex` — threads only contend when updating the *same* node's adjacency list
- **No external dependencies** beyond OpenMP

---

## References

- Subramanya et al., *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*, NeurIPS 2019

