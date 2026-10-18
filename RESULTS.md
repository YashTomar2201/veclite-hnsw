# VecLite results

Dataset: SIFT10K (10,000 base vectors, 128-d, 100 queries, L2). HNSW built with `ef_construction=200`.
Machine: AMD Ryzen 5 5500U laptop, Windows 11, g++ 13.1 (MSYS2 UCRT64), `-O3 -march=native`, single thread (pinned to one core, high priority).

**Method.** Each configuration is timed with 15 repeats per query, interleaved across the query set (so a repeat never hits a warm cache from the previous repeat of the same query), after a warm-up pass. Per-query latency = median of its 15 repeats; the reported median and p99 are across the 100 queries. The whole benchmark was run 3 times and `results.csv` holds the per-row median of those runs (`scripts/aggregate.py`). Recall is deterministic (identical across runs); latency varied by <=8% between runs for most rows and up to ~23% for a few. Chart: `recall_vs_latency.png`.

## Brute force baseline

| Recall@10 | Median | p99 | Memory |
|---|---|---|---|
| 100% | 772.7 us | 780.5 us | 5.12 MB |

On random Gaussian data (128-d, `bench_random`): 0.78 ms per query at 10K vectors, 7.8 ms at 100K.

## HNSW sweep (M x ef_search)

| M | ef_search | recall@1 | recall@10 | median us | p99 us | speedup |
|---|---|---|---|---|---|---|
| 8 | 10 | 0.820 | 0.789 | 18.7 | 33.0 | 41.3x |
| 8 | 20 | 0.940 | 0.911 | 31.2 | 53.8 | 24.8x |
| 8 | 40 | 0.980 | 0.964 | 48.8 | 76.8 | 15.8x |
| 8 | 80 | 1.000 | 0.989 | 81.6 | 120.2 | 9.5x |
| 8 | 160 | 1.000 | 0.997 | 144.5 | 199.8 | 5.3x |
| 8 | 320 | 1.000 | 0.999 | 259.3 | 349.6 | 3.0x |
| 16 | 10 | 0.960 | 0.911 | 27.5 | 43.9 | 28.1x |
| 16 | 20 | 0.990 | 0.966 | 43.4 | 69.1 | 17.8x |
| 16 | 40 | 1.000 | 0.997 | 66.6 | 98.8 | 11.6x |
| 16 | 80 | 1.000 | 1.000 | 110.0 | 158.7 | 7.0x |
| 16 | 160 | 1.000 | 1.000 | 185.8 | 264.4 | 4.2x |
| 16 | 320 | 1.000 | 1.000 | 332.1 | 451.7 | 2.3x |
| 32 | 10 | 0.990 | 0.975 | 39.9 | 66.8 | 19.4x |
| 32 | 20 | 1.000 | 0.997 | 62.3 | 93.2 | 12.4x |
| 32 | 40 | 1.000 | 0.999 | 95.5 | 139.1 | 8.1x |
| 32 | 80 | 1.000 | 1.000 | 153.3 | 239.5 | 5.0x |
| 32 | 160 | 1.000 | 1.000 | 243.6 | 354.8 | 3.2x |
| 32 | 320 | 1.000 | 1.000 | 395.5 | 555.8 | 2.0x |

Speedup = brute-force median (772.7 us) / HNSW median.

## Build cost

| M | Build time | Per insert | Memory (vectors + links) | Layers |
|---|---|---|---|---|
| 8 | 1.48 s | 148 us | 5.66 MB | 5 |
| 16 | 1.89 s | 189 us | 6.15 MB | 4 |
| 32 | 2.94 s | 294 us | 7.14 MB | 3 |

## Earlier phase: flat single-layer graph

10K random 128-d vectors, M=16, search always starts at node 0 (no hierarchy):

| ef | 10 | 20 | 50 | 100 | 200 | 400 |
|---|---|---|---|---|---|---|
| recall@10 | 0.397 | 0.541 | 0.732 | 0.870 | 0.951 | 0.984 |

Build took 4.4 s (brute-force linking, O(N^2)).

## Tests

15 test cases, 128,437 assertions, all passing (also re-run with bounds-checked STL, `-D_GLIBCXX_DEBUG`, with no errors). Includes SIFT10K recall@10 = 100% at `ef_search=100`, `M=16`, `ef_construction=200`.

## Takeaways

- Headline: M=16, `ef_search=40` gives 99.7% recall@10 at 66.6 us, 11.6x faster than brute force.
- At the same ~99.7% recall@10, M=16 (66.6 us) and M=32 (62.3 us) are about equally fast (within run-to-run noise); M=8 needs 144.5 us, about 2.2x more. M=32 builds 1.6x slower and uses more memory for no clear latency gain.
- Recall@1 saturates well before recall@10.
- Speedup depends on `ef_search`: 41x at the fastest setting, 2.0x at the slowest. Quote it together with recall.
- Benchmark methodology matters: my first method (5 back-to-back repeats of each query) reported HNSW latencies ~1.4x lower than the interleaved method, while brute force was unchanged. Cache warmth is the likely cause, but I did not isolate it from the simultaneous priority/pinning change. The numbers above use the stricter method.

## Audit and fixes

Checks run: strict warnings (`-Wall -Wextra -Wpedantic -Wshadow -Wconversion`), bounds-checked STL build of the full test suite (`-D_GLIBCXX_ASSERTIONS -D_GLIBCXX_DEBUG`, no errors; ASan/UBSan are not available on this MinGW toolchain), and a repeat-run reproducibility check of the benchmark.

| Finding | Fix |
|---|---|
| Latency varied up to 86% between two identical bench runs; recall was identical | Interleaved 15 repeats, high priority, pinned core, 3 runs aggregated by median (`scripts/aggregate.py`) |
| Old numbers were ~1.4x optimistic for HNSW (likely cache warmth) | All tables, chart, README regenerated with the stricter method |
| `M < 2` gave an infinite level multiplier (`1/ln(1)`) | `Hnsw` constructor throws `std::invalid_argument` for `M < 2`, `ef_construction < M`, `dim == 0` |
| Inner-product / cosine path was untested | New test: normalised vectors + `Metric::InnerProduct`, recall > 0.9 |
| No tests for empty index, `k > N`, `ef < k`, duplicate vectors, level distribution | Added in `tests/test_edge_cases.cpp` (6 new cases, 15 total) |
| Phase-1 random-data brute-force benchmark was lost when `bench.cpp` was rewritten | Restored as `bench/bench_random.cpp` |
| Missing `<algorithm>` in `visited.hpp`; `-Wconversion` warning in `recall.hpp` | Fixed |
| No way to pre-allocate | Added `Hnsw::reserve(n)` (used by the benchmark) |

## Suggested upgrades (not done yet)

| Upgrade | Why |
|---|---|
| AVX2/FMA distance with several accumulators | GCC only partly vectorises the in-order float reduction; distance is the hot loop |
| Heuristic neighbour selection (paper Algorithm 4) | Better recall per link, fewer unreachable nodes; the code already isolates `select_neighbors()` |
| Flat, fixed-stride layer-0 link storage + prefetch | Removes `vector<vector<vector>>` pointer chasing; most of the per-query cache misses |
| SIFT1M benchmark, more queries | SIFT10K saturates at 100% recall and has only 100 queries |
| Compare against `hnswlib` | Shows how far a from-scratch version is from production |

## Caveats

- SIFT10K is easy and has only 100 queries, so recall saturates at 100% quickly and p99 is noisy.
- Single laptop, single thread; numbers will vary on other machines.
- Simple neighbour selection (M closest); distance code is plain scalar C++ (GCC auto-vectorises only part of it, no explicit SIMD).
