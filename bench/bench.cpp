// Benchmark: HNSW (M x ef_search sweep) vs brute force on SIFT10K.
// Usage: bench [dataset_prefix] [out.csv]
//   default prefix: data/siftsmall/siftsmall_   (expects base/query .fvecs, groundtruth .ivecs)
// Honest-numbers rules: Release build, single thread, warm-up pass, median of 5 repeats.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

#include "veclite/brute_force.hpp"
#include "veclite/hnsw.hpp"
#include "veclite/io.hpp"

using namespace veclite;
using Clock = std::chrono::steady_clock;

constexpr size_t kK = 10;
constexpr int kReps = 15;

static double elapsed_us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

struct Latency { double median_us, p99_us; };

// Repeats are interleaved (rep-major, query-minor) so a noisy stretch of time (turbo/thermal
// change, background task) is spread over all queries instead of wrecking a single query's
// repeats. Per-query latency = median over repeats; reported median / p99 are across queries.
static Latency measure(size_t nq, const std::function<void(size_t)>& run_query) {
    for (size_t q = 0; q < nq; ++q) run_query(q);  // warm-up
    std::vector<std::vector<double>> t(nq, std::vector<double>(kReps));
    for (int r = 0; r < kReps; ++r)
        for (size_t q = 0; q < nq; ++q) {
            auto t0 = Clock::now();
            run_query(q);
            t[q][r] = elapsed_us(t0, Clock::now());
        }
    std::vector<double> per_query(nq);
    for (size_t q = 0; q < nq; ++q) {
        std::sort(t[q].begin(), t[q].end());
        per_query[q] = t[q][kReps / 2];
    }
    std::sort(per_query.begin(), per_query.end());
    size_t p99 = std::min(nq - 1, size_t(std::ceil(0.99 * nq)) - 1);
    return {per_query[nq / 2], per_query[p99]};
}

// recall@k against ground-truth ids (gt row = ids sorted nearest-first)
static double recall(const std::vector<Neighbor>& got, const int32_t* gt, size_t k) {
    size_t hits = 0;
    for (size_t i = 0; i < std::min(k, got.size()); ++i)
        for (size_t j = 0; j < k; ++j)
            if (int32_t(got[i].second) == gt[j]) { ++hits; break; }
    return double(hits) / k;
}

// Reduce scheduler noise: high priority, pinned to one core (best effort).
static void stabilise_process() {
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 1ull << 2);
#endif
}

int main(int argc, char** argv) {
    stabilise_process();
    std::string prefix = argc > 1 ? argv[1] : "data/siftsmall/siftsmall_";
    std::string out_path = argc > 2 ? argv[2] : "results.csv";

    size_t dim, n, dq, nq, dg, ng;
    auto base = read_fvecs(prefix + "base.fvecs", dim, n);
    auto queries = read_fvecs(prefix + "query.fvecs", dq, nq);
    auto gt = read_ivecs(prefix + "groundtruth.ivecs", dg, ng);
    std::printf("dataset: N=%zu dim=%zu queries=%zu\n", n, dim, nq);

    // Brute force: baseline timing, and a cross-check that the file's ground truth is right.
    BruteForce bf(dim);
    bf.add_batch(base.data(), n);
    size_t gt_mismatch = 0;
    for (size_t q = 0; q < nq; ++q)
        if (recall(bf.search(queries.data() + q * dim, kK), gt.data() + q * dg, kK) < 1.0) ++gt_mismatch;
    std::printf("ground truth vs our brute force: %zu/%zu queries differ (ties)\n", gt_mismatch, nq);
    Latency bf_lat = measure(nq, [&](size_t q) {
        volatile auto r = bf.search(queries.data() + q * dim, kK).size();
        (void)r;
    });
    std::printf("brute force: median %.1f us, p99 %.1f us\n", bf_lat.median_us, bf_lat.p99_us);

    FILE* csv = std::fopen(out_path.c_str(), "w");
    if (!csv) { std::perror("open csv"); return 1; }
    std::fprintf(csv, "algo,M,ef_construction,ef_search,recall@1,recall@10,median_us,p99_us,build_s,"
                      "build_us_per_insert,memory_mb\n");
    std::fprintf(csv, "brute_force,0,0,0,1.0000,1.0000,%.2f,%.2f,0,0,%.2f\n", bf_lat.median_us,
                 bf_lat.p99_us, n * dim * 4 / 1e6);

    const size_t ef_construction = 200;
    for (size_t M : {8, 16, 32}) {
        HnswParams p;
        p.M = M;
        p.ef_construction = ef_construction;
        Hnsw idx(dim, p);
        idx.reserve(n);
        auto t0 = Clock::now();
        for (size_t i = 0; i < n; ++i) idx.add(base.data() + i * dim);
        double build_s = elapsed_us(t0, Clock::now()) / 1e6;
        double mem_mb = idx.memory_bytes() / 1e6;
        std::printf("\nM=%zu  build %.2fs (%.0f us/insert)  memory %.1f MB  layers=%d\n", M, build_s,
                    build_s * 1e6 / n, mem_mb, idx.max_level() + 1);

        for (size_t ef : {10, 20, 40, 80, 160, 320}) {
            double r1 = 0, r10 = 0;
            for (size_t q = 0; q < nq; ++q) {
                auto got = idx.search(queries.data() + q * dim, kK, ef);
                r1 += recall(got, gt.data() + q * dg, 1);
                r10 += recall(got, gt.data() + q * dg, kK);
            }
            r1 /= nq;
            r10 /= nq;
            Latency lat = measure(nq, [&](size_t q) {
                volatile auto r = idx.search(queries.data() + q * dim, kK, ef).size();
                (void)r;
            });
            std::printf("  ef=%-3zu recall@1 %.3f recall@10 %.3f  median %6.1f us  p99 %6.1f us  speedup %5.1fx\n",
                        ef, r1, r10, lat.median_us, lat.p99_us, bf_lat.median_us / lat.median_us);
            std::fprintf(csv, "hnsw,%zu,%zu,%zu,%.4f,%.4f,%.2f,%.2f,%.3f,%.1f,%.2f\n", M, ef_construction,
                         ef, r1, r10, lat.median_us, lat.p99_us, build_s, build_s * 1e6 / n, mem_mb);
        }
    }
    std::fclose(csv);
    std::printf("\nwrote %s\n", out_path.c_str());
}
