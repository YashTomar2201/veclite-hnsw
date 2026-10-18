// Phase 2 experiment: recall of a flat (single-layer) graph vs brute force.
#include <chrono>
#include <cstdio>

#include "veclite/flat_graph.hpp"
#include "veclite/random.hpp"
#include "veclite/recall.hpp"

using namespace veclite;

int main(int argc, char** argv) {
    size_t n = argc > 1 ? std::atoi(argv[1]) : 10000, dim = argc > 2 ? std::atoi(argv[2]) : 128;
    size_t M = 16, k = 10, nq = 200;
    auto base = random_gaussian(n, dim, 1);
    auto queries = random_gaussian(nq, dim, 2);
    BruteForce bf(dim);
    bf.add_batch(base.data(), n);
    FlatGraph g(dim, M);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < n; ++i) g.add(base.data() + i * dim);
    std::printf("built N=%zu dim=%zu M=%zu in %.1fs\n", n, dim, M,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    for (size_t ef : {10, 20, 50, 100, 200, 400}) {
        double r = 0;
        for (size_t q = 0; q < nq; ++q) {
            const float* qv = queries.data() + q * dim;
            r += recall_at_k(bf.search(qv, k), g.search(qv, k, ef), k);
        }
        std::printf("  ef=%-4zu recall@10 = %.3f\n", ef, r / nq);
    }
}
