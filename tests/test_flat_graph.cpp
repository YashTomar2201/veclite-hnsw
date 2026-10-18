#include "doctest.h"
#include "veclite/flat_graph.hpp"
#include "veclite/random.hpp"
#include "veclite/recall.hpp"

using namespace veclite;

TEST_CASE("flat graph: degree cap and valid links") {
    const size_t n = 500, dim = 16, M = 8;
    auto base = random_gaussian(n, dim, 7);
    FlatGraph g(dim, M);
    for (size_t i = 0; i < n; ++i) g.add(base.data() + i * dim);
    for (const auto& l : g.links()) {
        CHECK(l.size() <= 2 * M);
        for (uint32_t nb : l) CHECK(nb < n);
    }
}

TEST_CASE("flat graph: >80% recall@10 on 10K random 128-d vectors at ef=100") {
    const size_t n = 10000, dim = 128, M = 16, k = 10, nq = 100;
    auto base = random_gaussian(n, dim, 1);
    auto queries = random_gaussian(nq, dim, 2);
    BruteForce bf(dim);
    bf.add_batch(base.data(), n);
    FlatGraph g(dim, M);
    for (size_t i = 0; i < n; ++i) g.add(base.data() + i * dim);

    double r10 = 0, r100 = 0;
    for (size_t q = 0; q < nq; ++q) {
        const float* qv = queries.data() + q * dim;
        auto truth = bf.search(qv, k);
        r10 += recall_at_k(truth, g.search(qv, k, 10), k);
        r100 += recall_at_k(truth, g.search(qv, k, 100), k);
    }
    CHECK(r100 / nq > 0.80);
    CHECK(r100 > r10);  // wider beam => better recall
}
