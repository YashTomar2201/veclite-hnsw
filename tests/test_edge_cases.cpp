#include <stdexcept>
#include <algorithm>
#include <cmath>

#include "doctest.h"
#include "veclite/hnsw.hpp"
#include "veclite/random.hpp"
#include "veclite/recall.hpp"

using namespace veclite;

TEST_CASE("hnsw: invalid parameters are rejected") {
    HnswParams p;
    p.M = 1;
    CHECK_THROWS_AS(Hnsw(8, p), std::invalid_argument);
    p.M = 16;
    p.ef_construction = 4;
    CHECK_THROWS_AS(Hnsw(8, p), std::invalid_argument);
    CHECK_THROWS_AS(Hnsw(0), std::invalid_argument);
}

TEST_CASE("hnsw: empty and tiny indexes") {
    Hnsw idx(4);
    float q[4] = {0, 0, 0, 0};
    CHECK(idx.search(q, 5, 50).empty());

    auto data = random_gaussian(3, 4, 5);
    for (size_t i = 0; i < 3; ++i) idx.add(data.data() + i * 4);
    auto r = idx.search(q, 10, 50);  // k > N: return everything
    CHECK(r.size() == 3);
    CHECK(std::is_sorted(r.begin(), r.end()));
}

TEST_CASE("hnsw: ef_search < k still returns k results") {
    auto base = random_gaussian(500, 8, 9);
    Hnsw idx(8);
    for (size_t i = 0; i < 500; ++i) idx.add(base.data() + i * 8);
    CHECK(idx.search(base.data(), 10, 1).size() == 10);
}

TEST_CASE("hnsw: duplicate vectors do not break the index") {
    const size_t dim = 8;
    auto base = random_gaussian(50, dim, 11);
    Hnsw idx(dim);
    for (int rep = 0; rep < 20; ++rep)
        for (size_t i = 0; i < 50; ++i) idx.add(base.data() + i * dim);
    auto r = idx.search(base.data(), 5, 50);
    REQUIRE(r.size() == 5);
    CHECK(r[0].first == doctest::Approx(0.f));
}

TEST_CASE("hnsw: cosine via normalised vectors + inner-product metric") {
    const size_t n = 3000, dim = 32, k = 10, nq = 50;
    auto base = random_gaussian(n, dim, 21);
    auto queries = random_gaussian(nq, dim, 22);
    for (size_t i = 0; i < n; ++i) normalize(base.data() + i * dim, dim);
    for (size_t i = 0; i < nq; ++i) normalize(queries.data() + i * dim, dim);

    BruteForce bf(dim, Metric::InnerProduct);
    bf.add_batch(base.data(), n);
    HnswParams p;
    p.ef_construction = 100;
    Hnsw idx(dim, p, Metric::InnerProduct);
    idx.reserve(n);
    for (size_t i = 0; i < n; ++i) idx.add(base.data() + i * dim);

    double r = 0;
    for (size_t i = 0; i < nq; ++i)
        r += recall_at_k(bf.search(queries.data() + i * dim, k), idx.search(queries.data() + i * dim, k, 100), k);
    CHECK(r / nq > 0.9);
}

TEST_CASE("hnsw: level distribution is geometric (P(level>=1) ~ 1/M)") {
    HnswParams p;
    p.M = 8;
    p.ef_construction = 20;
    Hnsw idx(2, p);
    auto base = random_gaussian(8000, 2, 3);
    for (size_t i = 0; i < 8000; ++i) idx.add(base.data() + i * 2);
    size_t above0 = 0;
    for (uint32_t i = 0; i < idx.size(); ++i) above0 += idx.level_of(i) >= 1;
    double frac = double(above0) / double(idx.size());
    CHECK(frac == doctest::Approx(1.0 / 8).epsilon(0.15));
}
