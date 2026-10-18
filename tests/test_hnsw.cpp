#include <cstdio>
#include <queue>

#include "doctest.h"
#include "veclite/hnsw.hpp"
#include "veclite/io.hpp"
#include "veclite/random.hpp"
#include "veclite/recall.hpp"

using namespace veclite;

namespace {
Hnsw build_random(size_t n, size_t dim, HnswParams p, uint32_t seed = 1) {
    auto base = random_gaussian(n, dim, seed);
    Hnsw idx(dim, p);
    for (size_t i = 0; i < n; ++i) idx.add(base.data() + i * dim);
    return idx;
}
}  // namespace

TEST_CASE("hnsw structure: valid ids, degree caps, layer consistency") {
    HnswParams p;
    p.M = 8;
    p.ef_construction = 50;
    auto idx = build_random(3000, 16, p);
    REQUIRE(idx.size() == 3000);
    CHECK(idx.max_level() >= 1);  // with 3000 nodes and M=8 we expect several layers
    CHECK(idx.level_of(idx.entry_point()) == idx.max_level());

    for (uint32_t id = 0; id < idx.size(); ++id) {
        for (int l = 0; l <= idx.level_of(id); ++l) {
            const auto& nbrs = idx.neighbors(id, l);
            CHECK(nbrs.size() <= idx.max_links(l));
            for (uint32_t nb : nbrs) {
                REQUIRE(nb < idx.size());
                CHECK(nb != id);
                CHECK(idx.level_of(nb) >= l);  // a layer-l link must land on a layer-l node
            }
        }
    }
}

TEST_CASE("hnsw: all nodes reachable from the entry point on layer 0") {
    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    auto idx = build_random(3000, 16, p);
    std::vector<char> seen(idx.size(), 0);
    std::queue<uint32_t> q;
    q.push(idx.entry_point());
    seen[idx.entry_point()] = 1;
    size_t count = 1;
    while (!q.empty()) {
        uint32_t u = q.front();
        q.pop();
        for (uint32_t v : idx.neighbors(u, 0))
            if (!seen[v]) { seen[v] = 1; ++count; q.push(v); }
    }
    CHECK(count == idx.size());
}

TEST_CASE("hnsw: recall rises with ef_search (random 128-d, 10K)") {
    const size_t n = 10000, dim = 128, k = 10, nq = 100;
    HnswParams p;
    p.M = 16;
    p.ef_construction = 100;
    auto idx = build_random(n, dim, p);
    auto base = random_gaussian(n, dim, 1);
    auto queries = random_gaussian(nq, dim, 2);
    BruteForce bf(dim);
    bf.add_batch(base.data(), n);

    double prev = 0, last = 0;
    for (size_t ef : {10, 40, 160}) {
        double r = 0;
        for (size_t qi = 0; qi < nq; ++qi) {
            const float* q = queries.data() + qi * dim;
            r += recall_at_k(bf.search(q, k), idx.search(q, k, ef), k);
        }
        r /= nq;
        CHECK(r >= prev - 0.01);  // monotone up to noise
        prev = last = r;
    }
    CHECK(last > 0.9);
}

TEST_CASE("hnsw: SIFT10K recall@10 > 95% at ef_search=100, M=16") {
    const std::string dir = std::string(VECLITE_DATA_DIR) + "/siftsmall/siftsmall_";
    size_t dim, n, dq, nq, dg, ng;
    std::vector<float> base, queries;
    std::vector<int32_t> gt;
    try {
        base = read_fvecs(dir + "base.fvecs", dim, n);
        queries = read_fvecs(dir + "query.fvecs", dq, nq);
        gt = read_ivecs(dir + "groundtruth.ivecs", dg, ng);
    } catch (const std::exception& e) {
        MESSAGE("SIFT10K not found, skipping: " << e.what());
        return;
    }
    REQUIRE(n == 10000);
    REQUIRE(dim == 128);

    HnswParams p;
    p.M = 16;
    p.ef_construction = 200;
    Hnsw idx(dim, p);
    for (size_t i = 0; i < n; ++i) idx.add(base.data() + i * dim);

    const size_t k = 10;
    double r = 0;
    for (size_t qi = 0; qi < nq; ++qi) {
        auto got = idx.search(queries.data() + qi * dim, k, 100);
        size_t hits = 0;
        for (auto& g : got)
            for (size_t j = 0; j < k; ++j)
                if (int32_t(g.second) == gt[qi * dg + j]) { ++hits; break; }
        r += double(hits) / k;
    }
    r /= nq;
    std::printf("  SIFT10K recall@10 (ef=100, M=16) = %.4f\n", r);
    CHECK(r > 0.95);
}
