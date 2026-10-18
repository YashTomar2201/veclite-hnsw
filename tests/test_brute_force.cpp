#include <algorithm>
#include <cmath>

#include "doctest.h"
#include "veclite/brute_force.hpp"
#include "veclite/random.hpp"

using namespace veclite;

TEST_CASE("l2_squared / inner_product / normalize basics") {
    float a[3] = {1, 2, 3}, b[3] = {4, 6, 3};
    CHECK(l2_squared(a, b, 3) == doctest::Approx(25.f));
    CHECK(inner_product(a, b, 3) == doctest::Approx(25.f));
    normalize(a, 3);
    CHECK(inner_product(a, a, 3) == doctest::Approx(1.f));
}

TEST_CASE("brute-force top-k matches a naive full sort (100 random queries)") {
    const size_t n = 2000, dim = 32, k = 10, nq = 100;
    auto base = random_gaussian(n, dim, 1);
    auto queries = random_gaussian(nq, dim, 2);
    BruteForce bf(dim);
    bf.add_batch(base.data(), n);
    REQUIRE(bf.size() == n);

    for (size_t qi = 0; qi < nq; ++qi) {
        const float* q = queries.data() + qi * dim;
        std::vector<Neighbor> all;
        for (size_t i = 0; i < n; ++i)
            all.emplace_back(l2_squared(q, base.data() + i * dim, dim), (uint32_t)i);
        std::sort(all.begin(), all.end());

        auto got = bf.search(q, k);
        REQUIRE(got.size() == k);
        for (size_t j = 0; j < k; ++j) CHECK(got[j].second == all[j].second);
    }
}

TEST_CASE("k larger than N returns everything, sorted") {
    auto base = random_gaussian(5, 4, 3);
    BruteForce bf(4);
    bf.add_batch(base.data(), 5);
    auto r = bf.search(base.data(), 10);
    REQUIRE(r.size() == 5);
    CHECK(r[0].second == 0);  // a point is its own nearest neighbour
    CHECK(std::is_sorted(r.begin(), r.end()));
}
