// Brute-force latency on random Gaussian data (the Phase-1 baseline numbers).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "veclite/brute_force.hpp"
#include "veclite/random.hpp"

using namespace veclite;
using Clock = std::chrono::steady_clock;

int main() {
    const size_t dim = 128, k = 10, nq = 100, reps = 5;
    for (size_t n : {10000u, 100000u}) {
        auto base = random_gaussian(n, dim, 1);
        auto queries = random_gaussian(nq, dim, 2);
        BruteForce bf(dim);
        bf.add_batch(base.data(), n);
        for (size_t q = 0; q < nq; ++q) bf.search(queries.data() + q * dim, k);  // warm-up
        std::vector<double> us;
        for (size_t r = 0; r < reps; ++r)
            for (size_t q = 0; q < nq; ++q) {
                auto t0 = Clock::now();
                volatile size_t sz = bf.search(queries.data() + q * dim, k).size();
                (void)sz;
                us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
            }
        std::sort(us.begin(), us.end());
        std::printf("brute force N=%zu dim=%zu: median %.1f us\n", n, dim, us[us.size() / 2]);
    }
}
