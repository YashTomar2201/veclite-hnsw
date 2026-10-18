#pragma once
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace veclite {

// n vectors of N(0,1) components, flat row-major: data[i * dim + j].
inline std::vector<float> random_gaussian(size_t n, size_t dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> data(n * dim);
    for (auto& x : data) x = nd(rng);
    return data;
}

}  // namespace veclite
