#pragma once
#include <algorithm>
#include <vector>

#include "veclite/brute_force.hpp"

namespace veclite {

// Fraction of the true top-k that appears in the returned list (both ranked lists).
inline double recall_at_k(const std::vector<Neighbor>& truth, const std::vector<Neighbor>& got,
                          size_t k) {
    size_t kt = std::min(k, truth.size()), hits = 0;
    for (size_t i = 0; i < std::min(k, got.size()); ++i)
        for (size_t j = 0; j < kt; ++j)
            if (got[i].second == truth[j].second) { ++hits; break; }
    return kt ? double(hits) / double(kt) : 1.0;
}

}  // namespace veclite
