#pragma once
// Phase 2 stepping stone: ONE layer, exact-kNN insertion, fixed entry point.
// Exists to show why a flat graph is not enough (see README / phase notes).
#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <vector>

#include "veclite/brute_force.hpp"
#include "veclite/distance.hpp"
#include "veclite/visited.hpp"

namespace veclite {

class FlatGraph {
public:
    FlatGraph(size_t dim, size_t M, Metric metric = Metric::L2)
        : dim_(dim), M_(M), M_max_(2 * M), metric_(metric) {}

    size_t size() const { return links_.size(); }

    void add(const float* v) {
        uint32_t id = static_cast<uint32_t>(size());
        // Find the M nearest existing nodes by brute force (O(N); HNSW fixes this later).
        std::vector<Neighbor> cands;
        cands.reserve(id);
        for (uint32_t i = 0; i < id; ++i) cands.emplace_back(dist(v, vec(i)), i);
        size_t m = std::min<size_t>(M_, cands.size());
        std::partial_sort(cands.begin(), cands.begin() + m, cands.end());

        data_.insert(data_.end(), v, v + dim_);
        links_.emplace_back();
        visited_.resize(id + 1);
        for (size_t j = 0; j < m; ++j) {
            uint32_t nb = cands[j].second;
            links_[id].push_back(nb);
            links_[nb].push_back(id);                       // reverse link
            if (links_[nb].size() > M_max_) shrink(nb);     // ...then cap the degree
        }
    }

    // Best-first search from `entry`, keeping the `ef` closest nodes seen.
    // candidates = MIN-heap (next node to expand is the closest);
    // results    = MAX-heap (top is the worst kept, the one to evict).
    std::vector<Neighbor> search_layer(const float* q, uint32_t entry, size_t ef) {
        using MinHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>;
        MinHeap candidates;
        std::priority_queue<Neighbor> results;
        visited_.next();

        float d0 = dist(q, vec(entry));
        candidates.emplace(d0, entry);
        results.emplace(d0, entry);
        visited_.mark(entry);

        while (!candidates.empty()) {
            Neighbor c = candidates.top();
            candidates.pop();
            if (results.size() >= ef && c.first > results.top().first) break;  // nothing closer left
            for (uint32_t nb : links_[c.second]) {
                if (visited_.visited(nb)) continue;
                visited_.mark(nb);
                float d = dist(q, vec(nb));
                if (results.size() < ef || d < results.top().first) {
                    candidates.emplace(d, nb);
                    results.emplace(d, nb);
                    if (results.size() > ef) results.pop();
                }
            }
        }
        std::vector<Neighbor> out(results.size());
        for (size_t i = out.size(); i-- > 0; results.pop()) out[i] = results.top();
        return out;
    }

    // Always starts from node 0: no hierarchy to give a better starting point.
    std::vector<Neighbor> search(const float* q, size_t k, size_t ef) {
        auto r = search_layer(q, 0, std::max(ef, k));
        if (r.size() > k) r.resize(k);
        return r;
    }

    const std::vector<std::vector<uint32_t>>& links() const { return links_; }

private:
    const float* vec(uint32_t id) const { return data_.data() + size_t(id) * dim_; }
    float dist(const float* a, const float* b) const { return distance(metric_, a, b, dim_); }

    void shrink(uint32_t node) {  // keep the M_max closest neighbours
        std::vector<Neighbor> v;
        for (uint32_t nb : links_[node]) v.emplace_back(dist(vec(node), vec(nb)), nb);
        std::partial_sort(v.begin(), v.begin() + M_max_, v.end());
        links_[node].clear();
        for (size_t i = 0; i < M_max_; ++i) links_[node].push_back(v[i].second);
    }

    size_t dim_, M_, M_max_;
    Metric metric_;
    std::vector<float> data_;
    std::vector<std::vector<uint32_t>> links_;
    VisitedList visited_;
};

}  // namespace veclite
