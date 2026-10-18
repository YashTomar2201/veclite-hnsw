#pragma once
#include <algorithm>
#include <cstdint>
#include <queue>
#include <utility>
#include <vector>

#include "veclite/distance.hpp"

namespace veclite {

using Neighbor = std::pair<float, uint32_t>;  // (distance, id)

// Exact search. Ground truth for every recall number in this project.
class BruteForce {
public:
    BruteForce(size_t dim, Metric metric = Metric::L2) : dim_(dim), metric_(metric) {}

    // Vectors live in ONE flat array (cache friendly), id = insertion order.
    uint32_t add(const float* v) {
        data_.insert(data_.end(), v, v + dim_);
        return static_cast<uint32_t>(size() - 1);
    }
    void add_batch(const float* v, size_t n) {
        data_.insert(data_.end(), v, v + n * dim_);
    }

    size_t size() const { return data_.size() / dim_; }
    size_t dim() const { return dim_; }
    const float* vector(uint32_t id) const { return data_.data() + id * dim_; }

    // Top-k by a max-heap of size k: O(N log k), the heap top is the worst kept.
    // Result sorted nearest-first.
    std::vector<Neighbor> search(const float* q, size_t k) const {
        std::priority_queue<Neighbor> heap;
        for (size_t i = 0; i < size(); ++i) {
            float d = distance(metric_, q, vector(static_cast<uint32_t>(i)), dim_);
            if (heap.size() < k) {
                heap.emplace(d, static_cast<uint32_t>(i));
            } else if (d < heap.top().first) {
                heap.pop();
                heap.emplace(d, static_cast<uint32_t>(i));
            }
        }
        std::vector<Neighbor> out(heap.size());
        for (size_t i = out.size(); i-- > 0; heap.pop()) out[i] = heap.top();
        return out;
    }

private:
    size_t dim_;
    Metric metric_;
    std::vector<float> data_;
};

}  // namespace veclite
