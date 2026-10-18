#pragma once
#include <cstdint>
#include <random>
#include <vector>

#include "veclite/brute_force.hpp"  // Neighbor
#include "veclite/distance.hpp"
#include "veclite/visited.hpp"

namespace veclite {

struct HnswParams {
    size_t M = 16;                // max links per node on layers >= 1 (layer 0 allows 2*M)
    size_t ef_construction = 200; // candidate list size while building
    uint32_t seed = 42;
};

// Hierarchical Navigable Small World index (Malkov & Yashunin).
// Throws std::invalid_argument if M < 2 (mL = 1/ln(M) would be infinite) or ef_construction < M.
// Not thread-safe: search() reuses an internal visited list.
class Hnsw {
public:
    Hnsw(size_t dim, HnswParams p = {}, Metric metric = Metric::L2);

    void reserve(size_t n);        // pre-allocate vector storage for n points
    uint32_t add(const float* v);  // Algorithm 1
    std::vector<Neighbor> search(const float* q, size_t k, size_t ef_search);  // Algorithm 5

    // Introspection (tests, benchmark memory accounting).
    size_t size() const { return levels_.size(); }
    size_t dim() const { return dim_; }
    int max_level() const { return max_level_; }
    uint32_t entry_point() const { return entry_; }
    int level_of(uint32_t id) const { return levels_[id]; }
    const std::vector<uint32_t>& neighbors(uint32_t id, int layer) const { return links_[id][layer]; }
    size_t max_links(int layer) const { return layer == 0 ? 2 * p_.M : p_.M; }
    const HnswParams& params() const { return p_; }
    size_t num_links() const;
    size_t memory_bytes() const { return data_.size() * sizeof(float) + num_links() * sizeof(uint32_t); }

private:
    const float* vec(uint32_t id) const { return data_.data() + size_t(id) * dim_; }
    float dist(const float* a, const float* b) const { return distance(metric_, a, b, dim_); }
    int random_level();
    // Best-first search on one layer (Algorithm 2). Result sorted nearest-first.
    std::vector<Neighbor> search_layer(const float* q, uint32_t entry, size_t ef, int layer);
    // Algorithm 3 (simple): the m closest of `cands` (already sorted nearest-first).
    std::vector<uint32_t> select_neighbors(const std::vector<Neighbor>& cands, size_t m) const;
    void shrink(uint32_t node, int layer);  // re-select the best max_links(layer)

    size_t dim_;
    HnswParams p_;
    Metric metric_;
    double mL_;
    std::mt19937 rng_;

    std::vector<float> data_;                                // flat: data_[id * dim + j]
    std::vector<int> levels_;                                // top layer of each node
    std::vector<std::vector<std::vector<uint32_t>>> links_;  // links_[node][layer]
    uint32_t entry_ = 0;
    int max_level_ = -1;
    VisitedList visited_;
};

}  // namespace veclite
