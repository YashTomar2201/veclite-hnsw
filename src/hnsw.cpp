#include "veclite/hnsw.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <queue>
#include <stdexcept>

namespace veclite {

Hnsw::Hnsw(size_t dim, HnswParams p, Metric metric)
    : dim_(dim), p_(p), metric_(metric), mL_(p.M >= 2 ? 1.0 / std::log(double(p.M)) : 0.0), rng_(p.seed) {
    if (dim == 0) throw std::invalid_argument("dim must be > 0");
    if (p.M < 2) throw std::invalid_argument("M must be >= 2");
    if (p.ef_construction < p.M) throw std::invalid_argument("ef_construction must be >= M");
}

void Hnsw::reserve(size_t n) {
    data_.reserve(n * dim_);
    levels_.reserve(n);
    links_.reserve(n);
    visited_.resize(n);
}

size_t Hnsw::num_links() const {
    size_t n = 0;
    for (const auto& node : links_)
        for (const auto& layer : node) n += layer.size();
    return n;
}

int Hnsw::random_level() {
    // u in (0, 1]; level = floor(-ln(u) * mL) gives P(level >= l) = M^-l.
    double u = 1.0 - std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    return static_cast<int>(std::floor(-std::log(u) * mL_));
}

std::vector<Neighbor> Hnsw::search_layer(const float* q, uint32_t entry, size_t ef, int layer) {
    using MinHeap = std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>;
    MinHeap candidates;                     // min-heap: closest unexplored first
    std::priority_queue<Neighbor> results;  // max-heap: worst kept result on top
    visited_.next();

    float d0 = dist(q, vec(entry));
    candidates.emplace(d0, entry);
    results.emplace(d0, entry);
    visited_.mark(entry);

    while (!candidates.empty()) {
        Neighbor c = candidates.top();
        candidates.pop();
        if (results.size() >= ef && c.first > results.top().first) break;
        for (uint32_t nb : links_[c.second][layer]) {
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

std::vector<uint32_t> Hnsw::select_neighbors(const std::vector<Neighbor>& cands, size_t m) const {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < cands.size() && i < m; ++i) out.push_back(cands[i].second);
    return out;
}

void Hnsw::shrink(uint32_t node, int layer) {
    auto& l = links_[node][layer];
    std::vector<Neighbor> v;
    v.reserve(l.size());
    for (uint32_t nb : l) v.emplace_back(dist(vec(node), vec(nb)), nb);
    std::sort(v.begin(), v.end());
    l = select_neighbors(v, max_links(layer));
}

uint32_t Hnsw::add(const float* v) {
    const uint32_t id = static_cast<uint32_t>(size());
    const int level = random_level();
    data_.insert(data_.end(), v, v + dim_);
    levels_.push_back(level);
    links_.emplace_back(level + 1);
    visited_.resize(id + 1);

    if (id == 0) {
        entry_ = 0;
        max_level_ = level;
        return id;
    }

    // Greedy descent (ef=1) through the layers above the new node's top layer.
    uint32_t cur = entry_;
    for (int l = max_level_; l > level; --l) cur = search_layer(v, cur, 1, l)[0].second;

    // From min(level, max_level) down to 0: find ef_construction candidates, link to the
    // best M both ways, and shrink any neighbour that overflowed.
    for (int l = std::min(level, max_level_); l >= 0; --l) {
        auto found = search_layer(v, cur, p_.ef_construction, l);
        links_[id][l] = select_neighbors(found, p_.M);
        for (uint32_t nb : links_[id][l]) {
            links_[nb][l].push_back(id);
            if (links_[nb][l].size() > max_links(l)) shrink(nb, l);
        }
        cur = found[0].second;  // closest found seeds the next layer down
    }

    if (level > max_level_) {
        entry_ = id;
        max_level_ = level;
    }
    return id;
}

std::vector<Neighbor> Hnsw::search(const float* q, size_t k, size_t ef_search) {
    if (size() == 0) return {};
    uint32_t cur = entry_;
    for (int l = max_level_; l >= 1; --l) cur = search_layer(q, cur, 1, l)[0].second;
    auto r = search_layer(q, cur, std::max(ef_search, k), 0);
    if (r.size() > k) r.resize(k);
    return r;
}

}  // namespace veclite
