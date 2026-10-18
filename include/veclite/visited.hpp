#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

namespace veclite {

// "Visited" set with O(1) reset: instead of clearing N flags per query, bump a
// generation counter; a node is visited iff its stamp equals the current one.
class VisitedList {
public:
    explicit VisitedList(size_t n = 0) : stamp_(n, 0) {}
    void resize(size_t n) { if (n > stamp_.size()) stamp_.resize(n, 0); }
    void next() {
        if (++gen_ == 0) {  // wrapped after 2^32 queries: really clear once
            std::fill(stamp_.begin(), stamp_.end(), 0);
            gen_ = 1;
        }
    }
    bool visited(uint32_t id) const { return stamp_[id] == gen_; }
    void mark(uint32_t id) { stamp_[id] = gen_; }

private:
    std::vector<uint32_t> stamp_;
    uint32_t gen_ = 0;
};

}  // namespace veclite
