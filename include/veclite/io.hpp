#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace veclite {

// TEXMEX .fvecs / .ivecs: each record = int32 d, then d float32 (or int32).
// Returns a flat row-major array; `dim` and `n` are filled in. Throws on error.
std::vector<float> read_fvecs(const std::string& path, size_t& dim, size_t& n);
std::vector<int32_t> read_ivecs(const std::string& path, size_t& dim, size_t& n);

}  // namespace veclite
