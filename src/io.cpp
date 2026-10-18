#include "veclite/io.hpp"

#include <cstdio>
#include <memory>
#include <stdexcept>

namespace veclite {
namespace {

template <typename T>
std::vector<T> read_vecs(const std::string& path, size_t& dim, size_t& n) {
    std::unique_ptr<FILE, int (*)(FILE*)> f(std::fopen(path.c_str(), "rb"), std::fclose);
    if (!f) throw std::runtime_error("cannot open " + path);

    int32_t d = 0;
    if (std::fread(&d, 4, 1, f.get()) != 1 || d <= 0) throw std::runtime_error("bad header: " + path);
    dim = static_cast<size_t>(d);

    std::fseek(f.get(), 0, SEEK_END);
    long bytes = std::ftell(f.get());
    size_t rec = 4 + dim * sizeof(T);
    if (bytes < 0 || static_cast<size_t>(bytes) % rec != 0) throw std::runtime_error("bad size: " + path);
    n = static_cast<size_t>(bytes) / rec;
    std::fseek(f.get(), 0, SEEK_SET);

    std::vector<T> out(n * dim);
    for (size_t i = 0; i < n; ++i) {
        int32_t di = 0;
        if (std::fread(&di, 4, 1, f.get()) != 1 || static_cast<size_t>(di) != dim ||
            std::fread(out.data() + i * dim, sizeof(T), dim, f.get()) != dim)
            throw std::runtime_error("truncated/inconsistent record in " + path);
    }
    return out;
}

}  // namespace

std::vector<float> read_fvecs(const std::string& p, size_t& dim, size_t& n) { return read_vecs<float>(p, dim, n); }
std::vector<int32_t> read_ivecs(const std::string& p, size_t& dim, size_t& n) { return read_vecs<int32_t>(p, dim, n); }

}  // namespace veclite
