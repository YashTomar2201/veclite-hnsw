#pragma once
#include <cmath>
#include <cstddef>

namespace veclite {

enum class Metric { L2, InnerProduct };

// Squared L2: ranking is identical to true L2, so we skip the sqrt.
inline float l2_squared(const float* a, const float* b, size_t dim) {
    float s = 0.f;
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

inline float inner_product(const float* a, const float* b, size_t dim) {
    float s = 0.f;
    for (size_t i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

// Scale to unit length in place. Cosine similarity == inner product of
// normalised vectors, so normalise once at insert time, never per comparison.
inline void normalize(float* v, size_t dim) {
    float n = std::sqrt(inner_product(v, v, dim));
    if (n > 0.f)
        for (size_t i = 0; i < dim; ++i) v[i] /= n;
}

// Everything in the library is "smaller = closer".
inline float distance(Metric m, const float* a, const float* b, size_t dim) {
    return m == Metric::L2 ? l2_squared(a, b, dim) : 1.f - inner_product(a, b, dim);
}

}  // namespace veclite
