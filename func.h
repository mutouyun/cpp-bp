#pragma once

#include <utility>
#include <cmath>

#include "matrix.h"

namespace BP {

template <typename Func, size_t N>
vector_t<N> && F(Func&& f, vector_t<N> && m1) {
    for (size_t i = 0; i < N; ++i) {
        m1[i] = f(m1[i]);
    }
    return std::move(m1);
}

template <typename Func, size_t N>
vector_t<N> F(Func&& f, vector_t<N> const & m1) {
    return F(f, vector_t(m1));
}

double sigmoid(double x) {
    return 1 / (1 + std::exp(-x));
}

double sigmoid_d(double x) {
    return x * (1 - x);
}

double heaviside(double x) {
    return (x < 0.0) ? 0.0 : 1.0;
}

} // namespace BP
