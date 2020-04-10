#pragma once

#include <cmath>
#include <cstddef>
#include <random>

namespace BP {

inline double exp(double x) noexcept {
    return std::exp(x);
}

inline double abs(double x) noexcept {
    return std::abs(x);
}

template <typename F>
double Fn(F&& f, double x) noexcept {
    return f(x);
}

template <typename F, typename M>
M Fn(F&& f, M const & x) noexcept {
    M out;
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = Fn(f, x[i]);
    }
    return out;
}

template <typename M>
auto mean(M const & x) {
    std::remove_cv_t<std::remove_reference_t<decltype(x[0])>> out {};
    for (std::size_t i = 0; i < x.size(); ++i) {
        out += x[i];
    }
    return out / x.size();
}

inline auto& random_engine() {
    thread_local std::default_random_engine gen(std::random_device{}());
    return gen;
}

template <typename M>
struct random_helper;

template <>
struct random_helper<double> {
    static double gen(double b, double e) {
        // [b, e)
        return std::uniform_real_distribution<double> { b, e } (random_engine());
    }
};

template <typename M>
struct random_helper {
    static M gen(double b, double e) {
        M out;
        for (std::size_t i = 0; i < out.size(); ++i) {
            out[i] = random_helper<std::decay_t<decltype(out[i])>>::gen(b, e);
        }
        return out;
    }
};

template <typename M>
inline M random(double b, double e) {
    return random_helper<M>::gen(b, e);
}

template <typename M>
M sigmoid(M const & x) {
    return 1.0 / (1.0 + Fn(exp, -x));
}

template <typename M>
M sigmoid_d(M const & x) {
    return x * (1.0 - x);
}

} // namespace BP
