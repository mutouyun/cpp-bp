#pragma once

#include <array>
#include <iostream>
#include <utility>
#include <type_traits>

namespace BP {

template <size_t N>
class vector_t : public std::array<double, N> {
    using base_t = std::array<double, N>;

public:
    template <typename... T>
    vector_t(T&&... args)
        : base_t { static_cast<double>(args)... } {}

    vector_t& operator+=(vector_t const & D) {
        return *this = *this + D;
    }

    vector_t& fill(double v) {
        base_t::fill(v);
        return *this;
    }
};

template <typename... T>
vector_t(T&&... args) -> vector_t<sizeof... (T)>;

template <size_t M, size_t N>
class matrix_t : public std::array<vector_t<N>, M> {
    using base_t = std::array<vector_t<N>, M>;

public:
    template <typename... T>
    matrix_t(T&&... args)
        : base_t { std::forward<T>(args)... } {}
};

template <template <size_t> class... Vec, size_t N>
matrix_t(Vec<N> const &... args) -> matrix_t<sizeof... (Vec), N>;

template <size_t M, size_t N>
matrix_t<N, M> transpose(matrix_t<M, N> const & m1) {
    matrix_t<N, M> output;
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            output[n][m] = m1[m][n];
        }
    }
    return output;
}

template <size_t N>
vector_t<N> operator+(vector_t<N> const & m1, vector_t<N> const & m2) {
    vector_t<N> output;
    for (size_t i = 0; i < N; ++i) {
        output[i] = m1[i] + m2[i];
    }
    return output;
}

template <size_t N>
vector_t<N> operator-(vector_t<N> const & m1, vector_t<N> const & m2) {
    vector_t<N> output;
    for (size_t i = 0; i < N; ++i) {
        output[i] = m1[i] - m2[i];
    }
    return output;
}

template <size_t N>
vector_t<N> operator*(vector_t<N> const & m1, vector_t<N> const & m2) {
    vector_t<N> output;
    for (size_t i = 0; i < N; ++i) {
        output[i] = m1[i] * m2[i];
    }
    return output;
}

template <size_t M, size_t N>
vector_t<M> operator*(matrix_t<M, N> const & m1, vector_t<N> const & m2) {
    vector_t<M> output;
    for (size_t m = 0; m < M; ++m) {
        output[m] = 0;
        for (size_t n = 0; n < N; ++n) {
            output[m] += m1[m][n] * m2[n];
        }
    }
    return output;
}

template <size_t M, size_t N, size_t O>
matrix_t<M, O> operator*(matrix_t<M, N> const & m1, matrix_t<N, O> const & m2) {
    matrix_t<M, O> output;
    for (size_t m = 0; m < M; ++m) {
        for (size_t o = 0; o < O; ++o) {
            output[m][o] = 0;
            for (size_t n = 0; n < N; ++n) {
                output[m][o] += m1[m][n] * m2[n][o];
            }
        }
    }
    return output;
}

template <size_t N>
std::ostream& operator<<(std::ostream& out, vector_t<N> const & vec) {
    out << "[";
    for (size_t n = 0; n < N - 1; ++n) {
        out << vec[n] << ", ";
    }
    out << vec[N - 1] << "]";
    return out;
}

template <size_t M, size_t N>
std::ostream& operator<<(std::ostream& out, matrix_t<M, N> const & mtx) {
    out << "[";
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N - 1; ++n) {
            out << mtx[m][n] << ", ";
        }
        out << mtx[m][N - 1] << ((m < M - 1) ? "\n " : "]");
    }
    return out;
}

} // namespace BP
