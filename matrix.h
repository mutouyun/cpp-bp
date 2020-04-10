#pragma once

#include <array>
#include <iostream>
#include <utility>
#include <type_traits>
#include <cstddef>

namespace BP {

/// definition

template <std::size_t N>
class vector_t : public std::array<double, N> {
    using base_t = std::array<double, N>;

public:
    template <typename... T>
    vector_t(T&&... args) noexcept
        : base_t { static_cast<double>(args)... } {}
};

template <typename... T>
vector_t(T&&... args) -> vector_t<sizeof...(T)>;

template <std::size_t M, std::size_t N>
class matrix_t : public std::array<vector_t<N>, M> {
    using base_t = std::array<vector_t<N>, M>;

public:
    template <typename... T>
    matrix_t(T&&... args) noexcept
        : base_t { std::forward<T>(args)... } {}
};

template <template <std::size_t> class... Vec, std::size_t N>
matrix_t(Vec<N> const &... args) -> matrix_t<sizeof...(Vec), N>;

/// output

inline std::ostream& operator<<(std::ostream& out, vector_t<0> const & vec) {
    return out << "[]";
}

template <std::size_t N>
std::ostream& operator<<(std::ostream& out, vector_t<N> const & vec) {
    out << "[ ";
    for (std::size_t i = 0; i < vec.size() - 1; ++i) {
        out << vec[i] << ", ";
    }
    out << vec.back();
    return out << " ]";
}

template <std::size_t M, std::size_t N>
std::ostream& operator<<(std::ostream& out, matrix_t<M, N> const & mtx) {
    out << "[" << std::endl;
    for (auto n : mtx) out << "  " << n << std::endl;
    return out << "]";
}

/// transpose

template <std::size_t N>
matrix_t<N, 1> operator~(vector_t<N> const & v) noexcept {
    matrix_t<N, 1> out;
    for (std::size_t i = 0; i < v.size(); ++i) {
        out[i][0] = v[i];
    }
    return out;
}

template <std::size_t M, std::size_t N>
matrix_t<N, M> operator~(matrix_t<M, N> const & m) noexcept {
    matrix_t<N, M> out;
    for (std::size_t i = 0; i < m.size(); ++i) {
        for (std::size_t j = 0; j < m[i].size(); ++j) {
            out[j][i] = m[i][j];
        }
    }
    return out;
}

/// operators

template <std::size_t N, typename F>
vector_t<N> operate(vector_t<N> const & v1, double d, F && op) noexcept {
    vector_t<N> out;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = op(v1[i], d);
    }
    return out;
}

template <std::size_t N, typename F>
vector_t<N> operate(double d, vector_t<N> const & v1, F && op) noexcept {
    vector_t<N> out;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = op(d, v1[i]);
    }
    return out;
}

template <std::size_t N, typename F>
vector_t<N> operate(vector_t<N> const & v1, vector_t<N> const & v2, F && op) noexcept {
    vector_t<N> out;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = op(v1[i], v2[i]);
    }
    return out;
}

template <std::size_t M, std::size_t N, typename V, typename F>
matrix_t<M, N> operate(matrix_t<M, N> const & m, V const & v, F && op) noexcept {
    matrix_t<M, N> out;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = operate(m[i], v, op);
    }
    return out;
}

template <typename V, std::size_t M, std::size_t N, typename F>
matrix_t<M, N> operate(V const & v, matrix_t<M, N> const & m, F && op) noexcept {
    matrix_t<M, N> out;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = operate(v, m[i], op);
    }
    return out;
}

template <std::size_t M, std::size_t N, typename F>
matrix_t<M, N> operate(matrix_t<M, N> const & m1, matrix_t<M, N> const & m2, F && op) noexcept {
    matrix_t<M, N> out;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = operate(m1[i], m2[i], op);
    }
    return out;
}

template <typename M1, typename M2>
auto operator+(M1 const & m1, M2 const & m2) noexcept {
    return operate(m1, m2, [](auto x, auto y) { return x + y; });
}

template <typename M1, typename M2>
auto operator-(M1 const & m1, M2 const & m2) noexcept {
    return operate(m1, m2, [](auto x, auto y) { return x - y; });
}

template <typename M1, typename M2>
auto operator*(M1 const & m1, M2 const & m2) noexcept {
    return operate(m1, m2, [](auto x, auto y) { return x * y; });
}

template <typename M1, typename M2>
auto operator/(M1 const & m1, M2 const & m2) noexcept {
    return operate(m1, m2, [](auto x, auto y) { return x / y; });
}

template <typename M>
M operator-(M const & m) noexcept { return 0.0 - m; }

template <typename M1, typename M2> M1 & operator +=(M1 & m1, M2 const & m2) noexcept { return (m1 = m1 + m2); }
template <typename M1, typename M2> M1 & operator -=(M1 & m1, M2 const & m2) noexcept { return (m1 = m1 - m2); }
template <typename M1, typename M2> M1 & operator *=(M1 & m1, M2 const & m2) noexcept { return (m1 = m1 * m2); }
template <typename M1, typename M2> M1 & operator /=(M1 & m1, M2 const & m2) noexcept { return (m1 = m1 / m2); }

/// dot product

template <std::size_t N>
double dot(vector_t<N> const & v1, vector_t<N> const & v2) noexcept {
    double out = 0.0;
    for (std::size_t i = 0; i < v1.size(); ++i) {
        out += v1[i] * v2[i];
    }
    return out;
}

template <std::size_t M, std::size_t N, std::size_t O>
matrix_t<M, O> dot(matrix_t<M, N> const & m1, matrix_t<N, O> const & m2) noexcept {
    matrix_t<M, O> out;
    for (std::size_t m = 0; m < m1.size(); ++m) {
        for (std::size_t o = 0; o < out[m].size(); ++o) {
            for (std::size_t n = 0; n < m2.size(); ++n) {
                out[m][o] += m1[m][n] * m2[n][o];
            }
        }
    }
    return out;
}

template <std::size_t M, std::size_t N>
matrix_t<M, N> dot(matrix_t<M, 1> const & m1, vector_t<N> const & v1) noexcept {
    matrix_t<M, N> out;
    for (size_t m = 0; m < m1.size(); ++m) {
        for (size_t n = 0; n < out[m].size(); ++n) {
            out[m][n] += m1[m][0] * v1[n];
        }
    }
    return out;
}

template <std::size_t N, std::size_t O>
vector_t<O> dot(vector_t<N> const & v1, matrix_t<N, O> const & m1) noexcept {
    vector_t<O> out;
    for (std::size_t o = 0; o < out.size(); ++o) {
        for (std::size_t n = 0; n < m1.size(); ++n) {
            out[o] += v1[n] * m1[n][o];
        }
    }
    return out;
}

} // namespace BP
