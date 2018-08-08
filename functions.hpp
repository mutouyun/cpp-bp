#pragma once

#include <array>
#include <vector>
#include <iostream>
#include <utility>
#include <type_traits>
#include <cmath>

namespace BP {

template <size_t N>
using vector_t = std::array<double, N>;

template <size_t M, size_t N>
using matrix_t = std::array<vector_t<N>, M>;

template <typename... T>
vector_t<sizeof... (T)> vector(T&&... args)
{
    return { static_cast<double>(args)... };
}

template <size_t N>
vector_t<N> vector(const std::vector<double>& m1)
{
    vector_t<N> output;
    for (size_t i = 0; i < std::min(N, m1.size()); ++i)
    {
        output[i] = m1[i];
    }
    return output;
}

template <size_t N>
vector_t<N> sigmoid(const vector_t<N>& m1)
{
    vector_t<N> output;
    for (size_t i = 0; i < N; ++i)
    {
        output[i] = 1 / (1 + std::exp(-m1[i]));
    }
    return output;
}

template <size_t N>
vector_t<N> sigmoid_d(const vector_t<N>& m1)
{
    vector_t<N> output;
    for (size_t i = 0; i < N; ++i)
    {
        output[i] = m1[i] * (1 - m1[i]);
    }
    return output;
}

template <size_t M, size_t N>
matrix_t<N, M> transpose(const matrix_t<M, N>& m1)
{
    matrix_t<N, M> output;
    for (size_t m = 0; m < M; ++m)
    {
        for (size_t n = 0; n < N; ++n)
        {
            output[n][m] = m1[m][n];
        }
    }
    return output;
}

template <size_t N>
vector_t<N> operator+(const vector_t<N>& m1, const vector_t<N>& m2)
{
    vector_t<N> output;
    for (size_t i = 0; i < N; ++i)
    {
        output[i] = m1[i] + m2[i];
    }
    return output;
}

template <size_t N>
vector_t<N> operator-(const vector_t<N>& m1, const vector_t<N>& m2)
{
    vector_t<N> output;
    for (size_t i = 0; i < N; ++i)
    {
        output[i] = m1[i] - m2[i];
    }
    return output;
}

template <size_t N>
vector_t<N> operator*(const vector_t<N>& m1, const vector_t<N>& m2)
{
    vector_t<N> output;
    for (size_t i = 0; i < N; ++i)
    {
        output[i] = m1[i] * m2[i];
    }
    return output;
}

template <size_t M, size_t N>
vector_t<M> operator*(const matrix_t<M, N>& m1, const vector_t<N>& m2)
{
    vector_t<M> output;
    for (size_t m = 0; m < M; ++m)
    {
        output[m] = 0;
        for (size_t n = 0; n < N; ++n)
        {
            output[m] += m1[m][n] * m2[n];
        }
    }
    return output;
}

template <size_t M, size_t N, size_t O>
matrix_t<M, O> operator*(const matrix_t<M, N>& m1, const matrix_t<N, O>& m2)
{
    matrix_t<M, O> output;
    for (size_t m = 0; m < M; ++m)
    {
        for (size_t o = 0; o < O; ++o)
        {
            output[m][o] = 0;
            for (size_t n = 0; n < N; ++n)
            {
                output[m][o] += m1[m][n] * m2[n][o];
            }
        }
    }
    return output;
}

template <size_t N>
std::ostream& operator<<(std::ostream& out, const vector_t<N>& vec)
{
    out << "[";
    for (size_t n = 0; n < N - 1; ++n)
    {
        out << vec[n] << ", ";
    }
    out << vec[N - 1] << "]";
    return out;
}

template <size_t M, size_t N>
std::ostream& operator<<(std::ostream& out, const matrix_t<M, N>& mtx)
{
    out << "[";
    for (size_t m = 0; m < M; ++m)
    {
        for (size_t n = 0; n < N - 1; ++n)
        {
            out << mtx[m][n] << ", ";
        }
        out << mtx[m][N - 1] << ((m < M - 1) ? "\n " : "]");
    }
    return out;
}

} // namespace BP
