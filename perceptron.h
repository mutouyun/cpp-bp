#pragma once

#include "matrix.h"
#include "func.h"

namespace BP {

template <std::size_t N>
class perceptron {
    vector_t<N> W_;
    double bias_;

    void adjust(vector_t<N> const & A, double delta) {
        for (std::size_t i = 0; i < N; ++i) {
            W_[i] += A[i] * delta;
        }
        bias_ += delta;
    }

public:
    perceptron() {
        W_.fill(1.0);
        bias_ = 1.0;
    }

    auto const & W() const noexcept {
        return W_;
    }

    double bias() const noexcept {
        return bias_;
    }

    template <size_t M>
    void train(matrix_t<M, N> const & X, vector_t<M> const & Y) {
        for (std::size_t i = 0; i < X.size(); ++i) {
            double P = calc(X[i]);
            double D = (Y[i] - P) * sigmoid_d(P);
            adjust(X[i], D);
        }
    }

    auto calc(vector_t<N> const & A) {
        double sum = bias_;
        for (double i : A * W_) sum += i;
        return sigmoid(sum);
    }
};

} // namespace BP
