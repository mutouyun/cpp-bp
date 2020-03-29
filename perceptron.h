#pragma once

#include "matrix.h"

namespace BP {

template <size_t N>
class perceptron {
    vector_t<N> W_;

public:
    perceptron() {
        W_.fill(0.5);
    }

    auto const & W() const noexcept {
        return W_;
    }

    template <size_t M>
    auto train(matrix_t<M, N> const & X, vector_t<M> const & Y) {
        auto P = sigmoid(X * W_);
        auto P_error = Y - P;
        auto P_delta = P_error * sigmoid_d(P);
        auto W_delta = transpose(X) * P_delta;
        W_ += W_delta;
        return P;
    }

    auto calc(vector_t<N> const & A) {
        return sigmoid(matrix_t { A } * W_);
    }
};

} // namespace BP
