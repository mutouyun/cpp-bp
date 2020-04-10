#pragma once

#include <cstddef>
#include <utility>
#include <iostream>

#include "func.h"

namespace BP {

template <class... Matrixes>
class network;

template <>
class network<> {
public:
    template <typename T>
    static decltype(auto) predict(T&& L) noexcept {
        return std::forward<T>(L);
    }

    template <
        std::size_t M, std::size_t N,
        template <std::size_t, std::size_t> class Mtx
    >
    static auto train(Mtx<M, N> const & X, Mtx<N, M> const & Y) noexcept {
        return ~Y - X;
    }
};

template <
    std::size_t M, std::size_t N,
    template <std::size_t, std::size_t> class Mtx,
    class... Mtxes
>
class network<Mtx<M, N>, Mtxes...> : public network<Mtxes...> {
    using base_t = network<Mtxes...>;

    Mtx<M, N> W_;

public:
    network()
        : W_(random<Mtx<M, N>>(-1, 1)) {}

    Mtx<M, N> const & W() const noexcept {
        return W_;
    }

    template <std::size_t M_, std::size_t N_>
    auto predict(Mtx<M_, N_> const & X) const {
        return base_t::predict(sigmoid(dot(X, W_)));
    }

    template <
        std::size_t N_,
        template <std::size_t> class Vec
    >
    auto predict(Vec<N_> const & X) const {
        return base_t::predict(sigmoid(dot(X, W_)));
    }

    template <std::size_t M_, std::size_t N_, std::size_t My>
    auto train(Mtx<M_, N_> const & X, Mtx<My, M_> const & Y) {
        auto L = sigmoid(dot(X, W_));
        auto D = base_t::train(L, Y); // cross-entropy
        auto R = dot(D, ~W_) * sigmoid_d(X);
        W_ += dot(~X, D);
        return R;
    }
};

inline std::ostream& operator<<(std::ostream& out, network<> const & n) noexcept {
    return out;
}

template <class Mtx, class... Mtxes>
std::ostream& operator<<(std::ostream& out, network<Mtx, Mtxes...> const & n) {
    return out << static_cast<network<Mtxes...> const &>(n)
               << "W" << sizeof...(Mtxes) + 1 << " = " << n.W() << std::endl;
}

template <class Network, std::size_t... N>
struct network_helper;

template <class... M, std::size_t N1, std::size_t N2>
struct network_helper<network<M...>, N1, N2> {
    using type = network<M..., matrix_t<N1, N2>>;
};

template <class... M, std::size_t N1, std::size_t N2, std::size_t N3, std::size_t... N>
struct network_helper<network<M...>, N1, N2, N3, N...> {
    using join = typename network_helper<network<M...>, N1, N2>::type;
    using type = typename network_helper<join, N2, N3, N...>::type;
};

template <std::size_t... N>
using network_t = typename network_helper<network<>, N...>::type;

template <typename T>
auto error_of(T const & X, T const & Y) noexcept {
    return network<>::train(X, Y);
}

} // namespace BP
