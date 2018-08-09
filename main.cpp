#include "functions.hpp"

int main()
{
    using namespace BP;

    matrix_t<4, 4> X
    {
        vector(5.1, 3.5, 1.4, 0.2),
        vector(4.9, 3.0, 1.4, 0.2),
        vector(6.2, 3.4, 5.4, 2.3),
        vector(5.9, 3.0, 5.1, 1.8)
    };

    auto Y = vector(0, 0, 1, 1);
    auto W = vector(0.5, 0.5, 0.5, 0.5);

    for (int i = 0; i < 100; ++i)
    {
        auto P = sigmoid(X * W);
        std::cout << (i + 1) << ":\tP = " << P << ",\tW = " << W << std::endl;
        auto P_error = Y - P;
        auto P_delta = P_error * sigmoid_d(P);
        auto W_delta = transpose(X) * P_delta;
        W = W + W_delta;
    }
    return 0;
}
