#include "functions.hpp"

int main(void)
{
    using namespace BP;

//    matrix_t<4, 4> X
//    {
//        vector(5.1, 3.5, 1.4, 0.2),
//        vector(4.9, 3.0, 1.4, 0.2),
//        vector(6.2, 3.4, 5.4, 2.3),
//        vector(5.9, 3.0, 5.1, 1.8)
//    };
//    auto Y = vector(0, 0, 1, 1);
//    auto W = vector(0.5, 0.5, 0.5, 0.5);

//    for (int i = 0; i < 1000; ++i)
//    {
//        auto P = sigmoid(X * W);
//        std::cout << (i + 1) << ":\tP = " << P << ",\tW = " << W << std::endl;
//        auto P_error = Y - P;
//        auto P_delta = P_error * sigmoid_d(P);
//        auto W_delta = transpose(X) * P_delta;
//        W = W + W_delta;
//    }

    matrix_t<4, 2> X
    {
        vector(0.1, 1),
        vector(0  , 0.8),
        vector(0.7, 1),
        vector(1.1, 0.8)
    };
    auto W = vector(0.5, 0.5);
    auto Y = vector(0, 0, 1, 1);

    for (int i = 0; i < 1000; ++i)
    {
        auto P = sigmoid(X * W);
        std::cout << (i + 1) << ":\tP = " << P << ",\tW = " << W << std::endl;
        auto P_error = Y - P;
        auto P_delta = P_error * sigmoid_d(P);
        auto W_delta = transpose(X) * P_delta;
        W = W + W_delta;
    }

    std::cout << "Test(0.2, 0.9):\tP = " << sigmoid(matrix_t<1, 2>{ vector(0.2, 0.9) } * W) << std::endl;
    std::cout << "Test(0.1, 1  ):\tP = " << sigmoid(matrix_t<1, 2>{ vector(0.1, 1  ) } * W) << std::endl;
    std::cout << "Test(0  , 1  ):\tP = " << sigmoid(matrix_t<1, 2>{ vector(0  , 1  ) } * W) << std::endl;
    std::cout << "Test(0.8, 0.9):\tP = " << sigmoid(matrix_t<1, 2>{ vector(0.8, 0.9) } * W) << std::endl;
    std::cout << "Test(0.9, 1  ):\tP = " << sigmoid(matrix_t<1, 2>{ vector(0.9, 1  ) } * W) << std::endl;
    std::cout << "Test(1  , 1  ):\tP = " << sigmoid(matrix_t<1, 2>{ vector(1  , 1  ) } * W) << std::endl;
    return 0;
}
