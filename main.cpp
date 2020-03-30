#include <iostream>

#include "matrix.h"
#include "perceptron.h"

int main(void) {
    using namespace BP;

    auto X = matrix_t {
        vector_t(0, 0),
        vector_t(0, 1),
        vector_t(1, 0),
        vector_t(1, 1)
    };
    auto Y = vector_t(0, 0, 0, 1);

    perceptron<2> P;
    for (int i = 0; i < 1000; ++i) {
        P.train(X, Y);
        std::cout << (i + 1) << ":\tW = " << P.W() << ",\tbias = " << P.bias() << std::endl;
    }

    std::cout << "Test(0  , 0  ):\tP = " << P.calc({ 0  , 0   }) << std::endl;
    std::cout << "Test(0  , 1  ):\tP = " << P.calc({ 0  , 1   }) << std::endl;
    std::cout << "Test(1  , 0  ):\tP = " << P.calc({ 1  , 0   }) << std::endl;
    std::cout << "Test(1  , 1  ):\tP = " << P.calc({ 1  , 1   }) << std::endl;
    std::cout << "Test(0.2, 0.9):\tP = " << P.calc({ 0.2, 0.9 }) << std::endl;
    std::cout << "Test(0.1, 1  ):\tP = " << P.calc({ 0.1, 1   }) << std::endl;
    std::cout << "Test(1  , 0.1):\tP = " << P.calc({ 1  , 0.1 }) << std::endl;
    std::cout << "Test(0.8, 0  ):\tP = " << P.calc({ 0.8, 0   }) << std::endl;
    std::cout << "Test(0.8, 0.9):\tP = " << P.calc({ 0.8, 0.9 }) << std::endl;
    std::cout << "Test(0.9, 1  ):\tP = " << P.calc({ 0.9, 1   }) << std::endl;
    std::cout << "Test(0.5, 0.5):\tP = " << P.calc({ 0.5, 0.5 }) << std::endl;
    return 0;
}
