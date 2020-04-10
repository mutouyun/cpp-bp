#include <iostream>

#include "matrix.h"
#include "func.h"
#include "network.h"

int main(void) {
    using namespace BP;

    matrix_t X = {
        vector_t { 1, 0, 0 },
        vector_t { 1, 0, 1 },
        vector_t { 1, 1, 0 },
        vector_t { 1, 1, 1 }
    };
    matrix_t Y = { 
        vector_t { 0, 1, 1, 1 }, // OR
        vector_t { 0, 0, 0, 1 }, // AND
        vector_t { 0, 1, 1, 0 }, // XOR
        vector_t { 1, 0, 0, 1 }  // XNOR
    };

    network_t<3, 3, 4> n;
    std::cout << n << std::endl;

    for (int i = 0; i < 1000; ++i) {
        n.train(X, Y);
        if (i % 100 == 0) {
            std::cout << mean( Fn(BP::abs, error_of(n.predict(X), Y)) ) << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "predict X: " << n.predict(X) << std::endl;

    std::cout << std::endl;
    std::cout << "predict [ 0.2, 0.1 ]: " << n.predict(vector_t { 1, 0.2, 0.1 }) << std::endl;
    std::cout << "predict [ 0.2, 0.9 ]: " << n.predict(vector_t { 1, 0.2, 0.9 }) << std::endl;
    std::cout << "predict [ 0.1, 1   ]: " << n.predict(vector_t { 1, 0.1, 1   }) << std::endl;
    std::cout << "predict [ 1  , 0.1 ]: " << n.predict(vector_t { 1, 1  , 0.1 }) << std::endl;
    std::cout << "predict [ 0.8, 0   ]: " << n.predict(vector_t { 1, 0.8, 0   }) << std::endl;
    std::cout << "predict [ 0.8, 0.9 ]: " << n.predict(vector_t { 1, 0.8, 0.9 }) << std::endl;
    std::cout << "predict [ 0.9, 1   ]: " << n.predict(vector_t { 1, 0.9, 1   }) << std::endl;
    std::cout << "predict [ 0.5, 0.5 ]: " << n.predict(vector_t { 1, 0.5, 0.5 }) << std::endl;

    return 0;
}
