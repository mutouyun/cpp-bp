#include "functions.hpp"

int main()
{
    using namespace BP;

    std::cout << sigmoid(vector(1, 2, 3, 4)) << std::endl;

    matrix_t<2, 3> mtx1
    {
        vector(1, 2, 3),
        vector(4, 5, 6)
    };
    std::cout << mtx1 << "\n transpose:\n" << transpose(mtx1) << std::endl;
    return 0;
}
