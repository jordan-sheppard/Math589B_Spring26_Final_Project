#include <cstdio>
#include <cstdlib>
#include "solver.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: ./solver theta phi alpha\n");
        return 1;
    }

    double theta = std::atof(argv[1]);
    double phi   = std::atof(argv[2]);
    double alpha = std::atof(argv[3]);

    Result r = solve(theta, phi, alpha);

    std::printf("%.10f %.10f %.10f\n", r.optimal_l1_init, r.optimal_l2_init, r.optimal_cost);
    return 0;
}
