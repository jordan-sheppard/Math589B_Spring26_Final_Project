#include "solver.hpp"

Result solve(double /*theta*/, double /*phi*/, double /*alpha*/) {
    // Placeholder so the harness can run without CUDA.
    Result r;
    r.l1 = 0.0;
    r.l2 = 0.0;
    r.cost = 1e100;
    return r;
}

