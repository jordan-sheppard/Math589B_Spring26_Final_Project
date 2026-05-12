// Entry point for the grader-facing driver: CLI supplies physical data for the damped-pendulum
// infinite-horizon optimal-control boundary-value setup (truncated multiple shooting + Newton
// inside `solve()`). States use first-order z = (θ, φ, …); see README "Problem and first-order model".
#include <cstdio>
#include <cstdlib>
#include "solver.hpp"

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: ./solver theta phi alpha\n");
        return 1;
    }

    // θ: angular position (radians) — one component of the prescribed boundary / goal sheet data
    //     paired with φ in the continuation search over wraps and homotopy sheets.
    double theta = std::atof(argv[1]);
    // φ = θ̇: angular velocity paired with θ in the same boundary specification (first-order state).
    double phi   = std::atof(argv[2]);
    // α > 0: linear damping coefficient in θ̈ = sin θ − α θ̇ + u cos θ (running cost uses the same α).
    double alpha = std::atof(argv[3]);

    // `solve` finds a feasible truncated-shooting trajectory satisfying defect + periodicity-style
    // constraints; returned scalars are costates λ₁, λ₂ (Pontryagin multipliers for θ, φ) at the
    // first shooting node and the accumulated objective J for the winning continuation branch.
    Result r = solve(theta, phi, alpha);

    // stdout: λ₁(0)  λ₂(0)  J 
    std::printf("%.10f %.10f %.10f\n", r.optimal_l1_init, r.optimal_l2_init, r.optimal_cost);
    return 0;
}
