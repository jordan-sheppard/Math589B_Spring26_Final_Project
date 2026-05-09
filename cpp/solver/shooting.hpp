#pragma once

#include <Eigen/Dense>

#include "types.hpp"

namespace pendulum {

struct ShootSettings {
    double T = 8.0;
    double dt = 2e-3;
    int max_iters = 30;

    double fd_eps = 1e-6;         // finite-difference step in costates (absolute)
    double lm_lambda0 = 1e-2;     // initial damping
    double lm_lambda_mul = 10.0;  // damping up/down factor

    double tol_resid = 1e-9;

    // Step safeguard
    double max_delta_norm = 10.0;
    int backtrack_max = 10;
};

struct ShootResult {
    Costate l0{};
    double cost = 0.0;
    Eigen::VectorXd resid = Eigen::VectorXd::Zero(0);  // terminal residual (size 2 or 4)
    int iters = 0;
    bool converged = false;
};

// Single-sheet shooting: solve for l0 so that x(T) ≈ 0.
ShootResult solveCostatesSingleSheetLM(const Params& p, const State& x0, const Costate& l0_init, const ShootSettings& s);

// Continuation in horizon T: solve sequentially for T_list[0], T_list[1], ...,
// warm-starting each stage from the previous stage's l0.
ShootResult solveCostatesSingleSheetLMContinuation(
    const Params& p,
    const State& x0,
    const Costate& l0_init,
    const ShootSettings& base,
    const Eigen::VectorXd& T_list);

}  // namespace pendulum

