#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "../cuda/forward_sim.cuh"
#include "../pendulum/types.hpp"

namespace pendulum {

struct ShootSettingsHost {
    double T = 8.0;
    double dt = 2e-3;
    int max_iters = 30;
    double fd_eps = 1e-6;
    double lm_lambda0 = 1e-2;
    double lm_lambda_mul = 10.0;
    double tol_resid = 1e-9;
    double max_delta_norm = 10.0;
    int backtrack_max = 10;
    bool use_variational_jacobian = true;
    IntegratorKind integrator = IntegratorKind::DP5;
    bool debug = false;
};

struct ShootResultHost {
    Costate l0{};
    double cost = 0.0;
    double resid[4]{};
    int resid_dim = 4;  // 2 or 4
    int iters = 0;
    bool converged = false;
};

inline double resid_inf(int dim, const double r[4]) {
    double m = 0.0;
    for (int i = 0; i < dim; ++i) {
        m = std::max(m, std::abs(r[i]));
    }
    return m;
}

ShootResultHost solve_costates_single_sheet_lm(
    const Params& p,
    const State& x0,
    const Costate& l0_init,
    const ShootSettingsHost& s,
    const double P[2][2]);

ShootResultHost solve_costates_single_sheet_lm_continuation(
    const Params& p,
    const State& x0,
    const Costate& l0_init,
    const ShootSettingsHost& base,
    const std::vector<double>& T_list,
    const double P[2][2]);

}  // namespace pendulum
