#include "shooting.hpp"

#include <algorithm>
#include <cmath>

#include "cost.hpp"
#include "dynamics.hpp"
#include "rk4.hpp"

namespace pendulum {

namespace {

struct ForwardSimOut {
    Eigen::Vector2d terminal_x = Eigen::Vector2d::Zero();
    double cost = 0.0;
};

PhasePoint rhsAsPhasePoint(const Params& p, const PhasePoint& z) {
    return asPhasePoint(hamiltonianRHS(p, z));
}

ForwardSimOut simulateForward(const Params& p, const State& x0, const Costate& l0, double T, double dt) {
    PhasePoint z;
    z.x = x0;
    z.l = l0;

    const int n = std::max(1, static_cast<int>(std::ceil(T / dt)));
    const double h = T / static_cast<double>(n);

    KahanSum J;
    double t = 0.0;
    for (int i = 0; i < n; ++i) {
        const double f0 = runningCost(z);
        const PhasePoint z_next = rk4Step<PhasePoint>(z, t, h, [&](double /*t*/, const PhasePoint& zz) {
            return rhsAsPhasePoint(p, zz);
        });
        const double f1 = runningCost(z_next);
        J.add(0.5 * h * (f0 + f1));
        z = z_next;
        t += h;
    }

    ForwardSimOut out;
    out.terminal_x = Eigen::Vector2d(z.x.theta, z.x.phi);
    out.cost = J.value();
    return out;
}

Eigen::Matrix2d finiteDiffJacobian(
    const Params& p,
    const State& x0,
    const Costate& l0,
    const Eigen::Vector2d& r0,
    double T,
    double dt,
    double eps) {
    Eigen::Matrix2d J;

    for (int j = 0; j < 2; ++j) {
        Costate lp = l0;
        if (j == 0) lp.l1 += eps;
        if (j == 1) lp.l2 += eps;
        const auto outp = simulateForward(p, x0, lp, T, dt);
        const Eigen::Vector2d rp = outp.terminal_x;
        J.col(j) = (rp - r0) / eps;
    }
    return J;
}

}  // namespace

ShootResult solveCostatesSingleSheetLM(const Params& p, const State& x0, const Costate& l0_init, const ShootSettings& s) {
    ShootResult best;
    best.l0 = l0_init;

    Costate l = l0_init;
    double lambda = s.lm_lambda0;

    auto out0 = simulateForward(p, x0, l, s.T, s.dt);
    Eigen::Vector2d r = out0.terminal_x;
    double cost = out0.cost;

    best.resid = r;
    best.cost = cost;
    best.converged = (r.lpNorm<Eigen::Infinity>() <= s.tol_resid);

    for (int iter = 0; iter < s.max_iters; ++iter) {
        const double rnorm = r.lpNorm<Eigen::Infinity>();
        if (rnorm <= s.tol_resid) {
            best.l0 = l;
            best.resid = r;
            best.cost = cost;
            best.converged = true;
            best.iters = iter;
            return best;
        }

        const Eigen::Matrix2d J = finiteDiffJacobian(p, x0, l, r, s.T, s.dt, s.fd_eps);
        const Eigen::Matrix2d A = J.transpose() * J + lambda * Eigen::Matrix2d::Identity();
        const Eigen::Vector2d g = J.transpose() * r;

        Eigen::Vector2d delta = Eigen::Vector2d::Zero();
        {
            Eigen::FullPivLU<Eigen::Matrix2d> lu(A);
            delta = lu.solve(-g);
        }

        Costate l_trial = l;
        l_trial.l1 += delta(0);
        l_trial.l2 += delta(1);

        const auto out_trial = simulateForward(p, x0, l_trial, s.T, s.dt);
        const Eigen::Vector2d r_trial = out_trial.terminal_x;
        const double rnorm_trial = r_trial.lpNorm<Eigen::Infinity>();

        // Accept/reject with simple damping update.
        if (rnorm_trial < rnorm) {
            l = l_trial;
            r = r_trial;
            cost = out_trial.cost;
            lambda = std::max(1e-16, lambda / s.lm_lambda_mul);
        } else {
            lambda = std::min(1e16, lambda * s.lm_lambda_mul);
        }

        // Track best-so-far
        const double best_norm = best.resid.lpNorm<Eigen::Infinity>();
        const double cur_norm = r.lpNorm<Eigen::Infinity>();
        if (cur_norm < best_norm) {
            best.l0 = l;
            best.resid = r;
            best.cost = cost;
            best.iters = iter + 1;
        }
    }

    best.converged = (best.resid.lpNorm<Eigen::Infinity>() <= s.tol_resid);
    return best;
}

}  // namespace pendulum

