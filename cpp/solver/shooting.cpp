#include "shooting.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "cost.hpp"
#include "dynamics.hpp"
#include "rk4.hpp"
#include "manifold_seed.hpp"

namespace pendulum {

namespace {

struct ForwardSimOut {
    Eigen::Vector2d terminal_x = Eigen::Vector2d::Zero();
    Eigen::Vector2d terminal_l = Eigen::Vector2d::Zero();
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
    out.terminal_l = Eigen::Vector2d(z.l.l1, z.l.l2);
    out.cost = J.value();
    return out;
}

Eigen::MatrixXd finiteDiffJacobianCentral(
    const Params& p,
    const State& x0,
    const Costate& l0,
    const Eigen::VectorXd& r0,
    double T,
    double dt,
    double eps,
    const Eigen::Matrix2d& P,
    bool use_manifold_resid) {
    const int m = static_cast<int>(r0.size());
    Eigen::MatrixXd J(m, 2);

    for (int j = 0; j < 2; ++j) {
        Costate lp = l0;
        Costate lm = l0;
        if (j == 0) {
            lp.l1 += eps;
            lm.l1 -= eps;
        }
        if (j == 1) {
            lp.l2 += eps;
            lm.l2 -= eps;
        }
        const auto outp = simulateForward(p, x0, lp, T, dt);
        const auto outm = simulateForward(p, x0, lm, T, dt);

        Eigen::VectorXd rp(m), rm(m);
        rp.head<2>() = outp.terminal_x;
        rm.head<2>() = outm.terminal_x;
        if (use_manifold_resid) {
            const Eigen::Vector2d mp = outp.terminal_l - P * outp.terminal_x;
            const Eigen::Vector2d mm = outm.terminal_l - P * outm.terminal_x;
            rp.tail<2>() = mp;
            rm.tail<2>() = mm;
        }

        J.col(j) = (rp - rm) / (2.0 * eps);
    }
    return J;
}

}  // namespace

ShootResult solveCostatesSingleSheetLM(const Params& p, const State& x0, const Costate& l0_init, const ShootSettings& s) {
    ShootResult best;
    best.l0 = l0_init;

    Costate l = l0_init;
    double lambda = s.lm_lambda0;

    // Stable-manifold matching at terminal time.
    const Eigen::Matrix2d P = stableManifoldSeedP(p.alpha);
    const bool use_manifold_resid = true;

    auto out0 = simulateForward(p, x0, l, s.T, s.dt);
    Eigen::VectorXd r(use_manifold_resid ? 4 : 2);
    r.head<2>() = out0.terminal_x;
    if (use_manifold_resid) {
        r.tail<2>() = out0.terminal_l - P * out0.terminal_x;
    }
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

        const double eps = std::max(1e-12, s.fd_eps);
        const Eigen::MatrixXd J = finiteDiffJacobianCentral(p, x0, l, r, s.T, s.dt, eps, P, use_manifold_resid);
        const Eigen::Matrix2d A = J.transpose() * J + lambda * Eigen::Matrix2d::Identity();
        const Eigen::Vector2d g = J.transpose() * r;

        Eigen::Vector2d delta = Eigen::Vector2d::Zero();
        {
            Eigen::FullPivLU<Eigen::Matrix2d> lu(A);
            delta = lu.solve(-g);
        }

        // Safeguard huge steps.
        const double dnorm = delta.norm();
        if (dnorm > s.max_delta_norm) {
            delta *= (s.max_delta_norm / dnorm);
        }

        bool accepted = false;
        Costate l_acc = l;
        Eigen::VectorXd r_acc = r;
        double cost_acc = cost;

        // Backtracking on step length to enforce decrease in residual.
        double step = 1.0;
        for (int bt = 0; bt <= s.backtrack_max; ++bt) {
            Costate l_trial = l;
            l_trial.l1 += step * delta(0);
            l_trial.l2 += step * delta(1);

            const auto out_trial = simulateForward(p, x0, l_trial, s.T, s.dt);
            Eigen::VectorXd r_trial(use_manifold_resid ? 4 : 2);
            r_trial.head<2>() = out_trial.terminal_x;
            if (use_manifold_resid) {
                r_trial.tail<2>() = out_trial.terminal_l - P * out_trial.terminal_x;
            }
            const double rnorm_trial = r_trial.lpNorm<Eigen::Infinity>();

            if (std::isfinite(rnorm_trial) && rnorm_trial < rnorm) {
                accepted = true;
                l_acc = l_trial;
                r_acc = r_trial;
                cost_acc = out_trial.cost;
                break;
            }
            step *= 0.5;
        }

        if (accepted) {
            l = l_acc;
            r = r_acc;
            cost = cost_acc;
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

ShootResult solveCostatesSingleSheetLMContinuation(
    const Params& p,
    const State& x0,
    const Costate& l0_init,
    const ShootSettings& base,
    const Eigen::VectorXd& T_list) {
    ShootResult best_overall;
    best_overall.l0 = l0_init;
    best_overall.resid = Eigen::Vector2d(std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
    best_overall.cost = std::numeric_limits<double>::infinity();

    Costate l = l0_init;
    for (int i = 0; i < T_list.size(); ++i) {
        ShootSettings s = base;
        s.T = T_list(i);

        const ShootResult stage = solveCostatesSingleSheetLM(p, x0, l, s);
        l = stage.l0;  // warm start next stage

        const double stage_norm = stage.resid.lpNorm<Eigen::Infinity>();
        const double best_norm = best_overall.resid.lpNorm<Eigen::Infinity>();
        if (stage_norm < best_norm) {
            best_overall = stage;
        }
    }

    return best_overall;
}

}  // namespace pendulum

