#include "shooting.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include "cost.hpp"
#include "dynamics.hpp"
#include "dp5.hpp"
#include "rk4.hpp"
#include "manifold_seed.hpp"

namespace pendulum {

namespace {

struct ForwardSimOut {
    Eigen::Vector2d terminal_x = Eigen::Vector2d::Zero();
    Eigen::Vector2d terminal_l = Eigen::Vector2d::Zero();
    Eigen::Matrix<double, 4, 2> dZ_dL0 = (Eigen::Matrix<double, 4, 2>() << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).finished();
    double cost = 0.0;
};

Eigen::Matrix4d jacobianDF(const Params& p, const PhasePoint& z) {
    const double th = z.x.theta;
    const double l1 = z.l.l1;
    const double l2 = z.l.l2;
    (void)l1;

    const double s = std::sin(th);
    const double c = std::cos(th);
    const double c2 = c * c;

    // d/dtheta of cos(theta)*sin(theta) = cos(2theta)
    const double cos2 = std::cos(2.0 * th);

    Eigen::Matrix4d A = Eigen::Matrix4d::Zero();
    // State: [theta, phi, l1, l2]

    // theta_dot = phi
    A(0, 1) = 1.0;

    // phi_dot = sin(theta) - alpha*phi - l2*cos^2(theta)
    A(1, 0) = c + 2.0 * l2 * c * s;
    A(1, 1) = -p.alpha;
    A(1, 3) = -c2;

    // l1_dot = -sin(theta) - l2*cos(theta) - l2^2*cos(theta)*sin(theta)
    A(2, 0) = -c + l2 * s - (l2 * l2) * cos2;
    A(2, 3) = -c - 2.0 * l2 * c * s;

    // l2_dot = -phi - l1 + alpha*l2
    A(3, 1) = -1.0;
    A(3, 2) = -1.0;
    A(3, 3) = p.alpha;

    return A;
}

struct AugState {
    PhasePoint z{};
    Eigen::Matrix<double, 4, 2> S = (Eigen::Matrix<double, 4, 2>() << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).finished();
    double J = 0.0;
};

inline AugState operator+(const AugState& a, const AugState& b) {
    AugState o;
    o.z = a.z + b.z;
    o.S = a.S + b.S;
    o.J = a.J + b.J;
    return o;
}

inline AugState operator*(double s, const AugState& a) {
    AugState o;
    o.z = s * a.z;
    o.S = s * a.S;
    o.J = s * a.J;
    return o;
}

// (a*s) not needed; keep only (s*a) to match rk4 usage.

ForwardSimOut simulateForward(
    const Params& p,
    const State& x0,
    const Costate& l0,
    double T,
    double dt,
    ShootSettings::Integrator integrator) {
    AugState a;
    a.z.x = x0;
    a.z.l = l0;
    a.S = (Eigen::Matrix<double, 4, 2>() << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0).finished();
    a.J = 0.0;

    const int n = std::max(1, static_cast<int>(std::ceil(T / dt)));
    const double h = T / static_cast<double>(n);

    KahanSum J;
    double t = 0.0;
    for (int i = 0; i < n; ++i) {
        const double f0 = runningCost(a.z);

        const auto rhs = [&](double /*t*/, const AugState& aa) {
            AugState d;
            const PhaseDeriv k = hamiltonianRHS(p, aa.z);
            d.z = asPhasePoint(k);
            const Eigen::Matrix4d A = jacobianDF(p, aa.z);
            d.S = A * aa.S;
            d.J = runningCost(aa.z);
            return d;
        };

        AugState a_next;
        if (integrator == ShootSettings::Integrator::DP5) {
            a_next = dp5Step<AugState>(a, t, h, rhs);
        } else {
            a_next = rk4Step<AugState>(a, t, h, rhs);
        }

        const double f1 = runningCost(a_next.z);
        J.add(0.5 * h * (f0 + f1));

        a = a_next;
        t += h;
    }

    ForwardSimOut out;
    out.terminal_x = Eigen::Vector2d(a.z.x.theta, a.z.x.phi);
    out.terminal_l = Eigen::Vector2d(a.z.l.l1, a.z.l.l2);
    out.dZ_dL0 = a.S;
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
    bool use_manifold_resid,
    ShootSettings::Integrator integrator) {
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
        const auto outp = simulateForward(p, x0, lp, T, dt, integrator);
        const auto outm = simulateForward(p, x0, lm, T, dt, integrator);

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

    auto out0 = simulateForward(p, x0, l, s.T, s.dt, s.integrator);
    Eigen::VectorXd r(use_manifold_resid ? 4 : 2);
    r.head<2>() = out0.terminal_x;
    if (use_manifold_resid) {
        r.tail<2>() = out0.terminal_l - P * out0.terminal_x;
    }
    double cost = out0.cost;

    best.resid = r;
    best.cost = cost;
    best.converged = (r.lpNorm<Eigen::Infinity>() <= s.tol_resid);

    if (s.debug) {
        std::fprintf(
            stderr,
            "[shoot] T=%.3f dt=%.3g init_l=(%.6g,%.6g) init_rinf=%.3e\n",
            s.T,
            s.dt,
            l.l1,
            l.l2,
            r.lpNorm<Eigen::Infinity>());
    }

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

        Eigen::MatrixXd J;
        if (s.use_variational_jacobian) {
            // Build dr/dl0 from variational sensitivity at time T.
            // z=[theta,phi,l1,l2]; S = dz/dl0 is 4x2.
            const Eigen::Matrix<double, 4, 2>& S = out0.dZ_dL0;
            const Eigen::Matrix<double, 2, 2> Sx = S.block<2, 2>(0, 0);
            const Eigen::Matrix<double, 2, 2> Sl = S.block<2, 2>(2, 0);

            if (use_manifold_resid) {
                J.resize(4, 2);
                J.block<2, 2>(0, 0) = Sx;
                J.block<2, 2>(2, 0) = Sl - P * Sx;
            } else {
                J = Sx;
            }
        } else {
            const double eps = std::max(1e-12, s.fd_eps);
            J = finiteDiffJacobianCentral(p, x0, l, r, s.T, s.dt, eps, P, use_manifold_resid, s.integrator);
        }
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
        double accepted_step = 0.0;

        // Backtracking on step length to enforce decrease in residual.
        double step = 1.0;
        for (int bt = 0; bt <= s.backtrack_max; ++bt) {
            Costate l_trial = l;
            l_trial.l1 += step * delta(0);
            l_trial.l2 += step * delta(1);

            const auto out_trial = simulateForward(p, x0, l_trial, s.T, s.dt, s.integrator);
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
                out0 = out_trial;  // keep sensitivity/cached terminal for next iteration
                accepted_step = step;
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

        if (s.debug) {
            std::fprintf(
                stderr,
                "[shoot] iter=%d rinf=%.3e -> %.3e accept=%d step=%.3g lm=%.3g l=(%.6g,%.6g)\n",
                iter,
                rnorm,
                r.lpNorm<Eigen::Infinity>(),
                accepted ? 1 : 0,
                accepted_step,
                lambda,
                l.l1,
                l.l2);
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

        // Adaptive continuation: stop once we're clearly converged.
        if (stage_norm <= s.tol_resid) {
            break;
        }
    }

    return best_overall;
}

}  // namespace pendulum

