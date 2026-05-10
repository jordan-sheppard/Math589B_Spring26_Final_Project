// Host-only Newton/LM shooting (matches cpp/solver/shooting.cpp).
// Uses plain arrays for linear algebra; Eigen appears only in manifold_seed.cpp.

#include "shoot_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "../cuda/forward_gpu.hpp"

namespace pendulum {

namespace {

inline void terminal_residual(
    const ForwardSimOut& o,
    const double P[2][2],
    bool use_manifold,
    double r[4],
    int* out_dim) {
    r[0] = o.terminal_x[0];
    r[1] = o.terminal_x[1];
    if (use_manifold) {
        const double th = o.terminal_x[0];
        const double ph = o.terminal_x[1];
        const double l1 = o.terminal_l[0];
        const double l2 = o.terminal_l[1];
        const double Px0 = P[0][0] * th + P[0][1] * ph;
        const double Px1 = P[1][0] * th + P[1][1] * ph;
        r[2] = l1 - Px0;
        r[3] = l2 - Px1;
        *out_dim = 4;
    } else {
        *out_dim = 2;
    }
}

inline void mat22_mul(const double A[2][2], const double B[2][2], double C[2][2]) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j];
        }
    }
}

inline bool solve_2x2(const double A[2][2], const double rhs[2], double x[2]) {
    const double a00 = A[0][0];
    const double a01 = A[0][1];
    const double a10 = A[1][0];
    const double a11 = A[1][1];
    const double det = a00 * a11 - a01 * a10;
    if (!(std::abs(det) > 1e-30)) {
        return false;
    }
    const double invdet = 1.0 / det;
    x[0] = invdet * (a11 * rhs[0] - a01 * rhs[1]);
    x[1] = invdet * (-a10 * rhs[0] + a00 * rhs[1]);
    return true;
}

void finite_diff_jacobian(
    const Params& p,
    const State& x0,
    const Costate& l0,
    double T,
    double dt,
    double eps,
    const double P[2][2],
    bool use_manifold,
    IntegratorKind integrator,
    double J[4][2],
    int* m_out) {
    const int m = use_manifold ? 4 : 2;
    *m_out = m;

    Costate lp0 = l0;
    Costate lm0 = l0;
    lp0.l1 += eps;
    lm0.l1 -= eps;
    Costate lp1 = l0;
    Costate lm1 = l0;
    lp1.l2 += eps;
    lm1.l2 -= eps;

    const Costate seeds[4] = {lp0, lm0, lp1, lm1};
    ForwardSimOut outs[4];
    forward_batch_cuda(p, x0, seeds, 4, T, dt, integrator, outs);

    for (int j = 0; j < 2; ++j) {
        const ForwardSimOut& outp = outs[2 * j + 0];
        const ForwardSimOut& outm = outs[2 * j + 1];
        double rp[4], rm[4];
        int dp = 0, dm = 0;
        terminal_residual(outp, P, use_manifold, rp, &dp);
        terminal_residual(outm, P, use_manifold, rm, &dm);
        (void)dp;
        (void)dm;

        for (int i = 0; i < m; ++i) {
            J[i][j] = (rp[i] - rm[i]) / (2.0 * eps);
        }
    }
}

void variational_jacobian(
    const ForwardSimOut& out0,
    const double P[2][2],
    bool use_manifold,
    double J[4][2],
    int* m_out) {
    const double S[4][2] = {
        {out0.dZ_dL0[0][0], out0.dZ_dL0[0][1]},
        {out0.dZ_dL0[1][0], out0.dZ_dL0[1][1]},
        {out0.dZ_dL0[2][0], out0.dZ_dL0[2][1]},
        {out0.dZ_dL0[3][0], out0.dZ_dL0[3][1]},
    };

    double Sx[2][2] = {
        {S[0][0], S[0][1]},
        {S[1][0], S[1][1]},
    };
    double Sl[2][2] = {
        {S[2][0], S[2][1]},
        {S[3][0], S[3][1]},
    };

    if (!use_manifold) {
        for (int i = 0; i < 2; ++i) {
            J[i][0] = Sx[i][0];
            J[i][1] = Sx[i][1];
        }
        *m_out = 2;
        return;
    }

    double PSx[2][2];
    mat22_mul(P, Sx, PSx);
    for (int i = 0; i < 2; ++i) {
        J[i][0] = Sx[i][0];
        J[i][1] = Sx[i][1];
        J[i + 2][0] = Sl[i][0] - PSx[i][0];
        J[i + 2][1] = Sl[i][1] - PSx[i][1];
    }
    *m_out = 4;
}

}  // namespace

ShootResultHost solve_costates_single_sheet_lm(
    const Params& p,
    const State& x0,
    const Costate& l0_init,
    const ShootSettingsHost& s,
    const double P[2][2]) {
    ShootResultHost best{};
    best.l0 = l0_init;

    const bool use_manifold = true;

    Costate l = l0_init;
    double lambda = s.lm_lambda0;

    ForwardSimOut out0{};
    forward_one_cuda(p, x0, l, s.T, s.dt, s.integrator, &out0);

    double r[4];
    int dim = 0;
    terminal_residual(out0, P, use_manifold, r, &dim);
    double cost = out0.cost;

    best.resid_dim = dim;
    for (int i = 0; i < dim; ++i) {
        best.resid[i] = r[i];
    }
    best.cost = cost;
    best.converged = (resid_inf(dim, r) <= s.tol_resid);

    if (s.debug) {
        std::fprintf(
            stderr,
            "[shoot] T=%.3f dt=%.3g init_l=(%.6g,%.6g) init_rinf=%.3e\n",
            s.T,
            s.dt,
            l.l1,
            l.l2,
            resid_inf(dim, r));
    }

    for (int iter = 0; iter < s.max_iters; ++iter) {
        const double rnorm = resid_inf(dim, r);
        if (rnorm <= s.tol_resid) {
            best.l0 = l;
            best.resid_dim = dim;
            for (int i = 0; i < dim; ++i) {
                best.resid[i] = r[i];
            }
            best.cost = cost;
            best.converged = true;
            best.iters = iter;
            return best;
        }

        double J[4][2]{};
        int m = 0;
        if (s.use_variational_jacobian) {
            variational_jacobian(out0, P, use_manifold, J, &m);
        } else {
            const double eps = std::max(1e-12, s.fd_eps);
            finite_diff_jacobian(p, x0, l, s.T, s.dt, eps, P, use_manifold, s.integrator, J, &m);
        }
        dim = m;

        // A = J^T J + lambda I  (2x2), g = J^T r (2)
        double H[2][2]{};
        H[0][0] = lambda;
        H[1][1] = lambda;
        double g0 = 0.0;
        double g1 = 0.0;

        for (int k = 0; k < m; ++k) {
            const double j0 = J[k][0];
            const double j1 = J[k][1];
            H[0][0] += j0 * j0;
            H[0][1] += j0 * j1;
            H[1][0] += j1 * j0;
            H[1][1] += j1 * j1;
            g0 += j0 * r[k];
            g1 += j1 * r[k];
        }

        double delta[2];
        const double rhs0 = -g0;
        const double rhs1 = -g1;
        const double rhst[2] = {rhs0, rhs1};
        if (!solve_2x2(H, rhst, delta)) {
            lambda = std::min(1e16, lambda * s.lm_lambda_mul);
            continue;
        }

        const double dnorm = std::hypot(delta[0], delta[1]);
        if (dnorm > s.max_delta_norm) {
            const double scl = s.max_delta_norm / dnorm;
            delta[0] *= scl;
            delta[1] *= scl;
        }

        bool accepted = false;
        Costate l_acc = l;
        double r_acc[4]{};
        int dim_acc = dim;
        double cost_acc = cost;
        double accepted_step = 0.0;

        const int ntrials = s.backtrack_max + 1;
        std::vector<Costate> trials(static_cast<std::size_t>(ntrials));
        std::vector<ForwardSimOut> trial_outs(static_cast<std::size_t>(ntrials));
        for (int bt = 0; bt < ntrials; ++bt) {
            const double mult = std::ldexp(1.0, -bt);
            trials[static_cast<std::size_t>(bt)].l1 = l.l1 + mult * delta[0];
            trials[static_cast<std::size_t>(bt)].l2 = l.l2 + mult * delta[1];
        }
        forward_batch_cuda(p, x0, trials.data(), ntrials, s.T, s.dt, s.integrator, trial_outs.data());

        for (int bt = 0; bt < ntrials; ++bt) {
            const ForwardSimOut& out_trial = trial_outs[static_cast<std::size_t>(bt)];
            double r_trial[4];
            int dtrial = 0;
            terminal_residual(out_trial, P, use_manifold, r_trial, &dtrial);
            const double rnorm_trial = resid_inf(dtrial, r_trial);

            if (std::isfinite(rnorm_trial) && rnorm_trial < rnorm) {
                accepted = true;
                l_acc.l1 = l.l1 + std::ldexp(1.0, -bt) * delta[0];
                l_acc.l2 = l.l2 + std::ldexp(1.0, -bt) * delta[1];
                dim_acc = dtrial;
                for (int i = 0; i < dim_acc; ++i) {
                    r_acc[i] = r_trial[i];
                }
                cost_acc = out_trial.cost;
                out0 = out_trial;
                accepted_step = std::ldexp(1.0, -bt);
                break;
            }
        }

        if (accepted) {
            l = l_acc;
            dim = dim_acc;
            for (int i = 0; i < dim; ++i) {
                r[i] = r_acc[i];
            }
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
                resid_inf(dim, r),
                accepted ? 1 : 0,
                accepted_step,
                lambda,
                l.l1,
                l.l2);
        }

        const double best_norm = resid_inf(best.resid_dim, best.resid);
        const double cur_norm = resid_inf(dim, r);
        if (cur_norm < best_norm) {
            best.l0 = l;
            best.resid_dim = dim;
            for (int i = 0; i < dim; ++i) {
                best.resid[i] = r[i];
            }
            best.cost = cost;
            best.iters = iter + 1;
        }
    }

    best.converged = (resid_inf(best.resid_dim, best.resid) <= s.tol_resid);
    return best;
}

ShootResultHost solve_costates_single_sheet_lm_continuation(
    const Params& p,
    const State& x0,
    const Costate& l0_init,
    const ShootSettingsHost& base,
    const std::vector<double>& T_list,
    const double P[2][2]) {
    ShootResultHost best_overall{};
    best_overall.l0 = l0_init;
    best_overall.resid_dim = 2;
    best_overall.resid[0] = std::numeric_limits<double>::infinity();
    best_overall.resid[1] = std::numeric_limits<double>::infinity();
    best_overall.cost = std::numeric_limits<double>::infinity();

    Costate l = l0_init;
    for (std::size_t i = 0; i < T_list.size(); ++i) {
        ShootSettingsHost st = base;
        st.T = T_list[i];

        const ShootResultHost stage = solve_costates_single_sheet_lm(p, x0, l, st, P);
        l = stage.l0;

        const double stage_norm = resid_inf(stage.resid_dim, stage.resid);
        const double best_norm = resid_inf(best_overall.resid_dim, best_overall.resid);
        if (stage_norm < best_norm) {
            best_overall = stage;
        }

        if (stage_norm <= st.tol_resid) {
            break;
        }
    }

    return best_overall;
}

}  // namespace pendulum
