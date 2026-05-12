#include "backward_manifold_seed.hpp"

#include <cmath>
#include <cstdio>

#include "core/manifold_seed.hpp"
#include "core/solver_debug.hpp"
#include "dynamics/pendulum_oc.cuh"

namespace {

inline void zero_mat(Mat4x4 &M) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            M(r, c) = 0.0;
        }
    }
}

/// RK4 on position-costate only (no variational matrix), for cheap host screening / forward fill.
__host__ inline VarState physics_deriv_zero_m(const VarState &z, const SystemParams &p) {
    VarState d;
    compute_state_physics(z, p, d);
    d.cost() = 0.0;
    zero_mat(d.M);
    return d;
}

__host__ inline VarState rk4_physics(const VarState &current, const SystemParams &params, double dt) {
    const double half_dt = 0.5 * dt;
    const VarState k1 = physics_deriv_zero_m(current, params);
    const VarState k2 = physics_deriv_zero_m(current + (k1 * half_dt), params);
    const VarState k3 = physics_deriv_zero_m(current + (k2 * half_dt), params);
    const VarState k4 = physics_deriv_zero_m(current + (k3 * dt), params);
    VarState next = current + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
    next.cost() = 0.0;
    zero_mat(next.M);
    return next;
}

__host__ inline double dist2_xy(double theta, double phi, double theta0, double phi0) {
    const double e_th = std::atan2(std::sin(theta - theta0), std::cos(theta - theta0));
    const double d_ph = phi - phi0;
    return e_th * e_th + d_ph * d_ph;
}

__host__ inline void running_min_update_state(const VarState &z, double theta0, double phi0,
                                              double &best_d2, double &best_l1, double &best_l2) {
    const double d2 = dist2_xy(z.theta(), z.phi(), theta0, phi0);
    if (d2 < best_d2) {
        best_d2 = d2;
        best_l1 = z.l1();
        best_l2 = z.l2();
    }
}

/// One backward ray: integrate from `z_start` for `n_steps` steps of size `dt_neg` (negative).
__host__ void backward_ray(const VarState &z_start, const SystemParams &sys, double dt_neg, int n_steps,
                           double theta0, double phi0, double &out_best_d2, double &out_best_l1,
                           double &out_best_l2) {
    VarState z = z_start;
    z.cost() = 0.0;
    zero_mat(z.M);

    out_best_d2 = 1e300;
    running_min_update_state(z, theta0, phi0, out_best_d2, out_best_l1, out_best_l2);

    for (int s = 0; s < n_steps; ++s) {
        z = rk4_physics(z, sys, dt_neg);
        if (!std::isfinite(z.theta()) || !std::isfinite(z.phi()) || !std::isfinite(z.l1()) ||
            !std::isfinite(z.l2())) {
            out_best_d2 = 1e300;
            return;
        }
        running_min_update_state(z, theta0, phi0, out_best_d2, out_best_l1, out_best_l2);
    }
}

__host__ bool forward_fill_nodes(const SystemParams &sys, const IntegratorParams &integ,
                                 const VarState &z0, std::vector<double> &out_flat) {
    const int N = sys.num_shooting_intervals;
    out_flat.resize(static_cast<size_t>(N * 4));

    VarState z = z0;
    z.cost() = 0.0;
    zero_mat(z.M);

    out_flat[0] = z.theta();
    out_flat[1] = z.phi();
    out_flat[2] = z.l1();
    out_flat[3] = z.l2();

    const double dt = integ.dt;
    for (int k = 1; k < N; ++k) {
        for (int step = 0; step < integ.num_steps; ++step) {
            z = rk4_physics(z, sys, dt);
            if (!std::isfinite(z.theta()) || !std::isfinite(z.phi()) || !std::isfinite(z.l1()) ||
                !std::isfinite(z.l2())) {
                return false;
            }
        }
        const int o = k * 4;
        out_flat[static_cast<size_t>(o + 0)] = z.theta();
        out_flat[static_cast<size_t>(o + 1)] = z.phi();
        out_flat[static_cast<size_t>(o + 2)] = z.l1();
        out_flat[static_cast<size_t>(o + 3)] = z.l2();
    }
    return true;
}

}  // namespace

bool build_ms_guess_from_backward_cloud(const SystemParams &sys, const IntegratorParams &integ,
                                        std::vector<double> &out_flat_nodes) {
    const int N = sys.num_shooting_intervals;
    if (N <= 0 || integ.num_steps <= 0) {
        return false;
    }

    const double dtheta_ic = sys.theta_init - sys.theta_goal;
    const double dphi_ic = sys.phi_init - sys.phi_goal;
    const double dx_norm =
        std::sqrt(dtheta_ic * dtheta_ic + dphi_ic * dphi_ic);
    const double eps_base =
        0.12 * std::max(1e-6, std::min(dx_norm, 0.85));

    double P[4];
    stable_manifold_P(sys.alpha, P);
    const double P11 = P[0], P12 = P[1], P21 = P[2], P22 = P[3];

    const int grid_n = 7;
    const int n_back_steps = 8 * integ.num_steps;
    const double dt_neg = -std::fabs(integ.dt);

    const bool dbg = math589_solver_debug_enabled();

    double global_best_d2 = 1e300;
    double global_l1 = 0.0;
    double global_l2 = 0.0;
    bool any_finite = false;

    for (int attempt = 0; attempt < 3; ++attempt) {
        const double eps = eps_base * std::pow(1.65, static_cast<double>(attempt));

        for (int i = 0; i < grid_n; ++i) {
            const double ti = grid_n == 1 ? 0.0 : (static_cast<double>(i) / static_cast<double>(grid_n - 1));
            const double sx = 2.0 * ti - 1.0;
            for (int j = 0; j < grid_n; ++j) {
                const double tj =
                    grid_n == 1 ? 0.0 : (static_cast<double>(j) / static_cast<double>(grid_n - 1));
                const double sy = 2.0 * tj - 1.0;
                const double d_th = sx * eps;
                const double d_ph = sy * eps;

                VarState zT;
                zT.theta() = sys.theta_goal + d_th;
                zT.phi() = sys.phi_goal + d_ph;
                zT.l1() = P11 * d_th + P12 * d_ph;
                zT.l2() = P21 * d_th + P22 * d_ph;
                zT.cost() = 0.0;
                zero_mat(zT.M);

                double ray_d2 = 1e300;
                double ray_l1 = 0.0;
                double ray_l2 = 0.0;
                backward_ray(zT, sys, dt_neg, n_back_steps, sys.theta_init, sys.phi_init, ray_d2, ray_l1,
                             ray_l2);

                if (ray_d2 < 1e290) {
                    any_finite = true;
                    if (ray_d2 < global_best_d2) {
                        global_best_d2 = ray_d2;
                        global_l1 = ray_l1;
                        global_l2 = ray_l2;
                    }
                }
            }
        }
    }

    if (!any_finite) {
        return false;
    }

    VarState z0;
    z0.theta() = sys.theta_init;
    z0.phi() = sys.phi_init;
    z0.l1() = global_l1;
    z0.l2() = global_l2;
    z0.cost() = 0.0;
    zero_mat(z0.M);

    if (!forward_fill_nodes(sys, integ, z0, out_flat_nodes)) {
        return false;
    }

    if (dbg) {
        std::fprintf(stderr,
                     "[MATH589][BACKSEED] best_d2=%.6g l1=%.8g l2=%.8g n_back_steps=%d grid=%d eps_base=%.6g\n",
                     global_best_d2, global_l1, global_l2, n_back_steps, grid_n, eps_base);
    }

    return true;
}
