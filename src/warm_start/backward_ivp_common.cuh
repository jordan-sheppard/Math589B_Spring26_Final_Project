#pragma once

#include "dynamics/pendulum_oc.cuh"

namespace warm_start {

constexpr int kPatchGrid = 49;

__host__ __device__ inline void varstate_zero_m(VarState &v) {
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            v.M(r, c) = 0.0;
        }
    }
}

__host__ __device__ inline VarState rk4_step_physics_only(const VarState &x_in, const SystemParams &params, double dt) {
    VarState x = x_in;
    varstate_zero_m(x);

    VarState k1, k2, k3, k4;
    compute_state_physics(x, params, k1);
    varstate_zero_m(k1);

    VarState x2 = x + k1 * (0.5 * dt);
    compute_state_physics(x2, params, k2);
    varstate_zero_m(k2);

    VarState x3 = x + k2 * (0.5 * dt);
    compute_state_physics(x3, params, k3);
    varstate_zero_m(k3);

    VarState x4 = x + k3 * dt;
    compute_state_physics(x4, params, k4);
    varstate_zero_m(k4);

    VarState out = x + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
    varstate_zero_m(out);
    return out;
}

__host__ __device__ inline double theta_phi_distance_wrapped(double theta, double phi, double theta0, double phi0) {
    double dtheta = theta - theta0;
    const double two_pi = 6.28318530717958647692;
    dtheta = dtheta - two_pi * rint(dtheta / two_pi);
    double dphi = phi - phi0;
    return sqrt(dtheta * dtheta + dphi * dphi);
}

__host__ __device__ inline double dist2_wrapped(double theta, double phi, double theta0, double phi0) {
    const double d = theta_phi_distance_wrapped(theta, phi, theta0, phi0);
    return d * d;
}

/// Small state near origin: linear combination of two stable 4-vectors (columns).
__host__ __device__ inline void origin_patch_state(double a, double b, const double col0[4], const double col1[4],
                                                   VarState &x) {
    x.theta() = col0[0] * a + col1[0] * b;
    x.phi() = col0[1] * a + col1[1] * b;
    x.l1() = col0[2] * a + col1[2] * b;
    x.l2() = col0[3] * a + col1[3] * b;
    x.cost() = 0.0;
    varstate_zero_m(x);
}

/// Physics-only backward IVP from origin patch; optional MS knot subsampling (length `4 * num_intervals`).
/// Returns wrapped (theta,phi) distance to `(target_th, target_ph)`.
__host__ __device__ inline double origin_patch_backward_to_targets(const SystemParams &sys_for_alpha, double dt,
                                                                 int num_intervals, int steps_per_interval,
                                                                 double a, double b, const double col0[4],
                                                                 const double col1[4], double target_th,
                                                                 double target_ph, double *out_traj_flat) {
    const int total_steps = num_intervals * steps_per_interval;
    VarState x;
    origin_patch_state(a, b, col0, col1, x);

    for (int s = 1; s <= total_steps; ++s) {
        x = rk4_step_physics_only(x, sys_for_alpha, -dt);

        if (out_traj_flat != nullptr) {
            for (int k = 0; k < num_intervals; ++k) {
                if (s == (num_intervals - k) * steps_per_interval) {
                    const int base = k * 4;
                    out_traj_flat[base + 0] = x.theta();
                    out_traj_flat[base + 1] = x.phi();
                    out_traj_flat[base + 2] = x.l1();
                    out_traj_flat[base + 3] = x.l2();
                }
            }
        }
    }

    return theta_phi_distance_wrapped(x.theta(), x.phi(), target_th, target_ph);
}

__host__ __device__ inline void patch_ab_from_ij(int i, int j, double radius, double &a, double &b) {
    const double g = static_cast<double>(kPatchGrid - 1);
    const double xi = -1.0 + 2.0 * static_cast<double>(i) / g;
    const double xj = -1.0 + 2.0 * static_cast<double>(j) / g;
    a = radius * xi;
    b = radius * xj;
}

} // namespace warm_start
