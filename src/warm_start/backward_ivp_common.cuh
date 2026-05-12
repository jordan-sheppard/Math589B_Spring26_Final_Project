#pragma once

#include "dynamics/pendulum_oc.cuh"

namespace warm_start {

constexpr int kRGridCount = 15;
constexpr int kABGridSize = 40;

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

/// Terminal state at forward time T: equilibrium at goal plus a*v1 + b*v2.
__host__ __device__ inline VarState make_terminal_state(const SystemParams &sys, double a, double b, const double v1[4],
                                                        const double v2[4]) {
    VarState x;
    x.theta() = sys.theta_goal + a * v1[0] + b * v2[0];
    x.phi() = sys.phi_goal + a * v1[1] + b * v2[1];
    x.l1() = a * v1[2] + b * v2[2];
    x.l2() = a * v1[3] + b * v2[3];
    x.cost() = 0.0;
    varstate_zero_m(x);
    return x;
}

/// Backward RK4 (physics only) from x(T). If `out_traj_flat` is non-null, length is 4 * num_intervals (MS node values).
/// Returns wrapped (theta,phi) distance to (theta_init, phi_init) at t=0.
__host__ __device__ inline double backward_ivp_to_ms_guess(const SystemParams &sys, double dt, int num_intervals,
                                                           int steps_per_interval, double a, double b,
                                                           const double v1[4], const double v2[4],
                                                           double *out_traj_flat) {
    const int total_steps = num_intervals * steps_per_interval;
    VarState x = make_terminal_state(sys, a, b, v1, v2);

    for (int s = 1; s <= total_steps; ++s) {
        x = rk4_step_physics_only(x, sys, -dt);

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

    return theta_phi_distance_wrapped(x.theta(), x.phi(), sys.theta_init, sys.phi_init);
}

__host__ __device__ inline int total_seed_count() { return kRGridCount * kABGridSize * kABGridSize; }

__host__ __device__ inline void decode_seed_index(int seed, int &ir, int &ia, int &ib) {
    ib = seed % kABGridSize;
    seed /= kABGridSize;
    ia = seed % kABGridSize;
    seed /= kABGridSize;
    ir = seed;
}

__host__ __device__ inline void ab_from_grid(int ia, int ib, double r, double &a, double &b) {
    const double g = static_cast<double>(kABGridSize - 1);
    a = -r + (2.0 * r) * (static_cast<double>(ia) / g);
    b = -r + (2.0 * r) * (static_cast<double>(ib) / g);
}

} // namespace warm_start
