#pragma once

// Helpers shared by CPU warm-start reconstruction and the GPU patch scoring kernel.
//
// Stable subspace / patch.  At the upright equilibrium the linearized closed-loop Hamiltonian
// Jacobian has a two-dimensional stable invariant subspace E^s (eigenvalues with Re λ < 0).
// Small states x ≈ a u0 + b u1 with (a,b) in a disk lie in that tangent plane and lie (to first
// order) on the nonlinear stable manifold; we use them as a two-parameter family of backward IVP
// initial data.
//
// Backward IVP.  With the optimal feedback substituted ("physics-only" RHS), the same vector field
// f(x) runs forward in time.  A backward trajectory x(t) with negative dt solves dx/dt = f(x) with
// reversed time: integrating from t = 0 at the patch toward t = -T is equivalent to shooting
// from a candidate initial condition on E^s and asking where the forward flow lands at horizon T
// in (theta, phi)—here we measure only the terminal angles against targets.
//
// Scoring metric.  theta is 2π-periodic; phi is treated as a real coordinate.  Distance uses the
// product geometry on the cylinder S^1 × R: shortest arc in theta plus Euclidean separation in phi
// (see dist2_wrapped for the squared metric used in ranking).
//
// Grid / radii.  patch_ab_from_ij maps a uniform (i,j) grid on [-1,1]^2 to (a,b) = radius * (xi,xj),
// i.e. a tensor-product discretization of the radius-scaled square in coefficient space.

#include "dynamics/pendulum_oc.cuh"

namespace warm_start {

// Patch resolution: (kPatchGrid)^2 samples of (a,b) per radius level (odd count centers a cell on 0).
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

// One explicit RK4 step of dx/dt = f(x); backward warm starts pass dt < 0 (same f, reversed time).
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

// Geodesic distance on S^1_theta × R_phi: |dtheta| minimized over 2πZ, then Euclidean in (dtheta,dphi).
__host__ __device__ inline double theta_phi_distance_wrapped(double theta, double phi, double theta0, double phi0) {
    double dtheta = theta - theta0;
    const double two_pi = 6.28318530717958647692;
    dtheta = dtheta - two_pi * rint(dtheta / two_pi); // principal wrap: dtheta ∈ (-π, π]
    double dphi = phi - phi0;
    return sqrt(dtheta * dtheta + dphi * dphi);
}

// Squared wrapped distance d^2 on S^1×R; used as the nonnegative score minimized in top-K search.
__host__ __device__ inline double dist2_wrapped(double theta, double phi, double theta0, double phi0) {
    const double d = theta_phi_distance_wrapped(theta, phi, theta0, phi0);
    return d * d;
}

// Physics state components within a generous magnitude bound (GPU patch scoring + host replay guards).
__host__ __device__ inline bool varstate_physics_finite(const VarState &x) {
    return isfinite(x.theta()) && isfinite(x.phi()) && isfinite(x.l1()) && isfinite(x.l2()) &&
           fabs(x.theta()) <= 1.0e9 && fabs(x.phi()) <= 1.0e9 && fabs(x.l1()) <= 1.0e9 && fabs(x.l2()) <= 1.0e9;
}

// RK4 backward integration of the physics-only field; returns squared wrapped terminal distance, or a huge
// sentinel if any step blows up (matches `patch_score_kernel` selection semantics).
__host__ __device__ inline double patch_backward_terminal_dist2_or_huge(VarState x, const SystemParams &sys_alpha_only,
                                                                        double dt, int num_intervals,
                                                                        int steps_per_interval, double theta_tgt,
                                                                        double phi_init_base) {
    const int total_steps = num_intervals * steps_per_interval;
    for (int s = 1; s <= total_steps; ++s) {
        x = rk4_step_physics_only(x, sys_alpha_only, -dt);
        if (!varstate_physics_finite(x)) {
            return 1.0e300;
        }
    }
    return dist2_wrapped(x.theta(), x.phi(), theta_tgt, phi_init_base);
}

/// Small state near origin: x ∈ span{col0,col1} ⊂ R^4, an orthonormal basis of the stable subspace slice.
__host__ __device__ inline void origin_patch_state(double a, double b, const double col0[4], const double col1[4],
                                                   VarState &x) {
    x.theta() = col0[0] * a + col1[0] * b;
    x.phi() = col0[1] * a + col1[1] * b;
    x.l1() = col0[2] * a + col1[2] * b;
    x.l2() = col0[3] * a + col1[3] * b;
    x.cost() = 0.0;
    varstate_zero_m(x);
}

// If `out_traj_flat` is non-null, record nodal samples when backward step `s` hits an MS knot time.
__host__ __device__ inline void origin_patch_record_ms_knots(const VarState &x, int s, int num_intervals,
                                                             int steps_per_interval, double *out_traj_flat) {
    if (out_traj_flat == nullptr) {
        return;
    }
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

/// Backward Cauchy problem: start on the stable patch at s=0, apply RK4 with step -dt for
/// `total_steps = num_intervals * steps_per_interval`.  Optional MS knot subsampling (length `4 * num_intervals`)
/// stores states at backward times matching forward multiple-shooting knots.
/// Returns the wrapped (theta,phi) distance to `(target_th, target_ph)` — the objective minimized by top-K search.
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
        origin_patch_record_ms_knots(x, s, num_intervals, steps_per_interval, out_traj_flat);
    }

    return theta_phi_distance_wrapped(x.theta(), x.phi(), target_th, target_ph);
}

// Affine map from grid indices (i,j) ∈ {0,…,G-1}^2 to (a,b) ∈ [-radius,radius]^2 (scaled reference square).
__host__ __device__ inline void patch_ab_from_ij(int i, int j, double radius, double &a, double &b) {
    const double g = static_cast<double>(kPatchGrid - 1);
    const double xi = -1.0 + 2.0 * static_cast<double>(i) / g;
    const double xj = -1.0 + 2.0 * static_cast<double>(j) / g;
    a = radius * xi;
    b = radius * xj;
}

// Row-major unravel of linear index over (well, radius, i, j); matches `patch_score_kernel` thread layout.
__host__ __device__ inline void decode_patch_linear_index(int idx, int grid_n, int n_rad, int &iw, int &ir, int &i,
                                                          int &j) {
    const int nij = grid_n * grid_n;
    int tmp = idx;
    const int ij = tmp % nij;
    tmp /= nij;
    ir = tmp % n_rad;
    tmp /= n_rad;
    iw = tmp;
    j = ij % grid_n;
    i = ij / grid_n;
}

} // namespace warm_start
