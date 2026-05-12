#pragma once

#include <cmath>

#include "core/solver_types.cuh"

__host__ __device__ inline void compute_state_physics(const VarState &state, const SystemParams &params,
                                                      VarState &ds) {
    double theta = state.theta();
    double phi = state.phi();
    double l1 = state.l1();
    double l2 = state.l2();
    double alpha = params.alpha;

    double sin_t = sin(theta);
    double cos_t = cos(theta);

    double cos_t_sq = cos_t * cos_t;
    double l2_sq = l2 * l2;
    double phi_sq = phi * phi;

    ds.theta() = phi;
    ds.phi() = sin_t - alpha * phi - l2 * cos_t_sq;
    ds.l1() = -l2_sq * cos_t * sin_t - l2 * cos_t - sin_t;
    ds.l2() = -phi - l1 + alpha * l2;

    ds.cost() = 1.0 - cos_t + 0.5 * phi_sq + 0.5 * l2_sq * cos_t_sq;
}

__host__ __device__ inline Mat4x4 compute_sensitivity_jacobian(const VarState &state,
                                                                const SystemParams &params) {
    double theta = state.theta();
    double phi = state.phi();
    double l1 = state.l1();
    double l2 = state.l2();
    double alpha = params.alpha;

    double sin_t = sin(theta);
    double cos_t = cos(theta);

    double cos_t_sq = cos_t * cos_t;
    double sin_t_sq = sin_t * sin_t;
    double l2_sq = l2 * l2;

    Mat4x4 A;

    A(0, 0) = 0.;
    A(0, 1) = 1.;
    A(0, 2) = 0.;
    A(0, 3) = 0.;

    A(1, 0) = cos_t + 2.0 * l2 * cos_t * sin_t;
    A(1, 1) = -alpha;
    A(1, 2) = 0.;
    A(1, 3) = -cos_t_sq;

    A(2, 0) = -(l2_sq * (cos_t_sq - sin_t_sq) - l2 * sin_t + cos_t);
    A(2, 1) = 0.;
    A(2, 2) = 0.;
    A(2, 3) = -(2.0 * l2 * cos_t * sin_t + cos_t);

    A(3, 0) = 0.;
    A(3, 1) = -1.;
    A(3, 2) = -1.;
    A(3, 3) = alpha;

    return A;
}

__host__ __device__ inline VarState get_derivatives(const VarState &state, const SystemParams &params) {
    VarState ds;

    compute_state_physics(state, params, ds);

    Mat4x4 A = compute_sensitivity_jacobian(state, params);
    ds.M = A * state.M;

    return ds;
}

/// `flow_sign = +1` forward time; `-1` backward time (negate ODE and variational block).
__host__ __device__ inline VarState get_derivatives_flow(const VarState &state, const SystemParams &params,
                                                         double flow_sign) {
    VarState ds;
    compute_state_physics(state, params, ds);
    ds.theta() *= flow_sign;
    ds.phi() *= flow_sign;
    ds.l1() *= flow_sign;
    ds.l2() *= flow_sign;
    ds.cost() *= flow_sign;

    Mat4x4 A = compute_sensitivity_jacobian(state, params);
    ds.M = (flow_sign * 1.0) * (A * state.M);

    return ds;
}
