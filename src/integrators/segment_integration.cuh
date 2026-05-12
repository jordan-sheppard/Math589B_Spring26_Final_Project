#pragma once

#include "core/solver_types.cuh"
#include "dynamics/pendulum_oc.cuh"

__host__ __device__ inline double compute_hamiltonian(const VarState &state, const SystemParams &params) {
    double theta = state.theta();
    double phi = state.phi();
    double l1 = state.l1();
    double l2 = state.l2();
    double alpha = params.alpha;

    double sin_t = sin(theta);
    double cos_t = cos(theta);

    double l2_sq = l2 * l2;
    double phi_sq = phi * phi;
    double cos_t_sq = cos_t * cos_t;

    return 1.0 - cos_t + 0.5 * phi_sq - 0.5 * l2_sq * cos_t_sq + l1 * phi + l2 * (sin_t - alpha * phi);
}

__host__ __device__ inline VarState rk4_step(const VarState &current, const SystemParams &params, double dt) {
    double half_dt = 0.5 * dt;

    VarState k1 = get_derivatives(current, params);

    VarState k2 = get_derivatives(current + (k1 * half_dt), params);

    VarState k3 = get_derivatives(current + (k2 * half_dt), params);

    VarState k4 = get_derivatives(current + (k3 * dt), params);

    VarState next_state = current + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
    return next_state;
}

__host__ __device__ inline VarState rk4_step_flow(const VarState &current, const SystemParams &params, double dt,
                                                  double flow_sign) {
    const double half_dt = 0.5 * dt;

    VarState k1 = get_derivatives_flow(current, params, flow_sign);
    VarState k2 = get_derivatives_flow(current + (k1 * half_dt), params, flow_sign);
    VarState k3 = get_derivatives_flow(current + (k2 * half_dt), params, flow_sign);
    VarState k4 = get_derivatives_flow(current + (k3 * dt), params, flow_sign);

    return current + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
}

__host__ __device__ inline SegmentEvaluation simulate_segment(const VarState &initial_guess,
                                                              const SystemParams &sys_params,
                                                              const IntegratorParams &int_params) {
    VarState current_state = initial_guess;

    current_state.cost() = 0.0;

#pragma unroll
    for (int r = 0; r < 4; r++) {
#pragma unroll
        for (int c = 0; c < 4; c++) {
            if (r == c) {
                current_state.M(r, c) = 1.0;
            } else {
                current_state.M(r, c) = 0.0;
            }
        }
    }

    double init_H = compute_hamiltonian(current_state, sys_params);

    const double flow_sign = int_params.backward_time ? -1.0 : 1.0;

    for (int step = 0; step < int_params.num_steps; step++) {
        current_state = rk4_step_flow(current_state, sys_params, int_params.dt, flow_sign);
    }

    SegmentEvaluation result;
    result.final_state = current_state;
    result.initial_hamiltonian = init_H;

    return result;
}
