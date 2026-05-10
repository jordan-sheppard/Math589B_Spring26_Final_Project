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

/// Same integrand as `compute_state_physics` cost slot (running cost density).
__host__ __device__ inline double running_cost_density(const VarState &state, const SystemParams &params) {
    double theta = state.theta();
    double phi = state.phi();
    double l2 = state.l2();
    double sin_t = sin(theta);
    double cos_t = cos(theta);
    double cos_t_sq = cos_t * cos_t;
    double l2_sq = l2 * l2;
    double phi_sq = phi * phi;
    (void)params;
    return 1.0 - cos_t + 0.5 * phi_sq + 0.5 * l2_sq * cos_t_sq;
}

__host__ __device__ inline void kahan_add(double &sum, double &comp, double y) {
    y -= comp;
    double t = sum + y;
    comp = (t - sum) - y;
    sum = t;
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

/// Cash–Karp fifth-order step (fixed step; embedded 4th order not used).
__host__ __device__ inline VarState rkck5_step(const VarState &y, const SystemParams &p, double h) {
    VarState k1 = get_derivatives(y, p);

    VarState k2 = get_derivatives(y + h * (1.0 / 5.0) * k1, p);

    VarState k3 = get_derivatives(y + h * ((3.0 / 40.0) * k1 + (9.0 / 40.0) * k2), p);

    VarState k4 = get_derivatives(y + h * ((3.0 / 10.0) * k1 + (-9.0 / 10.0) * k2 + (6.0 / 5.0) * k3), p);

    VarState k5 = get_derivatives(
        y + h * ((-11.0 / 54.0) * k1 + (5.0 / 2.0) * k2 + (-70.0 / 27.0) * k3 + (35.0 / 27.0) * k4), p);

    VarState k6 = get_derivatives(y + h * ((1631.0 / 55296.0) * k1 + (175.0 / 512.0) * k2 + (575.0 / 13824.0) * k3 +
                                             (44275.0 / 110592.0) * k4 + (253.0 / 4096.0) * k5),
                                  p);

    const double b1 = 37.0 / 378.0;
    const double b3 = 250.0 / 621.0;
    const double b4 = 125.0 / 594.0;
    const double b6 = 512.0 / 1771.0;

    return y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b6 * k6);
}

__host__ __device__ inline VarState integration_step(const VarState &current, const SystemParams &sys_params,
                                                     const IntegratorParams &int_params) {
    if (int_params.use_dp5) {
        return rkck5_step(current, sys_params, int_params.dt);
    }
    return rk4_step(current, sys_params, int_params.dt);
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

    double cost_sum = 0.0;
    double cost_comp = 0.0;
    const double dt = int_params.dt;

    for (int step = 0; step < int_params.num_steps; step++) {
        double fa = running_cost_density(current_state, sys_params);
        VarState next_state = integration_step(current_state, sys_params, int_params);
        double fb = running_cost_density(next_state, sys_params);
        double incr = 0.5 * dt * (fa + fb);
        kahan_add(cost_sum, cost_comp, incr);
        next_state.cost() = 0.0;
        current_state = next_state;
    }

    SegmentEvaluation result;
    result.final_state = current_state;
    result.final_state.cost() = cost_sum;
    result.initial_hamiltonian = init_H;

    return result;
}
