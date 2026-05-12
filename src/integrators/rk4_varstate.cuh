#pragma once

// Explicit RK4 on the augmented `VarState` ODE from `get_derivatives` / `get_derivatives_flow`
// (`dynamics/pendulum_oc.cuh`). Used by `simulate_segment` and kept here so the segment map
// file stays focused on initialization, time direction, and outputs.

#include "dynamics/pendulum_oc.cuh"

/// One explicit RK4 step (classical 4th order) on the augmented first-order system dz/dt = f(z) from
/// `get_derivatives`, with z = (x, c, M):  k1..k4 are evaluations of f;  z^{+} = z + (h/6)(k1 + 2k2 + 2k3 + k4).
/// Here h = `dt` is one substep; local truncation error is O(h^5) per step.
__host__ __device__ inline VarState rk4_step(const VarState &current, const SystemParams &params, double dt) {
    double half_dt = 0.5 * dt;

    VarState k1 = get_derivatives(current, params);

    VarState k2 = get_derivatives(current + (k1 * half_dt), params);

    VarState k3 = get_derivatives(current + (k2 * half_dt), params);

    VarState k4 = get_derivatives(current + (k3 * dt), params);

    VarState next_state = current + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
    return next_state;
}

/// Same RK4 tableau as `rk4_step`, but on f_flow = flow_sign · f so that flow_sign = +1 is forward physical
/// time and -1 reverses the ODE (backward IVP / decreasing time parameter) consistently for x, cost rate,
/// and dM/dt = flow_sign · A M.
__host__ __device__ inline VarState rk4_step_flow(const VarState &current, const SystemParams &params, double dt,
                                                  double flow_sign) {
    const double half_dt = 0.5 * dt;

    VarState k1 = get_derivatives_flow(current, params, flow_sign);
    VarState k2 = get_derivatives_flow(current + (k1 * half_dt), params, flow_sign);
    VarState k3 = get_derivatives_flow(current + (k2 * half_dt), params, flow_sign);
    VarState k4 = get_derivatives_flow(current + (k3 * dt), params, flow_sign);

    return current + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
}
