#pragma once

// Mathematical setting (multiple shooting on one segment):
//
//   Augmented first-order ODE.  Let x = (theta, phi, l1, l2)^T denote the physical-adjoint state from the
//   Pontryagin necessary conditions (see `compute_state_physics`).  Along a trajectory, append a scalar c
//   with dc/dt = L(x), the running cost/Lagrangian integrand, so c accumulates ∫ L dt.  Append also the
//   variational matrix M(t) ∈ R^{4×4} satisfying dM/dt = A(x(t)) M with A = ∂(x-dot)/∂x, so M(t) =
//   ∂x(t)/∂x(t0) for fixed parameters.  Together, VarState encodes (x, c, M) as one vector space object so
//   the same RK4 stages apply componentwise (including M), consistent with the chain rule on the flow map.
//
//   Segment map.  One shooting segment applies the discrete-time flow: z_{k+1} = Φ_h^{N}(z_k), where h is
//   the RK4 step `dt`, N = `num_steps`, and Φ_h is one explicit RK4 step on the augmented ODE.  With
//   `backward_time`, the same ODE is integrated in decreasing physical time (flow_sign = -1), i.e. the
//   vector field is negated for x, c, and the variational block.
//
//   Role in defects.  Multiple shooting forms continuity defects between the integrated endpoint of segment
//   i and the initial guess at node i+1 (and boundary conditions separately).  This file only evaluates
//   the segment flow; defect assembly lives in the shooting layer.
//
//   Initial conditions for c and M.  At the segment start, c = 0 (integral starts empty).  M = I because
//   ∂x(t0)/∂x(t0) is the identity; subsequent RK4 steps transport M along dM/dt = A M.

#include "integrators/rk4_varstate.cuh"

/// Pontryagin Hamiltonian H(theta, phi, l1, l2) for the same model as `compute_state_physics` (diagnostic).
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

/// Discrete segment flow map: z_end ≈ Φ_h^N(z_start) with z = (x, c, M), h = `int_params.dt`, N = `num_steps`.
/// Resets c to 0 at the segment origin so c_end approximates ∫_{segment} L(x(s)) ds (running cost along the
/// segment only).  Initializes M = I_4 so subsequent RK4 updates approximate ∂x_end/∂x_start for Newton.
__host__ __device__ inline SegmentEvaluation simulate_segment(const VarState &initial_guess,
                                                              const SystemParams &sys_params,
                                                              const IntegratorParams &int_params) {
    VarState current_state = initial_guess;

    // Running cost component is the time integral of L; per segment we start the integral at the node.
    current_state.cost() = 0.0;

    // M(t0) = I: sensitivity of current x w.r.t. x at this segment's initial time is the identity.
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

    // H at the shooting node (before integration); used as a diagnostic scalar, not in the defect.
    double init_H = compute_hamiltonian(current_state, sys_params);

    const double flow_sign = int_params.backward_time ? -1.0 : 1.0;

    // N RK4 substeps of size dt: approximates continuous flow over signed arc length ≈ N·dt in time parameter.
    for (int step = 0; step < int_params.num_steps; step++) {
        current_state = rk4_step_flow(current_state, sys_params, int_params.dt, flow_sign);
    }

    SegmentEvaluation result;
    result.final_state = current_state;
    result.initial_hamiltonian = init_H;

    return result;
}
