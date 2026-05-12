#pragma once

#include <vector>

#include "core/solver_types.cuh"

// Multiple shooting (MS) unknown: one stacked vector S in R^{4N}, N = num_shooting_intervals.
// Knot k stores (theta_k, phi_k, l1_k, l2_k) — configuration and costate at that mesh point in the
// indirect (PMP) formulation. Segment k integrates the Hamiltonian flow for a fixed horizon slice;
// interior defects enforce continuity of the full state across the chain; two boundary rows pin
// (theta,phi) at t=0 and two match (theta,phi) at the terminal time to (theta_goal, phi_goal).

/// Piecewise-linear guess in (theta,phi) across knots; co-states (l1,l2) start at zero (cheap cold start).
std::vector<double> compute_linear_initial_guess(const SystemParams &sys_params);

// Outer damped Newton driver: each inner step solves J(S) dS = -F(S) with F the shooting defect stack
// and J its Jacobian (see `defect_jacobian_host` / `newton_iteration`). Stops when ||F||_infty <
// newton_params.tolerance or after newton_params.max_iterations steps without meeting tolerance.

/// Newton loop on shooting defects; on success, `flat_node_guesses` is overwritten with the converged knot values.
OptimizationResult solve_multiple_shooting(std::vector<double> &flat_node_guesses,
                                           const SystemParams &sys_params,
                                           const IntegratorParams &int_params,
                                           const NewtonParams &newton_params);

/// Convenience overload: builds a linear initial guess and runs forward-time MS only (`backward_time` forced false).
OptimizationResult solve_multiple_shooting(SystemParams sys_params, IntegratorParams int_params,
                                           NewtonParams newton_params);
