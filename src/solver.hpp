// Top-level API: wires continuation + multiple shooting (see `driver/continuation_sheets.cu`).
// Mathematically, `solve` targets the same damped-pendulum OC/BVP as the `main` CLI:
//   given (θ, φ, α), search over angle wraps / continuation sheets and homotopy in scaled boundary
//   data, then refine a truncated-horizon multiple-shooting discretization of the state–costate flow.
#pragma once

#include <cstdio>
#include <vector>

#include "cuda/gpu_macros.cuh"
#include "core/host_buffers.hpp"
#include "core/solver_host_types.hpp"
#include "core/solver_types.cuh"
#include "integrators/segment_integration.cuh"
#include "shooting/defect_jacobian_host.hpp"
#include "shooting/gpu_eval_segments.hpp"
#include "shooting/multiple_shooting_solve.hpp"
#include "shooting/newton_iteration.hpp"

/// CPU driver: homotopy in scaled boundary data + sheet search; returns best MS solution found.
///
/// Parameters (same units/roles as `./solver theta phi alpha`):
///   `target_theta`   — θ (rad), angular position in the boundary / goal data fed to continuation.
///   `target_phi`     — φ = θ̇, angular velocity for that boundary specification.
///   `alpha`          — α > 0, damping in the pendulum dynamics and the eliminated-control running cost.
///
/// Return value: for the converged branch, `optimal_l1_init` / `optimal_l2_init` are costates
/// λ₁, λ₂ at the first shooting node; `optimal_cost` is the objective J (integrated Lagrangian).
Result solve(double target_theta, double target_phi, double alpha);
