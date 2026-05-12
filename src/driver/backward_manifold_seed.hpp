#pragma once

#include <vector>

#include "core/solver_types.cuh"

/// Build a multiple-shooting flat node guess (size `4 * num_shooting_intervals`) from a 2D grid
/// of linearized stable-manifold terminals near `theta_goal`, backward Hamiltonian flow with a
/// running minimum of squared distance to `(theta_init, phi_init)`, then forward fill along the
/// same ODE. Returns false only if dynamics produced no finite candidate (e.g. all NaN).
bool build_ms_guess_from_backward_cloud(const SystemParams &sys, const IntegratorParams &integ,
                                         std::vector<double> &out_flat_nodes);
