#pragma once

#include <vector>

#include "core/solver_types.cuh"

/// GPU sweep of backward physics-only RK4 trajectories from terminal eigen-combo seeds; returns MS flat
/// guess (size `4 * num_shooting_intervals`) subsampled from the best seed by wrapped (theta,phi) mismatch at t=0.
/// Returns empty vector if Eigen basis fails, CUDA fails, or no finite score was produced.
std::vector<double> compute_backward_eigen_ms_warm_start(const SystemParams &sys, const IntegratorParams &int_params);
