#pragma once

#include <vector>

#include "core/solver_types.cuh"

/// Multi-well GPU patch search (origin stable plane, dense radii / 49×49 grid) plus subsampled MS guesses.
/// Returns up to `top_k` flat trajectories (`4 * num_shooting_intervals` each), ordered by increasing GPU score
/// (wrapped (theta,phi) distance to the well-shifted target). Empty if Eigen/CUDA failure.
std::vector<std::vector<double>> compute_patch_topk_ms_warm_starts(const SystemParams &sys,
                                                                   const IntegratorParams &int_params,
                                                                   int top_k = 12);
