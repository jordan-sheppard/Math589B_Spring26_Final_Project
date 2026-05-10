#pragma once

#include <vector>

#include "core/solver_types.cuh"

std::vector<double> compute_linear_initial_guess(const SystemParams &sys_params);

OptimizationResult solve_multiple_shooting(std::vector<double> &flat_node_guesses,
                                           const SystemParams &sys_params,
                                           const IntegratorParams &int_params,
                                           const NewtonParams &newton_params);

OptimizationResult solve_multiple_shooting(SystemParams sys_params, IntegratorParams int_params,
                                           NewtonParams newton_params);
