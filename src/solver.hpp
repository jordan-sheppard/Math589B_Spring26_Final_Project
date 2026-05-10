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

Result solve(double target_theta, double target_phi, double alpha);
