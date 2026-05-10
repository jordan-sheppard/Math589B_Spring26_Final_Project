#pragma once

#include "core/host_buffers.hpp"
#include "core/solver_types.cuh"

IterationLog compute_newton_step(HDArrays &solver_arrays, const SystemParams &sys_params,
                                 const IntegratorParams &int_params);
