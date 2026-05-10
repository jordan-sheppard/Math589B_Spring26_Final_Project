#pragma once

#include "core/host_buffers.hpp"
#include "core/solver_types.cuh"

__global__ void multiple_shooting_kernel(DeviceArrays d, SystemParams sys_params, IntegratorParams int_params);

void evaluate_segments_on_gpu(HDArrays &solver_arrays, const SystemParams &sys_params,
                              const IntegratorParams &int_params);
