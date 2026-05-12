#pragma once

#include "core/host_buffers.hpp"
#include "core/solver_types.cuh"

// GPU work for MS: each thread independently time-marches one segment IVP (RK4 with `int_params.dt`
// and `int_params.num_steps` per segment — see `IntegratorParams` in `solver_types.cuh`). Outputs are
// segment endpoints and local sensitivity matrices consumed on the host when forming J and F.

__global__ void multiple_shooting_kernel(DeviceArrays d, SystemParams sys_params, IntegratorParams int_params);

void evaluate_segments_on_gpu(HDArrays &solver_arrays, const SystemParams &sys_params,
                              const IntegratorParams &int_params);
