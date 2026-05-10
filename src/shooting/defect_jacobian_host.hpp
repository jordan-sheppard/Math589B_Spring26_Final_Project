#pragma once

#include "core/host_buffers.hpp"
#include "core/solver_host_types.hpp"
#include "core/solver_types.cuh"

void build_global_system(const HDArrays &solver_arrays, const SystemParams &sys_params, SparseMat &J,
                         VectorXd &F);
