#pragma once

#include "core/host_buffers.hpp"
#include "core/solver_types.cuh"

// One Newton step for F(S)=0: (i) evaluate segment flows Phi_k and sensitivities on the GPU from the
// current knot vector S; (ii) on the CPU, form the square sparse Jacobian J = dF/dS (block structure
// from chain rule across shooting interfaces plus boundary rows); (iii) solve J dS = -F (full Newton,
// no line search); (iv) S <- S + dS on the host. `IterationLog` returns ||F||_infty before the update
// and ||dS||_2 for diagnostics.

IterationLog compute_newton_step(HDArrays &solver_arrays, const SystemParams &sys_params,
                                 const IntegratorParams &int_params);
