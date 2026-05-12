#pragma once

#include "core/host_buffers.hpp"
#include "core/solver_host_types.hpp"
#include "core/solver_types.cuh"

// Assemble F(S) and J(S)=dF/dS for the square multiple-shooting system F=0 in R^{4N}.
// Residual layout: rows (0..4N-1) group as (N-1) interior interface blocks of size 4 (continuity of
// the Hamiltonian state across the chain) plus one terminal block of 4 rows: initial (theta,phi)
// Dirichlet values and terminal (theta,phi) matching goals. Columns follow S = [S_0;...;S_{N-1}] with
// S_k = (theta_k, phi_k, l1_k, l2_k). Nonzeros are 4x4 flow sensitivities from VarState::M, identity
// blocks, and boundary-specific entries — assembled here on the CPU from GPU-produced segment endpoints.

void build_global_system(const HDArrays &solver_arrays, const SystemParams &sys_params,
                         const IntegratorParams &int_params, SparseMat &J, VectorXd &F);
