// CUDA error checking macro used after launches and memory ops (host-side only).
//
// Numerical-safety rationale: a silently-failed kernel (bad launch config, OOM,
// illegal memory access) leaves device buffers in an undefined state. If we
// continued, downstream host computations -- segment residuals F_k, the block
// Jacobian J, Newton steps J*dx = -F, and norms ||F||, ||dx|| -- would be
// computed from uninitialized/garbage memory, producing NaN/Inf or, worse,
// finite-but-wrong values that pass tolerance checks. Aborting at the first
// cudaError preserves the invariant that every floating-point result consumed
// by the solver was produced by a successful kernel, so convergence diagnostics
// (residual decrease, Newton contraction) reflect true numerics rather than
// memory corruption.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) std::exit(code);
    }
}
