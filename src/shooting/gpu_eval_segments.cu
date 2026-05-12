#include "shooting/gpu_eval_segments.hpp"

#include "cuda/gpu_macros.cuh"
#include "integrators/segment_integration.cuh"

// Kernel indexing vs MS convention: forward mode uses knot k as the initial condition for segment k.
// Backward mode uses knot k+1 for segments k = 0..N-2 so the discrete flow advances toward decreasing
// label index; the final segment k = N-1 still starts at knot N-1 (the left end of the horizon), matching
// `defect_jacobian_host` assembly.

/// One thread per shooting segment `k`: load segment initial state from `d.node_guesses`, integrate, write result[k].
/// When `backward_time` is true, segment `k` reads the *right* knot `(k+1)` so the IVP flows backward in the chain.
__global__ void multiple_shooting_kernel(DeviceArrays d, SystemParams sys_params, IntegratorParams int_params) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= sys_params.num_shooting_intervals) {
        return;
    }

    VarState initial_guess;
    const int N = sys_params.num_shooting_intervals;
    // Forward MS: segment k starts at knot k. Backward MS: interior segments start from knot k+1 (see defect assembly).
    const int base =
        (int_params.backward_time && k < N - 1) ? ((k + 1) * 4) : (k * 4);
    initial_guess.theta() = d.node_guesses[base + 0];
    initial_guess.phi() = d.node_guesses[base + 1];
    initial_guess.l1() = d.node_guesses[base + 2];
    initial_guess.l2() = d.node_guesses[base + 3];

    SegmentEvaluation result = simulate_segment(initial_guess, sys_params, int_params);

    d.segment_results[k] = result;
}

/// Host orchestration: H2D guesses, launch one block grid, sync, D2H segment endpoints.
// threads_per_block is a launch occupancy choice only; it does not change the mathematical MS problem.
void evaluate_segments_on_gpu(HDArrays &solver_arrays, const SystemParams &sys_params,
                              const IntegratorParams &int_params) {
    solver_arrays.copy_guesses_to_device();

    int threads_per_block = 256;
    int blocks_per_grid =
        (sys_params.num_shooting_intervals + threads_per_block - 1) / threads_per_block;

    multiple_shooting_kernel<<<blocks_per_grid, threads_per_block>>>(
        solver_arrays.get_device_arrays(), sys_params, int_params);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    solver_arrays.copy_results_to_host();
}
