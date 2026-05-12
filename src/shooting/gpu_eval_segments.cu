#include "shooting/gpu_eval_segments.hpp"

#include "cuda/gpu_macros.cuh"
#include "integrators/segment_integration.cuh"

__global__ void multiple_shooting_kernel(DeviceArrays d, SystemParams sys_params, IntegratorParams int_params) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if (k >= sys_params.num_shooting_intervals) {
        return;
    }

    VarState initial_guess;
    initial_guess.theta() = d.node_guesses[k * 4 + 0];
    initial_guess.phi() = d.node_guesses[k * 4 + 1];
    initial_guess.l1() = d.node_guesses[k * 4 + 2];
    initial_guess.l2() = d.node_guesses[k * 4 + 3];

    SegmentEvaluation result = simulate_segment(initial_guess, sys_params, int_params);

    d.segment_results[k] = result;
}

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
