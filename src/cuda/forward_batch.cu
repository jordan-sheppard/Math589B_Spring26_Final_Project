#include "forward_gpu.hpp"

#include <cuda_runtime.h>
#include <cstddef>

namespace pendulum {

namespace {

__global__ void pendulum_forward_batch_kernel(
    Params p,
    State x0,
    const Costate* __restrict__ d_seeds,
    double T,
    double dt,
    IntegratorKind integrator,
    ForwardSimOut* __restrict__ d_outs,
    int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    d_outs[i] = simulate_forward(p, x0, d_seeds[i], T, dt, integrator);
}

// Each host thread that runs an independent solve must have its own stream and buffers
// so concurrent LM workers do not stomp each other's device memory.
thread_local cudaStream_t tls_stream = nullptr;

inline void ensure_tls_stream() {
    if (tls_stream == nullptr) {
        cudaStreamCreateWithFlags(&tls_stream, cudaStreamNonBlocking);
    }
}

thread_local Costate* d_seeds_tl = nullptr;
thread_local ForwardSimOut* d_outs_tl = nullptr;
thread_local int cap_tl = 0;

}  // namespace

void forward_batch_cuda(
    const Params& p,
    const State& x0,
    const Costate* h_seeds,
    int n,
    double T,
    double dt,
    IntegratorKind kind,
    ForwardSimOut* h_outs) {
    ensure_tls_stream();

    if (n <= 0) {
        return;
    }
    if (n > cap_tl) {
        cudaFree(d_seeds_tl);
        cudaFree(d_outs_tl);
        d_seeds_tl = nullptr;
        d_outs_tl = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&d_seeds_tl), static_cast<std::size_t>(n) * sizeof(Costate));
        cudaMalloc(reinterpret_cast<void**>(&d_outs_tl), static_cast<std::size_t>(n) * sizeof(ForwardSimOut));
        cap_tl = n;
    }

    cudaMemcpyAsync(
        d_seeds_tl,
        h_seeds,
        static_cast<std::size_t>(n) * sizeof(Costate),
        cudaMemcpyHostToDevice,
        tls_stream);

    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    pendulum_forward_batch_kernel<<<blocks, threads, 0, tls_stream>>>(
        p, x0, d_seeds_tl, T, dt, kind, d_outs_tl, n);

    cudaMemcpyAsync(
        h_outs,
        d_outs_tl,
        static_cast<std::size_t>(n) * sizeof(ForwardSimOut),
        cudaMemcpyDeviceToHost,
        tls_stream);
    cudaStreamSynchronize(tls_stream);
}

}  // namespace pendulum
