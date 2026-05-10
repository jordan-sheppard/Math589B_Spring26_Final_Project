#pragma once

#include "../pendulum/types.hpp"
#include "forward_sim.cuh"

namespace pendulum {

// Run N independent augmented forward simulations on the GPU (one thread per trajectory).
// Uses a persistent device buffer resized as needed. No Eigen; device code lives in .cuh.
void forward_batch_cuda(
    const Params& p,
    const State& x0,
    const Costate* h_seeds,
    int n,
    double T,
    double dt,
    IntegratorKind kind,
    ForwardSimOut* h_outs);

inline void forward_one_cuda(
    const Params& p,
    const State& x0,
    const Costate& l0,
    double T,
    double dt,
    IntegratorKind kind,
    ForwardSimOut* out) {
    forward_batch_cuda(p, x0, &l0, 1, T, dt, kind, out);
}

}  // namespace pendulum
