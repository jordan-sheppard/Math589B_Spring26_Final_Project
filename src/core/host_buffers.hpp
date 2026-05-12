#pragma once

// Host/device buffers for the **discrete** multiple-shooting unknowns and the **continuous** IVP
// flow map evaluated on each shooting segment.
//
// Unknown vector x ∈ R^{4N} (N = num_intervals): one copy of the Pontryagin augmented state
// (θ, φ, ℓ₁, ℓ₂) at each shooting knot — angle, angular rate, and costates for the damped-pendulum
// OC model (see `pendulum_oc.cuh`). Layout is contiguous by knot:
//   indices [4k .. 4k+3] ↔ (θ_k, φ_k, ℓ₁_k, ℓ₂_k),  k = 0 … N−1.
// This is the same ordering used when assembling F and J on the host.
//
// `h_segment_results[k]` stores the **endpoint** of the forward (or backward-time) IVP that starts
// from knot k over one segment; together with the next knot’s x_{k+1} it defines the k-th defect.

#include <vector>

#include <cuda_runtime.h>

#include "core/solver_types.cuh"
#include "cuda/gpu_macros.cuh"

struct HDArrays {
    std::vector<double> h_node_guesses;              // x in R^{4N}; see layout comment above
    std::vector<SegmentEvaluation> h_segment_results; // Φ_k(x_k) — flow map image at segment end, one per k

    double *d_node_guesses;            // device mirror of `h_node_guesses` (same flat layout)
    SegmentEvaluation *d_segment_results; // device mirror of `h_segment_results`

    explicit HDArrays(int num_intervals);

    void copy_guesses_to_device();   // H2D before kernel
    void copy_results_to_host();    // D2H after synchronize (see gpu_eval_segments)

    DeviceArrays get_device_arrays() const;

    ~HDArrays();
};
