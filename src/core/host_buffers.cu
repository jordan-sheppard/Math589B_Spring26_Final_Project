#include "core/host_buffers.hpp"

// Allocate storage for x ∈ R^{4N}: each shooting knot k contributes four scalars (θ_k, φ_k, ℓ₁_k, ℓ₂_k)
// at flat indices [4k .. 4k+3]. Segment k’s IVP reads that 4-vector as its initial condition and writes
// `segment_results[k]` = flow map value at the segment’s terminal time.
HDArrays::HDArrays(int num_intervals) {
    int num_states = num_intervals * 4;

    h_node_guesses.resize(num_states, 0.0);
    h_segment_results.resize(num_intervals);

    gpuErrchk(cudaMalloc(&d_node_guesses, num_states * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_segment_results, num_intervals * sizeof(SegmentEvaluation)));
}

// Push current MS iterate x to device before parallel segment IVPs.
void HDArrays::copy_guesses_to_device() {
    size_t bytes = h_node_guesses.size() * sizeof(double);
    gpuErrchk(cudaMemcpy(d_node_guesses, h_node_guesses.data(), bytes, cudaMemcpyHostToDevice));
}

// Pull Φ_k(x_k) (and embedded sensitivities in `VarState::M`) after kernel sync for host assembly of F, J.
void HDArrays::copy_results_to_host() {
    size_t bytes = h_segment_results.size() * sizeof(SegmentEvaluation);
    gpuErrchk(cudaMemcpy(h_segment_results.data(), d_segment_results, bytes, cudaMemcpyDeviceToHost));
}

DeviceArrays HDArrays::get_device_arrays() const {
    return {d_node_guesses, d_segment_results};
}

HDArrays::~HDArrays() {
    gpuErrchk(cudaFree(d_node_guesses));
    gpuErrchk(cudaFree(d_segment_results));
}
