#include "core/host_buffers.hpp"

HDArrays::HDArrays(int num_intervals) {
    int num_states = num_intervals * 4;

    h_node_guesses.resize(num_states, 0.0);
    h_segment_results.resize(num_intervals);

    gpuErrchk(cudaMalloc(&d_node_guesses, num_states * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_segment_results, num_intervals * sizeof(SegmentEvaluation)));
}

void HDArrays::copy_guesses_to_device() {
    size_t bytes = h_node_guesses.size() * sizeof(double);
    gpuErrchk(cudaMemcpy(d_node_guesses, h_node_guesses.data(), bytes, cudaMemcpyHostToDevice));
}

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
