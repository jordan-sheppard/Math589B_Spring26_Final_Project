#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "core/solver_types.cuh"
#include "cuda/gpu_macros.cuh"

struct HDArrays {
    std::vector<double> h_node_guesses;
    std::vector<SegmentEvaluation> h_segment_results;

    double *d_node_guesses;
    SegmentEvaluation *d_segment_results;

    explicit HDArrays(int num_intervals);

    void copy_guesses_to_device();
    void copy_results_to_host();

    DeviceArrays get_device_arrays() const;

    ~HDArrays();
};
