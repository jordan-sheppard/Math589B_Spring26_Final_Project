#pragma once

#include "core/solver_types.cuh"

/// Evaluate a backward stable-patch grid on GPU.
/// - `basis.B` is a 4x2 basis in [theta,phi,l1,l2] (row-major packed).
/// - `wells_k` is an array of well indices k; theta_eff is theta_target - 2πk.
/// - Output array `out` has size (num_wells * grid_n * grid_n), row-major by [well][i][j].
void stable_patch_grid_backward_gpu(const SystemParams &sys,
                                    const StablePatchBasis &basis,
                                    const int *wells_k,
                                    int num_wells,
                                    const StablePatchGridSettings &gs,
                                    StablePatchCandidate *out);

