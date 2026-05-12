#pragma once

#include <vector>

#include "core/solver_types.cuh"

/// CPU-side top-K filtering by d2 per well.
/// Input `cands` is size (num_wells * grid_n * grid_n).
std::vector<StablePatchCandidate> stable_patch_topk_per_well(const StablePatchCandidate *cands,
                                                            int num_wells,
                                                            int grid_n,
                                                            int top_k);

/// CPU 2D Newton refinement on (a,b) for one candidate seed and one well.
/// Returns best refined (may be non-converged).
StablePatchRefineOut refine_candidate_newton_2d(const SystemParams &sys,
                                                const StablePatchBasis &basis,
                                                int well_k,
                                                double a0,
                                                double b0,
                                                const StablePatchNewtonSettings &ns,
                                                const StablePatchGridSettings &gs);

