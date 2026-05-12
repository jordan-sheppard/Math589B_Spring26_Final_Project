#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "core/solver_types.cuh"

namespace warm_start {

// Integer lifts k with theta_work = theta − 2πk: small de-duped set around lround(theta / 2π), used for
// S^1 sheet bookkeeping in continuation and backward-patch terminal targets (same probe order everywhere).
inline void theta_well_shift_candidates(double theta, std::vector<int> &out) {
    const double two_pi = 2.0 * std::acos(-1.0);
    const int k_round = static_cast<int>(std::lround(theta / two_pi));
    const int arr[] = {k_round, 0, k_round - 1, k_round + 1, k_round - 2, k_round + 2};
    out.clear();
    for (int k : arr) {
        if (std::find(out.begin(), out.end(), k) == out.end()) {
            out.push_back(k);
        }
    }
}

} // namespace warm_start

/// GPU-batched backward IVP warm starts for multiple shooting (see `backward_ivp_batch.cu`).
///
/// **Stable subspace.**  Linearizes the closed-loop Hamiltonian field at the upright equilibrium,
/// takes the two-dimensional stable invariant subspace \(E^s\) (eigenvalues with \(\Re\lambda<0\)),
/// and uses an orthonormal basis of \(E^s\) as coefficients \((a,b)\) for a small patch of states.
///
/// **Backward IVP.**  Each candidate integrates the same physics-only vector field as forward
/// segments but with negative step size, i.e. a backward Cauchy problem from the patch; terminal
/// \((\theta,\phi)\) are compared to the initial data after fixing the \(S^1\) ambiguity in \(\theta\).
///
/// **Wells / wraps / metric.**  Several integer windings ("wells") define targets
/// \(\theta_{\mathrm{tgt}}=\theta_{\mathrm{init}}-2\pi k\); scoring uses squared geodesic distance on
/// \(S^1\times\mathbb{R}\) in \((\theta,\phi)\) (see `dist2_wrapped`).
///
/// **Grid / radii / top-K.**  Tensor-product grids in \((a,b)\) at multiple radii discretize nested
/// squares in coefficient space; the GPU scores all combinations, then `partial_sort` extracts the
/// `top_k` smallest terminal mismatches for host replay into MS nodal warm starts.
///
/// Returns up to `top_k` trajectories, each a flat array of length `4 * num_shooting_intervals`
/// with nodal `(theta, phi, lambda1, lambda2)` sampled backward along the horizon (see
/// `origin_patch_backward_to_targets` in `backward_ivp_common.cuh`).
///
/// On Eigen failure or invalid dimensions, returns an empty vector.  Does not run Newton; the
/// driver passes these vectors into `solve_multiple_shooting` as initial guesses.
std::vector<std::vector<double>> compute_patch_topk_ms_warm_starts(const SystemParams &sys,
                                                                   const IntegratorParams &int_params,
                                                                   int top_k = 12);
