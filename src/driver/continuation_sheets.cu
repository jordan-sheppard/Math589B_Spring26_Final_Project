#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "core/manifold_seed.hpp"
#include "core/solver_debug.hpp"
#include "core/solver_types.cuh"
#include "shooting/patch_refine_newton.hpp"
#include "shooting/stable_patch_grid.hpp"

namespace {
constexpr double two_pi() { return 6.283185307179586476925286766559; }
}  // namespace

Result solve(double target_theta, double target_phi, double alpha) {
    // Settings (tuned later; start deterministic).
    StablePatchGridSettings gs;
    gs.wells_half_span = 2;
    gs.grid_n = 33;
    gs.grid_radius = 1e-2;
    gs.back_steps = 2000;
    gs.back_dt = 1e-3;
    gs.top_k_per_well = 16;

    StablePatchNewtonSettings ns;
    ns.max_iters = 20;
    ns.tol = 1e-8;
    ns.fd_eps = 1e-5;
    ns.backtrack_max = 12;
    ns.step_clip = 2.0;

    SystemParams sys;
    sys.alpha = alpha;
    sys.theta_init = target_theta;
    sys.phi_init = target_phi;
    sys.theta_goal = 0.0;  // not used by this pipeline (we use wells_k explicitly)
    sys.phi_goal = 0.0;
    sys.num_shooting_intervals = 0;

    const bool dbg = math589_solver_debug_enabled();

    // Wells (angle periodicity)
    const int k_round = static_cast<int>(std::lround(target_theta / two_pi()));
    std::vector<int> wells;
    for (int d = -gs.wells_half_span; d <= gs.wells_half_span; ++d) {
        wells.push_back(k_round + d);
    }

    if (dbg) {
        std::fprintf(stderr, "[MATH589][PATCH] theta=%.10g phi=%.10g alpha=%.10g k_round=%d wells=%zu grid_n=%d back_steps=%d\n",
                     target_theta, target_phi, alpha, k_round, wells.size(), gs.grid_n, gs.back_steps);
    }

    // Stable patch basis
    StablePatchBasis basis;
    stable_manifold_basis(alpha, basis.B);

    // GPU grid evaluate
    const int num_wells = static_cast<int>(wells.size());
    const int total = num_wells * gs.grid_n * gs.grid_n;
    std::vector<StablePatchCandidate> cands(static_cast<size_t>(std::max(0, total)));
    stable_patch_grid_backward_gpu(sys, basis, wells.data(), num_wells, gs, cands.data());

    // Top-K per well
    const std::vector<StablePatchCandidate> top =
        stable_patch_topk_per_well(cands.data(), num_wells, gs.grid_n, gs.top_k_per_well);

    if (dbg) {
        int valid = 0;
        for (const auto &c : cands) valid += (c.valid != 0);
        std::fprintf(stderr, "[MATH589][PATCH] candidates total=%d valid=%d top=%zu\n", total, valid, top.size());
    }

    // Refine and select by cost among converged; fallback to best-by-residual.
    bool found_conv = false;
    StablePatchRefineOut best_conv;
    best_conv.J = std::numeric_limits<double>::infinity();

    bool found_any = false;
    StablePatchRefineOut best_any;
    best_any.r_inf = 1e300;

    for (const auto &seed : top) {
        StablePatchRefineOut r = refine_candidate_newton_2d(sys, basis, seed.well_k, seed.a, seed.b, ns, gs);
        found_any = true;
        if (r.r_inf < best_any.r_inf) {
            best_any = r;
        }
        if (r.converged) {
            if (!found_conv || r.J < best_conv.J) {
                found_conv = true;
                best_conv = r;
            }
        }
    }

    Result out{};
    if (found_conv) {
        out.optimal_l1_init = best_conv.l1;
        out.optimal_l2_init = best_conv.l2;
        out.optimal_cost = best_conv.J;
        out.optimal_theta_wraps = k_round; // informational only
        out.final_theta_goal = two_pi() * static_cast<double>(k_round);
        if (dbg) {
            std::fprintf(stderr, "[MATH589][PATCH] SELECT converged r_inf=%.3e a=%.6g b=%.6g l=(%.10g,%.10g) J=%.10g\n",
                         best_conv.r_inf, best_conv.a, best_conv.b, best_conv.l1, best_conv.l2, best_conv.J);
        }
        return out;
    }

    if (found_any) {
        out.optimal_l1_init = best_any.l1;
        out.optimal_l2_init = best_any.l2;
        out.optimal_cost = std::isfinite(best_any.J) ? best_any.J : 0.0;
        out.optimal_theta_wraps = k_round;
        out.final_theta_goal = two_pi() * static_cast<double>(k_round);
        if (dbg) {
            std::fprintf(stderr, "[MATH589][PATCH] SELECT best_nonconverged r_inf=%.3e a=%.6g b=%.6g l=(%.10g,%.10g) J=%.10g\n",
                         best_any.r_inf, best_any.a, best_any.b, best_any.l1, best_any.l2, best_any.J);
        }
        return out;
    }

    // Last resort fallback: lambda ≈ P x, cost=0 (deterministic, non-zero in general).
    double P[4];
    stable_manifold_P(alpha, P);
    out.optimal_l1_init = P[0] * target_theta + P[1] * target_phi;
    out.optimal_l2_init = P[2] * target_theta + P[3] * target_phi;
    out.optimal_cost = 0.0;
    out.optimal_theta_wraps = k_round;
    out.final_theta_goal = two_pi() * static_cast<double>(k_round);
    if (dbg) {
        std::fprintf(stderr, "[MATH589][PATCH] FALLBACK linear P seed l=(%.10g,%.10g)\n",
                     out.optimal_l1_init, out.optimal_l2_init);
    }
    return out;
}
