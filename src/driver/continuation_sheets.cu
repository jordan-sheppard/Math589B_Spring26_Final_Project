#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/solver_types.cuh"
#include "shooting/multiple_shooting_solve.hpp"
#include "warm_start/backward_ivp_warmstart.hpp"

// `solve()` driver for a periodic-coordinate boundary-value problem (BVP) solved by multiple shooting.
//
// Sheet / wrap bookkeeping (S^1 periodicity in theta):
//   The configuration angle theta lives on a circle; numerically we work in R and relate physical angles
//   by theta_phys ~ theta + 2πk.  Candidates k in `k_candidates` pick representatives theta_work of the
//   *same* target direction after subtracting 2πk (a lift of the target onto the universal cover).  The
//   inner loop over `wrap` sets theta_goal = 2π·wrap, i.e. which equivalence class of the *terminal*
//   angle constraint is enforced—different sheets can correspond to the same physical endpoint modulo 2π
//   but different winding / branch of the periodic boundary map F(x)=0.
//
// Homotopy / continuation in boundary data:
//   We embed the true boundary pair (theta_init, phi_init) in a one-parameter family
//     (theta_init(s), phi_init(s)) = s · (theta_work, phi_target),  s ∈ (0,1],
//   starting from a small-norm IVP (s small) where the shooting map is better conditioned / has a larger
//   basin, then increase s toward 1 while reusing the converged trajectory as the Newton initial guess.
//   `ds` is the predictor step in s; failed Newton steps trigger a smaller `ds` (pseudo-arclength-style
//   backtracking in the homotopy parameter).  `MIN_CONTINUATION_STEP_SIZE` is the smallest allowed s-step
//   before abandoning that (sheet, wrap) branch.
//
// Warm starts:
//   `compute_patch_topk_ms_warm_starts` supplies several discrete shooting-node vectors (candidates in the
//   MS state space).  Each is first refined by *backward-time* multiple shooting (IVP integrated from the
//   goal backward), which often lands closer to a consistent defect surface than a forward-only guess;
//   homotopy continuation then uses *forward-time* MS on the same discretization.
//
// Arguments (boundary / model data for the underlying ODE BVP):
//   target_theta, target_phi — prescribed values at the “initial” end of the horizon in the lifted
//     coordinates used by the shooter (paired with fixed terminal data in `SystemParams`).
//   alpha — parameter entering the vector field (family of dynamics); held fixed during a `solve` call.

namespace {

double two_pi() { return 2.0 * std::acos(-1.0); }

void run_backward_warm_start_refinement(
    const std::vector<std::vector<double>> &warm_list,
    const SystemParams &candidate_params,
    const IntegratorParams &int_bwd,
    const IntegratorParams &int_params,
    const NewtonParams &newton_params,
    OptimizationResult &last_success,
    std::vector<double> &active_trajectory) {
    for (const std::vector<double> &seed_traj : warm_list) {
        std::vector<double> traj = seed_traj;
        OptimizationResult res =
            solve_multiple_shooting(traj, candidate_params, int_bwd, newton_params);
        if (res.success) {
            if (!last_success.success || res.r.optimal_cost < last_success.r.optimal_cost) {
                last_success = res;
                active_trajectory = traj;
            }
        }
    }

    if (!last_success.success) {
        active_trajectory = compute_linear_initial_guess(candidate_params);
        last_success =
            solve_multiple_shooting(active_trajectory, candidate_params, int_params, newton_params);
    }
}

void homotopy_continue_to_one(double theta_work,
                               double target_phi,
                               SystemParams &candidate_params,
                               double &current_s,
                               double &ds,
                               double min_continuation_step_size,
                               const IntegratorParams &int_params,
                               const NewtonParams &newton_params,
                               OptimizationResult &last_success,
                               std::vector<double> &active_trajectory) {
    while (current_s < 1.0) {
        double next_s = std::min(current_s + ds, 1.0);

        candidate_params.theta_init = next_s * theta_work;
        candidate_params.phi_init = next_s * target_phi;

        std::vector<double> candidate_trajectory = active_trajectory;

        OptimizationResult result =
            solve_multiple_shooting(candidate_trajectory, candidate_params, int_params, newton_params);

        if (result.success) {
            current_s = next_s;
            active_trajectory = candidate_trajectory;
            last_success = result;

            if (result.num_iterations <= 4) {
                ds *= 1.5;
            }
        } else {
            ds *= 0.5;

            if (ds < min_continuation_step_size) {
                break;
            }
        }
    }
}

void update_best_if_improved(const OptimizationResult &last_success,
                             int wrap,
                             double theta_goal,
                             bool &found_best,
                             double &best_cost,
                             Result &best_result) {
    if (!last_success.success) {
        return;
    }
    if (!found_best || last_success.r.optimal_cost < best_cost) {
        found_best = true;
        best_cost = last_success.r.optimal_cost;
        best_result = last_success.r;
        best_result.optimal_theta_wraps = wrap;
        best_result.final_theta_goal = theta_goal;
    }
}

} // namespace

Result solve(double target_theta, double target_phi, double alpha) {
    // Trivial equilibrium: zero boundary data gives the zero trajectory without invoking the shooter.
    if (std::fabs(target_theta) < 1.0e-14 && std::fabs(target_phi) < 1.0e-14) {
        return Result{};
    }

    // Multiple shooting: partition [0,T] into NUM_SHOOTING_INTERVALS segments; unknowns are values at
    // interior nodes so that segment IVPs match (defects → nonlinear equations solved by Newton below).
    const int NUM_SHOOTING_INTERVALS = 20;
    const double TOTAL_HORIZON = 16.0;
    const int NUM_INTEGRATION_STEPS = 128;
    // Fixed substep dt on each segment: refines the IVP flow map used inside each shooting interval.
    const double INTEGRATION_DT =
        TOTAL_HORIZON /
        (static_cast<double>(NUM_SHOOTING_INTERVALS) * static_cast<double>(NUM_INTEGRATION_STEPS));

    // Newton on the stacked defect + boundary residual map; tolerance is in the MS objective / residual norm.
    const int MAX_NEWTON_ITERATIONS = 25;
    const double NEWTON_TOL = 1e-9;

    // Homotopy step control in the boundary-scaling parameter s (see file header).
    const double MIN_CONTINUATION_STEP_SIZE = 1e-4;
    // |wrap| ≤ MAX_THETA_WRAPS: finite search over adjacent terminal-angle sheets theta_goal = 2π·wrap.
    const int MAX_THETA_WRAPS = 1;
    const double TWO_PI = two_pi();

    // Lifts of the target angle: integers k so theta_work = target_theta − 2πk lies near a convenient
    // principal range; different k can change which local minimizer / basin the homotopy enters.
    std::vector<int> k_candidates;
    warm_start::theta_well_shift_candidates(target_theta, k_candidates);

    // Boundary / ODE parameters for the shooting residual: alpha enters the vector field; phi_* are
    // velocity-like boundary components paired with theta_* on the periodic cylinder.
    SystemParams sys_params;
    sys_params.alpha = alpha;
    sys_params.phi_init = target_phi;
    sys_params.num_shooting_intervals = NUM_SHOOTING_INTERVALS;
    sys_params.phi_goal = 0.0;

    // Time-stepping of the segment flows (forward vs backward toggled later for warm-start geometry).
    IntegratorParams int_params;
    int_params.dt = INTEGRATION_DT;
    int_params.num_steps = NUM_INTEGRATION_STEPS;
    int_params.backward_time = false;

    // Stopping criteria for the Gauss–Newton / Newton iteration on shooting defects + boundaries.
    NewtonParams newton_params;
    newton_params.max_iterations = MAX_NEWTON_ITERATIONS;
    newton_params.tolerance = NEWTON_TOL;

    // Among all (k, wrap, homotopy branch) sheets, keep the MS solution with smallest optimal_cost
    // (shooting objective at convergence).
    Result best_result{};
    bool found_best = false;
    double best_cost = 1e300;

    // Outer loop: target-angle lift k (unwrapping on R before enforcing 2π periodicity in the model).
    for (int k : k_candidates) {
        const double theta_work = target_theta - TWO_PI * static_cast<double>(k);
        sys_params.theta_init = theta_work;

        // Inner loop: terminal theta sheet index wrap; shifts the goal angle in R by full turns while
        // leaving the dynamics unchanged—selects which preimage of the circle-valued terminal constraint
        // the BVP is written on.
        for (int wrap = -MAX_THETA_WRAPS; wrap <= MAX_THETA_WRAPS; ++wrap) {
            sys_params.theta_goal = wrap * TWO_PI;

            // Homotopy initial s0: start from a small boundary vector so the induced IVP is mild; the
            // threshold 0.05 fixes a radius in (theta, phi)-space for the first scaled problem F(x;s0)=0.
            double target_norm = std::sqrt(theta_work * theta_work + target_phi * target_phi);
            double current_s = 1.0;
            if (target_norm > 0.05) {
                current_s = 0.05 / target_norm;
            }
            // Predictor increment in s; adapted heuristically from Newton iteration counts (see below).
            double ds = 0.1;

            SystemParams candidate_params = sys_params;
            candidate_params.theta_init = current_s * theta_work;
            candidate_params.phi_init = current_s * target_phi;

            // kWarmTop: number of diverse MS node vectors proposed as nonlinear optimization seeds (local
            // patches in state space discretized on the shooting grid).
            constexpr int kWarmTop = 12;
            std::vector<std::vector<double>> warm_list =
                compute_patch_topk_ms_warm_starts(candidate_params, int_params, kWarmTop);

            IntegratorParams int_bwd = int_params;
            // Backward-time flow: integrate from the terminal constraint toward t=0 so each seed solves a
            // nearby IVP that is often easier to reconcile with the initial boundary in MS defects.
            int_bwd.backward_time = true;

            OptimizationResult last_success{};
            last_success.success = false;
            std::vector<double> active_trajectory = compute_linear_initial_guess(candidate_params);

            run_backward_warm_start_refinement(warm_list, candidate_params, int_bwd, int_params, newton_params,
                                               last_success, active_trajectory);

            if (!last_success.success) {
                continue;
            }

            homotopy_continue_to_one(theta_work, target_phi, candidate_params, current_s, ds,
                                     MIN_CONTINUATION_STEP_SIZE, int_params, newton_params, last_success,
                                     active_trajectory);

            update_best_if_improved(last_success, wrap, sys_params.theta_goal, found_best, best_cost,
                                   best_result);
        }
    }

    if (!found_best) {
        std::fprintf(stderr,
                     "solver: no continuation sheet converged (stdout may show 0 0 0). "
                     "Common on clusters: rebuild with Makefile CUDA_GENCODE matching the node GPU "
                     "(Ocelote P100 sm_60, Puma V100 sm_70), run on a GPU allocation, and use a CUDA "
                     "module that exists on that cluster (Ocelote: make CUDA_MODULE=cuda11/11.8 "
                     "MAX_HOST_GCC_MAJOR=11).\n");
        return best_result;
    }

    return best_result;
}
