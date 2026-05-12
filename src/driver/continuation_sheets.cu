#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/solver_types.cuh"
#include "shooting/multiple_shooting_solve.hpp"

Result solve(double target_theta, double target_phi, double alpha) {
    const int NUM_SHOOTING_INTERVALS = 20;

    const double INTEGRATION_DT = 0.025;
    const int NUM_INTEGRATION_STEPS = 10;

    const int MAX_NEWTON_ITERATIONS = 15;
    const double NEWTON_TOL = 1e-6;

    const double MIN_CONTINUATION_STEP_SIZE = 1e-4;
    const int MAX_THETA_WRAPS = 1;
    const double TWO_PI = 2.0 * acos(-1.0);

    SystemParams sys_params;
    sys_params.alpha = alpha;
    sys_params.theta_init = target_theta;
    sys_params.phi_init = target_phi;
    sys_params.num_shooting_intervals = NUM_SHOOTING_INTERVALS;
    sys_params.phi_goal = 0.0;

    IntegratorParams int_params;
    int_params.dt = INTEGRATION_DT;
    int_params.num_steps = NUM_INTEGRATION_STEPS;

    NewtonParams newton_params;
    newton_params.max_iterations = MAX_NEWTON_ITERATIONS;
    newton_params.tolerance = NEWTON_TOL;

    Result best_result{};
    bool found_best = false;
    double best_cost = 1e300;
    int best_wrap = 0;

    for (int wrap = -MAX_THETA_WRAPS; wrap <= MAX_THETA_WRAPS; ++wrap) {
        sys_params.theta_goal = wrap * TWO_PI;

        double target_norm = std::sqrt(target_theta * target_theta + target_phi * target_phi);
        double current_s = 1.0;
        if (target_norm > 0.05) {
            current_s = 0.05 / target_norm;
        }
        double ds = 0.1;

        // std::printf("\n=== Searching sheet wrap=%d, theta_goal=%.6f ===\n", wrap, sys_params.theta_goal);
        // std::printf("Starting Multiple Shooting Solver for Theta = %.6f, Phi = %.6f...\n",
        //            sys_params.theta_init, sys_params.phi_init);

        SystemParams candidate_params = sys_params;
        candidate_params.theta_init = current_s * target_theta;
        candidate_params.phi_init = current_s * target_phi;

        std::vector<double> active_trajectory = compute_linear_initial_guess(candidate_params);
        OptimizationResult last_success =
            solve_multiple_shooting(active_trajectory, candidate_params, int_params, newton_params);

        if (!last_success.success) {
            // std::printf("Failed to converge on the initial seed for wrap=%d!\n", wrap);
            continue;
        }

        while (current_s < 1.0) {
            double next_s = std::min(current_s + ds, 1.0);

            candidate_params.theta_init = next_s * target_theta;
            candidate_params.phi_init = next_s * target_phi;

            // std::printf("\n=== Adaptive Step: s = %.4f (Theta = %.4f, Phi = %.4f) ===\n", next_s,
            //            candidate_params.theta_init, candidate_params.phi_init);

            std::vector<double> candidate_trajectory = active_trajectory;

            OptimizationResult result =
                solve_multiple_shooting(candidate_trajectory, candidate_params, int_params, newton_params);

            if (result.success) {
                current_s = next_s;
                active_trajectory = candidate_trajectory;
                last_success = result;

                if (result.num_iterations <= 4) {
                    ds *= 1.5;
                    // std::printf("  -> Fast convergence! Increasing step size to ds = %.5f\n", ds);
                }
            } else {
                ds *= 0.5;
                // std::printf("  -> Step failed! Shrinking step size to ds = %.5f\n", ds);

                if (ds < MIN_CONTINUATION_STEP_SIZE) {
                    // std::printf("CRITICAL FAILURE: Manifold lost. ds too small.\n");
                    break;
                }
            }
        }

        if (last_success.success) {
            // std::printf("\n--- Finished wrap=%d with cost %.10f ---\n", wrap, last_success.r.optimal_cost);
            if (!found_best || last_success.r.optimal_cost < best_cost) {
                found_best = true;
                best_cost = last_success.r.optimal_cost;
                best_wrap = wrap;
                best_result = last_success.r;
                best_result.optimal_theta_wraps = wrap;
                best_result.final_theta_goal = sys_params.theta_goal;
            }
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

    // std::printf("\n>>> BEST SHEET: wrap=%d, theta_goal=%.6f, cost=%.10f <<<\n", best_wrap,
    //            best_result.final_theta_goal, best_result.optimal_cost);

    return best_result;
}
