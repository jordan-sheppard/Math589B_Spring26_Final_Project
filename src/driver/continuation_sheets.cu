#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/solver_types.cuh"
#include "shooting/multiple_shooting_solve.hpp"
#include "warm_start/backward_ivp_warmstart.hpp"

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
    int_params.backward_time = false;

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

        constexpr int kWarmTop = 12;
        std::vector<std::vector<double>> warm_list =
            compute_patch_topk_ms_warm_starts(candidate_params, int_params, kWarmTop);

        IntegratorParams int_bwd = int_params;
        int_bwd.backward_time = true;

        OptimizationResult last_success{};
        last_success.success = false;
        std::vector<double> active_trajectory = compute_linear_initial_guess(candidate_params);

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
        // #region agent log
        {
            long long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
            std::FILE *df = std::fopen("debug-a00cc2.log", "a");
            if (df) {
                std::fprintf(df,
                               "{\"sessionId\":\"a00cc2\",\"timestamp\":%lld,\"location\":"
                               "\"continuation_sheets.cu:no_best\",\"message\":\"solve_exit\",\"hypothesisId\":"
                               "\"H4\",\"data\":{\"theta\":%.17g,\"phi\":%.17g,\"alpha\":%.17g,\"found_best\":0}}\n",
                               ts, target_theta, target_phi, alpha);
                std::fclose(df);
            }
        }
        // #endregion
        return best_result;
    }

    // std::printf("\n>>> BEST SHEET: wrap=%d, theta_goal=%.6f, cost=%.10f <<<\n", best_wrap,
    //            best_result.final_theta_goal, best_result.optimal_cost);

    // #region agent log
    {
        long long ts =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch())
                .count();
        std::FILE *df = std::fopen("debug-a00cc2.log", "a");
        if (df) {
            std::fprintf(df,
                           "{\"sessionId\":\"a00cc2\",\"timestamp\":%lld,\"location\":"
                           "\"continuation_sheets.cu:ok\",\"message\":\"solve_exit\",\"hypothesisId\":\"H4\","
                           "\"data\":{\"theta\":%.17g,\"phi\":%.17g,\"alpha\":%.17g,\"found_best\":1,"
                           "\"best_wrap\":%d,\"theta_goal\":%.17g,\"l1\":%.17g,\"l2\":%.17g,\"cost\":%.17g}}\n",
                           ts, target_theta, target_phi, alpha, best_wrap, best_result.final_theta_goal,
                           best_result.optimal_l1_init, best_result.optimal_l2_init, best_result.optimal_cost);
            std::fclose(df);
        }
    }
    // #endregion

    return best_result;
}
