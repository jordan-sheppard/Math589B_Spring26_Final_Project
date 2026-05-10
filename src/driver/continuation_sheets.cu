#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

#include "core/solver_types.cuh"
#include "shooting/multiple_shooting_solve.hpp"

Result solve(double target_theta, double target_phi, double alpha) {
    const int NUM_SHOOTING_INTERVALS = 20;

    const double INTEGRATION_DT = 0.025;
    const int NUM_INTEGRATION_STEPS = 10;

    const int MAX_NEWTON_ITERATIONS = 25;
    const double NEWTON_TOL = 1e-6;

    const double MIN_CONTINUATION_STEP_SIZE = 1e-4;
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
    int_params.use_dp5 = false;

    NewtonParams newton_params;
    newton_params.max_iterations = MAX_NEWTON_ITERATIONS;
    newton_params.tolerance = NEWTON_TOL;
    newton_params.lm_mu_initial = 1e-4;
    newton_params.lm_mu_increase = 10.0;
    newton_params.lm_mu_decrease = 0.5;
    newton_params.lm_mu_min = 1e-14;
    newton_params.lm_mu_max = 1e10;
    newton_params.lm_max_subiterations = 15;
    newton_params.max_delta_norm = 5e2;
    newton_params.backtrack_max = 10;

    Result best_result;
    bool found_best = false;
    double best_cost = 1e300;
    int best_wrap = 0;

    const int center_wrap = (int)std::lround(target_theta / TWO_PI);
    const int span =
        std::max(3, std::min(16, 3 + (int)std::ceil(std::abs(target_phi) * 0.45)));
    std::set<int> wrap_candidates;
    wrap_candidates.insert(center_wrap);
    for (int d = 1; d <= span; ++d) {
        wrap_candidates.insert(center_wrap + d);
        wrap_candidates.insert(center_wrap - d);
    }

    for (int wrap : wrap_candidates) {
        sys_params.theta_goal = static_cast<double>(wrap) * TWO_PI;

        double target_norm = std::sqrt(target_theta * target_theta + target_phi * target_phi);
        double current_s = 1.0;
        if (target_norm > 0.05) {
            current_s = 0.05 / target_norm;
        }
        double ds = 0.1;

        SystemParams candidate_params = sys_params;
        candidate_params.theta_init = current_s * target_theta;
        candidate_params.phi_init = current_s * target_phi;

        std::vector<double> active_trajectory = compute_linear_initial_guess(candidate_params);
        OptimizationResult last_success =
            solve_multiple_shooting(active_trajectory, candidate_params, int_params, newton_params);

        if (!last_success.success) {
            continue;
        }

        while (current_s < 1.0) {
            double next_s = std::min(current_s + ds, 1.0);

            candidate_params.theta_init = next_s * target_theta;
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

                if (ds < MIN_CONTINUATION_STEP_SIZE) {
                    break;
                }
            }
        }

        if (last_success.success) {
            IntegratorParams polish_params = int_params;
            polish_params.num_steps = int_params.num_steps + 6;

            std::vector<double> polish_traj = active_trajectory;
            OptimizationResult polish_result =
                solve_multiple_shooting(polish_traj, candidate_params, polish_params, newton_params);
            if (polish_result.success) {
                last_success = polish_result;
                active_trajectory = polish_traj;
            }

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
        return best_result;
    }

    return best_result;
}
