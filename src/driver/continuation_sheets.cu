#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "backward_manifold_seed.hpp"
#include "core/solver_debug.hpp"
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
    std::vector<int> wrap_candidates;
    wrap_candidates.reserve(static_cast<size_t>(2 * span + 1));
    wrap_candidates.push_back(center_wrap);
    for (int d = 1; d <= span; ++d) {
        wrap_candidates.push_back(center_wrap + d);
        wrap_candidates.push_back(center_wrap - d);
    }

    const bool dbg_drv = math589_solver_debug_enabled();
    const bool use_ic_homotopy = math589_ic_homotopy_enabled();

    if (dbg_drv) {
        std::fprintf(stderr,
                     "[MATH589][DRIVER] target theta=%.10g phi=%.10g alpha=%.10g "
                     "center_wrap=%d span=%zu candidate_wraps_total=%zu ic_homotopy=%d\n",
                     target_theta, target_phi, alpha, center_wrap, static_cast<size_t>(span),
                     wrap_candidates.size(), use_ic_homotopy ? 1 : 0);
    }

    for (int wrap : wrap_candidates) {
        sys_params.theta_goal = static_cast<double>(wrap) * TWO_PI;

        if (dbg_drv) {
            std::fprintf(stderr, "[MATH589][DRIVER] --- sheet wrap=%d theta_goal=%.10g ---\n", wrap,
                         sys_params.theta_goal);
        }

        OptimizationResult last_success;
        last_success.success = false;
        std::vector<double> active_trajectory;

        if (use_ic_homotopy) {
            double target_norm = std::sqrt(target_theta * target_theta + target_phi * target_phi);
            double current_s = 1.0;
            if (target_norm > 0.05) {
                current_s = 0.05 / target_norm;
            }
            double ds = 0.1;

            SystemParams candidate_params = sys_params;
            candidate_params.theta_init = current_s * target_theta;
            candidate_params.phi_init = current_s * target_phi;

            active_trajectory = compute_linear_initial_guess(candidate_params);
            if (dbg_drv) {
                std::fprintf(stderr,
                             "[MATH589][DRIVER] homotopy initial s=%.6g theta_ic=%.6g phi_ic=%.6g\n",
                             current_s, candidate_params.theta_init, candidate_params.phi_init);
            }

            last_success =
                solve_multiple_shooting(active_trajectory, candidate_params, int_params, newton_params);

            if (!last_success.success) {
                if (dbg_drv) {
                    std::fprintf(stderr,
                                 "[MATH589][DRIVER] initial MS solve FAILED wrap=%d (skip sheet)\n",
                                 wrap);
                }
                continue;
            }

            while (current_s < 1.0) {
                double next_s = std::min(current_s + ds, 1.0);

                candidate_params.theta_init = next_s * target_theta;
                candidate_params.phi_init = next_s * target_phi;

                std::vector<double> candidate_trajectory = active_trajectory;

                if (dbg_drv) {
                    std::fprintf(stderr,
                                 "[MATH589][DRIVER] homotopy try next_s=%.6g theta_ic=%.6g phi_ic=%.6g "
                                 "ds=%.6g\n",
                                 next_s, candidate_params.theta_init, candidate_params.phi_init, ds);
                }

                OptimizationResult result = solve_multiple_shooting(candidate_trajectory, candidate_params,
                                                                    int_params, newton_params);

                if (result.success) {
                    current_s = next_s;
                    active_trajectory = candidate_trajectory;
                    last_success = result;

                    if (result.num_iterations <= 4) {
                        ds *= 1.5;
                    }
                } else {
                    ds *= 0.5;

                    if (dbg_drv) {
                        std::fprintf(stderr,
                                     "[MATH589][DRIVER] homotopy step REJECT shrink ds=%.6g "
                                     "(MS did not converge at this next_s)\n",
                                     ds);
                    }

                    if (ds < MIN_CONTINUATION_STEP_SIZE) {
                        break;
                    }
                }
            }

            if (last_success.success) {
                IntegratorParams polish_params = int_params;
                polish_params.num_steps = int_params.num_steps + 6;

                std::vector<double> polish_traj = active_trajectory;
                if (dbg_drv) {
                    std::fprintf(stderr,
                                 "[MATH589][DRIVER] polish segments steps %d -> %d\n",
                                 int_params.num_steps, polish_params.num_steps);
                }
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
                if (dbg_drv) {
                    std::fprintf(stderr,
                                 "[MATH589][DRIVER] sheet wrap=%d finished ok best_cost_so_far=%.10g "
                                 "best_wrap_track=%d\n",
                                 wrap, best_cost, best_wrap);
                }
            }
        } else {
            std::vector<double> cloud_guess;
            const bool cloud_ok =
                build_ms_guess_from_backward_cloud(sys_params, int_params, cloud_guess);
            if (cloud_ok) {
                active_trajectory = std::move(cloud_guess);
            } else {
                if (dbg_drv) {
                    std::fprintf(stderr,
                                 "[MATH589][DRIVER] backward cloud failed; linear guess fallback "
                                 "wrap=%d\n",
                                 wrap);
                }
                active_trajectory = compute_linear_initial_guess(sys_params);
            }

            last_success =
                solve_multiple_shooting(active_trajectory, sys_params, int_params, newton_params);

            if (!last_success.success) {
                if (dbg_drv) {
                    std::fprintf(stderr,
                                 "[MATH589][DRIVER] MS solve FAILED wrap=%d (skip sheet)\n", wrap);
                }
                continue;
            }

            IntegratorParams polish_params = int_params;
            polish_params.num_steps = int_params.num_steps + 6;

            std::vector<double> polish_traj = active_trajectory;
            if (dbg_drv) {
                std::fprintf(stderr,
                             "[MATH589][DRIVER] polish segments steps %d -> %d\n", int_params.num_steps,
                             polish_params.num_steps);
            }
            OptimizationResult polish_result =
                solve_multiple_shooting(polish_traj, sys_params, polish_params, newton_params);
            if (polish_result.success) {
                last_success = polish_result;
                active_trajectory = std::move(polish_traj);
            }

            if (!found_best || last_success.r.optimal_cost < best_cost) {
                found_best = true;
                best_cost = last_success.r.optimal_cost;
                best_wrap = wrap;
                best_result = last_success.r;
                best_result.optimal_theta_wraps = wrap;
                best_result.final_theta_goal = sys_params.theta_goal;
            }
            if (dbg_drv) {
                std::fprintf(stderr,
                             "[MATH589][DRIVER] sheet wrap=%d finished ok best_cost_so_far=%.10g "
                             "best_wrap_track=%d\n",
                             wrap, best_cost, best_wrap);
            }
        }
    }

    if (!found_best) {
        if (dbg_drv) {
            std::fprintf(stderr,
                         "[MATH589][DRIVER] solve() returning DEFAULT (found_best=false) -> zeros\n");
        }
        return best_result;
    }

    if (dbg_drv) {
        std::fprintf(stderr,
                     "[MATH589][DRIVER] BEST wrap=%d final_theta_goal=%.10g l1=%.10f l2=%.10f cost=%.10f\n",
                     best_wrap, best_result.final_theta_goal, best_result.optimal_l1_init,
                     best_result.optimal_l2_init, best_result.optimal_cost);
    }

    return best_result;
}
