#include "shooting/multiple_shooting_solve.hpp"

#include <cstdio>
#include <vector>

#include "core/host_buffers.hpp"
#include "core/solver_debug.hpp"
#include "core/manifold_seed.hpp"
#include "shooting/newton_iteration.hpp"

OptimizationResult solve_multiple_shooting(std::vector<double> &flat_node_guesses,
                                           const SystemParams &sys_params,
                                           const IntegratorParams &int_params,
                                           const NewtonParams &newton_params) {
    HDArrays solver_arrays(sys_params.num_shooting_intervals);

    solver_arrays.h_node_guesses = flat_node_guesses;

    int iteration = 0;
    double current_error = -1.0;
    bool converged = false;

    double lm_mu = newton_params.lm_mu_initial;

    const bool dbg = math589_solver_debug_enabled();
    if (dbg) {
        std::fprintf(stderr,
                   "[MATH589][MS] start N_seg=%d alpha=%.6g dt=%.6g steps=%d tol=%.3e "
                   "theta_init=%.6g phi_init=%.6g theta_goal=%.6g phi_goal=%.6g lm_mu=%.3e "
                   "max_newton=%d\n",
                   sys_params.num_shooting_intervals, sys_params.alpha, int_params.dt,
                   int_params.num_steps, newton_params.tolerance, sys_params.theta_init,
                   sys_params.phi_init, sys_params.theta_goal, sys_params.phi_goal, lm_mu,
                   newton_params.max_iterations);
    }

    while (iteration < newton_params.max_iterations) {

        IterationLog log =
            compute_newton_step(solver_arrays, sys_params, int_params, newton_params, lm_mu);

        if (!log.success) {
            // One failed LM Armijo/subproblem is common; tighten damping and keep iterating rather
            // than aborting the whole multiple-shooting solve immediately.
            lm_mu = std::min(newton_params.lm_mu_max,
                             std::max(newton_params.lm_mu_min, lm_mu * newton_params.lm_mu_increase));
            if (dbg) {
                std::fprintf(stderr,
                             "[MATH589][MS] iter=%d LM_SUBPROBLEM_REJECTED bumped_lm_mu=%.6e "
                             "(max|F| unchanged this outer step)\n",
                             iteration, lm_mu);
            }
            iteration++;
            if (iteration >= newton_params.max_iterations) {
                converged = false;
                break;
            }
            continue;
        }

        current_error = log.max_defect_norm;

        if (dbg) {
            std::fprintf(stderr,
                         "[MATH589][MS] iter=%d ok=1 step_norm=%.6e max|F|=%.6e lm_mu(now)=%.6e\n",
                         iteration, log.step_size_norm, current_error, lm_mu);
        }

        if (current_error < newton_params.tolerance) {
            converged = true;
            if (dbg) {
                std::fprintf(stderr, "[MATH589][MS] CONVERGED iter=%d max|F|=%.6e < tol\n", iteration,
                             current_error);
            }
            break;
        }

        iteration++;
    }

    if (dbg && !converged) {
        std::fprintf(stderr, "[MATH589][MS] END_NOT_CONVERGED iters_used=%d last_max|F|=%.6e\n",
                     iteration, current_error >= 0.0 ? current_error : -1.0);
    }

    OptimizationResult final_result;
    final_result.success = converged;
    final_result.num_iterations = iteration;
    final_result.final_error = (current_error >= 0.0) ? current_error : 1e9;

    final_result.r.optimal_l1_init = solver_arrays.h_node_guesses[2];
    final_result.r.optimal_l2_init = solver_arrays.h_node_guesses[3];

    double total_cost = 0.0;
    double cost_comp = 0.0;
    for (int k = 0; k < sys_params.num_shooting_intervals; k++) {
        double y = solver_arrays.h_segment_results[k].final_state.cost() - cost_comp;
        double t = total_cost + y;
        cost_comp = (t - total_cost) - y;
        total_cost = t;
    }
    final_result.r.optimal_cost = total_cost;

    if (converged) {
        flat_node_guesses = solver_arrays.h_node_guesses;
    }

    if (dbg) {
        std::fprintf(stderr,
                     "[MATH589][MS] summary converged=%d l1(0)=%.10f l2(0)=%.10f cost=%.10f "
                     "iters=%d final_err=%.6e\n",
                     converged ? 1 : 0, final_result.r.optimal_l1_init,
                     final_result.r.optimal_l2_init, final_result.r.optimal_cost, iteration,
                     final_result.final_error);
    }

    return final_result;
}

std::vector<double> compute_linear_initial_guess(const SystemParams &sys_params) {
    int N = sys_params.num_shooting_intervals;
    std::vector<double> guess(N * 4, 0.0);

    double P[4];
    stable_manifold_P(sys_params.alpha, P);
    const double P11 = P[0];
    const double P12 = P[1];
    const double P21 = P[2];
    const double P22 = P[3];

    for (int k = 0; k < N; k++) {
        const double denom = static_cast<double>(N > 0 ? N : 1);
        double fraction = static_cast<double>(k) / denom;

        double theta_k = sys_params.theta_init * (1.0 - fraction) + sys_params.theta_goal * fraction;
        double phi_k = sys_params.phi_init * (1.0 - fraction) + sys_params.phi_goal * fraction;

        guess[k * 4 + 0] = theta_k;
        guess[k * 4 + 1] = phi_k;

        guess[k * 4 + 2] = P11 * theta_k + P12 * phi_k;
        guess[k * 4 + 3] = P21 * theta_k + P22 * phi_k;
    }

    return guess;
}

OptimizationResult solve_multiple_shooting(SystemParams sys_params, IntegratorParams int_params,
                                           NewtonParams newton_params) {
    std::vector<double> initial_guess = compute_linear_initial_guess(sys_params);

    return solve_multiple_shooting(initial_guess, sys_params, int_params, newton_params);
}
