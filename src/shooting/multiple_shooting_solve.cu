#include "shooting/multiple_shooting_solve.hpp"

// Host-side MS driver: each `compute_newton_step` performs (1) parallel GPU IVPs for every segment to
// evaluate the discrete flow map Phi_k and its state-transition sensitivities packed in VarState::M,
// then (2) CPU assembly of F and J, sparse LU for the Newton correction dS, and (3) S <- S + dS.
// Convergence metric: max absolute component of F (algebraically ||F||_infty), compared to
// newton_params.tolerance. Loop cap: newton_params.max_iterations (counts completed Newton steps that
// did not already satisfy tolerance on entry).

#include <vector>

#include "core/host_buffers.hpp"
#include "shooting/newton_iteration.hpp"

/// Repeated Newton solves with shared `HDArrays`; cost is summed over segment endpoint `cost()` fields.
OptimizationResult solve_multiple_shooting(std::vector<double> &flat_node_guesses,
                                           const SystemParams &sys_params,
                                           const IntegratorParams &int_params,
                                           const NewtonParams &newton_params) {
    HDArrays solver_arrays(sys_params.num_shooting_intervals);

    solver_arrays.h_node_guesses = flat_node_guesses;

    int iteration = 0;
    double current_error = 1e9;
    bool converged = false;

    // `iteration` increments only when, after a Newton step, the defect norm still exceeds tolerance
    // (so `num_iterations` in the result is not always equal to the number of `compute_newton_step` calls).
    while (iteration < newton_params.max_iterations) {

        IterationLog log = compute_newton_step(solver_arrays, sys_params, int_params);

        if (!log.success) {
            converged = false;
            current_error = 1e9;
            break;
        }

        current_error = log.max_defect_norm;

        // MS is converged when the shooting residual (defect norm) is below the user tolerance.
        if (current_error < newton_params.tolerance) {
            converged = true;
            break;
        }

        iteration++;
    }

    OptimizationResult final_result;
    final_result.success = converged;
    final_result.num_iterations = iteration;
    final_result.final_error = current_error;

    final_result.r.optimal_l1_init = solver_arrays.h_node_guesses[2];
    final_result.r.optimal_l2_init = solver_arrays.h_node_guesses[3];

    // Running cost was accumulated inside each segment's `VarState::cost` at the terminal IVP state.
    double total_cost = 0.0;
    for (int k = 0; k < sys_params.num_shooting_intervals; k++) {
        total_cost += solver_arrays.h_segment_results[k].final_state.cost();
    }
    final_result.r.optimal_cost = total_cost;

    if (converged) {
        flat_node_guesses = solver_arrays.h_node_guesses;
    }

    return final_result;
}

/// Straight-line interpolation of boundary `(theta,phi)` in `sys_params`; `(l1,l2)=0` at every knot.
// N is the number of shooting segments (and knots); unknown length is 4N. Fraction k/N samples the
// straight path from (theta_init,phi_init) to (theta_goal,phi_goal) in the covering plane.
std::vector<double> compute_linear_initial_guess(const SystemParams &sys_params) {
    int N = sys_params.num_shooting_intervals;
    std::vector<double> guess(N * 4, 0.0);

    for (int k = 0; k < N; k++) {
        double fraction = static_cast<double>(k) / N;

        double theta_k = sys_params.theta_init * (1.0 - fraction) + sys_params.theta_goal * fraction;
        double phi_k = sys_params.phi_init * (1.0 - fraction) + sys_params.phi_goal * fraction;

        guess[k * 4 + 0] = theta_k;
        guess[k * 4 + 1] = phi_k;

        guess[k * 4 + 2] = 0.0;
        guess[k * 4 + 3] = 0.0;
    }

    return guess;
}

OptimizationResult solve_multiple_shooting(SystemParams sys_params, IntegratorParams int_params,
                                           NewtonParams newton_params) {
    int_params.backward_time = false;
    std::vector<double> initial_guess = compute_linear_initial_guess(sys_params);

    return solve_multiple_shooting(initial_guess, sys_params, int_params, newton_params);
}
