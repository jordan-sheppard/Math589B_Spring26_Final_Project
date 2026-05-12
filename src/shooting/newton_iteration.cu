#include "shooting/newton_iteration.hpp"

// Computational split: `evaluate_segments_on_gpu` is embarrassingly parallel over segments (device
// IVP + discrete sensitivity propagation inside `simulate_segment`); `build_global_system` couples
// segments through continuity and boundary conditions on the host (Eigen triplets). The linear algebra
// step uses the same ordering of unknowns as `h_node_guesses` (stacked knots).

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "core/solver_host_types.hpp"
#include "shooting/defect_jacobian_host.hpp"
#include "shooting/gpu_eval_segments.hpp"

IterationLog compute_newton_step(HDArrays &solver_arrays, const SystemParams &sys_params,
                                 const IntegratorParams &int_params) {
    IterationLog log;

    evaluate_segments_on_gpu(solver_arrays, sys_params, int_params);

    SparseMat J;
    VectorXd F;
    build_global_system(solver_arrays, sys_params, int_params, J, F);

    Eigen::SparseLU<SparseMat> solver;

    solver.compute(J);
    if (solver.info() != Eigen::Success) {
        log.success = false;
        return log;
    }

    // Full Newton: dS solves the linearized shooting equations J dS = -F(S), S in R^{4N}.
    VectorXd dS = solver.solve(-F);

    // Additive update on the MS unknown vector (same component order as F and columns of J).
    for (int i = 0; i < dS.size(); i++) {
        solver_arrays.h_node_guesses[i] += dS(i);
    }

    // Residual norm ||F||_infty (max shooting defect) and Euclidean step length ||dS||_2 for logging.
    log.max_defect_norm = F.lpNorm<Eigen::Infinity>();
    log.step_size_norm = dS.norm();

    return log;
}
