#include "shooting/newton_iteration.hpp"

#include <cstdio>

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
    build_global_system(solver_arrays, sys_params, J, F);

    Eigen::SparseLU<SparseMat> solver;

    solver.compute(J);
    if (solver.info() != Eigen::Success) {
        // printf("Eigen SparseLU failed to factorize the Jacobian!\n");
        log.success = false;
        return log;
    }

    VectorXd dS = solver.solve(-F);

    for (int i = 0; i < dS.size(); i++) {
        solver_arrays.h_node_guesses[i] += dS(i);
    }

    log.max_defect_norm = F.lpNorm<Eigen::Infinity>();
    log.step_size_norm = dS.norm();

    return log;
}
