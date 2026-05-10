#include "shooting/newton_iteration.hpp"

#include <algorithm>
#include <cmath>

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>
#include <Eigen/Dense>

#include "core/solver_host_types.hpp"
#include "shooting/defect_jacobian_host.hpp"
#include "shooting/gpu_eval_segments.hpp"

IterationLog compute_newton_step(HDArrays &solver_arrays, const SystemParams &sys_params,
                                 const IntegratorParams &int_params, const NewtonParams &newton_params,
                                 double &lm_mu) {
    IterationLog log;
    log.success = false;
    log.max_defect_norm = 0.0;
    log.step_size_norm = 0.0;

    std::vector<double> z_backup = solver_arrays.h_node_guesses;

    auto linf = [](const VectorXd &v) { return v.lpNorm<Eigen::Infinity>(); };

    evaluate_segments_on_gpu(solver_arrays, sys_params, int_params);

    SparseMat J;
    VectorXd F;
    build_global_system(solver_arrays, sys_params, J, F);

    const double r_norm_start = linf(F);
    log.max_defect_norm = r_norm_start;

    SparseMat Jt = J.transpose();
    SparseMat JtJ = Jt * J;
    VectorXd JtF = Jt * F;

    double mu = lm_mu;
    if (mu < newton_params.lm_mu_min) {
        mu = newton_params.lm_mu_min;
    }

    const int n = static_cast<int>(z_backup.size());

    for (int sub = 0; sub < newton_params.lm_max_subiterations; ++sub) {
        SparseMat A = JtJ;
        for (int i = 0; i < n; ++i) {
            A.coeffRef(i, i) += mu;
        }

        Eigen::SimplicialLDLT<SparseMat> ldlt;
        ldlt.compute(A);
        if (ldlt.info() != Eigen::Success) {
            mu = std::min(newton_params.lm_mu_max, mu * newton_params.lm_mu_increase);
            continue;
        }

        VectorXd delta = ldlt.solve(-JtF);
        if (ldlt.info() != Eigen::Success) {
            mu = std::min(newton_params.lm_mu_max, mu * newton_params.lm_mu_increase);
            continue;
        }

        const double dn = delta.norm();
        if (dn > newton_params.max_delta_norm && dn > 0.0) {
            delta *= newton_params.max_delta_norm / dn;
        }

        bool improved = false;
        double eta_best = 1.0;
        double best_residual = r_norm_start;
        VectorXd trial_best = z_backup;

        double eta = 1.0;
        for (int bt = 0; bt <= newton_params.backtrack_max; ++bt) {
            for (int i = 0; i < n; ++i) {
                solver_arrays.h_node_guesses[i] = z_backup[i] + eta * delta(i);
            }

            evaluate_segments_on_gpu(solver_arrays, sys_params, int_params);

            SparseMat J_try;
            VectorXd F_try;
            build_global_system(solver_arrays, sys_params, J_try, F_try);

            const double r_try = linf(F_try);
            if (r_try < best_residual) {
                best_residual = r_try;
                trial_best = solver_arrays.h_node_guesses;
                eta_best = eta;
                improved = true;
                if (r_try < r_norm_start) {
                    break;
                }
            }
            eta *= 0.5;
        }

        if (improved && best_residual < r_norm_start) {
            solver_arrays.h_node_guesses = trial_best;
            lm_mu = std::max(newton_params.lm_mu_min, mu * newton_params.lm_mu_decrease);
            log.success = true;
            log.max_defect_norm = best_residual;
            log.step_size_norm = eta_best * delta.norm();
            return log;
        }

        solver_arrays.h_node_guesses = z_backup;
        mu = std::min(newton_params.lm_mu_max, mu * newton_params.lm_mu_increase);
    }

    log.success = false;
    solver_arrays.h_node_guesses = z_backup;
    return log;
}
