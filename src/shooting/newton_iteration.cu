#include "shooting/newton_iteration.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "core/solver_debug.hpp"

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

    const bool dbg = math589_solver_debug_enabled();
    const bool lm_verb = dbg && math589_solver_debug_lm_verbose();
    if (dbg) {
        std::fprintf(stderr,
                     "[MATH589][LM] enter |r|_inf_before=%.6e (J size m=%lld n=%lld) lm_mu_in=%.6e\n",
                     r_norm_start, static_cast<long long>(J.rows()),
                     static_cast<long long>(J.cols()), lm_mu);
    }

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
        const bool factor_ok = (ldlt.info() == Eigen::Success);
        if (lm_verb) {
            std::fprintf(stderr,
                         "[MATH589][LM] damping_try sub=%d/%d mu=%.6e ldlt_compute_ok=%d\n",
                         sub + 1, newton_params.lm_max_subiterations, mu, factor_ok ? 1 : 0);
        }
        if (!factor_ok) {
            mu = std::min(newton_params.lm_mu_max, mu * newton_params.lm_mu_increase);
            continue;
        }

        VectorXd delta = ldlt.solve(-JtF);
        // Do not inspect ldlt.info() after solve(): it reflects the factorization in compute(),
        // and treating stale/failed info here wrongly rejects steps on some Eigen/toolchain builds.

        if (!delta.allFinite()) {
            if (lm_verb) {
                std::fprintf(stderr, "[MATH589][LM] delta non-finite; increasing mu\n");
            }
            mu = std::min(newton_params.lm_mu_max, mu * newton_params.lm_mu_increase);
            continue;
        }

        const double dn = delta.norm();
        if (dn > newton_params.max_delta_norm && dn > 0.0) {
            delta *= newton_params.max_delta_norm / dn;
        }

        double eta_best = 1.0;
        double best_residual = r_norm_start;
        std::vector<double> trial_best = z_backup;

        int bt_used = -1;
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
                bt_used = bt;
                if (best_residual <= r_norm_start * (1.0 - 1e-13)) {
                    break;
                }
            }
            eta *= 0.5;
        }

        if (lm_verb) {
            std::fprintf(stderr,
                         "[MATH589][LM] backtrack_eta_best=%.6e bt_index=%d |r|_after=%.6e "
                         "|r|_before=%.6e accepted_strict_descent=%d\n",
                         eta_best, bt_used, best_residual, r_norm_start,
                         (best_residual < r_norm_start) ? 1 : 0);
        }

        if (best_residual < r_norm_start) {
            solver_arrays.h_node_guesses = trial_best;
            lm_mu = std::max(newton_params.lm_mu_min, mu * newton_params.lm_mu_decrease);
            log.success = true;
            log.max_defect_norm = best_residual;
            log.step_size_norm = eta_best * delta.norm();
            if (dbg) {
                std::fprintf(stderr,
                             "[MATH589][LM] ACCEPTED |r|_inf %.6e -> %.6e step_norm=%.6e lm_mu_out=%.6e\n",
                             r_norm_start, best_residual, log.step_size_norm, lm_mu);
            }
            return log;
        }

        solver_arrays.h_node_guesses = z_backup;
        mu = std::min(newton_params.lm_mu_max, mu * newton_params.lm_mu_increase);
    }

    log.success = false;
    solver_arrays.h_node_guesses = z_backup;
    if (dbg) {
        std::fprintf(stderr,
                     "[MATH589][LM] FAILED all damping tries -> leave |r|_inf=%.6e lm_mu(now)=%.6e\n",
                     r_norm_start, mu);
    }
    return log;
}
