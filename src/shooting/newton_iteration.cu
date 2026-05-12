#include "shooting/newton_iteration.hpp"

#include <chrono>
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
    build_global_system(solver_arrays, sys_params, int_params, J, F);

    Eigen::SparseLU<SparseMat> solver;

    solver.compute(J);
    if (solver.info() != Eigen::Success) {
        // printf("Eigen SparseLU failed to factorize the Jacobian!\n");
        log.success = false;
        // #region agent log
        {
            static int nl = 0;
            if (nl++ < 24) {
                long long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count();
                std::FILE *df = std::fopen(
                    "/Users/jordan/math/math589/semester2/coding/Math589B_Spring26_Final_Project/.cursor/"
                    "debug-a00cc2.log",
                    "a");
                if (df) {
                    std::fprintf(df,
                                   "{\"sessionId\":\"a00cc2\",\"timestamp\":%lld,\"location\":"
                                   "\"newton_iteration.cu:lu_fail\",\"message\":\"sparse_lu\",\"hypothesisId\":"
                                   "\"H1\",\"data\":{\"backward_time\":%d,\"lu_info\":%d}}\n",
                                   ts, (int)int_params.backward_time, (int)solver.info());
                    std::fclose(df);
                }
            }
        }
        // #endregion
        return log;
    }

    VectorXd dS = solver.solve(-F);

    for (int i = 0; i < dS.size(); i++) {
        solver_arrays.h_node_guesses[i] += dS(i);
    }

    log.max_defect_norm = F.lpNorm<Eigen::Infinity>();
    log.step_size_norm = dS.norm();

    // #region agent log
    {
        static int nl = 0;
        if (nl++ < 48) {
            long long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                               std::chrono::system_clock::now().time_since_epoch())
                               .count();
            std::FILE *df = std::fopen(
                "/Users/jordan/math/math589/semester2/coding/Math589B_Spring26_Final_Project/.cursor/"
                "debug-a00cc2.log",
                "a");
            if (df) {
                std::fprintf(df,
                               "{\"sessionId\":\"a00cc2\",\"timestamp\":%lld,\"location\":"
                               "\"newton_iteration.cu:step\",\"message\":\"newton_step\",\"hypothesisId\":\"H1\","
                               "\"data\":{\"backward_time\":%d,\"maxF\":%.17g,\"stepNorm\":%.17g}}\n",
                               ts, (int)int_params.backward_time, log.max_defect_norm, log.step_size_norm);
                std::fclose(df);
            }
        }
    }
    // #endregion

    return log;
}
