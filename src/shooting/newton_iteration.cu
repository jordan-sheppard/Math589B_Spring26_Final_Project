#include "shooting/newton_iteration.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
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

namespace {
// Agent NDJSON log (session 976d44): opened relative to process cwd — run solver from repo dir on Colab
// so `math589_debug_976d44.log` lands next to `./solver`.
constexpr const char *kMath589AgentLog976d44 = "math589_debug_976d44.log";
constexpr const char *kDebugLogPath390801 = ".cursor/debug-390801.log";

inline void appendDebug390801(const char *run_id, const char *hypothesis_id, const char *location,
                              const char *message, const std::string &data_json) {
    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();
    std::ofstream lf(kDebugLogPath390801, std::ios::app);
    if (!lf) return;
    lf << "{\"sessionId\":\"390801\",\"runId\":\"" << run_id << "\",\"hypothesisId\":\""
       << hypothesis_id << "\",\"location\":\"" << location << "\",\"message\":\"" << message
       << "\",\"data\":" << data_json << ",\"timestamp\":" << ts << "}\n";
}
} // namespace

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

    const double r_inf_start = linf(F);
    const double r_l2_start = F.norm();
    log.max_defect_norm = r_inf_start;
    const int N = sys_params.num_shooting_intervals;
    const int n_state = 4 * N;
    double max_cont = 0.0;
    for (int i = 0; i < std::max(0, 4 * (N - 1)); ++i) {
        max_cont = std::max(max_cont, std::abs(F(i)));
    }
    const double max_bc =
        std::max(std::max(std::abs(F(std::max(0, n_state - 4))), std::abs(F(std::max(0, n_state - 3)))),
                 std::max(std::abs(F(std::max(0, n_state - 2))), std::abs(F(std::max(0, n_state - 1)))));
    const double max_man = std::max(std::abs(F(std::max(0, n_state))), std::abs(F(std::max(0, n_state + 1))));

    const bool dbg = math589_solver_debug_enabled();
    const bool lm_verb = dbg && math589_solver_debug_lm_verbose();
    if (dbg) {
        std::fprintf(stderr,
                     "[MATH589][LM] enter |r|_2=%.6e |r|_inf_before=%.6e (J m=%lld n=%lld) lm_mu=%.6e\n",
                     r_l2_start, r_inf_start, static_cast<long long>(J.rows()),
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
    const double rel_req = newton_params.lm_relative_reduction_min;
    // When ‖r‖∞ is only tens-hundreds × tolerance, fixed relative cuts (e.g. 1e-5) demand
    // sub-representable reductions and every LM try fails (wrap=0: stall ~1e-4 then reject).
    const double tol = std::max(newton_params.tolerance, 1e-18);
    const double rn = std::fabs(r_inf_start);
    double rel_scale = 1.0;
    if (rn > 0.0) {
        rel_scale = rn / (200.0 * tol);
        rel_scale = std::min(1.0, rel_scale);
        rel_scale = std::max(0.03, rel_scale);
    }
    const double rel_eff = rel_req * rel_scale;
    // Gauss–Newton/LM minimizes ‖F‖₂² merit; LM Armijo gates should match ‖F‖₂ (not ‖F‖∞),
    // or ‖F‖∞ can stall flat while ‖F‖₂ still decreases — see plateau in debug NDJSON rejects.
    const double accept_thresh_l2 = std::fma(-rel_eff, r_l2_start, r_l2_start);
    const double inf_relaxed_cap = r_inf_start + 5.0 * tol;

    const double l2_tol_ref =
        tol * tol * static_cast<double>(std::max(F.size(), static_cast<Eigen::Index>(1)));

    // #region agent log
    {
        std::ostringstream d;
        d << "{\"r_inf\":" << r_inf_start << ",\"r_l2\":" << r_l2_start << ",\"max_cont\":" << max_cont
          << ",\"max_bc\":" << max_bc << ",\"max_man\":" << max_man << ",\"J_rows\":" << J.rows()
          << ",\"J_cols\":" << J.cols() << ",\"lm_mu_in\":" << lm_mu << "}";
        appendDebug390801("pre-fix", "H2", "newton_iteration.cu:entry_blocks",
                          "residual_block_breakdown", d.str());
    }
    // #endregion

    // #region agent log
    {
        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
        std::ofstream lf(kMath589AgentLog976d44, std::ios::app);
        if (lf)
            lf << "{\"sessionId\":\"976d44\",\"hypothesisId\":\"H1\",\"location\":\"newton_iteration.cu:"
                  "LM_gate\",\"message\":\"rel_scaling\",\"timestamp\":" << ts
               << ",\"data\":{\"r_inf\":" << r_inf_start << ",\"r_l2\":" << r_l2_start
               << ",\"tol\":" << tol << ",\"rel_req\":" << rel_req << ",\"rel_scale\":" << rel_scale
               << ",\"rel_eff\":" << rel_eff << ",\"accept_thr_l2\":" << accept_thresh_l2 << "}}\n";
    }
    // #endregion

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
        // #region agent log
        {
            std::ostringstream d;
            d << "{\"sub\":" << sub << ",\"mu\":" << mu << ",\"delta_norm\":" << dn
              << ",\"JtF_inf\":" << JtF.lpNorm<Eigen::Infinity>() << "}";
            appendDebug390801("pre-fix", "H3", "newton_iteration.cu:delta_quality",
                              "lm_delta_quality", d.str());
        }
        // #endregion

        double eta_best = 1.0;
        double best_l2 = r_l2_start;
        double best_inf = r_inf_start;
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

            const double try_inf = linf(F_try);
            if (try_inf > inf_relaxed_cap) {
                eta *= 0.5;
                continue;
            }

            const double try_l2 = F_try.norm();
            const double l2_eps =
                std::numeric_limits<double>::epsilon() *
                std::fma(512.0, std::fabs(std::max(try_l2, best_l2)),
                         std::fma(512.0, r_l2_start, l2_tol_ref));

            const bool improves_l2_merit = try_l2 < best_l2 - l2_eps;
            const bool tie_better_inf =
                std::fabs(try_l2 - best_l2) <= l2_eps && try_inf < best_inf - l2_tol_ref;

            if (improves_l2_merit || tie_better_inf) {
                best_l2 = try_l2;
                best_inf = try_inf;
                trial_best = solver_arrays.h_node_guesses;
                eta_best = eta;
                bt_used = bt;
                if (best_l2 <= accept_thresh_l2) {
                    break;
                }
            }
            eta *= 0.5;
        }

        if (lm_verb) {
            std::fprintf(stderr,
                         "[MATH589][LM] backtrack_eta_best=%.6e bt_index=%d |r|_2_after=%.6e "
                         "|r|_inf_after=%.6e suf_L2_descent=%d\n",
                         eta_best, bt_used, best_l2, best_inf,
                         (best_l2 <= accept_thresh_l2) ? 1 : 0);
        }

        if (best_l2 <= accept_thresh_l2) {
            solver_arrays.h_node_guesses = trial_best;
            lm_mu = std::max(newton_params.lm_mu_min, mu * newton_params.lm_mu_decrease);
            log.success = true;
            log.max_defect_norm = best_inf;
            log.step_size_norm = eta_best * delta.norm();
            if (dbg) {
                std::fprintf(stderr,
                             "[MATH589][LM] ACCEPTED |r|_inf %.6e -> %.6e |r|_2 %.6e -> %.6e "
                             "step_norm=%.6e lm_mu_out=%.6e\n",
                             r_inf_start, best_inf, r_l2_start, best_l2, log.step_size_norm, lm_mu);
            }
            // #region agent log
            {
                std::ostringstream d390801;
                d390801 << "{\"sub\":" << sub << ",\"r_inf_before\":" << r_inf_start
                        << ",\"r_inf_after\":" << best_inf << ",\"r_l2_before\":" << r_l2_start
                        << ",\"r_l2_after\":" << best_l2 << ",\"accept_thr_l2\":" << accept_thresh_l2
                        << ",\"eta_best\":" << eta_best << ",\"mu\":" << mu << "}";
                appendDebug390801("pre-fix", "H1", "newton_iteration.cu:accept_gate",
                                  "lm_accept_with_threshold", d390801.str());

                auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
                std::ofstream lf(kMath589AgentLog976d44, std::ios::app);
                if (lf)
                    lf << "{\"sessionId\":\"976d44\",\"hypothesisId\":\"H2\",\"runId\":\"post_l2_merit\","
                          "\"location\":\"newton_iteration.cu:accept\",\"message\":\"lm_accepted\","
                          "\"timestamp\":"
                       << ts << ",\"data\":{\"sub\":" << sub << ",\"r_inf_before\":" << r_inf_start
                       << ",\"r_inf_after\":" << best_inf << ",\"r_l2_before\":" << r_l2_start
                       << ",\"r_l2_after\":" << best_l2 << ",\"accept_thr_l2\":" << accept_thresh_l2
                       << ",\"lm_mu_out\":" << lm_mu << "}}\n";
            }
            // #endregion
            return log;
        }

        // #region agent log
        {
            std::ostringstream d390801;
            d390801 << "{\"sub\":" << sub << ",\"best_l2\":" << best_l2 << ",\"best_inf\":" << best_inf
                    << ",\"accept_thr_l2\":" << accept_thresh_l2 << ",\"mu\":" << mu
                    << ",\"eta_best\":" << eta_best << ",\"bt_used\":" << bt_used << "}";
            appendDebug390801("pre-fix", "H1", "newton_iteration.cu:reject_gate",
                              "lm_reject_threshold_miss", d390801.str());

            auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
            std::ofstream lf(kMath589AgentLog976d44, std::ios::app);
            if (lf)
                lf << "{\"sessionId\":\"976d44\",\"hypothesisId\":\"H2\",\"runId\":\"post_l2_merit\","
                      "\"location\":\"newton_iteration.cu:reject_sub\","
                      "\"message\":\"lm_subiter_rejected\",\"timestamp\":"
                   << ts << ",\"data\":{\"sub\":" << sub << ",\"best_l2\":" << best_l2
                   << ",\"best_inf\":" << best_inf << ",\"accept_thr_l2\":" << accept_thresh_l2
                   << ",\"mu\":" << mu << "}}\n";
        }
        // #endregion

        solver_arrays.h_node_guesses = z_backup;
        mu = std::min(newton_params.lm_mu_max, mu * newton_params.lm_mu_increase);
    }

    log.success = false;
    solver_arrays.h_node_guesses = z_backup;
    if (dbg) {
        std::fprintf(stderr,
                     "[MATH589][LM] FAILED all damping tries -> leave |r|_inf=%.6e |r|_2=%.6e "
                     "lm_mu(now)=%.6e\n",
                     r_inf_start, r_l2_start, mu);
    }
    // #region agent log
    {
        std::ostringstream d390801;
        d390801 << "{\"r_inf\":" << r_inf_start << ",\"r_l2\":" << r_l2_start << ",\"mu_final\":" << mu
                << ",\"lm_subiters\":" << newton_params.lm_max_subiterations << "}";
        appendDebug390801("pre-fix", "H4", "newton_iteration.cu:fail_all",
                          "lm_failed_all_subiters", d390801.str());

        auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch())
                      .count();
        std::ofstream lf(kMath589AgentLog976d44, std::ios::app);
        if (lf)
            lf << "{\"sessionId\":\"976d44\",\"hypothesisId\":\"H3\",\"runId\":\"post_l2_merit\","
                  "\"location\":\"newton_iteration.cu:fail_all\",\"message\":\"lm_failed_all_sub\","
                  "\"timestamp\":"
               << ts << ",\"data\":{\"r_inf\":" << r_inf_start << ",\"r_l2\":" << r_l2_start
               << ",\"mu_final\":" << mu << "}}\n";
    }
    // #endregion
    return log;
}
