#include "sheet_search.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include "manifold_seed.hpp"

namespace pendulum {

namespace {

constexpr double twoPi() { return 6.283185307179586476925286766559; }

int computeRadius(const SheetSearchSettings& s, double phi0) {
    const int by_speed = static_cast<int>(std::ceil(std::abs(phi0) * s.m_radius_per_speed));
    int r = std::max(s.m_radius_min, by_speed);
    r = std::min(r, s.m_radius_max);
    return r;
}

double scoreCandidate(const ShootResult& r) {
    // Primary: residual infinity norm. Secondary: cost.
    // If not converged, still allow best-by-residual for debugging.
    const double rn = r.resid.lpNorm<Eigen::Infinity>();
    if (!std::isfinite(rn)) return std::numeric_limits<double>::infinity();
    // Small weight on cost to break ties among similarly feasible solutions.
    const double cost = std::isfinite(r.cost) ? r.cost : 1e300;
    return rn + 1e-9 * cost;
}

}  // namespace

SheetSearchResult solveWithSheetSearch(const Params& p, const State& x0, const SheetSearchSettings& s) {
    // Center sheet: nearest equilibrium 2π m to theta.
    const double m_center_real = x0.theta / twoPi();
    const int m_center = static_cast<int>(std::llround(m_center_real));

    const int radius = computeRadius(s, x0.phi);

    SheetSearchResult out;
    out.best_m = m_center;
    out.theta0_shifted = x0.theta - twoPi() * static_cast<double>(m_center);

    double best_score = std::numeric_limits<double>::infinity();
    const Eigen::Matrix2d P = stableManifoldSeedP(p.alpha);

    // Try center first, then expand outward.
    for (int d = 0; d <= radius; ++d) {
        for (int sign = -1; sign <= 1; sign += 2) {
            if (d == 0 && sign == 1) continue;  // avoid duplicate (0,+)
            const int m = m_center + sign * d;

            const double theta_shifted = x0.theta - twoPi() * static_cast<double>(m);
            const State x0m{.theta = theta_shifted, .phi = x0.phi};

            // Seed l0 ≈ P x0m
            const Eigen::Vector2d lvec = P * Eigen::Vector2d(x0m.theta, x0m.phi);
            const Costate l0_init{.l1 = lvec(0), .l2 = lvec(1)};

            if (s.debug) {
                std::fprintf(stderr, "[sheet] m=%d thetaShift=%.6g phi=%.6g seed=(%.6g,%.6g)\n", m, x0m.theta, x0m.phi, l0_init.l1, l0_init.l2);
            }

            // Adaptive multi-start:
            // - easy regimes: only the base seed
            // - hard regimes: add a small deterministic neighborhood
            const bool hard = (std::abs(x0m.phi) > 2.5) || (std::abs(x0m.theta) > 2.5);
            const double scale = std::max(0.2, 0.2 * std::max(std::abs(l0_init.l1), std::abs(l0_init.l2)));

            const Costate seeds_hard[] = {
                l0_init,
                Costate{l0_init.l1 + scale, l0_init.l2},
                Costate{l0_init.l1 - scale, l0_init.l2},
                Costate{l0_init.l1, l0_init.l2 + scale},
                Costate{l0_init.l1, l0_init.l2 - scale},
                Costate{1.4 * l0_init.l1, 1.4 * l0_init.l2},
                Costate{0.6 * l0_init.l1, 0.6 * l0_init.l2},
            };

            const Costate seeds_easy[] = {l0_init};

            const auto& seeds = hard ? seeds_hard : seeds_easy;
            const int seeds_n = hard ? static_cast<int>(sizeof(seeds_hard) / sizeof(seeds_hard[0]))
                                     : static_cast<int>(sizeof(seeds_easy) / sizeof(seeds_easy[0]));

            for (int si = 0; si < seeds_n; ++si) {
                const Costate& seed = seeds[si];
                if (s.debug && hard) {
                    std::fprintf(stderr, "[sheet]  seed[%d]=(%.6g,%.6g)\n", si, seed.l1, seed.l2);
                }
                ShootResult cand;
                if (s.T_schedule.size() > 0) {
                    cand = solveCostatesSingleSheetLMContinuation(p, x0m, seed, s.shoot, s.T_schedule);
                } else {
                    cand = solveCostatesSingleSheetLM(p, x0m, seed, s.shoot);
                }
                const double sc = scoreCandidate(cand);
                if (sc < best_score) {
                    best_score = sc;
                    out.best = cand;
                    out.best_m = m;
                    out.theta0_shifted = theta_shifted;
                }

                // Good-enough early exit.
                if (out.best.resid.size() > 0 &&
                    out.best.resid.lpNorm<Eigen::Infinity>() <= s.good_enough_resid) {
                    if (s.debug) {
                        std::fprintf(stderr, "[sheet] early-exit m=%d rinf=%.3e\n", out.best_m, out.best.resid.lpNorm<Eigen::Infinity>());
                    }
                    return out;
                }
            }
        }
    }

    return out;
}

}  // namespace pendulum

