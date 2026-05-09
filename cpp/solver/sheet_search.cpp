#include "sheet_search.hpp"

#include <algorithm>
#include <cmath>
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

            ShootResult cand;
            if (s.T_schedule.size() > 0) {
                cand = solveCostatesSingleSheetLMContinuation(p, x0m, l0_init, s.shoot, s.T_schedule);
            } else {
                cand = solveCostatesSingleSheetLM(p, x0m, l0_init, s.shoot);
            }
            const double sc = scoreCandidate(cand);
            if (sc < best_score) {
                best_score = sc;
                out.best = cand;
                out.best_m = m;
                out.theta0_shifted = theta_shifted;
            }
        }
    }

    return out;
}

}  // namespace pendulum

