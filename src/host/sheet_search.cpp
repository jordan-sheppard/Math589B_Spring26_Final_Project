#include "sheet_search.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include "manifold_seed.hpp"

namespace pendulum {

namespace {

constexpr double two_pi() { return 6.283185307179586476925286766559; }

int compute_radius(const SheetSearchSettingsHost& s, double phi0) {
    const int by_speed = static_cast<int>(std::ceil(std::abs(phi0) * s.m_radius_per_speed));
    int r = std::max(s.m_radius_min, by_speed);
    r = std::min(r, s.m_radius_max);
    return r;
}

double score_candidate(const ShootResultHost& r) {
    const double rn = resid_inf(r.resid_dim, r.resid);
    if (!std::isfinite(rn)) {
        return std::numeric_limits<double>::infinity();
    }
    const double cst = std::isfinite(r.cost) ? r.cost : 1e300;
    return rn + 1e-9 * cst;
}

}  // namespace

SheetSearchResultHost solve_with_sheet_search(const Params& p, const State& x0, const SheetSearchSettingsHost& s) {
    const double m_center_real = x0.theta / two_pi();
    const int m_center = static_cast<int>(std::llround(m_center_real));

    const int radius = compute_radius(s, x0.phi);

    SheetSearchResultHost out{};
    out.best_m = m_center;
    out.theta0_shifted = x0.theta - two_pi() * static_cast<double>(m_center);

    double best_score = std::numeric_limits<double>::infinity();
    double P[2][2];
    stable_manifold_seed_P(p.alpha, P);

    for (int d = 0; d <= radius; ++d) {
        for (int sign = -1; sign <= 1; sign += 2) {
            if (d == 0 && sign == 1) {
                continue;
            }
            const int m = m_center + sign * d;

            const double theta_shifted = x0.theta - two_pi() * static_cast<double>(m);
            const State x0m{.theta = theta_shifted, .phi = x0.phi};

            const double l1_seed = P[0][0] * x0m.theta + P[0][1] * x0m.phi;
            const double l2_seed = P[1][0] * x0m.theta + P[1][1] * x0m.phi;
            const Costate l0_init{.l1 = l1_seed, .l2 = l2_seed};

            if (s.debug) {
                std::fprintf(
                    stderr,
                    "[sheet] m=%d thetaShift=%.6g phi=%.6g seed=(%.6g,%.6g)\n",
                    m,
                    x0m.theta,
                    x0m.phi,
                    l0_init.l1,
                    l0_init.l2);
            }

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
            const Costate* seeds = hard ? seeds_hard : seeds_easy;
            const int seeds_n = hard ? static_cast<int>(sizeof(seeds_hard) / sizeof(seeds_hard[0]))
                                     : static_cast<int>(sizeof(seeds_easy) / sizeof(seeds_easy[0]));

            for (int si = 0; si < seeds_n; ++si) {
                const Costate& seed = seeds[si];
                if (s.debug && hard) {
                    std::fprintf(stderr, "[sheet]  seed[%d]=(%.6g,%.6g)\n", si, seed.l1, seed.l2);
                }

                ShootResultHost cand;
                if (!s.T_schedule.empty()) {
                    cand = solve_costates_single_sheet_lm_continuation(p, x0m, seed, s.shoot, s.T_schedule, P);
                } else {
                    cand = solve_costates_single_sheet_lm(p, x0m, seed, s.shoot, P);
                }

                const double sc = score_candidate(cand);
                if (sc < best_score) {
                    best_score = sc;
                    out.best = cand;
                    out.best_m = m;
                    out.theta0_shifted = theta_shifted;
                }

                if (out.best.resid_dim > 0 && resid_inf(out.best.resid_dim, out.best.resid) <= s.good_enough_resid) {
                    if (s.debug) {
                        std::fprintf(
                            stderr,
                            "[sheet] early-exit m=%d rinf=%.3e\n",
                            out.best_m,
                            resid_inf(out.best.resid_dim, out.best.resid));
                    }
                    return out;
                }
            }
        }
    }

    return out;
}

}  // namespace pendulum
