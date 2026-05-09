#include "solver.hpp"

#include "cpp/solver/cost.hpp"
#include "cpp/solver/dynamics.hpp"
#include "cpp/solver/manifold_seed.hpp"
#include "cpp/solver/rk4.hpp"
#include "cpp/solver/sheet_search.hpp"
#include "cpp/solver/shooting.hpp"
#include "cpp/solver/types.hpp"

namespace {
bool debugEnabled() { return std::getenv("PENDULUM_DEBUG") != nullptr; }
}  // namespace

Result solve(double theta, double phi, double alpha) {
    pendulum::Params p;
    p.alpha = alpha;

    const pendulum::State x0{.theta = theta, .phi = phi};

    pendulum::SheetSearchSettings ss;
    const bool dbg = debugEnabled();
    ss.debug = dbg;
    ss.shoot.T = 10.0;
    ss.shoot.dt = 1e-3;
    ss.shoot.max_iters = 40;
    ss.shoot.tol_resid = 1e-10;
    ss.shoot.fd_eps = 1e-5;
    ss.shoot.lm_lambda0 = 1e-2;
    ss.shoot.max_delta_norm = 5.0;
    ss.shoot.backtrack_max = 12;
    ss.shoot.debug = dbg;

    // Heuristic: with high initial speed, allow more sheet offsets.
    ss.m_radius_min = 2;
    ss.m_radius_max = 80;
    ss.m_radius_per_speed = 2.0;

    // Continuation in horizon to improve robustness (especially for phi != 0).
    ss.T_schedule.resize(4);
    ss.T_schedule << 2.0, 4.0, 6.0, 8.0;

    const pendulum::SheetSearchResult sol = pendulum::solveWithSheetSearch(p, x0, ss);

    Result r;
    r.l1 = sol.best.l0.l1;
    r.l2 = sol.best.l0.l2;
    r.cost = sol.best.cost;
    return r;
}

