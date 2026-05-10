#include "solver.hpp"

#include <cstdlib>

#include "host/sheet_search.hpp"

namespace {
bool debug_enabled() { return std::getenv("PENDULUM_DEBUG") != nullptr; }
}  // namespace

Result solve(double theta, double phi, double alpha) {
    pendulum::Params p;
    p.alpha = alpha;

    const pendulum::State x0{.theta = theta, .phi = phi};

    pendulum::SheetSearchSettingsHost ss;
    const bool dbg = debug_enabled();
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

    ss.m_radius_min = 2;
    ss.m_radius_max = 80;
    ss.m_radius_per_speed = 2.0;

    ss.T_schedule = {2.0, 4.0, 6.0, 8.0};

    const pendulum::SheetSearchResultHost sol = pendulum::solve_with_sheet_search(p, x0, ss);

    Result r;
    r.l1 = sol.best.l0.l1;
    r.l2 = sol.best.l0.l2;
    r.cost = sol.best.cost;
    return r;
}
