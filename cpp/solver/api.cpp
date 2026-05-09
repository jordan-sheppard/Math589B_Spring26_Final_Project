#include "solver.hpp"

#include "cpp/solver/cost.hpp"
#include "cpp/solver/dynamics.hpp"
#include "cpp/solver/manifold_seed.hpp"
#include "cpp/solver/rk4.hpp"
#include "cpp/solver/sheet_search.hpp"
#include "cpp/solver/shooting.hpp"
#include "cpp/solver/types.hpp"

Result solve(double theta, double phi, double alpha) {
    pendulum::Params p;
    p.alpha = alpha;

    const pendulum::State x0{.theta = theta, .phi = phi};

    pendulum::SheetSearchSettings ss;
    ss.shoot.T = 8.0;
    ss.shoot.dt = 2e-3;
    ss.shoot.max_iters = 35;
    ss.shoot.tol_resid = 1e-8;

    // Heuristic: with high initial speed, allow more sheet offsets.
    ss.m_radius_min = 6;
    ss.m_radius_max = 80;
    ss.m_radius_per_speed = 2.0;

    const pendulum::SheetSearchResult sol = pendulum::solveWithSheetSearch(p, x0, ss);

    Result r;
    r.l1 = sol.best.l0.l1;
    r.l2 = sol.best.l0.l2;
    r.cost = sol.best.cost;
    return r;
}

