#include "solver.hpp"

#include "cpp/solver/cost.hpp"
#include "cpp/solver/dynamics.hpp"
#include "cpp/solver/manifold_seed.hpp"
#include "cpp/solver/rk4.hpp"
#include "cpp/solver/shooting.hpp"
#include "cpp/solver/types.hpp"

Result solve(double theta, double phi, double alpha) {
    pendulum::Params p;
    p.alpha = alpha;

    const pendulum::State x0{.theta = theta, .phi = phi};

    // Stable-manifold seed: lambda0 ≈ P x0.
    const Eigen::Matrix2d P = pendulum::stableManifoldSeedP(alpha);
    const Eigen::Vector2d xvec(theta, phi);
    const Eigen::Vector2d lvec0 = P * xvec;
    const pendulum::Costate l0_init{.l1 = lvec0(0), .l2 = lvec0(1)};

    pendulum::ShootSettings s;
    s.T = 8.0;
    s.dt = 2e-3;
    s.max_iters = 35;
    s.tol_resid = 1e-8;

    const pendulum::ShootResult sol = pendulum::solveCostatesSingleSheetLM(p, x0, l0_init, s);

    Result r;
    r.l1 = sol.l0.l1;
    r.l2 = sol.l0.l2;
    r.cost = sol.cost;
    return r;
}

