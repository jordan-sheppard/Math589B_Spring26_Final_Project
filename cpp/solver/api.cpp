#include "solver.hpp"

#include "cpp/solver/cost.hpp"
#include "cpp/solver/dynamics.hpp"
#include "cpp/solver/rk4.hpp"
#include "cpp/solver/types.hpp"

namespace {

pendulum::PhasePoint rhsAsPhasePoint(const pendulum::Params& p, const pendulum::PhasePoint& z) {
    return pendulum::asPhasePoint(pendulum::hamiltonianRHS(p, z));
}

}  // namespace

Result solve(double theta, double phi, double alpha) {
    // Checkpoint 2 scaffolding:
    // - implements the dynamics + RK4 + cost accumulation pipeline
    // - does NOT yet solve for the correct initial costates
    pendulum::Params p;
    p.alpha = alpha;

    pendulum::PhasePoint z;
    z.x.theta = theta;
    z.x.phi = phi;
    z.l.l1 = 0.0;
    z.l.l2 = 0.0;

    const double T = 2.0;
    const double dt = 1e-3;
    const int n = static_cast<int>(T / dt);

    pendulum::KahanSum J;
    double t = 0.0;
    for (int i = 0; i < n; ++i) {
        const double f0 = pendulum::runningCost(z);
        const pendulum::PhasePoint z_next =
            pendulum::rk4Step<pendulum::PhasePoint>(z, t, dt, [&](double /*t*/, const pendulum::PhasePoint& zz) {
                return rhsAsPhasePoint(p, zz);
            });
        const double f1 = pendulum::runningCost(z_next);

        J.add(0.5 * dt * (f0 + f1));

        z = z_next;
        t += dt;
    }

    Result r;
    r.l1 = 0.0;
    r.l2 = 0.0;
    r.cost = J.value();
    return r;
}

