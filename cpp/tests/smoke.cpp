#include <cassert>
#include <cmath>
#include <cstdio>

#include "cpp/solver/dynamics.hpp"
#include "cpp/solver/rk4.hpp"
#include "cpp/solver/types.hpp"

int main() {
    pendulum::Params p;
    p.alpha = 0.2;

    // Origin should be an equilibrium for the full Hamiltonian system.
    pendulum::PhasePoint z0;
    z0.x.theta = 0.0;
    z0.x.phi = 0.0;
    z0.l.l1 = 0.0;
    z0.l.l2 = 0.0;

    const auto k0 = pendulum::hamiltonianRHS(p, z0);
    assert(std::abs(k0.dtheta) == 0.0);
    assert(std::abs(k0.dphi) == 0.0);
    assert(std::abs(k0.dl1) == 0.0);
    assert(std::abs(k0.dl2) == 0.0);

    // A small perturbation should produce a bounded RK4 step.
    pendulum::PhasePoint z = z0;
    z.x.theta = 1e-3;
    z.x.phi = -2e-3;
    z.l.l1 = 3e-3;
    z.l.l2 = -4e-3;

    const double dt = 1e-2;
    const auto rhs = [&](double /*t*/, const pendulum::PhasePoint& zz) {
        return pendulum::asPhasePoint(pendulum::hamiltonianRHS(p, zz));
    };

    const auto z1 = pendulum::rk4Step<pendulum::PhasePoint>(z, 0.0, dt, rhs);
    const double delta_theta = std::abs(z1.x.theta - z.x.theta);
    assert(std::isfinite(delta_theta));
    assert(delta_theta < 1.0);  // very weak sanity bound

    std::printf("smoke ok\n");
    return 0;
}

