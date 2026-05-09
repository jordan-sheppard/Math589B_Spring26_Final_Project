#include "dynamics.hpp"

#include <cmath>

namespace pendulum {

double uStar(const PhasePoint& z) {
    return -z.l.l2 * std::cos(z.x.theta);
}

double runningCost(const PhasePoint& z) {
    const double u = uStar(z);
    return (1.0 - std::cos(z.x.theta)) + 0.5 * z.x.phi * z.x.phi + 0.5 * u * u;
}

PhaseDeriv hamiltonianRHS(const Params& p, const PhasePoint& z) {
    const double th = z.x.theta;
    const double ph = z.x.phi;
    const double l1 = z.l.l1;
    const double l2 = z.l.l2;

    const double s = std::sin(th);
    const double c = std::cos(th);
    const double c2 = c * c;

    // From effective H:
    // theta_dot = ∂H/∂l1 = phi
    // phi_dot   = ∂H/∂l2 = sin(theta) - alpha*phi - l2*cos^2(theta)
    //
    // l1_dot = -∂H/∂theta
    //   ∂H/∂theta = sin(theta) + l2*cos(theta) + l2^2*cos(theta)*sin(theta)
    //
    // l2_dot = -∂H/∂phi
    //   ∂H/∂phi = phi + l1 - alpha*l2
    PhaseDeriv k;
    k.dtheta = ph;
    k.dphi = s - p.alpha * ph - l2 * c2;
    k.dl1 = -s - l2 * c - (l2 * l2) * c * s;
    k.dl2 = -ph - l1 + p.alpha * l2;
    return k;
}

}  // namespace pendulum

