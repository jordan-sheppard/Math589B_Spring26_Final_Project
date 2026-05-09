#pragma once

#include "types.hpp"

namespace pendulum {

// Optimal control: u* = -l2 cos(theta)
double uStar(const PhasePoint& z);

// Running cost integrand f0(x, u*).
double runningCost(const PhasePoint& z);

// Hamiltonian vector field (state-costate ODE) under u*.
// Uses the effective Hamiltonian from the project PDF.
PhaseDeriv hamiltonianRHS(const Params& p, const PhasePoint& z);

}  // namespace pendulum

