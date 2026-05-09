#pragma once

#include <type_traits>

namespace pendulum {

// Generic fixed-step RK4 integrator.
// Requirements:
// - StateT supports: +, scalar * (double)
// - rhs(t, z) returns a StateT representing dz/dt at (t,z)
template <class StateT, class RHS>
inline StateT rk4Step(const StateT& z, double t, double h, RHS&& rhs) {
    static_assert(std::is_copy_constructible_v<StateT>);

    const StateT k1 = rhs(t, z);
    const StateT k2 = rhs(t + 0.5 * h, z + (0.5 * h) * k1);
    const StateT k3 = rhs(t + 0.5 * h, z + (0.5 * h) * k2);
    const StateT k4 = rhs(t + h, z + h * k3);

    return z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

}  // namespace pendulum

