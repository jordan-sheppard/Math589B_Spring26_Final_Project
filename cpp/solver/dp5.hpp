#pragma once

#include <type_traits>

namespace pendulum {

// Fixed-step Dormand–Prince 5th-order integrator (DP5).
// This uses the same stage structure as the common RK45 method, but we only
// return the 5th-order solution and do NOT adapt step sizes.
//
// Requirements:
// - StateT supports: +, scalar * (double)
// - rhs(t, z) returns a StateT representing dz/dt at (t,z)
template <class StateT, class RHS>
inline StateT dp5Step(const StateT& y, double t, double h, RHS&& rhs) {
    static_assert(std::is_copy_constructible_v<StateT>);

    // c values
    constexpr double c2 = 1.0 / 5.0;
    constexpr double c3 = 3.0 / 10.0;
    constexpr double c4 = 4.0 / 5.0;
    constexpr double c5 = 8.0 / 9.0;

    // a matrix (lower triangular)
    constexpr double a21 = 1.0 / 5.0;

    constexpr double a31 = 3.0 / 40.0;
    constexpr double a32 = 9.0 / 40.0;

    constexpr double a41 = 44.0 / 45.0;
    constexpr double a42 = -56.0 / 15.0;
    constexpr double a43 = 32.0 / 9.0;

    constexpr double a51 = 19372.0 / 6561.0;
    constexpr double a52 = -25360.0 / 2187.0;
    constexpr double a53 = 64448.0 / 6561.0;
    constexpr double a54 = -212.0 / 729.0;

    constexpr double a61 = 9017.0 / 3168.0;
    constexpr double a62 = -355.0 / 33.0;
    constexpr double a63 = 46732.0 / 5247.0;
    constexpr double a64 = 49.0 / 176.0;
    constexpr double a65 = -5103.0 / 18656.0;

    constexpr double a71 = 35.0 / 384.0;
    constexpr double a73 = 500.0 / 1113.0;
    constexpr double a74 = 125.0 / 192.0;
    constexpr double a75 = -2187.0 / 6784.0;
    constexpr double a76 = 11.0 / 84.0;

    // b (5th order)
    constexpr double b1 = 35.0 / 384.0;
    constexpr double b3 = 500.0 / 1113.0;
    constexpr double b4 = 125.0 / 192.0;
    constexpr double b5 = -2187.0 / 6784.0;
    constexpr double b6 = 11.0 / 84.0;

    const StateT k1 = rhs(t, y);
    const StateT k2 = rhs(t + c2 * h, y + h * (a21 * k1));
    const StateT k3 = rhs(t + c3 * h, y + h * (a31 * k1 + a32 * k2));
    const StateT k4 = rhs(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
    const StateT k5 = rhs(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
    const StateT k6 = rhs(
        t + h,
        y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));
    // k7 is not needed for the 5th-order solution with these b's.

    return y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6);
}

}  // namespace pendulum

