#pragma once

#include <type_traits>

#include "pendulum_math.cuh"

namespace pendulum {

template <class RHS>
PEND_HD inline AugState rk4_step(const AugState& z, double t, double h, RHS&& rhs) {
    static_assert(std::is_copy_constructible_v<AugState>);

    const AugState k1 = rhs(t, z);
    const AugState k2 = rhs(t + 0.5 * h, z + (0.5 * h) * k1);
    const AugState k3 = rhs(t + 0.5 * h, z + (0.5 * h) * k2);
    const AugState k4 = rhs(t + h, z + h * k3);

    return z + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

template <class RHS>
PEND_HD inline AugState dp5_step(const AugState& y, double t, double h, RHS&& rhs) {
    static_assert(std::is_copy_constructible_v<AugState>);

    constexpr double c2 = 1.0 / 5.0;
    constexpr double c3 = 3.0 / 10.0;
    constexpr double c4 = 4.0 / 5.0;
    constexpr double c5 = 8.0 / 9.0;

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

    constexpr double b1 = 35.0 / 384.0;
    constexpr double b3 = 500.0 / 1113.0;
    constexpr double b4 = 125.0 / 192.0;
    constexpr double b5 = -2187.0 / 6784.0;
    constexpr double b6 = 11.0 / 84.0;

    const AugState k1 = rhs(t, y);
    const AugState k2 = rhs(t + c2 * h, y + h * (a21 * k1));
    const AugState k3 = rhs(t + c3 * h, y + h * (a31 * k1 + a32 * k2));
    const AugState k4 = rhs(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3));
    const AugState k5 = rhs(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4));
    const AugState k6 = rhs(t + h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5));

    return y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6);
}

}  // namespace pendulum
