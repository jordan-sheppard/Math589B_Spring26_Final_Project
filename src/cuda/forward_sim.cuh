#pragma once

#include <algorithm>
#include <cmath>

#include "kahan.cuh"
#include "pendulum_math.cuh"
#include "rk4_dp5.cuh"

namespace pendulum {

enum class IntegratorKind { RK4 = 0, DP5 = 1 };

struct ForwardSimOut {
    double terminal_x[2]{};
    double terminal_l[2]{};
    double dZ_dL0[4][2]{};
    double cost = 0.0;
};

PEND_HD inline void init_sensitivity_identity(double S[4][2]) {
    for (int i = 0; i < 4; ++i) {
        S[i][0] = 0.0;
        S[i][1] = 0.0;
    }
    S[2][0] = 1.0;
    S[3][1] = 1.0;
}

PEND_HD inline ForwardSimOut simulate_forward(
    const Params& p,
    const State& x0,
    const Costate& l0,
    double T,
    double dt,
    IntegratorKind integrator) {
    AugState a{};
    a.z.x = x0;
    a.z.l = l0;
    init_sensitivity_identity(a.S);
    a.Jq = 0.0;

    const int n = std::max(1, static_cast<int>(std::ceil(T / dt)));
    const double h = T / static_cast<double>(n);

    KahanSum J;
    double t = 0.0;
    for (int i = 0; i < n; ++i) {
        const double f0 = running_cost(a.z);

        const auto rhs = [&](double /*tt*/, const AugState& aa) {
            return aug_rhs(p, aa);
        };

        AugState a_next;
        if (integrator == IntegratorKind::DP5) {
            a_next = dp5_step(a, t, h, rhs);
        } else {
            a_next = rk4_step(a, t, h, rhs);
        }

        const double f1 = running_cost(a_next.z);
        J.add(0.5 * h * (f0 + f1));

        a = a_next;
        t += h;
    }

    ForwardSimOut out{};
    out.terminal_x[0] = a.z.x.theta;
    out.terminal_x[1] = a.z.x.phi;
    out.terminal_l[0] = a.z.l.l1;
    out.terminal_l[1] = a.z.l.l2;
    for (int r = 0; r < 4; ++r) {
        out.dZ_dL0[r][0] = a.S[r][0];
        out.dZ_dL0[r][1] = a.S[r][1];
    }
    out.cost = J.value();
    return out;
}

}  // namespace pendulum
