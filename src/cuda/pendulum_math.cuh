#pragma once

#include <cmath>

#include "../pendulum/types.hpp"
#include "host_device_macros.cuh"

namespace pendulum {

// Variational state: z in R^4, sensitivity S = dz/dl0 (4x2), accumulated running-cost proxy for quadrature.
struct AugState {
    PhasePoint z{};
    // S[i][j] = d z_i / d l0_j, i in [0,3], j in [0,1]
    double S[4][2]{};
    double Jq = 0.0;  // not the cost functional; quadrature increment uses runningCost(z)
};

PEND_HD inline double u_star(const PhasePoint& z) {
    return -z.l.l2 * std::cos(z.x.theta);
}

PEND_HD inline double running_cost(const PhasePoint& z) {
    const double u = u_star(z);
    return (1.0 - std::cos(z.x.theta)) + 0.5 * z.x.phi * z.x.phi + 0.5 * u * u;
}

PEND_HD inline PhaseDeriv hamiltonian_rhs(const Params& p, const PhasePoint& z) {
    const double th = z.x.theta;
    const double ph = z.x.phi;
    const double l1 = z.l.l1;
    const double l2 = z.l.l2;

    const double s = std::sin(th);
    const double c = std::cos(th);
    const double c2 = c * c;

    PhaseDeriv k;
    k.dtheta = ph;
    k.dphi = s - p.alpha * ph - l2 * c2;
    k.dl1 = -s - l2 * c - (l2 * l2) * c * s;
    k.dl2 = -ph - l1 + p.alpha * l2;
    return k;
}

// Jacobian of Hamiltonian flow w.r.t. z = [theta, phi, l1, l2], same layout as cpp/solver/shooting.cpp
PEND_HD inline void jacobian_df(const Params& p, const PhasePoint& z, double A[4][4]) {
    const double th = z.x.theta;
    const double l2 = z.l.l2;

    const double s = std::sin(th);
    const double c = std::cos(th);
    const double c2 = c * c;
    const double c2theta = std::cos(2.0 * th);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            A[i][j] = 0.0;
        }
    }

    A[0][1] = 1.0;
    A[1][0] = c + 2.0 * l2 * c * s;
    A[1][1] = -p.alpha;
    A[1][3] = -c2;
    A[2][0] = -c + l2 * s - (l2 * l2) * c2theta;
    A[2][3] = -c - 2.0 * l2 * c * s;
    A[3][1] = -1.0;
    A[3][2] = -1.0;
    A[3][3] = p.alpha;
}

PEND_HD inline void mat4_mul_mat42(const double A[4][4], const double S[4][2], double out[4][2]) {
    for (int i = 0; i < 4; ++i) {
        out[i][0] = A[i][0] * S[0][0] + A[i][1] * S[1][0] + A[i][2] * S[2][0] + A[i][3] * S[3][0];
        out[i][1] = A[i][0] * S[0][1] + A[i][1] * S[1][1] + A[i][2] * S[2][1] + A[i][3] * S[3][1];
    }
}

PEND_HD inline AugState aug_rhs(const Params& p, const AugState& aa) {
    AugState d{};
    const PhaseDeriv k = hamiltonian_rhs(p, aa.z);
    d.z = as_phase_point(k);
    double A[4][4];
    jacobian_df(p, aa.z, A);
    mat4_mul_mat42(A, aa.S, d.S);
    d.Jq = running_cost(aa.z);
    return d;
}

PEND_HD inline AugState operator+(const AugState& a, const AugState& b) {
    AugState o;
    o.z = a.z + b.z;
    for (int i = 0; i < 4; ++i) {
        o.S[i][0] = a.S[i][0] + b.S[i][0];
        o.S[i][1] = a.S[i][1] + b.S[i][1];
    }
    o.Jq = a.Jq + b.Jq;
    return o;
}

PEND_HD inline AugState operator*(double s, const AugState& a) {
    AugState o;
    o.z = s * a.z;
    for (int i = 0; i < 4; ++i) {
        o.S[i][0] = s * a.S[i][0];
        o.S[i][1] = s * a.S[i][1];
    }
    o.Jq = s * a.Jq;
    return o;
}

}  // namespace pendulum
