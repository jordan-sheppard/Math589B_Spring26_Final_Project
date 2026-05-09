#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace pendulum {

struct State {
    double theta = 0.0;
    double phi = 0.0;
};

struct Costate {
    double l1 = 0.0;
    double l2 = 0.0;
};

struct PhasePoint {
    State x{};
    Costate l{};
};

struct Params {
    double alpha = 0.0;
};

struct PhaseDeriv {
    double dtheta = 0.0;
    double dphi = 0.0;
    double dl1 = 0.0;
    double dl2 = 0.0;
};

inline PhasePoint operator+(const PhasePoint& a, const PhasePoint& b) {
    return PhasePoint{
        .x = State{a.x.theta + b.x.theta, a.x.phi + b.x.phi},
        .l = Costate{a.l.l1 + b.l.l1, a.l.l2 + b.l.l2},
    };
}

inline PhasePoint operator*(double s, const PhasePoint& a) {
    return PhasePoint{
        .x = State{s * a.x.theta, s * a.x.phi},
        .l = Costate{s * a.l.l1, s * a.l.l2},
    };
}

inline PhasePoint operator*(const PhasePoint& a, double s) { return s * a; }

inline PhasePoint asPhasePoint(const PhaseDeriv& k) {
    return PhasePoint{
        .x = State{k.dtheta, k.dphi},
        .l = Costate{k.dl1, k.dl2},
    };
}

inline double normInf(const PhaseDeriv& k) {
    double m = 0.0;
    m = std::max(m, std::abs(k.dtheta));
    m = std::max(m, std::abs(k.dphi));
    m = std::max(m, std::abs(k.dl1));
    m = std::max(m, std::abs(k.dl2));
    return m;
}

}  // namespace pendulum

