#pragma once

namespace pendulum {

struct KahanSum {
    double sum = 0.0;
    double c = 0.0;

    void add(double x);
    double value() const { return sum; }
};

}  // namespace pendulum

