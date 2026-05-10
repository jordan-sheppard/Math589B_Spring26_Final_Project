#pragma once

#include "host_device_macros.cuh"

namespace pendulum {

struct KahanSum {
    double sum = 0.0;
    double c = 0.0;

    PEND_HD void add(double x) {
        const double y = x - c;
        const double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }

    PEND_HD double value() const { return sum; }
};

}  // namespace pendulum
