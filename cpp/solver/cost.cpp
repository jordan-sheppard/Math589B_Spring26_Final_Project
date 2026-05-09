#include "cost.hpp"

namespace pendulum {

void KahanSum::add(double x) {
    const double y = x - c;
    const double t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

}  // namespace pendulum

