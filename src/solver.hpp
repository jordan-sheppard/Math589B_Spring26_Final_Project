#pragma once

#include <cstddef> // For std::size_t


struct Result {
    double l1;
    double l2;
    double cost;
};

Result solve(double theta, double phi, double alpha);
