#include "solver.hpp"
#include <cmath>


__host__ __device__
void evaluate_derivatives(const double y[5], double dy[5], double alpha) {
    // Evaluates the state/costate/cost derivatives at the current
    // controlled pendulum state/costate, and stores them in the array dy

    // Unpack the state vector
    double theta    = y[0];
    double phi      = y[1];
    double lambda_1 = y[2];
    double lambda_2 = y[3];

    // Precompute expensive functions 
    double sin_t  = sin(theta);
    double cos_t  = cos(theta);
    double cos2_t = cos_t * cos_t;
    double lambda_2_sq = lambda_2 * lambda_2;
    double phi_sq = phi * phi;

    // Evaluate RHS of the effective controlled pendulum dynamics
    dy[0] = phi;
    dy[1] = sin_t - alpha * phi - lambda_2 * cos2_t;
    dy[2] = -lambda_2_sq * cos_t * sin_t - lambda_2 * cos_t - sin_t;
    dy[3] = -phi - lambda_1 + alpha * lambda_2;
    dy[4] = 1.0 - cos_t + 0.5 * phi_sq + 0.5 * lambda_2_sq * cos2_t;
}






Result solve(double theta, double phi, double alpha) {
    Result r;
    r.l1 = 0.0;
    r.l2 = 0.0;
    r.cost = 1e100;
    return r;
}
