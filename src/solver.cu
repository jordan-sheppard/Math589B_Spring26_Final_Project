#include "solver.hpp"
#include <cmath>
#include <cstdlib>


__host__ __device__
StateVec evaluate_derivatives(const StateVec& y, const double alpha) {
    // Evaluates the state/costate/cost derivatives at the current
    // controlled pendulum state/costate, and stores them in a state vector
    
    StateVec dy;

    // Unpack the current state vector (y)
    double theta    = y.data[0];
    double phi      = y.data[1];
    double lambda_1 = y.data[2];
    double lambda_2 = y.data[3];

    // Precompute expensive functions 
    double sin_t  = sin(theta);
    double cos_t  = cos(theta);
    double cos2_t = cos_t * cos_t;
    double lambda_2_sq = lambda_2 * lambda_2;
    double phi_sq = phi * phi;

    // Evaluate RHS of the effective controlled pendulum dynamics
    dy.data[0] = phi;
    dy.data[1] = sin_t - alpha * phi - lambda_2 * cos2_t;
    dy.data[2] = -lambda_2_sq * cos_t * sin_t - lambda_2 * cos_t - sin_t;
    dy.data[3] = -phi - lambda_1 + alpha * lambda_2;
    dy.data[4] = 1.0 - cos_t + 0.5 * phi_sq + 0.5 * lambda_2_sq * cos2_t;

    return dy;
}

__host__ __device__
StateVec rk4_step(const StateVec& y, const double dt, const double alpha) {
    // Create arrays to hold intermediate/final values 
    StateVec k1, k2, k3, k4, y_next;

    // Step 1: k1 = f(y)
    evaluate_derivatives(y, alpha, k1);

    // Step 2: k2 = f(y + dt/2 * k1)
    evaluate_derivatives(y + ((0.5 * dt) * k1), alpha, k2);

    // Step 3: k3 = f(y + dt/2 * k2)
    evaluate_derivatives(y + ((0.5 * dt) * k2), alpha, k3);

    // Step 4: k4 = f(y + dt * k3)
    evaluate_derivatives(y + (dt * k3), alpha, k4);

    // Final step: y_next = y + dt/6 * (k1 + 2k2 + 2k3 + k4)
    y_next = y + ((k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0));
    return y_next;
}

StateVec run_simulation(const StateVec& y0, const double alpha, const double dt, const double T_max) {
    // Run simulation with a given set of initial conditions
    StateVec current_state = y0;
    
    // Run RK4 on this set of initial conditions
    int num_timesteps = T_max / dt;
    for (int i = 0; i < num_timesteps; ++i) {
        current_state = rk4_step(current_state, dt, alpha);
    }

    return StateVec;
}	



Result solve(double theta, double phi, double alpha) {
    Result r;
    r.l1 = 0.0;
    r.l2 = 0.0;
    r.cost = 1e100;
    return r;
}
