#include "solver.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>
#include <complex>

__host__ __device__
StateVec evaluate_derivatives(const StateVec& y, const double alpha) {
    // Evaluates the state/costate/cost derivatives at the current
    // controlled pendulum state/costate, and stores them in a state vector
    
    StateVec dy;

    // CUDA Hardware Optimization: Compute expensive sin & cos simultaneously!
    double sin_t, cos_t;
    sincos(y.theta, &sin_t, &cos_t);

    // Compute expensive squaring functions
    double cos2_t = cos_t * cos_t;
    double lambda_2_sq = y.lambda_2 * y.lambda_2;
    double phi_sq = y.phi * y.phi;

    // Evaluate RHS of the effective controlled pendulum dynamics
    dy.theta = y.phi;
    dy.phi = sin_t - alpha * y.phi - y.lambda_2 * cos2_t;
    dy.lambda_1 = -lambda_2_sq * cos_t * sin_t - y.lambda_2 * cos_t - sin_t;
    dy.lambda_2 = -y.phi - y.lambda_1 + alpha * y.lambda_2;
    dy.cost = 1.0 - cos_t + 0.5 * phi_sq + 0.5 * lambda_2_sq * cos2_t;

    return dy;
}

__host__ __device__
double evaluate_hamiltonian(const StateVec& y, const double alpha) {
    // CUDA Hardware Optimization: Compute expensive sin & cos simultaneously!
    double sin_t, cos_t;
    sincos(y.theta, &sin_t, &cos_t);

    // Compute expensive squaring functions
    double l2_sq = y.lambda_2 * y.lambda_2;
    double phi_sq = y.phi * y.phi;
    double cos_t_sq = cos_t * cos_t;

    // Evaluate Hamiltonian
    double hamiltonian = 1 - cos_t + 0.5*phi_sq - 0.5*l2_sq*cos_t_sq + y.lambda_1*y.phi + y.lambda_2*(sin_t - alpha*y.phi);
    return hamiltonian;
}

__host__ __device__
double wrap_theta(const double theta) {
    double sin_t, cos_t;
    sincos(theta, sin_t, cos_t);
    return std::atan2(sin_t, cos_t); // Wraps to (-pi, pi]
}

__host__ __device__
StateVec rk4_step(const StateVec& y, const double dt, const double alpha) {
    StateVec k1, k2, k3, k4;

    k1 = evaluate_derivatives(y, alpha);                        // Step 1: k1 = f(y)
    k2 = evaluate_derivatives(y + ((0.5 * dt) * k1), alpha);    // Step 2: k2 = f(y + dt/2 * k1)
    k3 = evaluate_derivatives(y + ((0.5 * dt) * k2), alpha);    // Step 3: k3 = f(y + dt/2 * k2)
    k4 = evaluate_derivatives(y + (dt * k3), alpha);            // Step 4: k4 = f(y + dt * k3)
    return y + ((k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0));  // Final step: y_next = y + dt/6 * (k1 + 2k2 + 2k3 + k4)
}


void compute_stable_eigenspace(const double alpha, StateVec& v1, StateVec& v2) {
    double theta_wrapped = std::atan2(std::sin(theta), std::cos(theta));  // Wraps to (-pi, pi]
    
    // Build linearization matrix about origin
    Eigen::Matrix4d A;
    A << 0.0,  1.0,    0.0, 0.0,
         1.0,  -alpha, 0.0, -1.0,
	     -1.0, 0.0,    0.0, -1.0,
	     0.0,  -1.0,  -1.0, alpha;

    // Solve for eigenvalues/eigenvectors of A
    Eigen::EigenSolver<Eigen::Matrix4d> solver(A);
    Eigen::Vector4cd eigenvalues = solver.eigenvalues();
    Eigen::Matrix4cd eigenvectors = solver.eigenvectors();

    // Isolate stable manifold eigenvalues/eigenvectors
    Eigen::Matrix<std::complex<double>, 4, 2> Vs;    // Columns are stable eigenvectors
    std::size_t col = 0;
    bool is_real = true;
    for (std::size_t i = 0; i < 4; ++i) {
        if (eigenvalues(i).real() < 0 && col < 2) {
            Vs.col(col) = eigenvectors.col(i);
	        col++;
            if (std::abs(eigenvalues(i).imag()) >= 1e-9) {
                is_real = false;
            }
	    }
    }
    for (int i = 0; i < NUM_STATE_DIMS - 1; ++i) {
        v1[i] = Vs(i, 0).real();
        v2[i] = is_real ? Vs(i, 1).real() : Vs(i, 0).imag();
    }
}

DeviceArrays allocate_device_arrays(int array_memory_size) {
    // TODO IMPLEMENT
}

HostArrays copy_device_arrays_to_host(const DeviceArrays& d, const int num_array_elements,
                                      const int array_memory_size) {
    // TODO IMPLEMENT
}

void free_device_arrays(DeviceArrays& d) {
    // TODO IMPLEMENT
}




Result solve(double theta, double phi, double alpha) {
    Result res;
    return res;
}
