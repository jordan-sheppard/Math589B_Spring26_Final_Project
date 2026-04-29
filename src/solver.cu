#include "solver.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>
#include <complex>

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Sparse>
#include <Eigen/Dense>


__host__ __device__ void compute_state_physics(
    const VarState& state,
    const SystemParams& params,
    VarState& ds
) {
    // 1. Pull values locally using the struct accessors ()
    double theta_val = state.theta();
    double phi_val = state.phi();
    double l1_val = state.l1();
    double l2_val = state.l2();

    // 2. Trust the compiler optimizer to fuse these automatically!
    double sin_t = sin(theta_val);
    double cos_t = cos(theta_val);

    // Precompute squared terms
    double cos2_t = cos_t * cos_t;
    double l2_sq = l2_val * l2_val;
    double phi_sq = phi_val * phi_val;

    // 3. Evaluate RHS of the effective controlled pendulum dynamics
    ds.theta() = phi_val;
    ds.phi()   = sin_t - params.alpha * phi_val - l2_val * cos2_t;
    ds.l1()    = -l2_sq * cos_t * sin_t - l2_val * cos_t - sin_t;
    ds.l2()    = -phi_val - l1_val + params.alpha * l2_val;
    
    // Running Cost
    ds.cost()  = 1.0 - cos_t + 0.5 * phi_sq + 0.5 * l2_sq * cos2_t;
}



