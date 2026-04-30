#include "solver.hpp"
#include <cmath>
#include <cstdlib>
#include <vector>
#include <complex>

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Sparse>
#include <Eigen/Dense>


// ********* ODE PHYSICS SIMULATION FUNCTIONS *********

__host__ __device__
void compute_state_physics(
    const VarState& state,
    const SystemParams& params,
    VarState& ds
) {
    // 1. Pull values locally using the struct accessors ()
    double theta = state.theta();
    double phi = state.phi();
    double l1 = state.l1();
    double l2 = state.l2();
    double alpha = params.alpha;

    // 2. Trust the compiler optimizer to fuse these automatically!
    double sin_t = sin(theta);
    double cos_t = cos(theta);

    // Precompute squared terms
    double cos_t_sq = cos_t * cos_t;
    double l2_sq = l2 * l2;
    double phi_sq = phi * phi;

    // 3. Evaluate RHS of the effective controlled pendulum dynamics
    ds.theta() = phi;
    ds.phi()   = sin_t - alpha * phi - l2 * cos_t_sq;
    ds.l1()    = -l2_sq * cos_t * sin_t - l2 * cos_t - sin_t;
    ds.l2()    = -phi - l1 + alpha * l2;
    
    // Running Cost
    ds.cost()  = 1.0 - cos_t + 0.5 * phi_sq + 0.5 * l2_sq * cos_t_sq;
}


__host__ __device__
Mat4x4 compute_sensitivity_jacobian(
    const VarState& state,
    const SystemParams& params
) {
    // 1. Pull values locally using the struct accessors ()
    double theta = state.theta();
    double phi = state.phi();
    double l1 = state.l1();
    double l2 = state.l2();
    double alpha = params.alpha;

    // 2. Trust the compiler optimizer to fuse these automatically!
    double sin_t = sin(theta);
    double cos_t = cos(theta);

    // Precompute squared terms
    double cos_t_sq = cos_t * cos_t;
    double sin_t_sq  = sin_t * sin_t;
    double l2_sq = l2 * l2;

    // 3. Create sensitivity Jacobian matrix A(s)
    Mat4x4 A;

    // Row 0
    A(0, 0) = 0.;
    A(0, 1) = 1.;
    A(0, 2) = 0.;
    A(0, 3) = 0.;

    // Row 1 
    A(1, 0) = cos_t + 2.0 * l2 * cos_t * sin_t;
    A(1, 1) = -alpha;
    A(1, 2) = 0.;
    A(1, 3) = -cos_t_sq;

    // Row 2
    A(2, 0) = -(l2_sq * (cos_t_sq - sin_t_sq) - l2 * sin_t + cos_t);
    A(2, 1) = 0.;
    A(2, 2) = 0.;
    A(2, 3) = -(2.0 * l2 * cos_t * sin_t + cos_t);

    // Row 3
    A(3, 0) = 0.;
    A(3, 1) = -1.;
    A(3, 2) = -1.;
    A(3, 3) = alpha;

    return A;
}

__host__ __device__
VarState get_derivatives(
    const VarState& state,
    const SystemParams& params
) {
    VarState ds;

    // Compute/store ds/dt
    compute_state_physics(state, params, ds);

    // Compute/store dM/dt 
    Mat4x4 A = compute_sensitivity_jacobian(state, params);
    ds.M = A * state.M;

    return ds;
}

__host__ __device__
double compute_hamiltonian(
    const VarState& state,
    const SystemParams& params
) {
    // Parse out needed quantities
    double theta = state.theta();
    double phi   = state.phi();
    double l1    = state.l1();
    double l2    = state.l2();
    double alpha = params.alpha;

    // Compute trig functions (hope compiler optimizes)
    double sin_t = sin(theta);
    double cos_t = cos(theta);

    // Compute squared functions
    double l2_sq = l2 * l2;
    double phi_sq = phi * phi;
    double cos_t_sq = cos_t * cos_t;

    // Compute and return hamiltonian
    double hamiltonian = 1.0 - cos_t + 0.5*phi_sq - 0.5*l2_sq*cos_t_sq + l1*phi + l2*(sin_t - alpha*phi);
    return hamiltonian;
}

// ---- INTEGRATION KERNELS/FUNCTIONS ----

// Takes a single microscopic RK4 step, updating both the 5D state and 4x4 sensitivity matrix
__host__ __device__
VarState rk4_step(
    const VarState& current, 
    const SystemParams& params, 
    double dt
) {
    double half_dt = 0.5 * dt;

    // k1 = f(y_n)
    VarState k1 = get_derivatives(current, params);

    // k2 = f(y_n + dt/2 * k1)
    VarState k2 = get_derivatives(current + (k1 * half_dt), params);

    // k3 = f(y_n + dt/2 * k2)
    VarState k3 = get_derivatives(current + (k2 * half_dt), params);

    // k4 = f(y_n + dt * k3)
    VarState k4 = get_derivatives(current + (k3 * dt), params);

    // y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    VarState next_state = current + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);
    return next_state;
}

// Loops rk4_step over the micro-grid and packages the final state/matrix + initial Hamiltonian
__host__ __device__
SegmentEvaluation simulate_segment(
    const VarState& initial_guess,
    const SystemParams& sys_params,
    const IntegratorParams& int_params
) {
    VarState current_state = initial_guess;

    // 1. Initialize running cost for this local segment to zero (at t=0)
    current_state.cost() = 0.0;

    // 2. Initialize sensitivity matrix (at t=0) to M = d(s(t))/d(s_initial) = d(s(0))/d(s(0)) = I
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            if (r == c) {
                current.M(r, c) = 1.0;
            } else {
                current.M(r, c) = 0.0;
            }
        }
    }

    // 3. Compute the Hamiltonian constraint at the START of the segment
    double init_H = compute_hamiltonian(current, sys_params);

    // 4. Integrate forward in time using RK4
    for (int step = 0; step < int_params.num_steps; step++) {
        current = rk4_step(current, sys_params, int_params.dt);
    }

    // 5. Package and return results 
    SegmentEvaluation result;
    result.final_state = current;
    result.initial_hamiltonian = init_H;

    return result;
}

// Each thread reads the guess for its corresponding node, integrates the segment, and writes to segment_results[k]
__global__ 
void multiple_shooting_kernel(
    DeviceArrays d,
    SystemParams sys_params, 
    IntegratorParams int_params
) {
    // 1. Calculate the global thread ID (which corresponds to segment 'k')
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Safety check: Ensure we don't launch threads out of bounds
    if (k >= sys_params.num_shooting_intervals) {
        return;
    }

    // 3. Extract the initial guess for this specific node from the flat array
    // Note: simulate_segment handles initializing cost to 0.0 and M to the Identity matrix
    VarState initial_guess;
    initial_guess.theta() = d.node_guesses[k * 4 + 0];
    initial_guess.phi()   = d.node_guesses[k * 4 + 1];
    initial_guess.l1()    = d.node_guesses[k * 4 + 2];
    initial_guess.l2()    = d.node_guesses[k * 4 + 3];

    // 4. Run the full RK4 integration and sensitivity propagation for this segment
    SegmentEvaluation result = simulate_segment(initial_guess, sys_params, int_params);

    // 5. Write the integrated state, matrices, and Hamiltonian back to global memory
    d.segment_results[k] = result;
}

// Launches the GPU kernel, waits for completion, and copies SegmentEvaluations back to the CPU
void evaluate_segments_on_gpu(
    HDArrays& solver_arrays, 
    const SystemParams& sys_params, 
    const IntegratorParams& int_params
) {
    // 1. Copy initial state guesses for each segment from CPU to GPU
    solver_arrays.copy_guesses_to_device();

    // 2. Configure Grid
    int threads_per_block = 256;
    int blocks_per_grid = (sys_params.num_shooting_intervals + threads_per_block - 1) / threads_per_block;

    // 3. Launch GPU Kernel
    multiple_shooting_kernel<<<blocks_per_grid, threads_per_block>>>(
        solver_arrays.get_device_arrays(), 
        sys_params, 
        int_params
    );
    gpuErrchk(cudaDeviceSynchronize());

    // 4. Retrieve evaluated state data from GPU; copy to CPU
    solver_arrays.copy_results_to_host();
}


