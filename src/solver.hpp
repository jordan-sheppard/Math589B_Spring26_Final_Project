#pragma once

#include <cuda_runtime.h>
#include <cstddef> // For std::size_t
#include <cstdio>
#include <vector>  // For std::vector

// The macro captures the file name and line number where a GPU error occurred
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// The inline function evaluates the returned CUDA error code
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      // Prints the human-readable error string from CUDA
      std::fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      
      // Kills the program if abort is true
      if (abort) std::exit(code);
   }
}

constexpr std::size_t NUM_STATE_DIMS = 5;

struct StateVec {
    double data[NUM_STATE_DIMS];

    __host__ __device__ StateVec() {}

    __host__ __device__ double get(int i) const { return data[i]; }
    __host__ __device__ void set(int i, double val) { data[i] = val; }

    __host__ __device__
    StateVec operator+(const StateVec& other) const {
        StateVec result;
	for (std::size_t i = 0; i < NUM_STATE_DIMS; ++i) {
            result.data[i] = this->data[i] + other.data[i];
	}
	return result;
    }

    __host__ __device__
    StateVec operator-(const StateVec& other) const {
        StateVec result;
        for (std::size_t i = 0; i < NUM_STATE_DIMS; ++i) {
            result.data[i] = this->data[i] - other.data[i];
        }
        return result;
    }

    __host__ __device__
    StateVec operator*(double scalar) const {
	StateVec result;
	for (std::size_t i = 0; i < NUM_STATE_DIMS; ++i) {
            result.data[i] = this->data[i] * scalar;
        }
        return result;
    }
};

// Make scalar multiplication okay on both sides
__host__ __device__
inline StateVec operator*(double scalar, const StateVec& vec) {
    return vec * scalar;
}


struct SimulationParams {
    // Friction parameters
    double alpha;                  // Friction constant
    double dt;                     // Timestep size
    long num_timesteps;            // Number of timesteps
    
    // Search Grid parameters
    double l1_init_guess;          // Initial (LQR) Guess for lambda_1 costate
    double l2_init_guess;          // Initial (LQR) Guess for lambda_2 costate
    double search_radius;          // Search radius to left/right/up/down of this guess
    double costate_step_size;      // How big each step is
    int grid_size;                 // How many entries are in each dimension

    // Initial pendulum state (fixed)
    double theta_init;             // Initial angle
    double phi_init;               // Initial angular velocity
};

struct DeviceArrays {
    double* costs;
    double* start_hamiltonians;
    double* end_hamiltonians;
    double* thetas;
    double* phis;
    double* l1s;
    double* l2s;
};

struct HostArrays {
    std::vector<double> costs;
    std::vector<double> start_hamiltonians;
    std::vector<double> end_hamiltonians;
    std::vector<double> thetas;
    std::vector<double> phis;
    std::vector<double> l1s;
    std::vector<double> l2s;
};

struct Result {
    double l1;
    double l2;
    double cost;
};

struct ContinuationResult {
    Result r;
    double min_abs_H;
};

ContinuationResult continuation_core(const SimulationParams& p);
ContinuationResult run_continuation(double theta_target, double phi_target, double alpha);
Result solve(double theta, double phi, double alpha);
