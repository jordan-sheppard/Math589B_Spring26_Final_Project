#pragma once

#include <cuda_runtime.h>
#include <cstddef>   // For std::size_t
#include <cstdio>
#include <cmath>     // For std::sqrt and std::abs
#include <algorithm> // For std::max
#include <vector>    // For std::vector

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

// Evaluates the L2 norm of an (x,y) point
__host__ __device__ inline double l2_norm(double x, double y) {
    return sqrt(x * x + y * y);
}

// Evaluates the L-infinity norm (maximum absolute value) of an (x,y) point
__host__ __device__ inline double linfty_norm(double x, double y) {
    return fmax(fabs(x), fabs(y));
}

constexpr std::size_t NUM_STATE_DIMS = 5;       // Number of state dimensions (2 state, 2 costate, 1 cost)

struct StateVec {
    double theta = 0.0;
    double phi = 0.0;
    double lambda_1 = 0.0;
    double lambda_2 = 0.0;
    double cost = 0.0;

    __host__ __device__ StateVec() {}

    // Overload the [] operator to return a reference to the right variable
    __host__ __device__
    double& operator[](int index) {
        // Note: In CUDA, switch statements compile down to very fast jump tables
        switch(index) {
            case 0: return theta;
            case 1: return phi;
            case 2: return lambda_1;
            case 3: return lambda_2;
            case 4: return cost;
            default: return cost; // Fallback
        }
    }
    
    // Also provide a const version for read-only access
    __host__ __device__
    const double& operator[](int index) const {
        switch(index) {
            case 0: return theta;
            case 1: return phi;
            case 2: return lambda_1;
            case 3: return lambda_2;
            case 4: return cost;
            default: return cost;
        }
    }

    __host__ __device__
    StateVec operator+(const StateVec& other) const {
        StateVec result;
        for (std::size_t i = 0; i < NUM_STATE_DIMS; ++i) {
            // Notice the (*this)[i] to properly call your overloaded operator
            result[i] = (*this)[i] + other[i];
        }
        return result;
    }

    __host__ __device__
    StateVec operator-(const StateVec& other) const {
        StateVec result;
        for (std::size_t i = 0; i < NUM_STATE_DIMS; ++i) {
            result[i] = (*this)[i] - other[i];
        }
        return result;
    }

    __host__ __device__
    StateVec operator*(double scalar) const {
        StateVec result;
        for (std::size_t i = 0; i < NUM_STATE_DIMS; ++i) {
            result[i] = (*this)[i] * scalar;
        }
        return result;
    }
};

// Make scalar multiplication okay on both sides
__host__ __device__
inline StateVec operator*(double scalar, const StateVec& vec) {
    return vec * scalar;
}

struct TrajectoryPoint {
    double time;
    StateVec state;
};

struct BackwardSweepParams {
    double alpha;                  // Friction constant
    double dt;                     // Timestep size (will be NEGATIVE)
    long num_timesteps;            // Number of timesteps
    int num_trajectories;          // How many points are in our seed ring
    double target_theta;           // The target initial angle
    double target_phi;             // The target initial angular velocity
    const StateVec* seed_ring;     // The actual seed states.
};

struct ForwardSweepParams {
    double alpha;                  // Friction constant
    double dt;                     // Timestep size (will be POSITIVE)
    long num_timesteps;            // Number of timesteps
    
    // The exact physical state we want to start the pendulum at
    double target_theta;
    double target_phi;

    // The interpolated guess from the Backward Sweep
    double l1_guess;
    double l2_guess;
    
    // Grid parameters
    double search_radius;          // Very small now! (e.g., 0.05)
    int grid_size;                 // E.g., 128 (for a 128x128 high-res zoom)
};

struct BackwardsTimeDeviceArrays {
    double* start_hamiltonians;
    double* end_hamiltonians;
    TrajectoryPoint* hit_points;
    StateVec* seed_ring;
};

struct BackwardsTimeHostArrays {
    std::vector<double> start_hamiltonians;
    std::vector<double> end_hamiltonians;
    std::vector<TrajectoryPoint> hit_points;
};

struct ForwardsTimeDeviceArrays {
    double* start_hamiltonians;
    double* end_hamiltonians;
    TrajectoryPoint* final_states;
};

struct ForwardsTimeHostArrays{
     std::vector<double> start_hamiltonians;
    std::vector<double> end_hamiltonians;
    std::vector<TrajectoryPoint> final_states;
}

struct Result {
    double l1;
    double l2;
    double cost;
};

Result solve(double theta, double phi, double alpha);
