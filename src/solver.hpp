#pragma once

#include <vector>
#include <cstdio>
#include <Eigen/Sparse>
#include <Eigen/Dense>

// Type aliases to keep signatures clean
typedef Eigen::SparseMatrix<double> SparseMat;
typedef Eigen::VectorXd VectorXd;

// ---- GPU Error Checking ----

// Macro that checks for any errors in a CUDA operation,
// and if it has one, captures the file name and line number where the error occurred
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// Evaluates any returned CUDA error code
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


struct Mat4x4 {
    double data[16];

    // --- 2D Accessors ---
    // Allows you to use M(row, col) instead of M[row * 4 + col]
    __host__ __device__ double& operator()(int r, int c) {
        return data[r * 4 + c];
    }
    __host__ __device__ const double& operator()(int r, int c) const {
        return data[r * 4 + c];
    }

    // --- Matrix Addition ---
    __host__ __device__
    Mat4x4 operator+(const Mat4x4& other) const {
        Mat4x4 result;
        #pragma unroll
        for(int i = 0; i < 16; i++) {
            result.data[i] = this->data[i] + other.data[i];
        }
        return result;
    }

    // --- Matrix Subtraction ---
    __host__ __device__
    Mat4x4 operator-(const Mat4x4& other) const {
        Mat4x4 result;
        #pragma unroll
        for(int i = 0; i < 16; i++) {
            result.data[i] = this->data[i] - other.data[i];
        }
        return result;
    }

    // --- Matrix Multiplication ---
    __host__ __device__
    Mat4x4 operator*(const Mat4x4& other) const {
        Mat4x4 result;
        
        // Force the CUDA compiler to flatten these loops into straight-line code for performance
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                result(r, c) = (*this)(r, 0) * other(0, c) + 
                               (*this)(r, 1) * other(1, c) + 
                               (*this)(r, 2) * other(2, c) + 
                               (*this)(r, 3) * other(3, c);
            }
        }
        return result;
    }

    // --- Scalar Multiplication --- 
    __host__ __device__ Mat4x4 operator*(double scalar) const {
        Mat4x4 result;
        #pragma unroll
        for(int i = 0; i < 16; i++) {
            result.data[i] = this->data[i] * scalar;
        }
        return result;
    }
};

// Makes scalar multiplication for Mat4x4 okay on both sides
__host__ __device__
inline Mat4x4 operator*(double scalar, const Mat4x4& M) {
    return M * scalar;
}


struct VarState {
    double s[5];     // 4D physical state [theta, phi, l1, l2] + cost
    Mat4x4 M;       // 4x4 sensitivity matrix to initial conditions

    // --- Inline Reference Accessors --
    __host__ __device__ double& theta() { return s[0]; }
    __host__ __device__ const double& theta() const { return s[0]; }

    __host__ __device__ double& phi() { return s[1]; }
    __host__ __device__ const double& phi() const { return s[1]; }

    __host__ __device__ double& l1() { return s[2]; }
    __host__ __device__ const double& l1() const { return s[2]; }

    __host__ __device__ double& l2() { return s[3]; }
    __host__ __device__ const double& l2() const { return s[3]; }

    __host__ __device__ double& cost() { return s[4]; }
    __host__ __device__ const double& cost() const { return s[4]; }

    // Overload Addition/Subtraction/Scalar Multiplication
    __host__ __device__ VarState operator+(const VarState& other) const {
        VarState result;

        // Add states
        #pragma unroll
        for(int i = 0; i < 5; i++) {
            result.s[i] = this->s[i] + other.s[i];
        }

        // Add sensitivity matrices
        result.M = this->M + other.M;
        return result;
    }

    __host__ __device__ VarState operator-(const VarState& other) const {
        VarState result;

        // Add states
        #pragma unroll
        for(int i = 0; i < 5; i++) {
            result.s[i] = this->s[i] - other.s[i];
        }

        // Add sensitivity matrices
        result.M = this->M - other.M;
        return result;
    }

    __host__ __device__ VarState operator*(double scalar) const {
        VarState result;

        // Scalar multiply states
        #pragma unroll
        for(int i = 0; i < 5; i++) {
            result.s[i] = this->s[i] * scalar;
        }

        // Scalar multiply matrices
        result.M = this->M * scalar;
        return result;
    }
};

// Makes scalar multiplication for VarState okay on both sides
__host__ __device__
inline VarState operator*(double scalar, const VarState& vec) {
    return vec * scalar;
}

// 1. Physics & Constraints
struct SystemParams {
    double alpha;
    double theta_init;
    double phi_init;
    int num_shooting_intervals;
};

// 2. Integration Grid (The RK4 settings)
struct IntegratorParams {
    double dt;
    int num_steps;
};

// 3. Root-Finding Settings (The Newton settings)
struct NewtonParams {
    int max_iterations;
    double tolerance;
};

// 4. What the GPU writes to global memory for segment 'k'
struct SegmentEvaluation {
    VarState final_state;        // Contains the final 5D state AND the 4x4 M matrix at end of segment
    double initial_hamiltonian;  // The energy constraint evaluated at the start of the segment
};

// 5. Device array addresses (for passing to the GPU)
struct DeviceArrays {
    double* node_guesses;
    SegmentEvaluation* segment_results;
};

// 6. To track the convergence history per iteration
struct IterationLog {
    bool success = true;            // Whether Newton step succeeded or not (default to true)
    double max_defect_norm;
    double step_size_norm;
};

// 7. Parameter results for things we're searching for
struct Result {
    double optimal_l1_init;
    double optimal_l2_init;
    double optimal_cost;
};

// 8. Output of iterative optimization method looking for a good result, above
struct OptimizationResult {
    // Newton optimization result
    bool success;
    int num_iterations;
    double final_error;
    Result r;
};



struct HDArrays {
    // 1. Host Memory (CPU)
    std::vector<double> h_node_guesses;
    std::vector<SegmentEvaluation> h_segment_results;

    // 2. Device Memory (GPU)
    double* d_node_guesses;
    SegmentEvaluation* d_segment_results;

    // 3. Constructor: Allocates CPU and GPU memory ONCE
    HDArrays(int num_intervals) {
        int num_states = num_intervals * 4;

        // Size the CPU vectors
        h_node_guesses.resize(num_states, 0.0);
        h_segment_results.resize(num_intervals);

        // Allocate the GPU memory
        gpuErrchk(cudaMalloc(&d_node_guesses, num_states * sizeof(double)));
        gpuErrchk(cudaMalloc(&d_segment_results, num_intervals * sizeof(SegmentEvaluation)));
    }

    // 4. Memory Transfer Methods
    void copy_guesses_to_device() {
        size_t bytes = h_node_guesses.size() * sizeof(double);
        gpuErrchk(cudaMemcpy(d_node_guesses, h_node_guesses.data(), bytes, cudaMemcpyHostToDevice));
    }

    void copy_results_to_host() {
        size_t bytes = h_segment_results.size() * sizeof(SegmentEvaluation);
        gpuErrchk(cudaMemcpy(h_segment_results.data(), d_segment_results, bytes, cudaMemcpyDeviceToHost));
    }

    DeviceArrays get_device_arrays() const {
        return {d_node_guesses, d_segment_results};
    }

    // 5. Destructor: Automatically frees GPU memory when the struct is destroyed
    ~HDArrays() {
        gpuErrchk(cudaFree(d_node_guesses));
        gpuErrchk(cudaFree(d_segment_results));
    }
};

// ---- Physics/derivative computation functions ----

// Computes the state derivatives/physics ds/dt (along with cost)
__host__ __device__ void compute_state_physics(
    const VarState& state,
    const SystemParams& params,
    VarState& ds
);

// Computes the Jacobian matrix A(s) for the sensitivities M, such that dM/dt = A(s) * M
__host__ __device__ Mat4x4 compute_sensitivity_jacobian(
    const VarState& state,
    const SystemParams& params
);

// Computes the derivatives for the entire system (states AND sensitivities)
__host__ __device__ VarState get_derivatives(
    const VarState& state,
    const SystemParams& params
);


// ---- Integration functions ----

// Evaluates the Hamiltonian energy constraint at a specific state
__host__ __device__ double compute_hamiltonian(
    const VarState& state,
    const SystemParams& params
);

// Takes a single microscopic RK4 step, updating both the 5D state and 4x4 sensitivity matrix
__host__ __device__ VarState rk4_step(
    const VarState& current,
    const SystemParams& params,
    double dt
);

// Loops rk4_step over the micro-grid and packages the final state/matrix + initial Hamiltonian
__host__ __device__ SegmentEvaluation simulate_segment(
    const VarState& initial_guess,
    const SystemParams& sys_params,
    const IntegratorParams& int_params
);

// ---- GPU Kernel ----

// Thread k reads the guess for node k, integrates the segment, and writes to segment_results[k]
__global__ void multiple_shooting_kernel(
    DeviceArrays d,
    SystemParams sys_params, 
    IntegratorParams int_params
);

 
// Launches the GPU kernel, waits for completion, and copies SegmentEvaluations back to the CPU
void evaluate_segments_on_gpu(
    HDArrays& solver_arrays, 
    const SystemParams& sys_params, 
    const IntegratorParams& int_params
);

// Translates the SegmentEvaluations into the global defect vector F and sparse Jacobian J
void build_global_system(
    const HDArrays& solver_arrays,
    const SystemParams& sys_params,
    SparseMat& J,
    VectorXd& F
);

// Executes a single iteration: 
// 1. Evaluates segments on GPU 
// 2. Builds J and F 
// 3. Solves J * dS = -F using Eigen::SparseLU 
// 4. Updates guesses array
IterationLog compute_newton_step(
    HDArrays& solver_arrays,            // Modified in place
    const SystemParams& sys_params, 
    const IntegratorParams& int_params
);

// Generates an initial guess (currently uses LQR and linear interpolation...)
std::vector<SegmentEvaluation> compute_linear_initial_guess(
    SystemParams sys_params
);

// The core loop that calls compute_newton_step until tolerance is met or max_iters is reached
// given an initial guess for the initial conditions of each segment
OptimizationResult solve_multiple_shooting(
    std::vector<double>& flat_node_guesses, // Input guesses (from LQR or Continuation)
    SystemParams sys_params, 
    IntegratorParams int_params, 
    NewtonParams newton_params
);

// An overloaded version of the core loop that has no initial guess, and computes
// a linear initial guess interpolating the initial and final states before
// then running the newton iteration.
OptimizationResult solve_multiple_shooting(
    SystemParams sys_params, 
    IntegratorParams int_params, 
    NewtonParams newton_params
);

// Main driver; guesses optimal initial costates and costs for given initial
// theta and phi, and friction parameter alpha, and runs continuation on it.
Result solve(double target_theta, double target_phi, double alpha);
