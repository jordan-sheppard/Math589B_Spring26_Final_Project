#pragma once

#include <cstddef> // For std::size_t


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

// This is exactly what the GPU writes to global memory for segment 'k'
struct SegmentEvaluation {
    VarState final_state;        // Contains the final 5D state AND the 4x4 M matrix at end of segment
    double initial_hamiltonian;  // The energy constraint evaluated at the start of the segment
};

// To track the convergence history per iteration
struct IterationLog {
    int iteration_number;
    double max_defect_norm;
    double step_size_norm;
};

// Parameter results for things we're searching for
struct Result {
    double optimal_l1_init;
    double optimal_l2_init;
    double optimal_cost;
};

// Output of iterative optimization method looking for a good result, above
struct OptimizationResult {
    // Newton optimization result
    bool success;
    int num_iterations;
    double final_error;
    Result r;
};




Result solve(double theta, double phi, double alpha);
