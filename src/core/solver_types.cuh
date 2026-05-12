#pragma once

struct Mat4x4 {
    double data[16];

    __host__ __device__ double &operator()(int r, int c) { return data[r * 4 + c]; }
    __host__ __device__ const double &operator()(int r, int c) const { return data[r * 4 + c]; }

    __host__ __device__ Mat4x4 operator+(const Mat4x4 &other) const {
        Mat4x4 result;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = this->data[i] + other.data[i];
        }
        return result;
    }

    __host__ __device__ Mat4x4 operator-(const Mat4x4 &other) const {
        Mat4x4 result;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = this->data[i] - other.data[i];
        }
        return result;
    }

    __host__ __device__ Mat4x4 operator*(const Mat4x4 &other) const {
        Mat4x4 result;
#pragma unroll
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                result(r, c) = (*this)(r, 0) * other(0, c) + (*this)(r, 1) * other(1, c) +
                               (*this)(r, 2) * other(2, c) + (*this)(r, 3) * other(3, c);
            }
        }
        return result;
    }

    __host__ __device__ Mat4x4 operator*(double scalar) const {
        Mat4x4 result;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = this->data[i] * scalar;
        }
        return result;
    }
};

__host__ __device__ inline Mat4x4 operator*(double scalar, const Mat4x4 &M) { return M * scalar; }

struct VarState {
    double s[5];
    Mat4x4 M;

    __host__ __device__ double &theta() { return s[0]; }
    __host__ __device__ const double &theta() const { return s[0]; }

    __host__ __device__ double &phi() { return s[1]; }
    __host__ __device__ const double &phi() const { return s[1]; }

    __host__ __device__ double &l1() { return s[2]; }
    __host__ __device__ const double &l1() const { return s[2]; }

    __host__ __device__ double &l2() { return s[3]; }
    __host__ __device__ const double &l2() const { return s[3]; }

    __host__ __device__ double &cost() { return s[4]; }
    __host__ __device__ const double &cost() const { return s[4]; }

    __host__ __device__ VarState operator+(const VarState &other) const {
        VarState result;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            result.s[i] = this->s[i] + other.s[i];
        }
        result.M = this->M + other.M;
        return result;
    }

    __host__ __device__ VarState operator-(const VarState &other) const {
        VarState result;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            result.s[i] = this->s[i] - other.s[i];
        }
        result.M = this->M - other.M;
        return result;
    }

    __host__ __device__ VarState operator*(double scalar) const {
        VarState result;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            result.s[i] = this->s[i] * scalar;
        }
        result.M = this->M * scalar;
        return result;
    }
};

__host__ __device__ inline VarState operator*(double scalar, const VarState &vec) { return vec * scalar; }

struct SystemParams {
    double alpha;
    double theta_init;
    double phi_init;
    double theta_goal;
    double phi_goal;
    int num_shooting_intervals;
};

struct IntegratorParams {
    double dt;
    int num_steps;
    bool backward_time;
};

struct NewtonParams {
    int max_iterations;
    double tolerance;
};

struct SegmentEvaluation {
    VarState final_state;
    double initial_hamiltonian;
};

struct DeviceArrays {
    double *node_guesses;
    SegmentEvaluation *segment_results;
};

struct IterationLog {
    bool success = true;
    double max_defect_norm;
    double step_size_norm;
};

struct Result {
    double optimal_l1_init;
    double optimal_l2_init;
    double optimal_cost;
    int optimal_theta_wraps = 0;
    double final_theta_goal = 0.0;
};

struct OptimizationResult {
    bool success;
    int num_iterations;
    double final_error;
    Result r;
};
