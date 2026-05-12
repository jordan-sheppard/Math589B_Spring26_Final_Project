#pragma once

// Shared GPU/CPU POD types for the **augmented** Pontryagin state used inside each shooting segment.
//
// Continuous-time objects: along a segment, (θ, φ, ℓ₁, ℓ₂) satisfy the first-order damped-pendulum
// Hamiltonian ODEs (θ̇ = φ, etc.; see `pendulum_oc.cuh`). Angles are radians; time is implicit in the
// driver’s choice of `IntegratorParams::dt` and `num_steps` (segment length ≈ `dt * num_steps` in
// the same time units as α). The scalar `cost` slot accumulates the **running** integrand of the
// objective (not the full Bolza functional until terminal terms are added elsewhere).
//
// `Mat4x4` stores the fundamental matrix block M(t) ∈ R^{4×4} with M(0) = I and Ṁ = A(t) M, where
// A = ∂(θ̇,φ̇,ℓ̇₁,ℓ̇₂)/∂(θ,φ,ℓ₁,ℓ₂) — i.e. M_ij = ∂x_i(t)/∂x_j(0) for the first four components x = (θ,φ,ℓ₁,ℓ₂).
// Storage is **row-major**: entry (r,c) lives at `data[r*4+c]` (C row index r, column c).

struct Mat4x4 {
    double data[16];

    __host__ __device__ double &operator()(int r, int c) { return data[r * 4 + c]; }
    __host__ __device__ const double &operator()(int r, int c) const { return data[r * 4 + c]; }

    __host__ __device__ Mat4x4 operator+(const Mat4x4 &other) const { return mat4_add(*this, other); }

    __host__ __device__ Mat4x4 operator-(const Mat4x4 &other) const { return mat4_sub(*this, other); }

    __host__ __device__ Mat4x4 operator*(const Mat4x4 &other) const { return mat4_matmul(*this, other); }

    __host__ __device__ Mat4x4 operator*(double scalar) const { return mat4_scale(*this, scalar); }

private:
    __host__ __device__ static Mat4x4 mat4_add(const Mat4x4 &a, const Mat4x4 &b) {
        Mat4x4 result;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = a.data[i] + b.data[i];
        }
        return result;
    }

    __host__ __device__ static Mat4x4 mat4_sub(const Mat4x4 &a, const Mat4x4 &b) {
        Mat4x4 result;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = a.data[i] - b.data[i];
        }
        return result;
    }

    __host__ __device__ static Mat4x4 mat4_matmul(const Mat4x4 &a, const Mat4x4 &b) {
        Mat4x4 result;
#pragma unroll
        for (int r = 0; r < 4; r++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                result(r, c) = a(r, 0) * b(0, c) + a(r, 1) * b(1, c) + a(r, 2) * b(2, c) + a(r, 3) * b(3, c);
            }
        }
        return result;
    }

    __host__ __device__ static Mat4x4 mat4_scale(const Mat4x4 &a, double scalar) {
        Mat4x4 result;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = a.data[i] * scalar;
        }
        return result;
    }
};

__host__ __device__ inline Mat4x4 operator*(double scalar, const Mat4x4 &M) { return M * scalar; }

struct VarState {
    double s[5]; // (θ, φ, ℓ₁, ℓ₂, running-cost integrand); first four are MS state/costate at current time
    Mat4x4 M;    // sensitivity of (θ,φ,ℓ₁,ℓ₂) w.r.t. their values at segment start — see `Mat4x4` note

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

    __host__ __device__ VarState operator+(const VarState &other) const { return var_add(*this, other); }

    __host__ __device__ VarState operator-(const VarState &other) const { return var_sub(*this, other); }

    __host__ __device__ VarState operator*(double scalar) const { return var_scale(*this, scalar); }

private:
    __host__ __device__ static VarState var_add(const VarState &a, const VarState &b) {
        VarState result;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            result.s[i] = a.s[i] + b.s[i];
        }
        result.M = a.M + b.M;
        return result;
    }

    __host__ __device__ static VarState var_sub(const VarState &a, const VarState &b) {
        VarState result;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            result.s[i] = a.s[i] - b.s[i];
        }
        result.M = a.M - b.M;
        return result;
    }

    __host__ __device__ static VarState var_scale(const VarState &a, double scalar) {
        VarState result;
#pragma unroll
        for (int i = 0; i < 5; i++) {
            result.s[i] = a.s[i] * scalar;
        }
        result.M = a.M * scalar;
        return result;
    }
};

__host__ __device__ inline VarState operator*(double scalar, const VarState &vec) { return vec * scalar; }

struct SystemParams {
    double alpha;              // viscous damping strength (same units as in the state ODE)
    double theta_init;
    double phi_init;
    double theta_goal;         // terminal θ_f; driver may add 2π wraps to pick a Riemann sheet
    double phi_goal;
    int num_shooting_intervals; // N in x ∈ R^{4N}; also number of segment IVPs in the standard MS mesh
};

struct IntegratorParams {
    double dt;            // RK4 increment in **continuous** time (fixed per segment in the driver)
    int num_steps;        // number of RK4 steps ⇒ segment length ΔT ≈ `dt * num_steps`
    bool backward_time;   // if true, IVP integrates toward decreasing time (warm-start / adjoint direction)
};

struct NewtonParams {
    int max_iterations;
    double tolerance; // typically ‖F‖∞ stopping tolerance on the discrete defect stack
};

struct SegmentEvaluation {
    VarState final_state;       // Φ_k(x_k): augmented state at segment end after `num_steps` RK4 steps
    double initial_hamiltonian; // H(x_k) at segment start — diagnostic; not part of the 4N defect unknowns
};

struct DeviceArrays {
    double *node_guesses;              // flat x ∈ R^{4N}; same (θ,φ,ℓ₁,ℓ₂)-per-knot layout as host
    SegmentEvaluation *segment_results; // length N; parallel write of Φ_k(x_k) from segment kernels
};

struct IterationLog {
    bool success = true;       // false if e.g. sparse LU factorization fails
    double max_defect_norm;    // ‖F(x)‖∞ — max component of stacked defects before the Newton correction
    double step_size_norm;     // ‖Δx‖₂ where Δx solves J Δx = −F in the same R^{4N} unknown ordering
};

struct Result {
    double optimal_l1_init;   // ℓ₁ at knot 0 — flat index 2 in x (first costate at initial shooting node)
    double optimal_l2_init;   // ℓ₂ at knot 0 — flat index 3
    double optimal_cost;      // Σ_k `final_state.cost()` at segment endpoints — MS bookkeeping of running cost
    int optimal_theta_wraps = 0;
    double final_theta_goal = 0.0; // θ_f including any 2π sheet offset used for this continuation solve
};

struct OptimizationResult {
    bool success;
    int num_iterations; // discrete Newton outer iterations on the MS defect system
    double final_error;   // ‖F‖∞ after the last iteration (same norm family as `IterationLog`)
    Result r;
};
