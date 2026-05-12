#pragma once

// =============================================================================
// Optimal-control dynamics (Pontryagin first-order necessary conditions)
// =============================================================================
//
// Physical angle and angular rate (first-order state for the pendulum):
//   theta  = θ,   phi = φ = dθ/dt.
//
// Damping coefficient (parameter, not a state component):
//   params.alpha = α  (appears in φ̇ and in λ̇₂ below).
//
// Co-states (costates for θ and φ; stored as `l1`, `l2` in `VarState`):
//   l1 = λ₁  adjoins θ,   l2 = λ₂  adjoins φ.
//
// Control model (algebraic; eliminated in closed form in this file):
//   Scalar control u enters the φ-equation scaled by cos θ and appears quadratically in the
//   running cost. Minimizing the pointwise Hamiltonian w.r.t. u gives u = −λ₂ cos θ, which is
//   what is already substituted into the RHS below (there is no separate u variable in code).
//
// Running cost integrand L (accumulated in `VarState::cost()` as ∫ L dt by the integrator):
//   L(θ, φ, λ₂) = (1 − cos θ) + ½ φ² + ½ λ₂² cos²θ
//   i.e. `ds.cost()` below is L(θ, φ, l2). The first two terms are a “lift” potential and
//   kinetic penalty in φ; the last is ½ u² with u = −λ₂ cos θ.
//
// State ODE (same four components as `compute_state_physics` writes into `ds`):
//   θ̇ = φ
//   φ̇ = sin θ − α φ − λ₂ cos²θ     (= sin θ − α φ + u cos θ with u = −λ₂ cos θ)
//   λ̇₁ = −λ₂² cos θ sin θ − λ₂ cos θ − sin θ   (= −∂H/∂θ for the minimized Hamiltonian H)
//   λ̇₂ = −φ − λ₁ + α λ₂                        (= −∂H/∂φ)
//
// Minimized Hamiltonian H(θ, φ, λ₁, λ₂) (used for diagnostics in `compute_hamiltonian` in
// `segment_integration.cuh`, not computed here):
//   H = (1 − cos θ) + ½ φ² − ½ λ₂² cos²θ + λ₁ φ + λ₂ (sin θ − α φ).
//
// First-order form used by RK4 / augmented integrators:
//   The propagator treats y = (θ, φ, λ₁, λ₂, ∫L ds) as an ODE in ℝ⁵ with `VarState::s[0..3]`
//   for (theta, phi, l1, l2) and `VarState::cost()` for the running-cost accumulator.
//   `compute_state_physics` supplies only the explicit part (θ̇, φ̇, λ̇₁, λ̇₂, L); sensitivity
//   propagation for ∂(θ,φ,λ₁,λ₂)/∂(θ,φ,λ₁,λ₂)_0 is added in `get_derivatives` via `M`.
//
// Sensitivity / variational equation:
//   Let x = (θ, φ, λ₁, λ₂)ᵀ. With ẋ = f(x), define M(t) = ∂x(t)/∂x(0) (4×4). Then
//     dM/dt = A(x) M,   A_ij = ∂f_i/∂x_j,
//   implemented as `Mat4x4 A = compute_sensitivity_jacobian(...)` and `ds.M = A * state.M`.
//   Rows/columns of `M` align with state components `(theta, phi, l1, l2)` in `solver_types.cuh`.

#include <cmath>

#include "core/solver_types.cuh"

namespace {

/// Shared sin/cos powers for the OC RHS and its sensitivity matrix.
struct OcTrigLocals {
    double sin_t;
    double cos_t;
    double cos_t_sq;
    double sin_t_sq;
};

__host__ __device__ inline OcTrigLocals oc_trig_locals(double theta) {
    OcTrigLocals tr;
    tr.sin_t = sin(theta);
    tr.cos_t = cos(theta);
    tr.cos_t_sq = tr.cos_t * tr.cos_t;
    tr.sin_t_sq = tr.sin_t * tr.sin_t;
    return tr;
}

__host__ __device__ inline void oc_write_physics_ds(const OcTrigLocals &tr, double phi, double l1,
                                                    double l2, double alpha, VarState &ds) {
    double l2_sq = l2 * l2;
    double phi_sq = phi * phi;

    // θ̇ = φ
    ds.theta() = phi;
    // φ̇ = sin θ − α φ − λ₂ cos²θ  (control already eliminated: u = −λ₂ cos θ)
    ds.phi() = tr.sin_t - alpha * phi - l2 * tr.cos_t_sq;
    // λ̇₁ = −∂H/∂θ with H the minimized Hamiltonian; depends on (θ, λ₂) only through L and f.
    ds.l1() = -l2_sq * tr.cos_t * tr.sin_t - l2 * tr.cos_t - tr.sin_t;
    // λ̇₂ = −∂H/∂φ = −φ − λ₁ + α λ₂
    ds.l2() = -phi - l1 + alpha * l2;

    // L = (1 − cos θ) + ½ φ² + ½ λ₂² cos²θ  (integrand for `cost()`)
    ds.cost() = 1.0 - tr.cos_t + 0.5 * phi_sq + 0.5 * l2_sq * tr.cos_t_sq;
}

__host__ __device__ inline Mat4x4 oc_build_sensitivity_A(const OcTrigLocals &tr, double alpha, double l2) {
    double l2_sq = l2 * l2;

    Mat4x4 A;

    // Row 0: ∂(θ̇)/∂(θ, φ, λ₁, λ₂) with θ̇ = φ  →  [0, 1, 0, 0]
    A(0, 0) = 0.;
    A(0, 1) = 1.;
    A(0, 2) = 0.;
    A(0, 3) = 0.;

    // Row 1: ∂(φ̇)/∂x for φ̇ = sin θ − α φ − λ₂ cos²θ
    A(1, 0) = tr.cos_t + 2.0 * l2 * tr.cos_t * tr.sin_t;
    A(1, 1) = -alpha;
    A(1, 2) = 0.;
    A(1, 3) = -tr.cos_t_sq;

    // Row 2: ∂(λ̇₁)/∂x for λ̇₁ = −λ₂² cos θ sin θ − λ₂ cos θ − sin θ
    A(2, 0) = -(l2_sq * (tr.cos_t_sq - tr.sin_t_sq) - l2 * tr.sin_t + tr.cos_t);
    A(2, 1) = 0.;
    A(2, 2) = 0.;
    A(2, 3) = -(2.0 * l2 * tr.cos_t * tr.sin_t + tr.cos_t);

    // Row 3: ∂(λ̇₂)/∂x for λ̇₂ = −φ − λ₁ + α λ₂  →  [0, −1, −1, α]
    A(3, 0) = 0.;
    A(3, 1) = -1.;
    A(3, 2) = -1.;
    A(3, 3) = alpha;

    return A;
}

} // namespace

/// Time derivative of the Pontryagin state–cost pair: (θ̇, φ̇, λ̇₁, λ̇₂) and running cost L into `ds`.
/// Does not update `M`; see `get_derivatives` for the coupled sensitivity ODE dM/dt = A M.
__host__ __device__ inline void compute_state_physics(const VarState &state, const SystemParams &params,
                                                      VarState &ds) {
    OcTrigLocals tr = oc_trig_locals(state.theta());
    oc_write_physics_ds(tr, state.phi(), state.l1(), state.l2(), params.alpha, ds);
}

/// Jacobian A = ∂f/∂x with f = (θ̇, φ̇, λ̇₁, λ̇₂)ᵀ and x = (θ, φ, λ₁, λ₂)ᵀ = (theta, phi, l1, l2)ᵀ.
/// Entry A(r,c) is ∂(component r of f)/∂(component c of x); used in dM/dt = A M for `state.M`.
__host__ __device__ inline Mat4x4 compute_sensitivity_jacobian(const VarState &state,
                                                                const SystemParams &params) {
    OcTrigLocals tr = oc_trig_locals(state.theta());
    return oc_build_sensitivity_A(tr, params.alpha, state.l2());
}

namespace {

/// Augmented RHS: physics slot derivatives (optionally flipped by `flow_sign`) and Ṁ = flow_sign · A M.
__host__ __device__ inline VarState oc_augmented_rhs(const VarState &state, const SystemParams &params,
                                                     double flow_sign) {
    VarState ds;

    compute_state_physics(state, params, ds);
    ds.theta() *= flow_sign;
    ds.phi() *= flow_sign;
    ds.l1() *= flow_sign;
    ds.l2() *= flow_sign;
    ds.cost() *= flow_sign;

    Mat4x4 A = compute_sensitivity_jacobian(state, params);
    ds.M = flow_sign * (A * state.M);

    return ds;
}

} // namespace

/// Full augmented RHS for forward time: `compute_state_physics` plus variational block dM/dt = A(x) M.
/// Here M(t) = ∂x(t)/∂x(0) with x = (theta, phi, l1, l2); initial M = I is set in the integrator.
__host__ __device__ inline VarState get_derivatives(const VarState &state, const SystemParams &params) {
    return oc_augmented_rhs(state, params, 1.0);
}

/// Same as `get_derivatives` but for an arbitrary time direction: τ = flow_sign · t.
/// With flow_sign = −1 (backward IVP), ẋ and dM/dτ both pick up a minus so the chain rule for
/// ∂x/∂x₀ along reversed τ matches negating the forward generator A.
__host__ __device__ inline VarState get_derivatives_flow(const VarState &state, const SystemParams &params,
                                                         double flow_sign) {
    return oc_augmented_rhs(state, params, flow_sign);
}
