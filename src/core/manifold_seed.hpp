#pragma once

/// Stable-manifold linear map λ ≈ P x at the upright equilibrium, using the Hamiltonian
/// Hessian at the origin and the stable subspace of C = J_symp ∇²H(0). Stores row-major 2×2.
void stable_manifold_P(double alpha, double P[4]);
