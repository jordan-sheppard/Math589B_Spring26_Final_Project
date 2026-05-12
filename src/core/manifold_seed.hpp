#pragma once

/// Stable-manifold linear map λ ≈ P x at the upright equilibrium, using the Hamiltonian
/// Hessian at the origin and the stable subspace of C = J_symp ∇²H(0). Stores row-major 2×2.
void stable_manifold_P(double alpha, double P[4]);

/// Two real basis vectors for the local 2D stable subspace of the Hamiltonian linearization,
/// in phase coordinates [theta, phi, l1, l2]. Row-major 4×2 packed as B[8] = {B1; B2} columns.
void stable_manifold_basis(double alpha, double B[8]);
