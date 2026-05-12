// Host-only linear algebra for the multiple-shooting Newton step on the CPU (Eigen).
//
// Mathematical objects (discrete MS unknowns live in R^{4N} with N = num_shooting_intervals):
// - `VectorXd` stacks residuals F(x) (defects between segment endpoints and node matching)
//   and/or Newton corrections Δx in the same component order as the flat node vector
//   [ (θ,φ,ℓ₁,ℓ₂) at knot 0 | … | (θ,φ,ℓ₁,ℓ₂) at knot N−1 ] — see `HDArrays` / `solver_types.cuh`.
// - `SparseMat` is the Jacobian ∂F/∂x in a sparse layout suitable for direct factorization;
//   row/column indices align with that same global unknown ordering (block structure follows
//   segment coupling, not changed here).
#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

typedef Eigen::SparseMatrix<double> SparseMat;
typedef Eigen::VectorXd VectorXd;
