#pragma once

#include <Eigen/Dense>

namespace pendulum {

// Computes a 2x2 matrix P such that near the origin on the stable manifold,
//   lambda ≈ P * x,  where x=[theta, phi]^T and lambda=[l1, l2]^T.
//
// This follows the course PDF approach:
// 1) Build Hamiltonian matrix C = J * Hess(H) at 0.
// 2) Take stable eigenvectors Vs (eigs with Re(λ) < 0).
// 3) Partition Vs = [Vs1; Vs2] with Vs1,V2 each 2xs.
// 4) P = Vs2 * inv(Vs1).
Eigen::Matrix2d stableManifoldSeedP(double alpha);

}  // namespace pendulum

