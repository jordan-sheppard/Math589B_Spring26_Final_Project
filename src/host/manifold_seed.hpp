#pragma once

namespace pendulum {

// Stable manifold linearization l ≈ P x at the origin (2x2 row-major: P[row][col]).
void stable_manifold_seed_P(double alpha, double P[2][2]);

}  // namespace pendulum
