#include "core/manifold_seed.hpp"

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <cmath>

/// Ordering [θ, φ, λ₁, λ₂] for H after eliminating u (matches `compute_hamiltonian` in segment_integration.cuh).
static Eigen::Matrix4d hessian_H_at_origin(double alpha) {
    Eigen::Matrix4d H;
    H << 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, -alpha, 0.0, 1.0, 0.0, 0.0, 1.0, -alpha, 0.0, -1.0;
    return H;
}

void stable_manifold_P(double alpha, double P[4]) {
    Eigen::Matrix4d Hess = hessian_H_at_origin(alpha);

    Eigen::Matrix4d Js;
    Js << 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0;

    Eigen::Matrix4d C = Js * Hess;

    Eigen::ComplexEigenSolver<Eigen::Matrix4d> ces;
    ces.compute(C);

    const Eigen::Vector4cd &evals = ces.eigenvalues();
    const Eigen::Matrix4cd &evecs = ces.eigenvectors();

    // Build a real basis for the 2D stable subspace (Re λ < 0).
    Eigen::Matrix<double, 4, 4> W;
    W.setZero();
    int col = 0;

    for (int i = 0; i < 4 && col < 2; ++i) {
        std::complex<double> lam = evals(i);
        if (lam.real() >= -1e-10) {
            continue;
        }
        if (lam.imag() < -1e-10) {
            continue;
        }

        if (std::abs(lam.imag()) < 1e-10) {
            // Real eigenvector
            W.col(col++) = evecs.col(i).real();
        } else if (lam.imag() > 1e-10) {
            // Complex pair: use Re(v) and Im(v) as real basis
            Eigen::Vector4d vr = evecs.col(i).real();
            Eigen::Vector4d vi = evecs.col(i).imag();
            if (col < 2) {
                W.col(col++) = vr;
            }
            if (col < 2) {
                W.col(col++) = vi;
            }
        }
    }

    if (col < 2) {
        // Fallback: identity (degenerate numerics)
        P[0] = 1.0;
        P[1] = 0.0;
        P[2] = 0.0;
        P[3] = 1.0;
        return;
    }

    Eigen::Matrix2d Wtop = W.block<2, 2>(0, 0);
    Eigen::Matrix2d Wbot = W.block<2, 2>(2, 0);
    Eigen::Matrix2d Pm = Wbot * Wtop.fullPivLu().solve(Eigen::Matrix2d::Identity());

    P[0] = Pm(0, 0);
    P[1] = Pm(0, 1);
    P[2] = Pm(1, 0);
    P[3] = Pm(1, 1);
}
