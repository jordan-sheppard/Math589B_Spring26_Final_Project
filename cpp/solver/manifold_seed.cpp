#include "manifold_seed.hpp"

#include <complex>
#include <stdexcept>

namespace pendulum {

namespace {

// Effective Hamiltonian at (theta,phi,l1,l2) after substituting u*.
// H = (1-cosθ) + 1/2 φ^2 - 1/2 l2^2 cos^2θ + l1 φ + l2 (sinθ - α φ)
//
// We only need Hessian at 0.
Eigen::Matrix4d hessianAtZero(double alpha) {
    // Ordering: [theta, phi, l1, l2]
    Eigen::Matrix4d H = Eigen::Matrix4d::Zero();

    // ∂²/∂theta² (1-cosθ) at 0 = cos(0)=1
    H(0, 0) = 1.0;

    // 1/2 φ^2 => second derivative wrt phi is 1
    H(1, 1) = 1.0;

    // l1 φ => mixed second derivatives: d/dl1 d/dphi = 1
    H(1, 2) = 1.0;
    H(2, 1) = 1.0;

    // l2 sinθ => mixed second derivatives: d/dl2 d/dtheta = cos(0)=1
    H(0, 3) = 1.0;
    H(3, 0) = 1.0;

    // l2 (-α φ) => mixed: d/dl2 d/dphi = -alpha
    H(1, 3) = -alpha;
    H(3, 1) = -alpha;

    // -1/2 l2^2 cos^2θ: at 0, cos^2=1 => second derivative wrt l2 is -1
    H(3, 3) = -1.0;

    return H;
}

Eigen::Matrix4d canonicalJ() {
    Eigen::Matrix4d J = Eigen::Matrix4d::Zero();
    J.block<2, 2>(0, 2) = Eigen::Matrix2d::Identity();
    J.block<2, 2>(2, 0) = -Eigen::Matrix2d::Identity();
    return J;
}

}  // namespace

Eigen::Matrix2d stableManifoldSeedP(double alpha) {
    const Eigen::Matrix4d Hess = hessianAtZero(alpha);
    const Eigen::Matrix4d J = canonicalJ();
    const Eigen::Matrix4d C = J * Hess;

    Eigen::ComplexEigenSolver<Eigen::Matrix4d> ces(C);
    if (ces.info() != Eigen::Success) {
        throw std::runtime_error("eig failed for Hamiltonian matrix");
    }

    const Eigen::Vector4cd evals = ces.eigenvalues();
    const Eigen::Matrix4cd evecs = ces.eigenvectors();

    // Pick the two eigenvectors with Re(λ) < 0.
    Eigen::Matrix<std::complex<double>, 4, 2> Vs;
    int cnt = 0;
    for (int i = 0; i < 4; ++i) {
        if (std::real(evals(i)) < 0.0) {
            if (cnt >= 2) break;
            Vs.col(cnt) = evecs.col(i);
            ++cnt;
        }
    }

    if (cnt != 2) {
        throw std::runtime_error("expected exactly 2 stable eigenvalues at origin");
    }

    const Eigen::Matrix2cd Vs1 = Vs.block<2, 2>(0, 0);
    const Eigen::Matrix2cd Vs2 = Vs.block<2, 2>(2, 0);

    const Eigen::Matrix2cd P = Vs2 * Vs1.inverse();
    // P should be real for this Hamiltonian structure; take real part.
    return P.real();
}

}  // namespace pendulum

