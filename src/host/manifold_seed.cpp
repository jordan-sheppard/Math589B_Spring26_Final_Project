// Host-only: Eigen is used here to match cpp/solver/manifold_seed.cpp.
// Not included from any __device__ translation unit.

#include "manifold_seed.hpp"

#include <complex>
#include <stdexcept>

#include <Eigen/Dense>

namespace pendulum {

namespace {

Eigen::Matrix4d hessian_at_zero(double alpha) {
    Eigen::Matrix4d H = Eigen::Matrix4d::Zero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    H(1, 2) = 1.0;
    H(2, 1) = 1.0;
    H(0, 3) = 1.0;
    H(3, 0) = 1.0;
    H(1, 3) = -alpha;
    H(3, 1) = -alpha;
    H(3, 3) = -1.0;
    return H;
}

Eigen::Matrix4d canonical_J() {
    Eigen::Matrix4d J = Eigen::Matrix4d::Zero();
    J.block<2, 2>(0, 2) = Eigen::Matrix2d::Identity();
    J.block<2, 2>(2, 0) = -Eigen::Matrix2d::Identity();
    return J;
}

}  // namespace

void stable_manifold_seed_P(double alpha, double P[2][2]) {
    const Eigen::Matrix4d Hess = hessian_at_zero(alpha);
    const Eigen::Matrix4d J = canonical_J();
    const Eigen::Matrix4d C = J * Hess;

    Eigen::ComplexEigenSolver<Eigen::Matrix4d> ces(C);
    if (ces.info() != Eigen::Success) {
        throw std::runtime_error("eig failed for Hamiltonian matrix");
    }

    const Eigen::Vector4cd evals = ces.eigenvalues();
    const Eigen::Matrix4cd evecs = ces.eigenvectors();

    Eigen::Matrix<std::complex<double>, 4, 2> Vs;
    int cnt = 0;
    for (int i = 0; i < 4; ++i) {
        if (std::real(evals(i)) < 0.0) {
            if (cnt >= 2) {
                break;
            }
            Vs.col(cnt) = evecs.col(i);
            ++cnt;
        }
    }

    if (cnt != 2) {
        throw std::runtime_error("expected exactly 2 stable eigenvalues at origin");
    }

    const Eigen::Matrix2cd Vs1 = Vs.block<2, 2>(0, 0);
    const Eigen::Matrix2cd Vs2 = Vs.block<2, 2>(2, 0);
    const Eigen::Matrix2cd Pm = Vs2 * Vs1.inverse();
    const Eigen::Matrix2d Pr = Pm.real();

    P[0][0] = Pr(0, 0);
    P[0][1] = Pr(0, 1);
    P[1][0] = Pr(1, 0);
    P[1][1] = Pr(1, 1);
}

}  // namespace pendulum
