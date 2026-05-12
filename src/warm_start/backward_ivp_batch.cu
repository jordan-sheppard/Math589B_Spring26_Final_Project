#include "warm_start/backward_ivp_warmstart.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <vector>

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "cuda/gpu_macros.cuh"
#include "warm_start/backward_ivp_common.cuh"

__constant__ double c_ws_v1[4];
__constant__ double c_ws_v2[4];

namespace {

bool fill_log_radii(double *out, int n) {
    if (n <= 0) {
        return false;
    }
    const double lo = 1e-10;
    const double hi = 1e-3;
    if (n == 1) {
        out[0] = hi;
        return true;
    }
    for (int i = 0; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n - 1);
        out[i] = std::exp(std::log(lo) + t * (std::log(hi) - std::log(lo)));
    }
    return true;
}

/// Two independent directions for the two eigenvalues with smallest real part (stable subspace approximation).
bool compute_stable_plane_vectors(const SystemParams &sys, double v1[4], double v2[4]) {
    VarState eq;
    eq.theta() = sys.theta_goal;
    eq.phi() = sys.phi_goal;
    eq.l1() = 0.0;
    eq.l2() = 0.0;

    Mat4x4 AJ = compute_sensitivity_jacobian(eq, sys);
    Eigen::Matrix4d A;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            A(r, c) = AJ(r, c);
        }
    }

    Eigen::EigenSolver<Eigen::Matrix4d> es(A);
    if (es.info() != Eigen::Success) {
        return false;
    }

    std::array<int, 4> perm{0, 1, 2, 3};
    std::sort(perm.begin(), perm.end(), [&](int i, int j) {
        return es.eigenvalues()(i).real() < es.eigenvalues()(j).real();
    });

    Eigen::Vector4d u0, u1;
    bool ok = false;

    const int i0 = perm[0];
    const std::complex<double> lam0 = es.eigenvalues()(i0);
    const Eigen::Vector4cd vec0 = es.eigenvectors().col(i0);

    if (std::abs(lam0.imag()) < 1e-12) {
        u0 = vec0.real();
        if (u0.norm() < 1e-14) {
            return false;
        }
        u0.normalize();

        for (int t = 1; t < 4; ++t) {
            const int j = perm[t];
            const std::complex<double> lamj = es.eigenvalues()(j);
            const Eigen::Vector4cd vecj = es.eigenvectors().col(j);
            if (std::abs(lamj.imag()) < 1e-12) {
                u1 = vecj.real();
                u1 -= u0 * u0.dot(u1);
                if (u1.norm() > 1e-10) {
                    u1.normalize();
                    ok = true;
                    break;
                }
            }
        }
        if (!ok) {
            for (int t = 1; t < 4; ++t) {
                const int j = perm[t];
                if (std::abs(es.eigenvalues()(j).imag()) > 1e-12) {
                    u1 = es.eigenvectors().col(j).imag();
                    u1 -= u0 * u0.dot(u1);
                    if (u1.norm() > 1e-10) {
                        u1.normalize();
                        ok = true;
                        break;
                    }
                }
            }
        }
    } else {
        u0 = vec0.real();
        u1 = vec0.imag();
        if (u0.norm() < 1e-14 || u1.norm() < 1e-14) {
            return false;
        }
        u0.normalize();
        u1 -= u0 * u0.dot(u1);
        if (u1.norm() < 1e-10) {
            return false;
        }
        u1.normalize();
        ok = true;
    }

    if (!ok) {
        return false;
    }
    for (int k = 0; k < 4; ++k) {
        v1[k] = u0[k];
        v2[k] = u1[k];
    }
    return true;
}

__global__ void backward_warm_start_score_kernel(SystemParams sys, double dt, int num_intervals, int steps_per_interval,
                                                 const double *r_tab, double *scores) {
    const int seed = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = warm_start::total_seed_count();
    if (seed >= total) {
        return;
    }

    int ir = 0;
    int ia = 0;
    int ib = 0;
    warm_start::decode_seed_index(seed, ir, ia, ib);
    const double r = r_tab[ir];
    double a = 0.0;
    double b = 0.0;
    warm_start::ab_from_grid(ia, ib, r, a, b);

    scores[seed] = warm_start::backward_ivp_to_ms_guess(sys, dt, num_intervals, steps_per_interval, a, b, c_ws_v1,
                                                         c_ws_v2, nullptr);
}

} // namespace

std::vector<double> compute_backward_eigen_ms_warm_start(const SystemParams &sys, const IntegratorParams &int_params) {
    double v1[4];
    double v2[4];
    if (!compute_stable_plane_vectors(sys, v1, v2)) {
        return {};
    }

    const int N = sys.num_shooting_intervals;
    const int steps = int_params.num_steps;
    if (N <= 0 || steps <= 0) {
        return {};
    }

    std::array<double, warm_start::kRGridCount> r_host{};
    if (!fill_log_radii(r_host.data(), warm_start::kRGridCount)) {
        return {};
    }

    const int num_seeds = warm_start::total_seed_count();

    double *d_scores = nullptr;
    double *d_r_tab = nullptr;
    gpuErrchk(cudaMalloc(&d_scores, static_cast<size_t>(num_seeds) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_r_tab, static_cast<size_t>(warm_start::kRGridCount) * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_r_tab, r_host.data(), warm_start::kRGridCount * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_ws_v1, v1, sizeof(double) * 4));
    gpuErrchk(cudaMemcpyToSymbol(c_ws_v2, v2, sizeof(double) * 4));

    const int threads = 256;
    const int blocks = (num_seeds + threads - 1) / threads;
    backward_warm_start_score_kernel<<<blocks, threads>>>(sys, int_params.dt, N, steps, d_r_tab, d_scores);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    std::vector<double> h_scores(static_cast<size_t>(num_seeds));
    gpuErrchk(cudaMemcpy(h_scores.data(), d_scores, static_cast<size_t>(num_seeds) * sizeof(double),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_scores));
    gpuErrchk(cudaFree(d_r_tab));

    int best_seed = -1;
    double best = std::numeric_limits<double>::infinity();
    for (int s = 0; s < num_seeds; ++s) {
        const double sc = h_scores[static_cast<size_t>(s)];
        if (!std::isfinite(sc)) {
            continue;
        }
        if (sc < best) {
            best = sc;
            best_seed = s;
        }
    }

    if (best_seed < 0 || !std::isfinite(best)) {
        return {};
    }

    int ir = 0;
    int ia = 0;
    int ib = 0;
    warm_start::decode_seed_index(best_seed, ir, ia, ib);
    const double r = r_host[static_cast<size_t>(ir)];
    double a = 0.0;
    double b = 0.0;
    warm_start::ab_from_grid(ia, ib, r, a, b);

    std::vector<double> traj(static_cast<size_t>(4 * N));
    warm_start::backward_ivp_to_ms_guess(sys, int_params.dt, N, steps, a, b, v1, v2, traj.data());

    return traj;
}
