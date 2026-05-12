#include "warm_start/backward_ivp_warmstart.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <vector>

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "cuda/gpu_macros.cuh"
#include "warm_start/backward_ivp_common.cuh"

namespace {

void fill_linearization_at_origin(double alpha, Eigen::Matrix4d &A) {
    A << 0.0, 1.0, 0.0, 0.0, 1.0, -alpha, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, -1.0, alpha;
}

bool stable_columns_from_A(const Eigen::Matrix4d &A, double col0[4], double col1[4]) {
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
        col0[k] = u0[k];
        col1[k] = u1[k];
    }
    return true;
}

void build_well_shifts(double theta_init, std::vector<int> &out) {
    const double two_pi = 2.0 * acos(-1.0);
    const int k_round = static_cast<int>(std::lround(theta_init / two_pi));
    const int arr[] = {k_round, 0, k_round - 1, k_round + 1, k_round - 2, k_round + 2};
    out.clear();
    for (int k : arr) {
        if (std::find(out.begin(), out.end(), k) == out.end()) {
            out.push_back(k);
        }
    }
}

__constant__ double c_patch_col0[4];
__constant__ double c_patch_col1[4];

__global__ void patch_score_kernel(double *scores, SystemParams sys_alpha_only, int n_wells, const int *d_wells,
                                   int n_rad, const double *d_radii, double theta_init_base, double phi_init_base,
                                   double dt, int num_intervals, int steps_per_interval) {
    const int grid_n = warm_start::kPatchGrid;
    const int nij = grid_n * grid_n;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_wells * n_rad * nij;
    if (idx >= total) {
        return;
    }

    int tmp = idx;
    const int ij = tmp % nij;
    tmp /= nij;
    const int ir = tmp % n_rad;
    tmp /= n_rad;
    const int iw = tmp;

    const int j = ij % grid_n;
    const int i = ij / grid_n;
    const double radius = d_radii[ir];
    double a = 0.0;
    double b = 0.0;
    warm_start::patch_ab_from_ij(i, j, radius, a, b);

    const double two_pi = 6.28318530717958647692;
    const double theta_tgt = theta_init_base - two_pi * static_cast<double>(d_wells[iw]);

    VarState x;
    warm_start::origin_patch_state(a, b, c_patch_col0, c_patch_col1, x);

    const int total_steps = num_intervals * steps_per_interval;
    int ok = 1;
    for (int s = 1; s <= total_steps; ++s) {
        x = warm_start::rk4_step_physics_only(x, sys_alpha_only, -dt);
        if (!isfinite(x.theta()) || !isfinite(x.phi()) || !isfinite(x.l1()) || !isfinite(x.l2()) ||
            fabs(x.theta()) > 1.0e9 || fabs(x.phi()) > 1.0e9 || fabs(x.l1()) > 1.0e9 || fabs(x.l2()) > 1.0e9) {
            ok = 0;
            break;
        }
    }

    if (ok) {
        scores[idx] = warm_start::dist2_wrapped(x.theta(), x.phi(), theta_tgt, phi_init_base);
    } else {
        scores[idx] = 1.0e300;
    }
}

void decode_patch_index(int idx, int grid_n, int n_rad, int &iw, int &ir, int &i, int &j) {
    const int nij = grid_n * grid_n;
    int tmp = idx;
    const int ij = tmp % nij;
    tmp /= nij;
    ir = tmp % n_rad;
    tmp /= n_rad;
    iw = tmp;
    j = ij % grid_n;
    i = ij / grid_n;
}

} // namespace

std::vector<std::vector<double>> compute_patch_topk_ms_warm_starts(const SystemParams &sys,
                                                                   const IntegratorParams &int_params,
                                                                   int top_k) {
    std::vector<std::vector<double>> out;

    double col0[4];
    double col1[4];
    Eigen::Matrix4d A;
    fill_linearization_at_origin(sys.alpha, A);
    if (!stable_columns_from_A(A, col0, col1)) {
        return out;
    }

    const int N = sys.num_shooting_intervals;
    const int steps = int_params.num_steps;
    if (N <= 0 || steps <= 0 || top_k <= 0) {
        return out;
    }

    std::vector<int> wells;
    build_well_shifts(sys.theta_init, wells);
    const int n_wells = static_cast<int>(wells.size());
    const int grid_n = warm_start::kPatchGrid;
    const int nij = grid_n * grid_n;

    static const double kRadiiHost[] = {1.0e-10, 3.0e-10, 1.0e-9,  3.0e-9,  1.0e-8,  3.0e-8, 1.0e-7,  3.0e-7,
                                        1.0e-6,  3.0e-6,  1.0e-5,  3.0e-5,  1.0e-4,  3.0e-4,  1.0e-3};
    const int n_rad = static_cast<int>(sizeof(kRadiiHost) / sizeof(kRadiiHost[0]));

    const int total = n_wells * n_rad * nij;

    SystemParams sys_kernel = sys;
    sys_kernel.theta_init = 0.0;
    sys_kernel.phi_init = 0.0;
    sys_kernel.theta_goal = 0.0;
    sys_kernel.phi_goal = 0.0;
    sys_kernel.num_shooting_intervals = N;

    int *d_wells = nullptr;
    double *d_radii = nullptr;
    double *d_scores = nullptr;
    gpuErrchk(cudaMalloc(&d_wells, static_cast<size_t>(n_wells) * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_radii, static_cast<size_t>(n_rad) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_scores, static_cast<size_t>(total) * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_wells, wells.data(), static_cast<size_t>(n_wells) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_radii, kRadiiHost, static_cast<size_t>(n_rad) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_patch_col0, col0, sizeof(double) * 4));
    gpuErrchk(cudaMemcpyToSymbol(c_patch_col1, col1, sizeof(double) * 4));

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    patch_score_kernel<<<blocks, threads>>>(d_scores, sys_kernel, n_wells, d_wells, n_rad, d_radii, sys.theta_init,
                                              sys.phi_init, int_params.dt, N, steps);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    std::vector<double> h_scores(static_cast<size_t>(total));
    gpuErrchk(cudaMemcpy(h_scores.data(), d_scores, static_cast<size_t>(total) * sizeof(double),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_scores));
    gpuErrchk(cudaFree(d_radii));
    gpuErrchk(cudaFree(d_wells));

    std::vector<int> order(static_cast<size_t>(total));
    std::iota(order.begin(), order.end(), 0);
    const int k_take = std::min(top_k, total);
    std::partial_sort(order.begin(), order.begin() + k_take, order.end(), [&](int a, int b) {
        return h_scores[static_cast<size_t>(a)] < h_scores[static_cast<size_t>(b)];
    });

    SystemParams sys_alpha = sys_kernel;
    sys_alpha.alpha = sys.alpha;

    for (int t = 0; t < k_take; ++t) {
        const int idx = order[static_cast<size_t>(t)];
        int iw = 0;
        int ir = 0;
        int ii = 0;
        int jj = 0;
        decode_patch_index(idx, grid_n, n_rad, iw, ir, ii, jj);
        const double radius = kRadiiHost[ir];
        double a = 0.0;
        double b = 0.0;
        warm_start::patch_ab_from_ij(ii, jj, radius, a, b);
        const double two_pi = 2.0 * acos(-1.0);
        const double theta_tgt = sys.theta_init - two_pi * static_cast<double>(wells[static_cast<size_t>(iw)]);

        std::vector<double> traj(static_cast<size_t>(4 * N));
        warm_start::origin_patch_backward_to_targets(sys_alpha, int_params.dt, N, steps, a, b, col0, col1, theta_tgt,
                                                      sys.phi_init, traj.data());
        out.push_back(std::move(traj));
    }

    return out;
}
