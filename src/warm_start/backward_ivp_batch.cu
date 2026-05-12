#include "warm_start/backward_ivp_warmstart.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include "cuda/gpu_macros.cuh"
#include "warm_start/backward_ivp_common.cuh"

// -----------------------------------------------------------------------------
// Backward IVP patch search for multiple-shooting warm starts
// -----------------------------------------------------------------------------
//
// Motivation.  Multiple shooting with Newton needs a trajectory-shaped initial
// guess for the nodal states (theta, phi, lambda1, lambda2) on each subinterval.
// Near the upright equilibrium the stable manifold of the Hamiltonian saddle is
// approximately a two-dimensional subspace in R^4.  We parameterize small initial
// states in that plane (coefficients a,b), integrate the *state* ODE backward in
// time (negative dt) with the optimal feedback u* = -lambda2 cos(theta) already
// substituted (same RHS as forward Hamiltonian flow, "physics-only" in code), and
// score how close the terminal (theta, phi) is to a target after accounting for
// 2*pi wraps in theta.  Cheap parallel candidates on the GPU identify good (a,b);
// the host then replays the same RK4 backward path to emit subsampled MS knots.
//
// Wells / wraps.  Only (theta, phi) enter the terminal mismatch; theta lives on a
// circle, so a physical target is an equivalence class theta ~ theta + 2πk.  The
// search enumerates a small set of integer shifts k ("wells") so each candidate
// compares its terminal angle to theta_init − 2πk (and phi_init) — covering the
// principal lift of theta_init and neighboring windings so the warm start is not
// trapped in the wrong 2π sheet relative to the stored initial data.
//
// Scoring metric.  Each thread evaluates the squared wrapped distance
// d^2(theta,phi) on S^1 × R (see dist2_wrapped): a nonnegative proxy for terminal
// angle error, zero iff the backward endpoint matches the chosen lift exactly in
// both coordinates.  Non-finite or exploded states receive a large sentinel so
// they never rank among the smallest scores.
//
// Grid / radii.  For each radius r, (a,b) sample the square [−r,r]^2 in coefficient
// space (tensor product grid).  Multiple exponentially spaced r explore from the
// linear regime outward before the tangent-plane model of W^s becomes inaccurate.
//
// Top-K selection.  After a dense GPU evaluation, indices are partially sorted so
// the K smallest scores appear first — an O(N log K) selection of approximate
// global minimizers over the discrete search space; each is replayed on the host
// to recover full nodal trajectories for the solver.
//
// This file does *not* run Newton on the patch; Newton is applied later by the
// driver when solving the shooting defects.  See README "Backward IVP warm start".
// -----------------------------------------------------------------------------

namespace {

// Jacobian Df(0) of the closed-loop Hamiltonian state field at the origin
// (theta,phi,l1,l2) = 0 for the current alpha.  Hyperbolic structure: two
// eigenvalues with Re λ < 0 span the stable subspace E^s used for the patch.
// Ordering matches VarState / code.
void fill_linearization_at_origin(double alpha, Eigen::Matrix4d &A) {
    A << 0.0, 1.0, 0.0, 0.0, 1.0, -alpha, 0.0, -1.0, -1.0, 0.0, 0.0, -1.0, 0.0, -1.0, -1.0, alpha;
}

void sort_eigen_indices_by_real_part(const Eigen::EigenSolver<Eigen::Matrix4d> &es, std::array<int, 4> &perm) {
    perm = {0, 1, 2, 3};
    std::sort(perm.begin(), perm.end(), [&](int i, int j) {
        return es.eigenvalues()(i).real() < es.eigenvalues()(j).real();
    });
}

// Second stable direction from remaining *real* eigenmodes (orthogonal to u0).
bool stable_u1_from_other_real_modes(const Eigen::EigenSolver<Eigen::Matrix4d> &es, const std::array<int, 4> &perm,
                                     const Eigen::Vector4d &u0, Eigen::Vector4d &u1) {
    for (int t = 1; t < 4; ++t) {
        const int j = perm[t];
        if (std::abs(es.eigenvalues()(j).imag()) >= 1e-12) {
            continue;
        }
        u1 = es.eigenvectors().col(j).real();
        u1 -= u0 * u0.dot(u1);
        if (u1.norm() > 1e-10) {
            u1.normalize();
            return true;
        }
    }
    return false;
}

// If no second real mode works, use imaginary parts of complex modes to span the real invariant plane.
bool stable_u1_from_imaginary_parts(const Eigen::EigenSolver<Eigen::Matrix4d> &es, const std::array<int, 4> &perm,
                                    const Eigen::Vector4d &u0, Eigen::Vector4d &u1) {
    for (int t = 1; t < 4; ++t) {
        const int j = perm[t];
        if (std::abs(es.eigenvalues()(j).imag()) <= 1e-12) {
            continue;
        }
        u1 = es.eigenvectors().col(j).imag();
        u1 -= u0 * u0.dot(u1);
        if (u1.norm() > 1e-10) {
            u1.normalize();
            return true;
        }
    }
    return false;
}

bool stable_u0_u1_real_leading_eigenvalue(const Eigen::EigenSolver<Eigen::Matrix4d> &es,
                                          const std::array<int, 4> &perm, const int i0, Eigen::Vector4d &u0,
                                          Eigen::Vector4d &u1) {
    const std::complex<double> lam0 = es.eigenvalues()(i0);
    const Eigen::Vector4cd vec0 = es.eigenvectors().col(i0);

    if (std::abs(lam0.imag()) >= 1e-12) {
        return false;
    }
    u0 = vec0.real();
    if (u0.norm() < 1e-14) {
        return false;
    }
    u0.normalize();
    if (stable_u1_from_other_real_modes(es, perm, u0, u1)) {
        return true;
    }
    return stable_u1_from_imaginary_parts(es, perm, u0, u1);
}

bool stable_u0_u1_complex_leading_eigenvalue(const Eigen::Vector4cd &vec0, Eigen::Vector4d &u0, Eigen::Vector4d &u1) {
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
    return true;
}

// From A, extract an orthonormal basis {u0,u1} of E^s: take eigenvectors with
// smallest Re λ (stable spectral half-plane).  Real stable modes yield real
// directions; a complex conjugate stable pair contributes Re v, Im v after
// Gram–Schmidt so both lie in the real invariant plane.  These become __constant__
// columns for device kernels and identical host replay.
bool stable_columns_from_A(const Eigen::Matrix4d &A, double col0[4], double col1[4]) {
    Eigen::EigenSolver<Eigen::Matrix4d> es(A);
    if (es.info() != Eigen::Success) {
        return false;
    }

    std::array<int, 4> perm;
    sort_eigen_indices_by_real_part(es, perm);

    Eigen::Vector4d u0, u1;
    const int i0 = perm[0];
    const std::complex<double> lam0 = es.eigenvalues()(i0);
    const Eigen::Vector4cd vec0 = es.eigenvectors().col(i0);

    bool ok = false;
    if (std::abs(lam0.imag()) < 1e-12) {
        ok = stable_u0_u1_real_leading_eigenvalue(es, perm, i0, u0, u1);
    } else {
        ok = stable_u0_u1_complex_leading_eigenvalue(vec0, u0, u1);
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

__constant__ double c_patch_col0[4];
__constant__ double c_patch_col1[4];

// One thread = one Cartesian product candidate (well k, radius r, grid cell (i,j)).
// Backward IVP: classical RK4 on the 4D state (rk4_step_physics_only) with dt → −dt,
// i.e. one explicit fourth-order step of the reversed-time flow.  Objective: squared wrapped
// Riemannian distance on S^1×R to (theta_tgt, phi_init); blow-ups → +∞ surrogate.
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

    int iw = 0;
    int ir = 0;
    int i = 0;
    int j = 0;
    warm_start::decode_patch_linear_index(idx, grid_n, n_rad, iw, ir, i, j);
    const double radius = d_radii[ir];
    double a = 0.0;
    double b = 0.0;
    warm_start::patch_ab_from_ij(i, j, radius, a, b);

    const double two_pi = 6.28318530717958647692;
    const double theta_tgt = theta_init_base - two_pi * static_cast<double>(d_wells[iw]);

    VarState x;
    warm_start::origin_patch_state(a, b, c_patch_col0, c_patch_col1, x);

    scores[idx] = warm_start::patch_backward_terminal_dist2_or_huge(x, sys_alpha_only, dt, num_intervals,
                                                                      steps_per_interval, theta_tgt, phi_init_base);
}

bool build_initial_patch_columns(double alpha, double col0[4], double col1[4]) {
    Eigen::Matrix4d A;
    fill_linearization_at_origin(alpha, A);
    return stable_columns_from_A(A, col0, col1);
}

const double *patch_search_radii_host(int *n_rad_out) {
    static const double radii[] = {1.0e-10, 3.0e-10, 1.0e-9,  3.0e-9,  1.0e-8,  3.0e-8, 1.0e-7,  3.0e-7,
                                   1.0e-6,  3.0e-6,  1.0e-5,  3.0e-5,  1.0e-4,  3.0e-4,  1.0e-3};
    *n_rad_out = static_cast<int>(sizeof(radii) / sizeof(radii[0]));
    return radii;
}

SystemParams strip_initial_and_goals_for_kernel(const SystemParams &sys, int num_intervals) {
    SystemParams sys_kernel = sys;
    sys_kernel.theta_init = 0.0;
    sys_kernel.phi_init = 0.0;
    sys_kernel.theta_goal = 0.0;
    sys_kernel.phi_goal = 0.0;
    sys_kernel.num_shooting_intervals = num_intervals;
    return sys_kernel;
}

// Allocates device buffers, launches `patch_score_kernel`, copies scores back, frees temporaries.
void run_gpu_patch_score_search(const SystemParams &sys_kernel, int n_wells, const std::vector<int> &wells, int n_rad,
                                const double *radii_host, double theta_init, double phi_init, double dt, int N, int steps,
                                int total_candidates, const double col0[4], const double col1[4],
                                std::vector<double> &h_scores_out) {
    int *d_wells = nullptr;
    double *d_radii = nullptr;
    double *d_scores = nullptr;
    gpuErrchk(cudaMalloc(&d_wells, static_cast<size_t>(n_wells) * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_radii, static_cast<size_t>(n_rad) * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_scores, static_cast<size_t>(total_candidates) * sizeof(double)));
    gpuErrchk(cudaMemcpy(d_wells, wells.data(), static_cast<size_t>(n_wells) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_radii, radii_host, static_cast<size_t>(n_rad) * sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpyToSymbol(c_patch_col0, col0, sizeof(double) * 4));
    gpuErrchk(cudaMemcpyToSymbol(c_patch_col1, col1, sizeof(double) * 4));

    const int threads = 256;
    const int blocks = (total_candidates + threads - 1) / threads;
    patch_score_kernel<<<blocks, threads>>>(d_scores, sys_kernel, n_wells, d_wells, n_rad, d_radii, theta_init,
                                            phi_init, dt, N, steps);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    h_scores_out.resize(static_cast<size_t>(total_candidates));
    gpuErrchk(cudaMemcpy(h_scores_out.data(), d_scores, static_cast<size_t>(total_candidates) * sizeof(double),
                         cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(d_scores));
    gpuErrchk(cudaFree(d_radii));
    gpuErrchk(cudaFree(d_wells));
}

void partial_sort_topk_indices(const std::vector<double> &h_scores, int top_k, std::vector<int> &order, int &k_take) {
    const int total = static_cast<int>(h_scores.size());
    order.resize(static_cast<size_t>(total));
    std::iota(order.begin(), order.end(), 0);
    k_take = std::min(top_k, total);
    std::partial_sort(order.begin(), order.begin() + k_take, order.end(), [&](int a, int b) {
        return h_scores[static_cast<size_t>(a)] < h_scores[static_cast<size_t>(b)];
    });
}

// Host replay of the top-K GPU candidates into MS-shaped nodal warm starts.
void append_host_replayed_topk_trajectories(std::vector<std::vector<double>> &out, const std::vector<int> &order,
                                            int k_take, int grid_n, int n_rad, const double *radii_host,
                                            const std::vector<int> &wells, const SystemParams &sys,
                                            const SystemParams &sys_alpha, const IntegratorParams &int_params, int N,
                                            int steps, const double col0[4], const double col1[4]) {
    for (int t = 0; t < k_take; ++t) {
        const int idx = order[static_cast<size_t>(t)];
        int iw = 0;
        int ir = 0;
        int ii = 0;
        int jj = 0;
        warm_start::decode_patch_linear_index(idx, grid_n, n_rad, iw, ir, ii, jj);
        const double radius = radii_host[ir];
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
}

} // namespace

std::vector<std::vector<double>> compute_patch_topk_ms_warm_starts(const SystemParams &sys,
                                                                   const IntegratorParams &int_params,
                                                                   int top_k) {
    std::vector<std::vector<double>> out;

    double col0[4];
    double col1[4];
    if (!build_initial_patch_columns(sys.alpha, col0, col1)) {
        return out;
    }

    const int N = sys.num_shooting_intervals;
    const int steps = int_params.num_steps;
    if (N <= 0 || steps <= 0 || top_k <= 0) {
        return out;
    }

    std::vector<int> wells;
    warm_start::theta_well_shift_candidates(sys.theta_init, wells);
    const int n_wells = static_cast<int>(wells.size());
    const int grid_n = warm_start::kPatchGrid;
    const int nij = grid_n * grid_n;

    int n_rad = 0;
    const double *radii_host = patch_search_radii_host(&n_rad);
    const int total = n_wells * n_rad * nij;

    const SystemParams sys_kernel = strip_initial_and_goals_for_kernel(sys, N);

    std::vector<double> h_scores;
    run_gpu_patch_score_search(sys_kernel, n_wells, wells, n_rad, radii_host, sys.theta_init, sys.phi_init,
                               int_params.dt, N, steps, total, col0, col1, h_scores);

    std::vector<int> order;
    int k_take = 0;
    partial_sort_topk_indices(h_scores, top_k, order, k_take);

    SystemParams sys_alpha = sys_kernel;
    sys_alpha.alpha = sys.alpha;

    append_host_replayed_topk_trajectories(out, order, k_take, grid_n, n_rad, radii_host, wells, sys, sys_alpha,
                                           int_params, N, steps, col0, col1);

    return out;
}
