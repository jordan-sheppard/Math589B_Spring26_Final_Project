#include "shooting/stable_patch_grid.hpp"

#include <cmath>

#include <cuda_runtime.h>

#include "cuda/gpu_macros.cuh"
#include "dynamics/pendulum_oc.cuh"

namespace {

__host__ __device__ inline bool finite4(const VarState &z) {
    return isfinite(z.theta()) && isfinite(z.phi()) && isfinite(z.l1()) && isfinite(z.l2());
}

__host__ __device__ inline VarState deriv_physics_only(const VarState &z, const SystemParams &p) {
    VarState d;
    compute_state_physics(z, p, d);
    d.cost() = 0.0;
    // M unused; leave as-is.
    return d;
}

__host__ __device__ inline VarState rk4_physics_only(const VarState &y, const SystemParams &p, double h) {
    const double hh = 0.5 * h;
    const VarState k1 = deriv_physics_only(y, p);
    const VarState k2 = deriv_physics_only(y + hh * k1, p);
    const VarState k3 = deriv_physics_only(y + hh * k2, p);
    const VarState k4 = deriv_physics_only(y + h * k3, p);
    VarState yn = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    yn.cost() = 0.0;
    return yn;
}

__host__ __device__ inline double two_pi() { return 6.283185307179586476925286766559; }

__host__ __device__ inline double d2_target(double theta_end, double phi_end, double theta_eff, double phi_t) {
    const double dth = theta_end - theta_eff;
    const double dph = phi_end - phi_t;
    return dth * dth + dph * dph;
}

__host__ __device__ inline double r_inf_target(double theta_end, double phi_end, double theta_eff, double phi_t) {
    const double dth = theta_end - theta_eff;
    const double dph = phi_end - phi_t;
    return fmax(fabs(dth), fabs(dph));
}

__global__ void stable_patch_grid_kernel(SystemParams sys,
                                         StablePatchBasis basis,
                                         const int *wells_k,
                                         int num_wells,
                                         StablePatchGridSettings gs,
                                         StablePatchCandidate *out) {
    const int grid_total = gs.grid_n * gs.grid_n;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_wells * grid_total;
    if (tid >= total) return;

    const int well_idx = tid / grid_total;
    const int idx = tid - well_idx * grid_total;
    const int i = idx / gs.grid_n;
    const int j = idx - i * gs.grid_n;

    const int k = wells_k[well_idx];
    const double theta_eff = sys.theta_init - two_pi() * static_cast<double>(k);
    const double phi_t = sys.phi_init;

    const double t_i = (gs.grid_n == 1) ? 0.0 : (static_cast<double>(i) / static_cast<double>(gs.grid_n - 1));
    const double t_j = (gs.grid_n == 1) ? 0.0 : (static_cast<double>(j) / static_cast<double>(gs.grid_n - 1));
    const double a = (2.0 * t_i - 1.0) * gs.grid_radius;
    const double b = (2.0 * t_j - 1.0) * gs.grid_radius;

    // y0 = a*B1 + b*B2 (in phase coordinates around equilibrium at 0)
    VarState y0;
    y0.theta() = a * basis.B[0] + b * basis.B[1];
    y0.phi() = a * basis.B[2] + b * basis.B[3];
    y0.l1() = a * basis.B[4] + b * basis.B[5];
    y0.l2() = a * basis.B[6] + b * basis.B[7];
    y0.cost() = 0.0;

    const double h = -fabs(gs.back_dt);
    const double habs = fabs(h);
    double J = 0.0;

    VarState y = y0;
    int valid = finite4(y) ? 1 : 0;
    if (!valid) {
        StablePatchCandidate c;
        c.well_k = k;
        c.a = a;
        c.b = b;
        c.valid = 0;
        c.d2 = 1e300;
        c.r_residual = 1e300;
        out[tid] = c;
        return;
    }

    for (int s = 0; s < gs.back_steps; ++s) {
        VarState dy;
        compute_state_physics(y, sys, dy);
        const double f0 = dy.cost();

        VarState y1 = rk4_physics_only(y, sys, h);
        if (!finite4(y1)) {
            valid = 0;
            break;
        }
        VarState dy1;
        compute_state_physics(y1, sys, dy1);
        const double f1 = dy1.cost();
        J += 0.5 * habs * (f0 + f1);
        y = y1;
    }

    StablePatchCandidate c;
    c.well_k = k;
    c.a = a;
    c.b = b;
    c.theta_end = y.theta();
    c.phi_end = y.phi();
    c.l1_end = y.l1();
    c.l2_end = y.l2();
    c.J = J;
    c.valid = valid;
    c.d2 = valid ? d2_target(c.theta_end, c.phi_end, theta_eff, phi_t) : 1e300;
    c.r_residual = valid ? r_inf_target(c.theta_end, c.phi_end, theta_eff, phi_t) : 1e300;
    out[tid] = c;
}

}  // namespace

void stable_patch_grid_backward_gpu(const SystemParams &sys,
                                    const StablePatchBasis &basis,
                                    const int *wells_k,
                                    int num_wells,
                                    const StablePatchGridSettings &gs,
                                    StablePatchCandidate *out) {
    const int grid_total = gs.grid_n * gs.grid_n;
    const int total = num_wells * grid_total;
    if (total <= 0) return;

    int *d_wells = nullptr;
    StablePatchCandidate *d_out = nullptr;
    gpuErrchk(cudaMalloc(&d_wells, static_cast<size_t>(num_wells) * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_out, static_cast<size_t>(total) * sizeof(StablePatchCandidate)));

    gpuErrchk(cudaMemcpy(d_wells, wells_k, static_cast<size_t>(num_wells) * sizeof(int),
                         cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    stable_patch_grid_kernel<<<blocks, threads>>>(sys, basis, d_wells, num_wells, gs, d_out);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(out, d_out, static_cast<size_t>(total) * sizeof(StablePatchCandidate),
                         cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_out));
    gpuErrchk(cudaFree(d_wells));
}

