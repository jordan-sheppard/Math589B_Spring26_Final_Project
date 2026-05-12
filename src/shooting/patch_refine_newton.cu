#include "shooting/patch_refine_newton.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "dynamics/pendulum_oc.cuh"

namespace {

inline double two_pi() { return 6.283185307179586476925286766559; }

inline bool finite4(const VarState &z) {
    return std::isfinite(z.theta()) && std::isfinite(z.phi()) && std::isfinite(z.l1()) && std::isfinite(z.l2());
}

inline VarState deriv_physics_only(const VarState &z, const SystemParams &p) {
    VarState d;
    compute_state_physics(z, p, d);
    d.cost() = 0.0;
    return d;
}

inline VarState rk4_physics_only(const VarState &y, const SystemParams &p, double h) {
    const double hh = 0.5 * h;
    const VarState k1 = deriv_physics_only(y, p);
    const VarState k2 = deriv_physics_only(y + hh * k1, p);
    const VarState k3 = deriv_physics_only(y + hh * k2, p);
    const VarState k4 = deriv_physics_only(y + h * k3, p);
    VarState yn = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    yn.cost() = 0.0;
    return yn;
}

inline void pack_y0(const StablePatchBasis &basis, double a, double b, VarState &y0) {
    // basis.B is row-major 4x2: [theta phi l1 l2] rows, [B1 B2] cols
    y0.theta() = a * basis.B[0] + b * basis.B[1];
    y0.phi() = a * basis.B[2] + b * basis.B[3];
    y0.l1() = a * basis.B[4] + b * basis.B[5];
    y0.l2() = a * basis.B[6] + b * basis.B[7];
    y0.cost() = 0.0;
}

inline bool integrate_backward_endpoint(const SystemParams &sys,
                                        const StablePatchBasis &basis,
                                        int well_k,
                                        double a,
                                        double b,
                                        const StablePatchGridSettings &gs,
                                        VarState &y_end,
                                        double &J_out) {
    VarState y0;
    pack_y0(basis, a, b, y0);
    if (!finite4(y0)) return false;

    const double h = -std::fabs(gs.back_dt);
    const double habs = std::fabs(h);
    double J = 0.0;
    VarState y = y0;

    for (int s = 0; s < gs.back_steps; ++s) {
        VarState dy;
        compute_state_physics(y, sys, dy);
        const double f0 = dy.cost();

        VarState y1 = rk4_physics_only(y, sys, h);
        if (!finite4(y1)) return false;
        VarState dy1;
        compute_state_physics(y1, sys, dy1);
        const double f1 = dy1.cost();

        J += 0.5 * habs * (f0 + f1);
        y = y1;
    }

    (void)well_k; // well handled in residual, not in integration (equilibrium at 0)
    y_end = y;
    J_out = J;
    return true;
}

inline void residual_R(const SystemParams &sys, int well_k, const VarState &y_end, double R[2]) {
    const double theta_eff = sys.theta_init - two_pi() * static_cast<double>(well_k);
    const double phi_t = sys.phi_init;
    R[0] = y_end.theta() - theta_eff;
    R[1] = y_end.phi() - phi_t;
}

inline double rinf2(const double R[2]) { return std::max(std::fabs(R[0]), std::fabs(R[1])); }

inline bool solve2x2(const double A[2][2], const double rhs[2], double x[2]) {
    const double det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (!(std::fabs(det) > 1e-30)) return false;
    const double inv = 1.0 / det;
    x[0] = inv * ( A[1][1] * rhs[0] - A[0][1] * rhs[1]);
    x[1] = inv * (-A[1][0] * rhs[0] + A[0][0] * rhs[1]);
    return true;
}

} // namespace

std::vector<StablePatchCandidate> stable_patch_topk_per_well(const StablePatchCandidate *cands,
                                                            int num_wells,
                                                            int num_radii,
                                                            int grid_n,
                                                            int top_k) {
    const int per_well = grid_n * grid_n;
    const int slice = num_wells * per_well;
    std::vector<StablePatchCandidate> out;
    out.reserve(static_cast<size_t>(std::max(0, num_wells * top_k)));

    for (int w = 0; w < num_wells; ++w) {
        std::vector<StablePatchCandidate> v;
        v.reserve(static_cast<size_t>(per_well * std::max(1, num_radii)));
        for (int ri = 0; ri < num_radii; ++ri) {
            const StablePatchCandidate *base = cands + ri * slice + w * per_well;
            for (int i = 0; i < per_well; ++i) {
                if (base[i].valid && std::isfinite(base[i].r_residual)) {
                    v.push_back(base[i]);
                }
            }
        }
        std::sort(v.begin(), v.end(), [](const StablePatchCandidate &a, const StablePatchCandidate &b) {
            if (a.r_residual != b.r_residual) return a.r_residual < b.r_residual;
            if (a.J != b.J) return a.J < b.J;
            const double na = std::fabs(a.a) + std::fabs(a.b);
            const double nb = std::fabs(b.a) + std::fabs(b.b);
            return na < nb;
        });
        const int keep = std::min<int>(top_k, static_cast<int>(v.size()));
        for (int i = 0; i < keep; ++i) out.push_back(v[static_cast<size_t>(i)]);
    }
    return out;
}

StablePatchRefineOut refine_candidate_newton_2d(const SystemParams &sys,
                                                const StablePatchBasis &basis,
                                                int well_k,
                                                double a0,
                                                double b0,
                                                const StablePatchNewtonSettings &ns,
                                                const StablePatchGridSettings &gs) {
    StablePatchRefineOut out;
    out.converged = 0;
    out.iters = 0;
    out.a = a0;
    out.b = b0;
    out.r_inf = 1e300;

    double a = a0;
    double b = b0;
    double best_r = 1e300;
    double best_a = a0, best_b = b0;
    VarState best_y{};
    double best_J = std::numeric_limits<double>::infinity();

    for (int it = 0; it < ns.max_iters; ++it) {
        VarState y;
        double J = 0.0;
        if (!integrate_backward_endpoint(sys, basis, well_k, a, b, gs, y, J)) {
            break;
        }
        double R[2];
        residual_R(sys, well_k, y, R);
        const double r = rinf2(R);
        if (r < best_r) {
            best_r = r;
            best_a = a;
            best_b = b;
            best_y = y;
            best_J = J;
        }

        if (r <= ns.tol) {
            out.converged = 1;
            out.iters = it + 1;
            out.r_inf = r;
            out.a = a;
            out.b = b;
            out.l1 = y.l1();
            out.l2 = y.l2();
            out.J = J;
            return out;
        }

        // Centered finite-difference Jacobian ∂R/∂(a,b)
        const double ea = ns.fd_eps;
        const double eb = ns.fd_eps;
        VarState ya_p, ya_m, yb_p, yb_m;
        double Ja_p = 0.0, Ja_m = 0.0, Jb_p = 0.0, Jb_m = 0.0;
        if (!integrate_backward_endpoint(sys, basis, well_k, a + ea, b, gs, ya_p, Ja_p)) break;
        if (!integrate_backward_endpoint(sys, basis, well_k, a - ea, b, gs, ya_m, Ja_m)) break;
        if (!integrate_backward_endpoint(sys, basis, well_k, a, b + eb, gs, yb_p, Jb_p)) break;
        if (!integrate_backward_endpoint(sys, basis, well_k, a, b - eb, gs, yb_m, Jb_m)) break;
        double Ra_p[2], Ra_m[2], Rb_p[2], Rb_m[2];
        residual_R(sys, well_k, ya_p, Ra_p);
        residual_R(sys, well_k, ya_m, Ra_m);
        residual_R(sys, well_k, yb_p, Rb_p);
        residual_R(sys, well_k, yb_m, Rb_m);

        double A[2][2];
        const double inv2a = 1.0 / (2.0 * ea);
        const double inv2b = 1.0 / (2.0 * eb);
        A[0][0] = (Ra_p[0] - Ra_m[0]) * inv2a;
        A[1][0] = (Ra_p[1] - Ra_m[1]) * inv2a;
        A[0][1] = (Rb_p[0] - Rb_m[0]) * inv2b;
        A[1][1] = (Rb_p[1] - Rb_m[1]) * inv2b;

        double rhs[2] = {-R[0], -R[1]};
        double delta[2];
        if (!solve2x2(A, rhs, delta)) break;

        // Clip step
        delta[0] = std::max(-ns.step_clip, std::min(ns.step_clip, delta[0]));
        delta[1] = std::max(-ns.step_clip, std::min(ns.step_clip, delta[1]));

        // Backtracking line search
        double s = 1.0;
        bool accepted = false;
        for (int bt = 0; bt <= ns.backtrack_max; ++bt) {
            const double at = a + s * delta[0];
            const double bt_ = b + s * delta[1];
            VarState yt;
            double Jt = 0.0;
            if (!integrate_backward_endpoint(sys, basis, well_k, at, bt_, gs, yt, Jt)) {
                s *= 0.5;
                continue;
            }
            double Rt[2];
            residual_R(sys, well_k, yt, Rt);
            const double rt = rinf2(Rt);
            if (rt < r) {
                a = at;
                b = bt_;
                accepted = true;
                break;
            }
            s *= 0.5;
        }
        if (!accepted) break;
        out.iters = it + 1;
    }

    // Best-so-far
    out.converged = 0;
    out.iters = std::max(out.iters, 1);
    out.r_inf = best_r;
    out.a = best_a;
    out.b = best_b;
    out.l1 = best_y.l1();
    out.l2 = best_y.l2();
    out.J = best_J;
    return out;
}

