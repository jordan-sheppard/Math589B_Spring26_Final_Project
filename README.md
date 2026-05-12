# Math 589B Final Programming Project (Spring 2026)

CUDA implementation of **multiple shooting** with **Newton** (GPU segment integration, host sparse LU) for a damped-pendulum **infinite-horizon** optimal-control model. The driver in `src/driver/continuation_sheets.cu` searches angle wraps and goal sheets, **homotopes** scaled boundary data, **warm-starts** nodal trajectories with a **backward IVP patch** search (`src/warm_start/`), then refines with shooting.

---

## Build and run

From the project root (see `Makefile` for cluster module names and `CUDA_GENCODE`):

- Build: `make` produces the `solver` binary (loads `eigen` and `CUDA_MODULE`, checks host GCC vs `MAX_HOST_GCC_MAJOR`).
- Run: `./solver theta phi alpha`
- Usage on failure: `main` prints `usage: ./solver theta phi alpha` if the argument count is wrong.

If no continuation path converges, `solve()` may print a **stderr** hint about GPU architecture (`CUDA_GENCODE`), running on a GPU allocation, and CUDA modules on UA clusters (same text as in `continuation_sheets.cu`).

---

## Problem and first-order model

Angular position $\theta$, angular velocity $\phi = \dot\theta$, damping $\alpha > 0$, and scalar control $u$. The second-order dynamics used in the grader specification are equivalent to

$$
\ddot{\theta} = \sin(\theta) - \alpha \dot{\theta} + u\cos(\theta),
$$

with running cost $(1-\cos\theta) + \tfrac{1}{2}\phi^2 + \tfrac{1}{2}u^2$.

The code works in **first-order** form $z = (\theta,\phi,\lambda_1,\lambda_2)^\top$ (costates $\lambda_1,\lambda_2$ for $\theta$ and $\phi$; stored as `l1`, `l2` in `VarState`). With the feedback implied by Pontryagin’s conditions,

$$
u = -\lambda_2 \cos(\theta),
$$

the **state/costate ODE** implemented in `compute_state_physics` (`src/dynamics/pendulum_oc.cuh`) is

$$
\begin{aligned}
\dot\theta &= \phi,\\
\dot\phi &= \sin(\theta) - \alpha\phi - \lambda_2\cos^2(\theta),\\
\dot\lambda_1 &= -\lambda_2^2\cos(\theta)\sin(\theta) - \lambda_2\cos(\theta) - \sin(\theta),\\
\dot\lambda_2 &= -\phi - \lambda_1 + \alpha\lambda_2.
\end{aligned}
$$

The running cost **integrand** accumulated in `VarState::cost()` matches the original Lagrangian after eliminating $u$:

$$
L(\theta,\phi,\lambda_2) = (1-\cos\theta) + \tfrac{1}{2}\phi^2 + \tfrac{1}{2}\lambda_2^2\cos^2(\theta).
$$

**Hamiltonian** diagnostics use `compute_hamiltonian` in `src/integrators/segment_integration.cuh` (same model).

---

## Course task and program output

**Given:** $\theta$, $\phi$, $\alpha$.

**Return (via `Result` / stdout):** $\lambda_1$, $\lambda_2$ at the **first shooting node**, and a scalar cost $J$.

`src/main.cu` calls `solve(theta, phi, alpha)` from `src/solver.hpp` / `continuation_sheets.cu` and prints three numbers with ten decimal places:

```text
lambda1 lambda2 J
```

(`Result` also carries `optimal_theta_wraps` and `final_theta_goal` for the winning sheet; `main` does not print them.)

**Degenerate input:** if $|\theta|$ and $|\phi|$ are both below `1e-14`, `solve` returns zeros without running the pipeline.

---

## Truncated multiple shooting (what the numerics solve)

- **Intervals:** `NUM_SHOOTING_INTERVALS = 20`
- **Horizon:** `TOTAL_HORIZON = 16`
- **RK4 substeps per segment:** `NUM_INTEGRATION_STEPS = 128`
- **Time step:** $\texttt{dt} = \texttt{TOTAL\_HORIZON} / (\texttt{NUM\_SHOOTING\_INTERVALS} \cdot \texttt{NUM\_INTEGRATION\_STEPS})$ (same `IntegratorParams::dt` everywhere, including the warm start).

Each segment uses **RK4 on the augmented state** (`rk4_step_flow` in `segment_integration.cuh`): physics from `get_derivatives_flow` (which applies `IntegratorParams::backward_time` as a sign on the vector field and variational block) and a $4\times 4$ sensitivity $M$ with $M(t_0)=\mathbb{I}$.

**Defects and Newton** (`src/shooting/defect_jacobian_host.cu`, `newton_iteration.cu`):

- Unknowns: flat nodal values, length `4 * N`, layout $(\theta,\phi,\lambda_1,\lambda_2)$ per knot.
- **Forward** shooting (`backward_time == false`): segment $k$ starts at knot $k$; interior defects match segment end to knot $k{+}1$; last rows pin $(\theta,\phi)$ at knot $0$ to `(theta_init, phi_init)` and match terminal $(\theta,\phi)$ to `(theta_goal, phi_goal)`.
- **Backward** shooting (`backward_time == true`): kernel indexing and defect rows follow the “backward chain” branch in `defect_jacobian_host.cu` (segment $k$ reads knot $k{+}1$ when $k < N-1$).

Each Newton step: `evaluate_segments_on_gpu` → `build_global_system` → **sparse LU** on $J\,\Delta = -F$, full update of node guesses (`newton_iteration.cu`, no line search).

**Stopping:** `NewtonParams` from the driver: `max_iterations = 25`, `tolerance = 1e-9` (infinity norm of $F$).

**Printed $J$:** after convergence, `solve_multiple_shooting` sets `optimal_cost` to the **sum** over segments of each segment’s **terminal** `VarState::cost()` (running-cost integral accumulated along that segment). This is the objective associated with the **truncated** horizon discretization, not a closed-form infinite-horizon value.

---

## Driver: angle wraps, sheets, homotopy, warm start

Implemented in `solve()` in `src/driver/continuation_sheets.cu`.

1. **Principal $\theta$ wrap and neighbors**  
   Candidate integers $k$ are de-duplicated from  
   $\{\mathrm{round}(\theta_{\mathrm{tgt}}/2\pi), 0, \mathrm{round}-1, \mathrm{round}+1, \mathrm{round}-2, \mathrm{round}+2\}$.  
   For each $k$, the working angle is $\theta_{\mathrm{work}} = \theta_{\mathrm{tgt}} - 2\pi k$.

2. **Goal sheets**  
   For each $k$, `theta_goal` runs over `wrap * TWO_PI` with `wrap ∈ {-MAX_THETA_WRAPS, …, MAX_THETA_WRAPS}` and `MAX_THETA_WRAPS = 1` (so $\theta_{\mathrm{goal}} \in \{-2\pi, 0, 2\pi\}$). Terminal $\phi$ is pinned to `phi_goal = 0`.

3. **Homotopy parameter** $s \in (0,1]$  
   Let $r = \|(\theta_{\mathrm{work}}, \phi_{\mathrm{tgt}})\|_2$. If $r > 0.05$, start at $s = 0.05/r$; otherwise $s = 1$.  
   Boundary data for the inner solve uses $(\theta,\phi)_{\mathrm{init}} = s\,(\theta_{\mathrm{work}}, \phi_{\mathrm{tgt}})$ at the start knot and the same `(theta_goal, phi_goal)` at the end.

4. **Warm list**  
   `compute_patch_topk_ms_warm_starts(candidate_params, int_params, kWarmTop)` with `kWarmTop = 12` (`backward_ivp_warmstart.hpp` / `backward_ivp_batch.cu`).

5. **Backward MS first**  
   For each returned seed (and the same `NewtonParams`), `solve_multiple_shooting` is called with a copy of the trajectory and `IntegratorParams` identical except `backward_time = true`. The best successful result (lowest `optimal_cost`) is kept; its updated knot vector seeds the homotopy loop.

6. **Fallback**  
   If no warm seed converges, the code uses `compute_linear_initial_guess` (linear interpolation of $(\theta,\phi)$ from init to goal, costates zero) and **forward** MS (`backward_time = false`).

7. **Homotopy loop**  
   While $s < 1$: propose $s_{\mathrm{next}} = \min(s + \Delta s, 1)$, update init boundary to $s_{\mathrm{next}}(\theta_{\mathrm{work}}, \phi_{\mathrm{tgt}})$, re-solve forward MS from the previous knot vector.  
   - On success: $s \leftarrow s_{\mathrm{next}}$, update trajectory; if `num_iterations <= 4`, $\Delta s \leftarrow 1.5\,\Delta s$.  
   - On failure: $\Delta s \leftarrow 0.5\,\Delta s$; stop if $\Delta s < 10^{-4}$ (`MIN_CONTINUATION_STEP_SIZE`).  
   Initial $\Delta s = 0.1$.

8. **Best result**  
   Across all $(k,\ \mathrm{wrap})$ branches, keep the converged outcome with smallest `optimal_cost`.

---

## Backward IVP patch warm start (mathematics and numerics)

**Goal:** produce up to `top_k` candidate flat vectors of length `4 * N` (nodal $(\theta,\phi,\lambda_1,\lambda_2)$ along the shooting grid) that lie near the **stable manifold** of the Hamiltonian saddle at the upright, before Newton.

**Linearization** at $z=0$: $A \in \mathbb{R}^{4\times 4}$ is built in `fill_linearization_at_origin` (`backward_ivp_batch.cu`) from the closed-loop field at $(\theta,\phi,\lambda_1,\lambda_2)=0$ for the current `alpha`. Two **real orthonormal** directions spanning the stable invariant subspace are extracted with `Eigen::EigenSolver` (`stable_columns_from_A`): prefer real stable eigenvectors; otherwise use real/imaginary parts of a complex stable pair.

**Patch points:** $z_0(a,b) = a u_0 + b u_1$ with $(a,b)$ from a **square grid** of side `warm_start::kPatchGrid` (= **49**) points in $[-1,1]^2$, scaled by a **radius** $r$ so $a = r\,\xi_i$, $b = r\,\eta_j$ (`patch_ab_from_ij` in `backward_ivp_common.cuh`).

**Radii:** fixed log-spaced list of **16** values from `1e-10` to `1e-3` (see `kRadiiHost` in `backward_ivp_batch.cu`).

**Backward integration:** classical **RK4** on the **physics-only** RHS (`rk4_step_physics_only`: same `compute_state_physics` as forward segments, **no** variational/cost coupling in the stages). Steps use effective $-\texttt{dt}$ over `N * num_steps` steps, matching the MS substep count.

**$\theta$ ambiguity:** the same six integer shifts as in `build_well_shifts` are tried so $\theta_{\mathrm{tgt}} = \theta_{\mathrm{init}} - 2\pi k_{\mathrm{well}}$ aligns sheets.

**Scoring:** for finite trajectories, **squared** wrapped Euclidean distance in $(\theta,\phi)$ to $(\theta_{\mathrm{tgt}}, \phi_{\mathrm{init}})$ (`dist2_wrapped` / `theta_phi_distance_wrapped` in `backward_ivp_common.cuh`). Divergent/non-finite states get score `1e300`.

**GPU:** `patch_score_kernel` assigns one thread per triple (well shift, radius, grid cell); stable directions in `__constant__` memory. **Host:** `std::partial_sort` for the smallest scores; winners are **replayed** on the host via `origin_patch_backward_to_targets`, which fills the flat `4*N` vector at MS knot times (subsample condition `s == (N-k)*steps_per_interval` in `backward_ivp_common.cuh`).

This phase **does not** solve the BVP; it only supplies initial guesses for `solve_multiple_shooting`.

---

## File map (high level)

| Area | Main files |
|------|------------|
| Entry | `src/main.cu` |
| API / includes | `src/solver.hpp` |
| Continuation + `solve()` | `src/driver/continuation_sheets.cu` |
| Warm start | `src/warm_start/backward_ivp_warmstart.hpp`, `backward_ivp_batch.cu`, `backward_ivp_common.cuh` |
| Dynamics | `src/dynamics/pendulum_oc.cuh` |
| RK4 / segment | `src/integrators/segment_integration.cuh` |
| MS + cost | `src/shooting/multiple_shooting_solve.cu` |
| Newton step | `src/shooting/newton_iteration.cu`, `gpu_eval_segments.cu` |
| Defect Jacobian | `src/shooting/defect_jacobian_host.cu` |

---

## Summary constants (for reproducing behavior)

| Symbol / name | Value |
|---------------|--------|
| `NUM_SHOOTING_INTERVALS` | 20 |
| `TOTAL_HORIZON` | 16.0 |
| `NUM_INTEGRATION_STEPS` | 128 |
| `MAX_NEWTON_ITERATIONS` | 25 |
| `NEWTON_TOL` | `1e-9` |
| `MAX_THETA_WRAPS` | 1 |
| `kWarmTop` | 12 |
| `kPatchGrid` | 49 |
| Homotopy start scale | `0.05` when $r > 0.05$ |
| Initial $\Delta s$ | `0.1` |
| `MIN_CONTINUATION_STEP_SIZE` | `1e-4` |
