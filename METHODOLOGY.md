# METHODOLOGY: CPU Pendulum Optimal-Control Solver

This document describes the mathematics implemented in the C++ codebase under `cpp/`, how components are structured file-by-file, and how those choices relate to the course problem (controlled pendulum, infinite-horizon cost, stable manifold viewpoint).

The public callable API is unchanged from the assignment scaffold: `solve(theta, phi, alpha)` is declared in `src/solver.hpp` and implemented in `cpp/solver/api.cpp`.

---

## 1. Problem statement (consistent with README and PDF)

### 1.1 Dynamics

The controlled pendulum (angle $\theta$ from the **upward** vertical, $\varphi = \dot\theta$) is

$$
\ddot{\theta} = \sin\theta - \alpha\dot{\theta} + u\cos\theta,
$$

with friction parameter $\alpha \ge 0$ and scalar control $u$.

First-order state:

$$
x = (\theta, \varphi) \in \mathbb{R}^2.
$$

### 1.2 Running cost and objective

The infinite-horizon objective is

$$
J = \int_0^\infty \Bigl[(1-\cos\theta) + \tfrac12 \varphi^2 + \tfrac12 u^2\Bigr]\,\mathrm{d}t .
$$

In code we denote costates $\lambda_1$ (paired with $\theta$) and $\lambda_2$ (paired with $\varphi$).

### 1.3 Pontryagin / necessary conditions

Define the Hamiltonian (before eliminating $u$) as in the course notes. Minimizing the Hamiltonian pointwise over $u$ gives the optimal feedback

$$
u^\ast = -\lambda_2\cos\theta .
$$

Substituting $u^\ast$ into the Hamiltonian yields the **effective Hamiltonian** $H(x,\lambda)$.

The implementation in `cpp/solver/dynamics.cpp` evolves the Hamiltonian dynamics in standard form:

- State equations: $\dot{\theta}=\partial_{\lambda_1}H$, $\dot{\varphi}=\partial_{\lambda_2}H$.
- Costate equations: $\dot{\lambda}_1=-\partial_{\theta}H$, $\dot{\lambda}_2=-\partial_{\varphi}H$.

Explicitly (matching the coded RHS):

$$
\begin{aligned}
\dot{\theta} &= \varphi, \\
\dot{\varphi} &= \sin\theta - \alpha\varphi - \lambda_2\cos^2\theta, \\
\dot{\lambda}_1 &= -\sin\theta - \lambda_2\cos\theta - \lambda_2^2\cos\theta\sin\theta, \\
\dot{\lambda}_2 &= -\varphi - \lambda_1 + \alpha\lambda_2 .
\end{aligned}
$$

Together with $\lambda_1(0),\lambda_2(0)$ free (to satisfy transversality indirectly via the stable-manifold targeting below), integrating this ODE yields a trajectory $(x(t),\lambda(t))$ consistent with Pontryagin’s construction for infinite-horizon problems when the trajectory lies on the **stable manifold** of the equilibrium at the upright position.

---

## 2. What we compute

Given $(\theta_0,\varphi_0,\alpha)$, the executable must output three numbers:

- $\lambda_1(0)$, $\lambda_2(0)$, and $J$ (the value functional for that optimal solution).

Numerically:

- $\lambda_1(0),\lambda_2(0)$ are found by solving a **shooting / root-finding** problem built from forward integration on a truncated horizon $T$.
- $J$ is approximated by the definite integral $\int_0^T [(1-\cos\theta)+\tfrac12\varphi^2+\tfrac12(u^\ast)^2]\,\mathrm{d}t$ along the trajectory (no separate tail-correction term is implemented in code at present).

---

## 3. Stable manifold seed: linear map $P$

Near the equilibrium $(\theta,\varphi,\lambda)=(0,0,0)$, the stable manifold of the saddle-type Hamiltonian equilibrium is approximated locally by correlating $\lambda$ with $x$:

$$
\lambda \approx P x ,\qquad P \in \mathbb{R}^{2\times 2}.
$$

The course PDF derives $P$ from the **Hamiltonian matrix** $C = J \, \nabla^2 H(0)$ where $J$ is the canonical $4\times 4$ symplectic permutation used in MATLAB:

$$
J = \begin{pmatrix}
0_{2\times 2} & I_{2\times 2} \\
-I_{2\times 2} & 0_{2\times 2}
\end{pmatrix}.
$$

Eigenvectors belonging to eigenvalues with **negative real parts** span the stable subspace. Partitioning stacked eigenvectors as $[v_\theta,v_\varphi,v_{\lambda_1},v_{\lambda_2}]^\top$:

$$
\begin{pmatrix}
V_{s1} \\
V_{s2}
\end{pmatrix} \quad (2\times 2 \text{ blocks}),
\qquad
P = V_{s2} V_{s1}^{-1}.
$$

Implementation: **`cpp/solver/manifold_seed.cpp`** builds $\nabla^2 H$ at zero analytically from the effective $H$ (matching the Hessian ordering $[\theta,\varphi,\lambda_1,\lambda_2]$), forms $C = J \, \mathrm{Hess}(H)$, computes eigenpairs with Eigen’s complex eigensolver, selects the **two** stable modes, builds $P$, and returns **`P.real()`**.

This $P$ is used for:

1. **Warm initial guess**: $\lambda(0)\approx P(\theta_{\mathrm{sheet}},\varphi_0)^{\!\top}$ after shifting $\theta$ to a sheet (see §5).
2. **Augmented residual** at time $T$: enforce both small state and manifold consistency $\lambda(T)\approx P x(T)$ (see §4.2).

---

## 4. Shooting formulation

### 4.1 Augmented IVP and cost accumulation

Define the phase vector $z = (\theta,\varphi,\lambda_1,\lambda_2)^{\!\top}$ and denote its RHS $F_\alpha(z)$ (`hamiltonianRHS`).

Given initial state $x_0$ and initial costate $\ell_0 = (\lambda_1(0),\lambda_2(0))$:

- evolve $z(0)=(x_0,\ell_0)$ forward to time $T$;
- accumulate

$$
J \approx \int_0^T f_0(z(t))\,\mathrm{d}t,
\quad
f_0 = (1-\cos\theta)+\tfrac12\varphi^2+\tfrac12 (\lambda_2\cos\theta)^2
\quad (\text{because }u^\ast=-\lambda_2\cos\theta).
$$

**Cost integration** uses **trapezoidal quadrature per step**. The running sum applies **Kahan compensated summation** (`KahanSum` in **`cpp/solver/cost.hpp/.cpp`**) to reduce drift over long horizons.

### 4.2 Residual $r$ (dimension 4)

Let $x(T)=(\theta(T),\varphi(T))$ and $\lambda(T)=(\lambda_1(T),\lambda_2(T))$. With fixed $T$ from the course’s finite-horizon shooting viewpoint, define

$$
r =
\begin{bmatrix}
x(T) \\
\lambda(T) - P\,x(T)
\end{bmatrix}
\in\mathbb{R}^4 .
$$

The first block drives the state toward $0$ at horizon $T$; the second block enforces (linearized) stable-manifold consistency at $T$.

The shooting problem is:

$$
\text{Find } \ell_0 \quad \text{such that}\quad \|r(\ell_0)\|_{\infty}\le \varepsilon_{\mathrm{tol}} .
$$

**Note:** $\varepsilon_{\mathrm{tol}}$ near $10^{-10}$ in `api.cpp` is a stringent internal target on this surrogate residual; the grader compares printed outputs against reference values under its own tol (typically $10^{-6}$).

### 4.3 Jacobian $\partial r/\partial \lambda(0)$ via sensitivity (variational) equations

Rather than finite-differences (expensive and noisy), the code integrates the **variational equations** jointly with $z(t)$:

$$
\frac{\mathrm{d}}{\mathrm{d}t}\frac{\partial z}{\partial \ell_0} = D_z F_\alpha(z(t))\,\frac{\partial z}{\partial \ell_0},\qquad 
\frac{\partial z}{\partial \ell_0}(0)=\begin{bmatrix}
0_{2\times 2}\\ I_{2\times 2}
\end{bmatrix}.
$$

Implementation detail: the Jacobian $A(t)= D_z F$ is spelled out explicitly in **`jacobianDF`** in **`cpp/solver/shooting.cpp`** using the algebraic derivatives consistent with **`hamiltonianRHS`**.

The augmented state $\texttt{AugState}=\{z, S\in\mathbb{R}^{4\times 2}\}$ is propagated with either **RK4** or fixed-step **DP5** (§6). At $T$, extract blocks $S_{\theta,\varphi}(T)$, $S_{\lambda}(T)$ and assemble

$$
\frac{\partial r}{\partial \ell_0} =
\begin{bmatrix}
S_{\theta,\varphi}(T) \\
S_{\lambda}(T) - P\,S_{\theta,\varphi}(T)
\end{bmatrix}
\in\mathbb{R}^{4\times 2}.
$$

Fallback: **`use_variational_jacobian=false`** uses central finite differences of $r$ w.r.t. $\ell_0$ (still available for debugging).

### 4.4 Levenberg–Marquardt with backtracking

At iterate $k$, approximate Gauss–Newton normal equations:

$$
(J^\top J + \lambda I)\delta = -J^\top r,
\quad
J\in\mathbb{R}^{4\times 2},\ \lambda>0\ \text{LM damping.}
$$

- Solve with **`Eigen::FullPivLU`** on $2\times 2$ (stable for small matrices).
- **Clip** $\|\delta\|$ (`max_delta_norm`).
- **Backtracking** scales $\delta$ until $\|r\|_\infty$ strictly decreases versus the previous iterate (until `backtrack_max` halvings).

Damping adjusts on accept/reject: reduce $\lambda$ on success, multiply by `lm_lambda_mul` on failure.

### 4.5 Continuation in horizon $T$

To improve convergence for nonlinear sensitivity, **`solveCostatesSingleSheetLMContinuation`** runs a chain of horizons $T_1<T_2<\cdots$ (given as `Eigen::VectorXd T_schedule`). Each stage **warm-starts** from the $\ell_0$ returned by the prior stage.

**Adaptive cut:** after a stage, if $\|r\|_\infty \le$ `tol_resid`, continuation stops early (subsequent horizons are skipped).

Across stages, **`best_overall`** tracks the iterate with smallest $\|r\|_\infty$** seen **(defensive bookkeeping if a later horizon temporarily worsens the residual metric).

---

## 5. Multi-sheet search (equilibria modulo $2\pi$)

Angles are not forcibly wrapped in-state; equivalences $\theta$ and $\theta+2\pi k$ motivate **multiple equilibrium sheets**.

For integer $m$, define shifted angle

$$
\theta^{(m)} = \theta_0 - 2\pi m ,\qquad \varphi\text{ unchanged}.
$$

- **Sheet center:** $m_0=\mathrm{round}(\theta_0/(2\pi))$.
- **Search radius:** heuristic `max(m_radius_min, ceil(|phi| * m_radius_per_speed))`, capped by `m_radius_max`.

For each candidate $m$, traverse offsets `d = 0,1,...,radius` alternating signs (center-first shell expansion).

**Per sheet:**

1. Build linear seed $\ell_{\mathrm{init}}\approx P(\theta^{(m)},\varphi_0)^{\!\top}$ (reuse same $P$).
2. **Adaptive multi-start:** if $\max(|\theta^{(m)}|,|\varphi|) > 2.5$ use a small deterministic set (base seed plus axis perturbations scaled from $\ell_{\mathrm{init}}$, plus scale factors $0.6$ and $1.4$); otherwise **only** the base seed.
3. Run shooting with optional continuation (`T_schedule` non-empty in `api.cpp`).
4. **Score** candidates primary by $\|r\|_\infty$ and secondarily cost with tiny coefficient (`scoreCandidate`).
5. **Good-enough early exit:** if best $\|r\|_\infty\le$ `good_enough_resid` (default $10^{-10}$), return immediately (no remaining sheets/seeds).

**Debug hooks:** Setting environment variable **`PENDULUM_DEBUG=1`** enables `fprintf` diagnostics on **stderr** only (never stdout—the grader line stays clean).

---

## 6. Time integration inside forward simulation

Integration uses a fixed number of substeps $n=\lceil T/dt\rceil$ and step length $h=T/n$ (so the nominal `dt` is an upper bound, not necessarily exact divisor).

### 6.1 RK4 (**`cpp/solver/rk4.hpp`**)

Fourth-order classical Runge–Kutta applied to $\texttt{AugState}$.

### 6.2 Fixed-step DP5 (**`cpp/solver/dp5.hpp`**)

Dormand–Prince tableau (seven stages internally) evaluates the **order-5** update **without adaptive step sizing**. This preserves uniform control flow—a common requirement for SIMD/GPU parallelism later.

Switch: **`ShootSettings::integrator`** selects `RK4` vs **`DP5` (current default)**.

---

## 7. Repository layout relevant to this methodology

### 7.1 C++ pendulum solver (CPU)

| File | Role |
|------|------|
| **`src/solver.hpp`** | Public **`Result`** struct and **`solve(theta, phi, alpha)`** declaration (shared ABI with CUDA scaffold). |
| **`cpp/main.cpp`** | Parses CLI `$\theta,\varphi,\alpha$`, prints `l1 l2 cost` at 10 decimals (matches grading contract). |
| **`cpp/solver/api.cpp`** | Wires `solve(...)`: **`SheetSearchSettings`** thresholds, **`T_schedule`**, **`PENDULUM_DEBUG`** flag propagation. Single source of tuning defaults today. |
| **`cpp/solver/types.hpp`** | **`State`, `Costate`, `PhasePoint`, `PhaseDeriv**, operator overloads for vector-field stepping. |
| **`cpp/solver/dynamics.hpp/.cpp`** | **`uStar`**, **`runningCost`**, **`hamiltonianRHS`** defining the effective $H$ flow. |
| **`cpp/solver/cost.hpp/.cpp`** | **`KahanSum`** helper for $\int f_0\,\mathrm{d}t$. |
| **`cpp/solver/rk4.hpp`** | Generic template RK4 **`rk4Step`**. |
| **`cpp/solver/dp5.hpp`** | Generic template fixed-step **DP5** **`dp5Step`**. |
| **`cpp/solver/manifold_seed.hpp/.cpp`** | **`stableManifoldSeedP(alpha)`** via $J\,\mathrm{Hess}(H)$ eigendecomposition at origin. |
| **`cpp/solver/shooting.hpp/.cpp`** | Forward simulation **`simulateForward`** (state + sensitivities **`dZ/dL0`**), Jacobian helpers, **`solveCostatesSingleSheetLM`** and **`solveCostatesSingleSheetLMContinuation`**. |
| **`cpp/solver/sheet_search.hpp/.cpp`** | Multi-sheet **`solveWithSheetSearch`**: indexing $m$, seeds, continuation hook, scoring, early exit, debug prints. |

### 7.2 Tests and scaffolding

| File | Role |
|------|------|
| **`cpp/tests/smoke.cpp`** | Minimal deterministic checks (origin RHS null, RK4 perturbation sanity). |
| **`Makefile`** | Default **`solver`** builds CPU sources listing `cpp/...`; optional **`cuda`** target compiles **`src/*.cu`**. |

### 7.3 Not primary to this METHODOLOGY (but present)

| Path | Role |
|------|------|
| **`Eigen/main.cpp`** | Small Eigen eigenvalue demonstration (standalone; not wired to `./solver`). |
| **`tools/run_grader_conf.py`** | Local harness parses **`grader.conf`**, invokes **`BUILD_WITH`** and compares outputs. |

---

## 8. Design rationale and GPU portability notes

- **Variational Jacobian** removes per-iteration Jacobian finite-diff cost (critical when many candidates exist).
- **Fixed-step DP5** trades more RHS evaluations per step for accuracy and remains ** SIMD/GPU-friendly** (no branching on step rejection).
- **Multi-sheet × multi-start** is **embarrassingly parallel** in principle: assign one GPU thread block/warp per $(m,\text{seed})$ trajectory, replicate fixed-step integration, then reduce residuals.
- **`PENDULUM_DEBUG`** confines diagnostics to stderr so **`stdout`** remains machine-gradable JSON-like triple-output.

Future refinements discussed in iteration (not necessarily in code yet): quadratic tail corrections for $\int_{T}^{\infty}$, tighter polish passes (Richardson extrapolation between $h$ and $h/2$), merit functions beyond $\|r\|_\infty$ inside LM/backtracking.

---

## 9. Symbols quick reference

| Symbol | Meaning in code |
|--------|----------------|
| $\theta,\varphi$ | `theta`, `phi` |
| $\lambda_1,\lambda_2$ | `l1`, `l2` |
| $P$ | `stableManifoldSeedP(alpha)` $\to \texttt{Eigen::Matrix2d}$ |
| $z$ | `PhasePoint` $(x,\lambda)$, stacked as $\mathbb{R}^4$ in sensitivity |
| $S=\partial z/\partial\ell_0$ | **`ForwardSimOut::dZ_dL0`** |
| $r$ | **`ShootResult::resid`**, $\mathbb{R}^4$ if manifold residual on |
