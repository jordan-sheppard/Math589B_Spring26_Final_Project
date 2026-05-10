# METHODOLOGY: CUDA Multiple Shooting + Levenberg–Marquardt

This document specifies the mathematical methodology for the CUDA implementation under `src/`: parallel **multiple shooting** on the Hamiltonian flow for the course pendulum optimal-control problem, a **rectangular** residual of dimension **m = n + 2**, and **damped Gauss–Newton / Levenberg–Marquardt least squares** at each outer iteration. **Terminal manifold consistency** at the truncated horizon is enforced via **two extra scalar equations** alongside continuity and boundary matching, without row weighting or penalty parameters.

The public API is the assignment scaffold: `solve(theta, phi, alpha)` in [`src/solver.hpp`](src/solver.hpp), implemented by the driver in [`src/driver/continuation_sheets.cu`](src/driver/continuation_sheets.cu).

---

## 1. Problem statement

### 1.1 Dynamics

The controlled pendulum (angle $\theta$ from the **upward** vertical, $\varphi = \dot\theta$) is

$$
\ddot{\theta} = \sin\theta - \alpha\dot{\theta} + u\cos\theta ,
$$

with friction parameter $\alpha \ge 0$ and scalar control $u$. First-order state $x = (\theta,\varphi) \in \mathbb{R}^2$.

### 1.2 Running cost and objective

The infinite-horizon objective is

$$
J = \int_0^\infty \Bigl[(1-\cos\theta) + \tfrac12 \varphi^2 + \tfrac12 u^2\Bigr]\,\mathrm{d}t .
$$

### 1.3 Pontryagin necessary conditions and eliminated Hamiltonian flow

Minimizing the Hamiltonian pointwise over $u$ gives the optimal feedback

$$
u^\ast = -\lambda_2\cos\theta .
$$

Substituting into the adjoint equations yields an autonomous ODE for the **phase vector**

$$
z = (\theta,\varphi,\lambda_1,\lambda_2)^\top \in \mathbb{R}^4 .
$$

The implementation in [`src/dynamics/pendulum_oc.cuh`](src/dynamics/pendulum_oc.cuh) uses the following eliminated-$u$ right-hand side (consistent ordering of components):

$$
\begin{aligned}
\dot{\theta} &= \varphi, \\
\dot{\varphi} &= \sin\theta - \alpha\varphi - \lambda_2\cos^2\theta, \\
\dot{\lambda}_1 &= -\sin\theta - \lambda_2\cos\theta - \lambda_2^2\cos\theta\sin\theta, \\
\dot{\lambda}_2 &= -\varphi - \lambda_1 + \alpha\lambda_2 .
\end{aligned}
$$

Denote this field by $F_\alpha(z)$. Trajectories that converge to the upright equilibrium and satisfy infinite-horizon optimality are sought on the **stable manifold** of the saddle-type equilibrium at $(\theta,\varphi,\lambda)=(0,0,0)$; the truncated-horizon numerical method approximates that selection via boundary conditions and the linear map $P$ in §3–§5.

### 1.4 Running cost density along a trajectory

With $u^\ast = -\lambda_2\cos\theta$, the integrand for $J$ along a trajectory is

$$
f_0(z) = (1-\cos\theta) + \tfrac12 \varphi^2 + \tfrac12 (\lambda_2\cos\theta)^2 .
$$

---

## 2. What we compute

Given $(\theta_0,\varphi_0,\alpha)$, the executable prints three numbers:

- $\lambda_1(0)$, $\lambda_2(0)$ at the **physical** initial state (after any sheet bookkeeping in the driver), and  
- $J$, approximated by accumulating the running cost along the converged discrete trajectory on a **truncated** horizon implied by the shooting discretization.

Numerically, $\lambda_1(0)$ and $\lambda_2(0)$ are the costate components at **node 0** of the multiple-shooting chain once the nonlinear least-squares problem is solved.

---

## 3. Stable manifold linear map $P$

Near $(\theta,\varphi,\lambda)=(0,0,0)$, the stable manifold of the Hamiltonian equilibrium is approximated locally by correlating costates with position:

$$
\lambda \approx P x ,\qquad P \in \mathbb{R}^{2\times 2},\quad x=(\theta,\varphi)^\top .
$$

**Construction.** Let $H(z)$ denote the effective Hamiltonian after eliminating $u$. Form the Hessian $\nabla^2 H$ at the origin in the variable ordering

$$
(\theta,\varphi,\lambda_1,\lambda_2).
$$

Define the canonical symplectic matrix

$$
J_{\mathrm{symp}} =
\begin{pmatrix}
0_{2\times 2} & I_{2\times 2} \\
-I_{2\times 2} & 0_{2\times 2}
\end{pmatrix},
$$

and the **Hamiltonian matrix**

$$
C = J_{\mathrm{symp}}\,\nabla^2 H(0) .
$$

Compute the eigenpairs of $C$. The eigenvectors whose eigenvalues have **negative real part** span the **stable subspace** (dimension two). Stack those two eigenvectors and partition them into blocks corresponding to $(\theta,\varphi)$ and $(\lambda_1,\lambda_2)$:

$$
\begin{pmatrix} V_{s1} \\ V_{s2} \end{pmatrix},
\qquad V_{s1}, V_{s2} \in \mathbb{C}^{2\times 2}.
$$

Define

$$
P = \operatorname{real}\bigl(V_{s2}\, V_{s1}^{-1}\bigr).
$$

In code, $P$ is computed **on the host** (e.g. with Eigen’s complex eigensolver) once per outer solve or once per $\alpha$. It is used for:

1. **Optional warm starts** for nodal costates (e.g. $\lambda_k^{(0)} \approx P\, x_k^{(0)}$ on the chosen $\theta$-sheet).  
2. **Two extra residual rows** at the **terminal forward image** $\hat z_{N-1}$ (§5.3), enforcing $\lambda \approx P x$ at the truncated horizon. All residual rows share the same weight in the least-squares objective (§5.3).

---

## 4. Multiple shooting discretization

### 4.1 Nodes, segments, and flow map

Fix **N** shooting intervals and **N** nodes

$$
z_k \in \mathbb{R}^4,\quad k=0,\ldots,N-1 .
$$

Node components are $z_k = (\theta_k,\varphi_k,\lambda_{1,k},\lambda_{2,k})^\top$.

For each segment $k$, let $\Phi_k(z_k)$ be the **forward Hamiltonian flow map** obtained by integrating $\dot z = F_\alpha(z)$ from initial condition $z_k$ over a fixed local horizon (fixed $\Delta t$ and fixed number of substeps per segment in [`src/integrators/segment_integration.cuh`](src/integrators/segment_integration.cuh)). Integration is performed **in parallel on the GPU**: one CUDA thread (or indexed launch) per segment.

Define the **segment endpoint**

$$
\hat z_k := \Phi_k(z_k).
$$

### 4.2 Cost functional along segments

Along each segment, accumulate $f_0(z)$ from §1.4. The printed total $J$ is the sum of contributions over segments (finer quadrature or compensated summation may be added in implementation without changing the BVP above).

---

## 5. Residual vector $r$: square core + two manifold rows

Stack all unknowns in order:

$$
Z = \begin{bmatrix} z_0 \\ z_1 \\ \vdots \\ z_{N-1} \end{bmatrix} \in \mathbb{R}^n,\qquad n = 4N .
$$

### 5.1 Continuity defects (interior interfaces)

For $k=0,\ldots,N-2$, require **continuity** of the full phase vector across interfaces:

$$
r^{(\mathrm{cont})}_k := \hat z_k - z_{k+1} \in \mathbb{R}^4 .
$$

This yields **4(N−1)** scalar equations.

### 5.2 Boundary conditions (physical endpoints)

The implementation fixes the **initial physical state** and targets a **terminal physical state** at the **end of the last segment’s forward map**:

- **Initial:** match $(\theta,\varphi)$ at node $0$ to $(\theta_0,\varphi_0)$ supplied by the driver (possibly after sheet shifting / continuation in the outer loop).  
- **Terminal:** match $(\theta,\varphi)$ at $\hat z_{N-1}$ to $(\theta^\star,\varphi^\star)$ encoded by `theta_goal`, `phi_goal` in [`SystemParams`](src/core/solver_types.cuh).

That adds **4** scalar equations (two for the initial node, two involving the **forward** terminal state $\hat z_{N-1}$).

Organize the **square** part as $r^{(\mathrm{sq})} \in \mathbb{R}^{4N}$: block rows for each segment index $k=0,\ldots,N-1$, where blocks $k=0,\ldots,N-2$ carry $r^{(\mathrm{cont})}_k$, and the **last** block encodes the four boundary conditions (initial $(\theta,\varphi)$ on $z_0$ and terminal $(\theta,\varphi)$ on $\hat z_{N-1}$). This matches the sparsity pattern produced in [`src/shooting/defect_jacobian_host.cu`](src/shooting/defect_jacobian_host.cu).

### 5.3 Two additional manifold rows (unweighted)

Let $x(\hat z_{N-1})$ denote $(\theta,\varphi)$ extracted from $\hat z_{N-1}$, and $\lambda(\hat z_{N-1})$ the costate pair. Define

$$
r^{(\mathrm{man})} :=
\begin{bmatrix}
\lambda_1(\hat z_{N-1}) \\
\lambda_2(\hat z_{N-1})
\end{bmatrix}
-
P
\begin{bmatrix}
\theta(\hat z_{N-1}) \\
\varphi(\hat z_{N-1})
\end{bmatrix}
\in \mathbb{R}^2 .
$$

The **full** residual is

$$
r(Z) :=
\begin{bmatrix}
r^{(\mathrm{sq})}(Z) \\ r^{(\mathrm{man})}(Z)
\end{bmatrix}
\in \mathbb{R}^m,\qquad m = 4N + 2 = n + 2 .
$$

**Design choice:** there are **no** scalar weights: every row of $r$ enters $\tfrac12\|r\|_2^2$ with coefficient $1$. Overdetermination is resolved by **least squares** and **LM damping** (§6).

---

## 6. Jacobian $J = \partial r / \partial Z$ and LM least squares

### 6.1 Segment sensitivities

Along segment $k$, propagate the variational matrix

$$
M_k(t) := \frac{\partial \Phi_k(z_k)}{\partial z_k} \in \mathbb{R}^{4\times 4},
$$

via the linearized ODE $\dot M_k = A(z_k(t))\, M_k$, $M_k(0)=I$, where $A(z) = D_z F_\alpha(z)$ is implemented as [`compute_sensitivity_jacobian`](src/dynamics/pendulum_oc.cuh). At segment end, $\partial \hat z_k / \partial z_k = M_k^{\mathrm{end}}$.

This yields **sparse** Jacobian entries for continuity defects (derivatives of $\hat z_k - z_{k+1}$ with respect to $z_k$ and $z_{k+1}$).

### 6.2 Rows for $r^{(\mathrm{man})}$

Write $\hat z_{N-1} = \Phi_{N-1}(z_{N-1})$. Let $e_\theta,e_\varphi,e_{\lambda_1},e_{\lambda_2}$ be the standard basis vectors of $\mathbb{R}^4$ aligned with $(\theta,\varphi,\lambda_1,\lambda_2)$. For $\ell \in \{1,2\}$,

$$
\frac{\partial}{\partial z_{N-1}}
\Bigl(
\lambda_\ell(\hat z_{N-1}) - \sum_{j=1}^2 P_{\ell j}\, x_j(\hat z_{N-1})
\Bigr)
=
e_{\lambda_\ell}^\top M_{N-1}^{\mathrm{end}}
-
P_{\ell 1}\, e_{\theta}^\top M_{N-1}^{\mathrm{end}}
-
P_{\ell 2}\, e_{\varphi}^\top M_{N-1}^{\mathrm{end}} .
$$

Each manifold row is a fixed linear combination of the **rows** of $M_{N-1}^{\mathrm{end}}$ and contributes Jacobian entries only in columns belonging to **node $N-1$**.

### 6.3 Levenberg–Marquardt damping (normal equations)

At iteration $i$, let $J = J(Z^{(i)}) \in \mathbb{R}^{m\times n}$ and $r = r(Z^{(i)})$. The step $\delta \in \mathbb{R}^n$ solves the **damped normal equations**

$$
\bigl(J^\top J + \mu I\bigr)\, \delta = -J^\top r ,
$$

with $\mu > 0$. Typical safeguards:

- **Damping schedule:** increase $\mu$ when the step fails a merit test, decrease on success.  
- **Step clipping:** bound $\|\delta\|$ by a maximum norm.  
- **Backtracking:** scale $\delta$ until $\|r\|_\infty$ or $\|r\|_2$ decreases relative to the previous iterate.

For modest $N$, forming $J^\top J$ or solving the normal equations with dense Cholesky / `LDLT` on an $n\times n$ system on the host is reasonable ($n = 4N$, e.g. $N=20 \Rightarrow n=80$). The expensive work remains segment integration on the GPU.

This formulation treats **overdetermined** residuals ($m>n$) directly. A square Newton solve $J\delta=-r$ would not apply when $m>n$.

---

## 7. Outer algorithm (conceptual)

1. **Initialize** nodal guesses $Z^{(0)}$ (e.g. linear interpolation in $(\theta,\varphi)$ and optional $P$-based costate seed).  
2. **GPU:** evaluate all segments in parallel; copy $\hat z_k$ and $M_k^{\mathrm{end}}$ to host.  
3. **Host:** assemble $r$ and sparse $J$ of size $m\times n$.  
4. **Host:** compute LM step $\delta$; update $Z \leftarrow Z + \delta$ (with clipping/backtracking).  
5. Repeat until $\|r\|_\infty \le \varepsilon_{\mathrm{tol}}$ or iteration cap.

The driver may wrap this loop in **homotopy** on initial conditions and **multi-sheet** search in $\theta$ modulo $2\pi$—see [`src/driver/continuation_sheets.cu`](src/driver/continuation_sheets.cu).

---

## 8. Symbols quick reference

| Symbol | Meaning |
|--------|---------|
| $z,\hat z$ | Phase point; forward image after segment integration |
| $Z$ | Stacked nodal unknowns $(z_0,\ldots,z_{N-1})$ |
| $n$ | $4N$ |
| $m$ | $n+2$ |
| $P$ | Stable-manifold linear map $\mathbb{R}^2\to\mathbb{R}^2$ for $(\lambda_1,\lambda_2)$ vs $(\theta,\varphi)$ |
| $M_k^{\mathrm{end}}$ | $4\times 4$ sensitivity $\partial \hat z_k / \partial z_k$ |
| $\mu$ | LM damping parameter |
| $J,r$ | Jacobian and residual for LM–LS |

---

## 9. Implementation map (intended code locations)

| Idea | Target location (conceptual) |
|------|-------------------------------|
| Hamiltonian RHS + $A=D_z F$ | [`src/dynamics/pendulum_oc.cuh`](src/dynamics/pendulum_oc.cuh) |
| RK (or future DP5) segment + $M$ | [`src/integrators/segment_integration.cuh`](src/integrators/segment_integration.cuh) |
| Parallel segment eval | [`src/shooting/gpu_eval_segments.cu`](src/shooting/gpu_eval_segments.cu) |
| Build $r$, $J$ ($m\times n$) | [`src/shooting/defect_jacobian_host.cu`](src/shooting/defect_jacobian_host.cu) |
| LM normal-equation solve | [`src/shooting/newton_iteration.cu`](src/shooting/newton_iteration.cu) |
| Outer MS loop | [`src/shooting/multiple_shooting_solve.cu`](src/shooting/multiple_shooting_solve.cu) |
| Continuation / sheets | [`src/driver/continuation_sheets.cu`](src/driver/continuation_sheets.cu) |
| Host $P(\alpha)$ | Eigen-based helper (new translation unit or header), called from driver or assembly |

---

## 10. Optional refinements

- Fixed-step **DP5** instead of RK4 for smaller time-discretization bias.  
- **Continuation** in total horizon (schedule of segment counts or $\Delta t$).  
- **Trapezoidal quadrature** and **compensated summation** for $J$.  
- Richer **multi-start** sheet search for large $|(\theta,\varphi)|$.

These are numerical refinements; the core BVP definition remains **rectangular** $r$, sparse $J$, LM–LS, and **m = n + 2** with unweighted manifold rows.
