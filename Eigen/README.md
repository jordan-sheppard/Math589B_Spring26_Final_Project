# Eigen Eigenvalue Demo

This is a minimal example showing how to compute eigenvalues using the
Eigen C++ library.

---

## 🧠 Why Eigen?

We use **Eigen** because:

- It is **header-only** (no complicated linking)
- It works reliably on HPC systems
- It avoids configuration issues common with LAPACK/BLAS

This lets you focus on **math and algorithms**, not build problems.

---

## ⚙️ How to build

### Step 1: Load Eigen (on HPC)

On the HPC system, run:

```bash
module load eigen
# Eigen Eigenvalue Demo

This is a minimal example showing how to compute eigenvalues using Eigen.

## Why Eigen?

- Header-only (no linking headaches)
- Works reliably on HPC
- Avoids LAPACK configuration issues

## Build

```bash
make
