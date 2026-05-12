TARGET = solver
NVCC   = nvcc

# `module` is a shell function on many clusters; one shell for the whole `all` recipe.
SHELL := /bin/bash
.ONESHELL:

# Embed SASS for the GPUs you run on. UA Ocelote: P100 (sm_60); UA Puma: V100 (sm_70).
# If `solver` prints all zeros, a common cause is kernels built for the wrong arch (no match → failed launch).
# Override: make CUDA_GENCODE="-gencode arch=compute_80,code=sm_80"  # A100, etc.
CUDA_GENCODE ?= -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70
NVCCFLAGS = -O2 -Isrc $(CUDA_GENCODE) -diag-suppress=550,20012

# CUDA module name varies by cluster; run `module avail cuda` on a compute node. UA Puma often has
# cuda12/*; UA Ocelote often only has cuda11/11.8 — use:
#   make CUDA_MODULE=cuda11/11.8 MAX_HOST_GCC_MAJOR=11
CUDA_MODULE ?= cuda12
# Maximum host GCC major nvcc accepts depends on CUDA (e.g. 11 for CUDA 11.8; 13 for CUDA 12.4+).
# If nvcc still errors, lower this or pick a newer CUDA minor via CUDA_MODULE.
MAX_HOST_GCC_MAJOR ?= 13
# If default gcc is too new, try these modules in order until gcc is acceptable (site-specific names).
HOST_COMPILER_MODULES ?= gnu13 gnu13/13.2.0 gnu12 gnu12/12.2.0 gnu11 gnu11/11.3.0 gnu10 gnu10/10.3.0 gcc/11 compiler/gcc/11
# Optional: force the host compiler nvcc uses, e.g. `make NVCC_CCBIN=/opt/ohpc/pub/compiler/gcc/11.3.0/bin`
NVCC_CCBIN ?=

SRC = \
	src/main.cu \
	src/core/host_buffers.cu \
	src/warm_start/backward_ivp_batch.cu \
	src/shooting/gpu_eval_segments.cu \
	src/shooting/defect_jacobian_host.cu \
	src/shooting/newton_iteration.cu \
	src/shooting/multiple_shooting_solve.cu \
	src/driver/continuation_sheets.cu

# `all` loads modules first, then sets EIGEN_INC in the *same* shell (see below).
# Do not use $(shell pkg-config ...) at Makefile parse time: that runs before `module load`.
#
# Eigen must appear as $(EIGEN_INC)/Eigen/Dense. Many HPC "eigen" modules set EIGEN3_INCLUDE_DIR
# or only expose pkg-config after load; /usr/include/eigen3 often does not exist on compute nodes.
.PHONY: all test clean

$(TARGET): $(SRC) Makefile
	module load eigen
	module load $(CUDA_MODULE)
	gccver=$$(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion 2>/dev/null || echo 0); \
	gcc_major=$${gccver%%.*}; \
	if [ -z "$$gcc_major" ] || ! [ "$$gcc_major" -eq "$$gcc_major" ] 2>/dev/null; then gcc_major=99; fi; \
	if [ "$$gcc_major" -gt $(MAX_HOST_GCC_MAJOR) ]; then \
		for mod in $(HOST_COMPILER_MODULES); do \
			module load "$$mod" 2>/dev/null || true; \
			gccver=$$(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion 2>/dev/null || echo 0); \
			gcc_major=$${gccver%%.*}; \
			if [ -n "$$gcc_major" ] && [ "$$gcc_major" -eq "$$gcc_major" ] 2>/dev/null && [ "$$gcc_major" -le $(MAX_HOST_GCC_MAJOR) ]; then \
				break; \
			fi; \
		done; \
	fi; \
	gccver=$$(gcc -dumpfullversion 2>/dev/null || gcc -dumpversion 2>/dev/null || echo unknown); \
	gcc_major=$${gccver%%.*}; \
	if [ -z "$$gcc_major" ] || ! [ "$$gcc_major" -eq "$$gcc_major" ] 2>/dev/null; then gcc_major=99; fi; \
	if [ "$$gcc_major" -gt $(MAX_HOST_GCC_MAJOR) ]; then \
		echo "nvcc ($(CUDA_MODULE)) rejects host GCC major > $(MAX_HOST_GCC_MAJOR); found GCC $$gccver ($$(command -v gcc 2>/dev/null))." >&2; \
		echo "Try: module avail cuda gnu   then make CUDA_MODULE=... MAX_HOST_GCC_MAJOR=... matching that CUDA release." >&2; \
		echo "Or: make NVCC_CCBIN=/path/to/acceptable-g++/bin" >&2; \
		echo "Last resort: add -allow-unsupported-compiler to NVCCFLAGS (not recommended)." >&2; \
		exit 1; \
	fi; \
	NVCC_EXTRA=""; \
	if [ -n "$(strip $(NVCC_CCBIN))" ]; then NVCC_EXTRA="-ccbin $(NVCC_CCBIN)"; fi; \
	EIGEN_INC_RESOLVED=""; \
	if [ -n "$$EIGEN_INC" ] && [ -f "$$EIGEN_INC/Eigen/Dense" ]; then \
		EIGEN_INC_RESOLVED="$$EIGEN_INC"; \
	elif [ -n "$$EIGEN3_INCLUDE_DIR" ] && [ -f "$$EIGEN3_INCLUDE_DIR/Eigen/Dense" ]; then \
		EIGEN_INC_RESOLVED="$$EIGEN3_INCLUDE_DIR"; \
	else \
		for pc in eigen3 eigen Eigen3; do \
			d=$$(pkg-config --variable=includedir "$$pc" 2>/dev/null); \
			if [ -n "$$d" ] && [ -f "$$d/Eigen/Dense" ]; then EIGEN_INC_RESOLVED="$$d"; break; fi; \
			pfx=$$(pkg-config --variable=prefix "$$pc" 2>/dev/null); \
			if [ -n "$$pfx" ] && [ -f "$$pfx/include/eigen3/Eigen/Dense" ]; then \
				EIGEN_INC_RESOLVED="$$pfx/include/eigen3"; break; \
			fi; \
		done; \
	fi; \
	if [ -z "$$EIGEN_INC_RESOLVED" ]; then \
		for flag in $$(pkg-config --cflags-only-I eigen3 2>/dev/null) $$(pkg-config --cflags-only-I eigen 2>/dev/null); do \
			case "$$flag" in \
				-I*) d="$${flag#-I}"; \
					if [ -f "$$d/Eigen/Dense" ]; then EIGEN_INC_RESOLVED="$$d"; break; \
					elif [ -f "$$d/eigen3/Eigen/Dense" ]; then EIGEN_INC_RESOLVED="$$d/eigen3"; break; \
					fi ;; \
			esac; \
		done; \
	fi; \
	if [ -z "$$EIGEN_INC_RESOLVED" ]; then \
		for d in /usr/include/eigen3 /usr/local/include/eigen3; do \
			if [ -f "$$d/Eigen/Dense" ]; then EIGEN_INC_RESOLVED="$$d"; break; fi; \
		done; \
	fi; \
	if [ -z "$$EIGEN_INC_RESOLVED" ] || [ ! -f "$$EIGEN_INC_RESOLVED/Eigen/Dense" ]; then \
		echo "Could not find Eigen headers (expect .../Eigen/Dense). After 'module load eigen', try:" >&2; \
		echo "  export EIGEN_INC=/path/containing/Eigen   # directory that contains Eigen/Dense" >&2; \
		echo "or: export EIGEN3_INCLUDE_DIR=/same/path" >&2; \
		exit 1; \
	fi; \
	$(NVCC) $(NVCCFLAGS) $$NVCC_EXTRA -isystem "$$EIGEN_INC_RESOLVED" $(SRC) -o $(TARGET)

all: $(TARGET)

test: $(TARGET)
	python tools/run_grader.py

clean:
	rm -f $(TARGET)
