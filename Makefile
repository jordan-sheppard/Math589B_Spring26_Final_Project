TARGET = solver
NVCC   = nvcc

# `module` is a shell function on many clusters; one shell for the whole `all` recipe.
SHELL := /bin/bash
.ONESHELL:

# NVCCFLAGS = -O2 -arch=sm_60
NVCCFLAGS = -g -G -O0 -arch=sm_60 -Isrc

SRC = \
	src/main.cu \
	src/core/host_buffers.cu \
	src/shooting/gpu_eval_segments.cu \
	src/shooting/defect_jacobian_host.cu \
	src/shooting/newton_iteration.cu \
	src/shooting/multiple_shooting_solve.cu \
	src/driver/continuation_sheets.cu

# `all` loads modules first, then sets EIGEN_INC in the *same* shell (see below).
# Do not use $(shell pkg-config ...) at Makefile parse time: that runs before `module load`.
all:
	module load eigen
	module load cuda11/11.8
	if [ -z "$$EIGEN_INC" ]; then \
		EIGEN_INC=$$(pkg-config --variable=includedir eigen3 2>/dev/null); \
	fi
	if [ -z "$$EIGEN_INC" ]; then \
		EIGEN_INC=/usr/include/eigen3; \
	fi
	$(NVCC) $(NVCCFLAGS) -isystem "$$EIGEN_INC" $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
