TARGET = solver

# CPU build (default for macOS without CUDA)
CXX ?= clang++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic

EIGEN_CFLAGS := $(shell pkg-config --cflags eigen3 2>/dev/null)
ifeq ($(strip $(EIGEN_CFLAGS)),)
EIGEN_CFLAGS :=
ifneq ("$(wildcard /opt/homebrew/include/eigen3/Eigen/Core)","")
EIGEN_CFLAGS += -I/opt/homebrew/include/eigen3
else ifneq ("$(wildcard /usr/local/include/eigen3/Eigen/Core)","")
EIGEN_CFLAGS += -I/usr/local/include/eigen3
else ifneq ("$(wildcard /usr/include/eigen3/Eigen/Core)","")
EIGEN_CFLAGS += -I/usr/include/eigen3
endif
endif

CPU_SRC = \
	cpp/main.cpp \
	cpp/solver/api.cpp \
	cpp/solver/cost.cpp \
	cpp/solver/dynamics.cpp \
	cpp/solver/manifold_seed.cpp \
	cpp/solver/shooting.cpp \
	cpp/solver/sheet_search.cpp

all: cpu

cpu:
	$(CXX) $(CXXFLAGS) -Isrc -I. $(EIGEN_CFLAGS) $(CPU_SRC) -o $(TARGET)

# CUDA build (for CUDA machines only): ODE / sensitivities are CUDA-compiled
# (__host__/__device__) with no Eigen on device; Eigen is only used in
# src/host/manifold_seed.cpp for the stable-manifold seed P.
NVCC ?= nvcc
NVCCFLAGS ?= -O2 -std=c++20 -Isrc -I.
CUDA_SRC = \
	src/main.cu \
	src/solver.cu \
	src/cuda/forward_batch.cu \
	src/host/manifold_seed.cpp \
	src/host/shooting_host.cpp \
	src/host/sheet_search.cpp

# std::thread (sheet_search) — pass pthread to the host toolchain; nvcc rejects bare -pthread.
CUDA_HOST_PTHREAD ?= -Xcompiler -pthread

cuda:
	$(NVCC) $(NVCCFLAGS) $(CUDA_HOST_PTHREAD) $(EIGEN_CFLAGS) $(CUDA_SRC) -o $(TARGET)

SMOKE_TARGET = smoke
smoke:
	$(CXX) $(CXXFLAGS) -Isrc -I. $(EIGEN_CFLAGS) cpp/tests/smoke.cpp cpp/solver/cost.cpp cpp/solver/dynamics.cpp -o $(SMOKE_TARGET)
	./$(SMOKE_TARGET)

grade-local: all
	python3 tools/run_grader_conf.py

clean:
	rm -f $(TARGET)
	rm -f $(SMOKE_TARGET)
