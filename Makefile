TARGET = solver

# CPU build (default for macOS without CUDA)
CXX ?= clang++
CXXFLAGS ?= -O2 -std=c++20 -Wall -Wextra -Wpedantic

# Checkpoint 1: build a tiny CPU stub so the grader harness can run locally.
CPU_SRC = cpp_stub/main.cpp cpp_stub/solver_stub.cpp

all: cpu

cpu:
	$(CXX) $(CXXFLAGS) -Isrc $(CPU_SRC) -o $(TARGET)

# CUDA build (for CUDA machines only)
NVCC ?= nvcc
NVCCFLAGS ?= -O2
CUDA_SRC = src/main.cu src/solver.cu

cuda:
	$(NVCC) $(NVCCFLAGS) $(CUDA_SRC) -o $(TARGET)

grade-local: all
	python3 tools/run_grader_conf.py

clean:
	rm -f $(TARGET)
