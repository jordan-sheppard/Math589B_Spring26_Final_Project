TARGET = solver
NVCC   = nvcc

# `module` is a shell function on many clusters; bash keeps one shell for the full recipe line.
SHELL := /bin/bash

# NVCCFLAGS = -O2 -arch=sm_60
NVCCFLAGS = -g -G -O0 -arch=sm_60 -Isrc

EIGEN_INC ?= $(shell pkg-config --cflags eigen3 2>/dev/null | sed 's/-I//')

ifeq ($(EIGEN_INC),)
EIGEN_INC = /usr/include/eigen3
endif

SRC = \
	src/main.cu \
	src/core/host_buffers.cu \
	src/shooting/gpu_eval_segments.cu \
	src/shooting/defect_jacobian_host.cu \
	src/shooting/newton_iteration.cu \
	src/shooting/multiple_shooting_solve.cu \
	src/driver/continuation_sheets.cu

all:
	module load eigen && module load cuda11/11.8 && $(NVCC) $(NVCCFLAGS) -isystem $(EIGEN_INC) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
