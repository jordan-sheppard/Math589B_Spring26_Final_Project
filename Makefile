TARGET = solver
NVCC   = nvcc

# `module` is a shell function on many clusters; one shell for the whole `all` recipe.
SHELL := /bin/bash
.ONESHELL:

# NVCCFLAGS = -O2 -arch=sm_60
NVCCFLAGS = -O2 -Isrc

SRC = \
	src/main.cu \
	src/core/manifold_seed.cu \
	src/shooting/stable_patch_grid.cu \
	src/shooting/patch_refine_newton.cu \
	src/driver/continuation_sheets.cu

# `all` loads modules first, then sets EIGEN_INC in the *same* shell (see below).
# Do not use $(shell pkg-config ...) at Makefile parse time: that runs before `module load`.
#
# Eigen must appear as $(EIGEN_INC)/Eigen/Dense. Many HPC "eigen" modules set EIGEN3_INCLUDE_DIR
# or only expose pkg-config after load; /usr/include/eigen3 often does not exist on compute nodes.
.PHONY: all test clean

$(TARGET): $(SRC) Makefile
	module load eigen
	module load cuda11/11.8
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
	$(NVCC) $(NVCCFLAGS) -isystem "$$EIGEN_INC_RESOLVED" $(SRC) -o $(TARGET)

all: $(TARGET)

test: $(TARGET)
	python tools/run_grader.py

clean:
	rm -f $(TARGET)
