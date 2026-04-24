TARGET = solver
NVCC   = nvcc
NVCCFLAGS = -O2 -arch=sm_60
EIGEN_INC ?= $(shell pkg-config --cflags eigen3 2>/dev/null | sed 's/-I//')

ifeq ($(EIGEN_INC),)
EIGEN_INC = /usr/include/eigen3
endif

SRC = src/main.cu src/solver.cu

all:
	$(NVCC) $(NVCCFLAGS) -isystem $(EIGEN_INC) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
