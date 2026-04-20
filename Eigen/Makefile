CXX = g++
CXXFLAGS = -O0 -std=c++17 -Wall

EIGEN_INC ?= $(shell pkg-config --cflags eigen3 2>/dev/null | sed 's/-I//')

ifeq ($(EIGEN_INC),)
EIGEN_INC = /usr/include/eigen3
endif

all: main

main: main.cpp
	$(CXX) $(CXXFLAGS) -I$(EIGEN_INC) $< -o $@

clean:
	rm -f main
