#pragma once

#include <cstddef> // For std::size_t

constexpr std::size_t NUM_STATE_DIMS = 5;

struct StateVec {
    double data[NUM_STATE_DIMS];

    __host__ __device__ StateVec() {}

    __host__ __device__ double get(int i) const { return data[i]; }
    __host__ __device__ void set(int i, double val) { data[i] = val; }

    __host__ __device__
    StateVec operator+(const StateVec& other) const {
        StateVec result;
	for (std::size_t i = 0; i < NUM_STATE_DIMS; ++i) {
            result.data[i] = this->data[i] + other.data[i];
	}
	return result;
    }

    __host__ __device__
    StateVec operator-(const StateVec& other) const {
        StateVec result;
        for (std::size_t i = 0; i < NUM_STATE_DIMS; ++i) {
            result.data[i] = this->data[i] - other.data[i];
        }
        return result;
    }

    __host__ __device__
    StateVec operator*(double scalar) const {
	StateVec result;
	for (std::size_t i = 0; i < NUM_STATE_DIMS; ++i) {
            result.data[i] = this->data[i] * scalar;
        }
        return result;
    }
};

// Make scalar multiplication okay on both sides
__host__ __device__
inline StateVec operator*(double scalar, const StateVec& vec) {
    return vec * scalar;
}

struct Result {
    double l1;
    double l2;
    double cost;
};

Result solve(double theta, double phi, double alpha);
