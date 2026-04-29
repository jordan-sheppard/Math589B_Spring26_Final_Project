#pragma once

#include <cstddef> // For std::size_t


struct Mat4x4 {
    float data[16];

    // --- 2D Accessors ---
    // Allows you to use M(row, col) instead of M[row * 4 + col]
    __host__ __device__ float& operator()(int r, int c) {
        return data[r * 4 + c];
    }
    __host__ __device__ const float& operator()(int r, int c) const {
        return data[r * 4 + c];
    }

    // --- Matrix Multiplication ---
    __host__ __device__
    Mat4x4 operator*(const Mat4x4& other) const {
        Mat4x4 result;
        
        // Force the CUDA compiler to flatten these loops into straight-line code for performance
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                result(r, c) = (*this)(r, 0) * other(0, c) + 
                               (*this)(r, 1) * other(1, c) + 
                               (*this)(r, 2) * other(2, c) + 
                               (*this)(r, 3) * other(3, c);
            }
        }
        return result;
    }
};


struct VarState {
    float s[5];     // 4D physical state [theta, phi, l1, l2] + cost
    float M[16];    // 4x4 sensitivity matrix to initial conditions

    // --- Inline Reference Accessors --

    __host__ __device__ float& theta() { return s[0]; }
    __host__ __device__ const float& theta() const { return s[0]; }

    __host__ __device__ float& phi() { return s[1]; }
    __host__ __device__ const float& phi() const { return s[1]; }

    __host__ __device__ float& l1() { return s[2]; }
    __host__ __device__ const float& l1() const { return s[2]; }

    __host__ __device__ float& l2() { return s[3]; }
    __host__ __device__ const float& l2() const { return s[3]; }

    __host__ __device__ float& cost() { return s[4]; }
    __host__ __device__ const float& cost() const { return s[4]; }

    // Overload Addition/Subtraction/Scalar Multiplication
    __host__ __device__ VarState operator+(const VarState& other) const {
        VarState result;
        for(int i = 0; i < 5; i++) result.s[i] = this->s[i] + other.s[i];
        for(int i = 0; i < 16; i++) result.M[i] = this->M[i] + other.M[i];
        return result;
    }

    __host__ __device__ VarState operator-(const VarState& other) const {
        VarState result;
        for(int i = 0; i < 5; i++) result.s[i] = this->s[i] - other.s[i];
        for(int i = 0; i < 16; i++) result.M[i] = this->M[i] - other.M[i];
        return result;
    }

    __host__ __device__ VarState operator*(float scalar) const {
        VarState result;
        for(int i = 0; i < 5; i++) result.s[i] = this->s[i] * scalar;
        for(int i = 0; i < 16; i++) result.M[i] = this->M[i] * scalar;
        return result;
    }
};

// Makes scalar multiplication for VarState okay on both sides
__host__ __device__
inline VarState operator*(double scalar, const VarSate& vec) {
    return vec * scalar;
}


struct Result {
    double l1;
    double l2;
    double cost;
};

Result solve(double theta, double phi, double alpha);
