#pragma once

// Plain C++/CUDA dual compilation for headers shared by host (.cpp/.cu) code.
#ifndef __CUDACC__
#define PEND_HD
#define PEND_HOST
#else
#include <cuda_runtime.h>
#define PEND_HD __host__ __device__
#define PEND_HOST __host__
#endif
