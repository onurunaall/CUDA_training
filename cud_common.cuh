#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>

// Macro for CUDA error checking
#define CUDA_ERROR_CHECK(call) { cudaCheckError((call), __FILE__, __LINE__); }

inline void cudaCheckError(cudaError_t errorCode, const char* file, int line, bool abort = true) {
    if (errorCode != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s | File: %s | Line: %d\n", cudaGetErrorString(errorCode), file, line);
        if (abort) exit(errorCode);
    }
}