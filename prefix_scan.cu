#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include "cuda_common.cuh"

#define BLOCK_SIZE 128 // Number of threads per block

// Macro for CUDA error checking
#define CUDA_ERROR_CHECK(call) { cudaCheckError((call), __FILE__, __LINE__); }

inline void cudaCheckError(cudaError_t errorCode, const char* file, int line, bool abort = true) {
    if (errorCode != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s | File: %s | Line: %d\n", cudaGetErrorString(errorCode), file, line);
        if (abort) exit(errorCode);
    }
}

// Kernel for performing block-level scan with auxiliary array
__global__ void blockScanKernel(int* input, int* blockSums, int inputSize) {
    int localIdx = threadIdx.x; // Thread index within the block
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x; // Global index in the input array

    extern __shared__ int sharedMemory[];

    // Load data into shared memory
    sharedMemory[localIdx] = (globalIdx < inputSize) ? input[globalIdx] : 0;
    __syncthreads();

    // Up-sweep (reduce) phase
    int step = 1;
    for (int depth = 1; depth < blockDim.x; depth *= 2) {
        if ((localIdx + 1) % (step * 2) == 0) {
            sharedMemory[localIdx] += sharedMemory[localIdx - step];
        }
        step *= 2;
        __syncthreads();
    }

    // Save the block sum for further processing
    if (localIdx == blockDim.x - 1) {
        blockSums[blockIdx.x] = sharedMemory[localIdx];
    }

    // Down-sweep phase
    if (localIdx == blockDim.x - 1) {
        sharedMemory[localIdx] = 0;
    }
    __syncthreads();

    step /= 2;
    for (int depth = blockDim.x / 2; depth > 0; depth /= 2) {
        if ((localIdx + 1) % (step * 2) == 0) {
            int temp = sharedMemory[localIdx - step];
            sharedMemory[localIdx - step] = sharedMemory[localIdx];
            sharedMemory[localIdx] += temp;
        }
        step /= 2;
        __syncthreads();
    }

    // Write results back to the input array
    if (globalIdx < inputSize) {
        input[globalIdx] = sharedMemory[localIdx];
    }
}

// Kernel for adding block-level results to input
__global__ void addBlockSumsKernel(int* input, int* blockSums, int inputSize) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && globalIdx < inputSize) {
        input[globalIdx] += blockSums[blockIdx.x - 1];
    }
}

// Host function to perform efficient scan
void runEfficientScan(int argc, char** argv) {
    int inputSize = 1 << 23; // Default input size (8 million elements)
    if (argc > 1) {
        inputSize = 1 << atoi(argv[1]);
    }

    int inputBytes = inputSize * sizeof(int);
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int blockSumsBytes = gridDim.x * sizeof(int);

    // Allocate host memory
    int* h_input = (int*)malloc(inputBytes);
    int* h_result = (int*)malloc(inputBytes);

    // Initialize input array
    for (int i = 0; i < inputSize; ++i) {
        h_input[i] = 1;
    }

    printf("Launching kernel with gridDim.x = %d, blockDim.x = %d, memory = %d MB\n",
           gridDim.x, blockDim.x, inputBytes / (1024 * 1024));

    // Allocate device memory
    int *d_input, *d_blockSums;
    CUDA_ERROR_CHECK(cudaMalloc(&d_input, inputBytes));
    CUDA_ERROR_CHECK(cudaMalloc(&d_blockSums, blockSumsBytes));

    // Copy input data to device
    CUDA_ERROR_CHECK(cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice));

    // Launch scan kernel
    blockScanKernel<<<gridDim, blockDim, BLOCK_SIZE * sizeof(int)>>>(d_input, d_blockSums, inputSize);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Perform summation of block sums
    addBlockSumsKernel<<<gridDim, blockDim>>>(d_input, d_blockSums, inputSize);
    CUDA_ERROR_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_ERROR_CHECK(cudaMemcpy(h_result, d_input, inputBytes, cudaMemcpyDeviceToHost));

    printf("Final element value: %d\n", h_result[inputSize - 1]);

    // Free device and host memory
    CUDA_ERROR_CHECK(cudaFree(d_input));
    CUDA_ERROR_CHECK(cudaFree(d_blockSums));
    free(h_input);
    free(h_result);

    CUDA_ERROR_CHECK(cudaDeviceReset());
}
