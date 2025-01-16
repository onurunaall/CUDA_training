#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <iostream>
#include "cuda_common.cuh"

#define BLOCK_SIZE 64 // Number of threads per block

__global__ void sparseMatrixScan(int* input, int* outputIndexArray, int* auxiliaryArray, int inputSize) {
    int localIdx = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int localInput[BLOCK_SIZE];

    // Initialize local input array with 1 for non-zero values and 0 otherwise
    localInput[localIdx] = (input[globalIdx] > 0) ? 1 : 0;
    __syncthreads();

    // Reduction phase (up-sweep)
    int depth = ceilf(log2f(BLOCK_SIZE));
    for (int i = 1; i <= depth; ++i) {
        int step = 1 << i;
        int offset = 1 << (i - 1);
        if (((localIdx + 1) % step) == 0) {
            localInput[localIdx] += localInput[localIdx - offset];
        }
        __syncthreads();
    }

    // Reset the last element for down-sweep
    if (localIdx == BLOCK_SIZE - 1) {
        localInput[localIdx] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int i = depth; i > 0; --i) {
        int step = 1 << i;
        int offset = 1 << (i - 1);
        if (((localIdx + 1) % step) == 0) {
            int temp = localInput[localIdx];
            localInput[localIdx] += localInput[localIdx - offset];
            localInput[localIdx - offset] = temp;
        }
        __syncthreads();
    }

    // Store block sum in auxiliary array
    if (localIdx == BLOCK_SIZE - 1) {
        auxiliaryArray[blockIdx.x] = localInput[localIdx];
    }

    outputIndexArray[globalIdx] = localInput[localIdx];
}

__global__ void sparseMatrixSummation(int* outputIndexArray, int* auxiliaryArray, int inputSize) {
    int localIdx = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int localInput[BLOCK_SIZE];

    localInput[localIdx] = outputIndexArray[globalIdx];
    __syncthreads();

    // Add the prefix sums from previous blocks
    for (int i = 0; i < blockIdx.x; ++i) {
        localInput[localIdx] += auxiliaryArray[i];
    }

    outputIndexArray[globalIdx] = localInput[localIdx];
}

__global__ void sparseMatrixCompact(int* input, int* output, int* outputIndexArray, int arraySize) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx > 0 && globalIdx < arraySize) {
        if (outputIndexArray[globalIdx] != outputIndexArray[globalIdx - 1]) {
            output[outputIndexArray[globalIdx]] = input[globalIdx - 1];
        }
    }
}

void runSparseMatrix() {
    int inputSize = 1 << 7; // Input size (128 elements)
    int inputByteSize = inputSize * sizeof(int);
    dim3 block(BLOCK_SIZE);
    dim3 grid((inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    int auxiliaryByteSize = grid.x * sizeof(int);

    // Allocate host memory
    int* hInput = (int*)malloc(inputByteSize);
    int* hOutputIndex = (int*)malloc(inputByteSize);
    int* hAuxiliary = (int*)malloc(auxiliaryByteSize);

    // Initialize input data
    for (int i = 0; i < inputSize; ++i) {
        hInput[i] = (i % 5 == 0) ? i : 0;
    }

    int *dInput, *dOutputIndexArray, *dAuxiliaryArray, *dOutput;
    cudaMalloc(&dInput, inputByteSize);
    cudaMalloc(&dOutputIndexArray, inputByteSize);
    cudaMalloc(&dAuxiliaryArray, auxiliaryByteSize);
    cudaMalloc(&dOutput, inputByteSize);

    cudaMemcpy(dInput, hInput, inputByteSize, cudaMemcpyHostToDevice);

    // Launch the sparse matrix scan kernel
    sparseMatrixScan<<<grid, block>>>(dInput, dOutputIndexArray, dAuxiliaryArray, inputSize);
    cudaDeviceSynchronize();

    // Perform summation of block sums
    sparseMatrixSummation<<<grid, block>>>(dOutputIndexArray, dAuxiliaryArray, inputSize);
    cudaDeviceSynchronize();

    // Perform compaction
    sparseMatrixCompact<<<grid, block>>>(dInput, dOutput, dOutputIndexArray, inputSize);
    cudaDeviceSynchronize();

    // Copy output back to host
    int* hOutput = (int*)malloc(inputByteSize);
    cudaMemcpy(hOutput, dOutput, inputByteSize, cudaMemcpyDeviceToHost);

    // Print compacted output
    for (int i = 0; i < hOutput[inputSize - 1]; ++i) {
        printf("%d\n", hOutput[i]);
    }

    // Free device and host memory
    cudaFree(dInput);
    cudaFree(dOutputIndexArray);
    cudaFree(dAuxiliaryArray);
    cudaFree(dOutput);
    free(hInput);
    free(hAuxiliary);
    free(hOutputIndex);
    free(hOutput);
}
