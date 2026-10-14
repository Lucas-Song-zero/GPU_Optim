#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void gemm_coalescing(int M, int N, int K, float alpha ,const float *A, const float *B, float beta, float *C){
    const uint cRow = blockIdx.x * blockDim.x + threadIdx.y;
    const uint cCol = blockIdx.y * blockDim.y + threadIdx.x;
    // use this way to cluster threads that access the sequential B's row-wise elements rather than A's non-sequential  col-wise elements

    if(cRow < M && cCol < N){
        float tmp = 0.0;
        for(int i=0; i<K; ++i){
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}