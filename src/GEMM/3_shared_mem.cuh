#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const int BM, const int BK, const int BN>
__global__ void gemm_shared_mem(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
    // the base row and col shift for this threadBlock
    const uint cRow_base = blockIdx.x;
    const uint cCol_base = blockIdx.y;

    // allocate buffer for current block in L1 Cache (Shared Memory)
    // this block is partioning K dim, avoiding allocating too much shared memory
    // so the BK is a factor influencing memory allocation and access efficiency.
    __shared__ float A_bk[BM * BK];
    __shared__ float B_bk[BK * BN];

    // BM and BN are max values for threadIdx.x and threadIdx.y respectively
    // they are also the blockDim, as well as the dividers for gridDim (ceil_div(M/BM), ceil_div(N/BN))
    const uint tRow = threadIdx.y; 
    const uint tCol = threadIdx.x;

    // A should shift for (cRow_base * BM) rows, so += (cRow_base * BM) * K
    // K is for row shifting, each row contains K elements
    // B should shift for (cCol_base * BN) cols, so just += (cCol_base * BN)
    // *1 is for col shifting
    A += cRow_base * BM * K;
    B += cCol_base * BN;
    C += cRow_base * BM * N + cCol_base * BN;

    // bkIdx is for partitioning the K dim
    float tmp = 0.0;
    for(int bkIdx = 0 ; bkIdx < K; bkIdx += BK){
        // let each thread load one of the elements in A & B
        // make the tCol the consecutive index so that we make sure memory coalescing.
        A_bk[tRow * BK + tCol] = A[tRow * K + tCol];
        B_bk[tRow * BN + tCol] = B[tRow * N + tCol];

        // to make sure data ready, we need to block and synchronize
        __syncthreads();
        A += BK;
        B += BK * N;

        // execute the dotproduct on the currently block cached in shared mem
        for(int i=0; i<BK; ++i){
            tmp += A_bk[tRow * BK + i] * B_bk[i * BN + tCol];
        }
        // again we need sync to avoid faster threads load new block and overwrite data we still need
        __syncthreads();
    }
    C[tRow * N + tCol] = alpha * tmp + beta * C[tRow * N + tCol];
}