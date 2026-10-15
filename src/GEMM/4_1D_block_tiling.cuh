#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BK, const int BN, const int TM>
__global__ void gemm_1d_block_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C){
    // the same as gemm_shared_mem
    // gridDim still ceil(M/BM), ceil(N/BN)
    // but blockDim changes to ceil(BM/TM), BN/1
    // 1 because we currently adopt 1D block tiling
    const uint cRow_base = blockIdx.x;
    const uint cCol_base = blockIdx.y;
    __shared__ float A_bk[BM * BK];
    __shared__ float B_bk[BK * BN];
    // loading thread parameters
    const uint aRow = threadIdx.x / BK;
    const uint aCol = threadIdx.x % BK;
    const uint bRow = threadIdx.x / BN;
    const uint bCol = threadIdx.x % BN;
    A += cRow_base * BM * K;
    B += cCol_base * BN;
    C += cRow_base * BM * N + cCol_base * BN;
    const uint Astride = CEIL_DIV(BK*TM,BN);
    const uint Bstride = CEIL_DIV(BK*TM,BM);
    // computing thread parameters
    const uint cRow = threadIdx.x / BN;
    const uint cCol = threadIdx.x % BN;

    // initialize TM temporal registers
    float tmp[TM] = {0.0};
    for(int bkIdx = 0; bkIdx < K; bkIdx+=BK){
        // let each thread load several elements in A and B
        // if BM!=BN, the number of elements needed to be loaded is different for A and B
        for(int i=0; i<Astride; ++i){
            A_bk[(aRow * Astride + i) * BK + aCol] = A[(aRow * Astride + i) * K + aCol];
        }
        for(int i=0; i<Bstride; ++i){
            B_bk[(bRow * Bstride + i) * BN + bCol] = B[(bRow * Bstride + i) * N + bCol];
        }
        
        // synchronize
        __syncthreads();
        A += BK;
        B += BK * N;

        for(int i=0; i<BK; ++i){
            // we make use of B_bk's reuse by computating multiple rows simultaneously
            // we can cache B_bk's element in a tmp register
            float tmpB = B_bk[i * BN + cCol];
            for(int tmpIdx=0; tmpIdx < TM; ++tmpIdx){
                tmp[tmpIdx] += A_bk[(cRow * TM + tmpIdx) * BK + i] * tmpB;
            }
        }
        __syncthreads();
    }
    for(int tmpIdx=0; tmpIdx < TM; ++tmpIdx){
        C[(cRow * TM + tmpIdx) * N + cCol] = alpha * tmp[tmpIdx] + beta * C[(cRow * TM + tmpIdx) * N + cCol];
    }
}