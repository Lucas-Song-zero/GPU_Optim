#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_2d_block_tiling(int M, int N, int K, float alpha, const float* A, const float* B, float beta, float *C){
    const uint cRow_base = blockIdx.x;
    const uint cCol_base = blockIdx.y;
    __shared__ float A_bk[BM * BK];
    __shared__ float B_bk[BK * BN];
    // in TM=8, TN=8, BM=32, BN=64, BK=64 setting
    // threadIdx max is 32, thus miss half cols of the block
    // therefore, we need new loading pointers
    uint aRow = 0;
    uint bRow = 0;
    uint aCol = 0;
    uint bCol = 0;
    A += cRow_base * BM * K;
    B += cCol_base * BN;
    C += cRow_base * BM * N + cCol_base * BN;
    // computing thread parameters
    const uint cRow = threadIdx.x / (BN/TN);
    const uint cCol = threadIdx.x % (BN/TN);

    // initialize temporal registers
    float tmp[TM*TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    for(int bkIdx = 0; bkIdx < K; bkIdx+=BK){
        // let each thread load several elements in A and B
        // if BM!=BN, the number of elements needed to be loaded is different for A and B
        // for some settings, aCol or bCol might be less than BK or BN
        // so we finally need amortize the A and B matrix and use strides
        for(int loadOffset=0; loadOffset < CEIL_DIV(BM*BK, blockDim.x); ++loadOffset){
            aRow = (threadIdx.x + loadOffset*blockDim.x) / BK;
            aCol = (threadIdx.x + loadOffset*blockDim.x) % BK;
            A_bk[threadIdx.x + loadOffset*blockDim.x] = A[aRow * K + aCol];
        }
        for(int loadOffset=0; loadOffset < CEIL_DIV(BK*BN, blockDim.x); ++loadOffset){
            bRow = (threadIdx.x + loadOffset*blockDim.x) / BN;
            bCol = (threadIdx.x + loadOffset*blockDim.x) % BN;
            B_bk[threadIdx.x + loadOffset*blockDim.x] = B[bRow * N + bCol];
        }
        // synchronize
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // calculate per-thread results
        for(int i=0; i<BK; ++i){
            // cache A and B into regs
            for(uint tmpRow=0; tmpRow<TM; ++tmpRow){
                regM[tmpRow] = A_bk[(cRow * TM + tmpRow) * BK + i];
            }
            for(uint tmpCol=0; tmpCol<TN; ++tmpCol){
                regN[tmpCol] = B_bk[i * BN + cCol * TN + tmpCol];
            }
            // calculate outer product results
            for(uint resRow=0; resRow < TM; ++resRow){
                for(uint resCol=0; resCol < TN; ++resCol){
                    tmp[resRow * TN + resCol] += regM[resRow] * regN[resCol];
                }
            }
        }
        __syncthreads();
    }
    // write back the results
    for(uint resRow=0; resRow<TM; ++resRow){
        for(uint resCol=0; resCol<TN; ++resCol){
            C[(cRow * TM + resRow) * N + cCol * TN + resCol] =
                alpha * tmp[resRow * TN + resCol] + 
                beta * C[(cRow * TM + resRow) * N + cCol * TN + resCol];
        }
    }
}