#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void gemm_vectorize(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    const uint cRow_base = blockIdx.x;
    const uint cCol_base = blockIdx.y;
    __shared__ float A_bk[BM * BK];
    __shared__ float B_bk[BK * BN];
    uint aRow = threadIdx.x / (BK/4);
    uint bRow = threadIdx.x / (BN/4);
    uint aCol = threadIdx.x % (BK/4);
    uint bCol = threadIdx.x % (BN/4);
    A += cRow_base * BM * K;
    B += cCol_base * BN;
    C += cRow_base * BM * N + cCol_base * BN;
    const uint cRow = threadIdx.x / (BN/TN);
    const uint cCol = threadIdx.x % (BN/TN);

    // initialize registers and cache blocks
    float tmp[TM*TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    assert(CEIL_DIV(BM*BK, blockDim.x)>=4);
    assert(CEIL_DIV(BK*BN, blockDim.x)>=4);
    for(int bkIdx=0; bkIdx < K; bkIdx += BK){
        if(aRow * BK + aCol * 4 < BM*BK){
            reinterpret_cast<float4 *>(&A_bk[aRow * BK + aCol * 4])[0] = 
                reinterpret_cast<float4 *>(&A[aRow * K + aCol * 4])[0];
        }
        if(bRow * BN + bCol * 4 < BK*BN){
            reinterpret_cast<float4 *>(&B_bk[bRow * BN + bCol * 4])[0] =
                reinterpret_cast<float4 *>(&B[bRow * N + bCol * 4])[0];
        }

        __syncthreads();

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

    // write back , also using vectorize
    for(uint resRow=0; resRow < TM; ++resRow){
        for(uint resCol=0; resCol < TN; resCol += 4){
            float4 tmpC = reinterpret_cast<float4 *>(&C[(cRow * TM + resRow) * N + cCol * TN + resCol])[0];
            tmpC.x = alpha * tmp[resRow * TN + resCol] + beta * tmpC.x;
            tmpC.y = alpha * tmp[resRow * TN + resCol + 1] + beta * tmpC.y;
            tmpC.z = alpha * tmp[resRow * TN + resCol + 2] + beta * tmpC.z;
            tmpC.w = alpha * tmp[resRow * TN + resCol + 3] + beta * tmpC.w;
            // write back
            reinterpret_cast<float4 *>(&C[(cRow * TM + resRow) * N + cCol * TN + resCol])[0] = tmpC;
        }
    }
}