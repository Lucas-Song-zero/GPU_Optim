#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN, const int WMITER, const int WNITER>
__global__ void gemm_warp_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){

    const uint cRow_base = blockIdx.x;
    const uint cCol_base = blockIdx.y;
    __shared__ float A_bk[BM * BK];
    __shared__ float B_bk[BK * BN];
    A += cRow_base * BM * K;
    B += cCol_base * BN;
    C += cRow_base * BM * N + cCol_base * BN;
    // we first calculate the inner warp-tile row and col index
    // we assign each warp a block of BM * BN size
    // so for each thread in the warp, we have the index:
    // threadIdx.x/32 = warp index
    const uint wRow = (threadIdx.x/32) / (BN/(TN*WNITER));
    const uint wCol = (threadIdx.x/32) % (BN/(TN*WNITER));
    const uint cRow = (threadIdx.x%32) / (WNITER);
    const uint cCol = (threadIdx.x%32) % (WNITER);
    uint aRow = 0, bRow = 0, aCol = 0, bCol = 0;

    float tmp[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    // block tile the K dim
    for(int bkIdx=0; bkIdx < K; bkIdx+=BK){
        for(int i=0; i<CEIL_DIV(BM*BK, 4*blockDim.x); ++i){
            if(threadIdx.x + blockDim.x * i < BM*BK/4){
                aRow = (threadIdx.x + blockDim.x * i) / (BK/4);
                aCol = (threadIdx.x + blockDim.x * i) % (BK/4);
                reinterpret_cast<float4 *>(&A_bk[aRow * BK + aCol * 4])[0] =
                    reinterpret_cast<float4 *>(&A[aRow * K + aCol * 4])[0];
            }
        }
        for(int i=0; i<CEIL_DIV(BK*BN, 4*blockDim.x); ++i){
            if(threadIdx.x + blockDim.x * i < BK*BN/4){
                bRow = (threadIdx.x + blockDim.x * i) / (BN/4);
                bCol = (threadIdx.x + blockDim.x * i) % (BN/4);
                reinterpret_cast<float4 *>(&B_bk[bRow * BN + bCol * 4])[0] =
                    reinterpret_cast<float4 *>(&B[bRow * N + bCol * 4])[0];
            }
        }
        __syncthreads();

        A += BK;
        B += BK * N;

        // calculate per-thread results: each thread computes one TM x TN tile
        for(uint i=0; i<BK; ++i){
            for(uint tmpRow=0; tmpRow < TM; ++tmpRow){
                regM[tmpRow] = A_bk[(wRow * (TM*WMITER) + cRow * TM + tmpRow) * BK + i];
            }
            for(uint tmpCol=0; tmpCol < TN; ++tmpCol){
                regN[tmpCol] = B_bk[i * BN + wCol * (TN*WNITER) + cCol * TN + tmpCol];
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
    for(uint resRow=0; resRow < TM; ++resRow){
        for(uint resCol=0; resCol < TN; ++resCol){
            C[(wRow * TM * WMITER + cRow * TM + resRow) * N + (wCol * TN * WNITER) + cCol * TN + resCol] =
                alpha * tmp[resRow * TN + resCol] +
                beta * C[(wRow * TM * WMITER + cRow * TM + resRow) * N + (wCol * TN * WNITER) + cCol * TN + resCol];
        }
    }
}