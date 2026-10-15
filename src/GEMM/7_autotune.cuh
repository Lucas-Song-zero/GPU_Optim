#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int THREADS_NUM, const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__(THREADS_NUM)
gemm_tunable(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    const uint cRow_base = blockIdx.x;
    const uint cCol_base = blockIdx.y;
    __shared__ float A_bk[BM * BK];
    __shared__ float B_bk[BK * BN];
    A += cRow_base * BM * K;
    B += cCol_base * BN;
    C += cRow_base * BM * N + cCol_base * BN;
    const uint cRow = threadIdx.x / (BN/TN);
    const uint cCol = threadIdx.x % (BN/TN);
    uint aRow = 0, aCol = 0, bRow = 0, bCol = 0;

    // declaire and initialize temporal regs
    float tmp[TM*TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};
    // block tile the K dimension
    for(int bkIdx=0; bkIdx < K; bkIdx += BK){
        // cache A_bk and B_bk, A has been shifted to right row and B has been shifted to right col
        // A_bk = A[0:BM-1 ; bkIdx:bkIdx+BK-1], B_bk = B[bkIdx:bkIdx+BK-1 ; 0:BN-1]
        // so A_bk and B_bk are respectively size of BM*BK and BK*BN
        // we add here a config macro VEC_CONFIG to open vectorize or not
        #if VEC_CONFIG == 0 // not open vectorize
        for(int i=0; threadIdx.x + blockDim.x * i<BM*BK; ++i){
            aRow = (threadIdx.x + blockDim.x * i) / BK;
            aCol = (threadIdx.x + blockDim.x * i) % BK;
            A_bk[threadIdx.x + blockDim.x * i] = A[aRow * K + aCol];
        }
        for(int i=0; threadIdx.x + blockDim.x * i<BK*BN; ++i){
            bRow = (threadIdx.x + blockDim.x * i) / BN;
            bCol = (threadIdx.x + blockDim.x * i) % BN;
            B_bk[threadIdx.x + blockDim.x * i] = B[bRow * N + bCol];
        }
        #elif VEC_CONFIG == 1 // use vectorize
        for(int i=0; i<CEIL_DIV(BM*BK,4*blockDim.x); ++i){
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
        #endif
        // sync
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
                regN[tmpCol] = B_bk[i * BN + (cCol * TN + tmpCol)];
            }
            // caclulate outer product results and accumlate them
            for(uint resRow = 0; resRow < TM; ++resRow){
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