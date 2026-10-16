#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int WM, const int WN, const int LaneSizeN, const int TM, const int TN>
__global__ void gemm_warp_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    const uint ctaRow = blockIdx.x;
    const uint ctaCol = blockIdx.y;
    __shared__ float A_bk[BM * BK];
    __shared__ float B_bk[BK * BN];
    const uint ctaShift_C = ctaRow * BM * N + ctaCol * BN;
    // we have BM/WM * BN/WN warps in total
    const uint warpIdx = threadIdx.x / 32;
    const uint laneIdx = threadIdx.x % 32;
    const uint warpRow = warpIdx / (BN/WN);
    const uint warpCol = warpIdx % (BN/WN);
    const uint laneRow = laneIdx / LaneSizeN;
    const uint laneCol = laneIdx % LaneSizeN;
    const uint warpSubTileM = 32/LaneSizeN * TM;
    const uint warpSubTileN = LaneSizeN * TN;
    const uint wmIterMax = WM / warpSubTileM;
    const uint wnIterMax = WN / warpSubTileN;
    uint aRow = 0;
    uint bRow = 0;
    uint aCol = 0;
    uint bCol = 0;
    
    // shift A B and C pointers
    A += ctaRow * BM * K;
    B += ctaCol * BN;
    C += ctaShift_C;
    
    float tmp[wmIterMax * TM * wnIterMax * TN] = {0.0};
    float regM[wmIterMax * TM] = {0.0};
    float regN[wnIterMax * TN] = {0.0};
    
    for(int bkIdx=0; bkIdx < K; bkIdx += BK){
        for(int i=0; i<CEIL_DIV(BM*BK, 4*blockDim.x); ++i){
            if(threadIdx.x + blockDim.x * i < BM*BK/4){
                aRow = (threadIdx.x + blockDim.x * i) / (BK/4);
                aCol = (threadIdx.x + blockDim.x * i) % (BK/4);
                // store A_bk transposed as [BK][BM]: float4 read from global, scalar scatter into shared
                float4 aVal = reinterpret_cast<float4 *>(&A[aRow * K + aCol * 4])[0];
                A_bk[(aCol * 4 + 0) * BM + aRow] = aVal.x;
                A_bk[(aCol * 4 + 1) * BM + aRow] = aVal.y;
                A_bk[(aCol * 4 + 2) * BM + aRow] = aVal.z;
                A_bk[(aCol * 4 + 3) * BM + aRow] = aVal.w;
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

        // calculate each warp tile
        #pragma unroll
        for(uint i=0; i<BK; ++i){
            // first cache all regM and regN;
            #pragma unroll
            for(uint wmIter=0; wmIter < wmIterMax; ++wmIter){
                #pragma unroll
                for(uint tmpRow=0; tmpRow < TM; ++tmpRow){
                    regM[wmIter * TM + tmpRow] = A_bk[i * BM + (warpRow * WM + wmIter * warpSubTileM + laneRow * TM + tmpRow)];
                }
            }
            #pragma unroll
            for(uint wnIter=0; wnIter < wnIterMax; ++wnIter){
                #pragma unroll
                for(uint tmpCol=0; tmpCol < TN; ++tmpCol){
                    regN[wnIter * TN + tmpCol] = B_bk[i * BN + (warpCol * WN + wnIter * warpSubTileN + laneCol * TN + tmpCol)];
                }
            }

            // calculate outer product results
            #pragma unroll
            for(uint resRow=0; resRow < TM * wmIterMax; ++resRow){
                #pragma unroll
                for(uint resCol=0; resCol < TN * wnIterMax; ++resCol){
                    tmp[resRow * (TN*wnIterMax) + resCol] += regM[resRow] * regN[resCol];
                }
            }
        }
        __syncthreads();
    }
    // write back the results, we need to separate tmp back to different warp SubTiles.
    for(uint wmIter=0; wmIter < wmIterMax; ++wmIter){
        for(uint wnIter=0; wnIter < wnIterMax; ++wnIter){
            for(uint resRow=0; resRow < TM; ++resRow){
                for(uint resCol=0; resCol < TN; ++resCol){
                    C[(warpRow * WM + wmIter * warpSubTileM + laneRow * TM + resRow) * N + (warpCol * WN + wnIter * warpSubTileN + laneCol * TN + resCol)] =
                        alpha * tmp[(wmIter * TM + resRow) * (wnIterMax * TN) + (wnIter * TN + resCol)] +
                        beta * C[(warpRow * WM + wmIter * warpSubTileM + laneRow * TM + resRow) * N + (warpCol * WN + wnIter * warpSubTileN + laneCol * TN + resCol)];
                }
            }
        }
    }
}