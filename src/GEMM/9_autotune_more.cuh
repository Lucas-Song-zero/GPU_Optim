#pragma once

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Load A/B tiles from global memory into shared memory (vectorized float4).
//
// A is read coalesced along K and scattered *transposed* into A_bk[BK][BM]:
// the inner-loop smem read then becomes a stride-1 (row) access instead of the
// strided column access of the untransposed [BM][BK] layout, which avoids the
// 8-way bank conflict.
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int LaneSizeN, const int TM, const int TN>
__device__ void loadFromGmem(int K, int N, float *A, const float *B,
                             float *A_bk, float *B_bk) {
  // A: BM x BK tile, vectorized along BK (float4), transposed store [BK][BM]
  for (uint i = 0; i < CEIL_DIV(BM * BK, 4 * blockDim.x); ++i) {
    uint idx = threadIdx.x + blockDim.x * i;
    if (idx < BM * BK / 4) {
      uint aRow = idx / (BK / 4);
      uint aCol = idx % (BK / 4);
      float4 aVal = reinterpret_cast<float4 *>(&A[aRow * K + aCol * 4])[0];
      A_bk[(aCol * 4 + 0) * BM + aRow] = aVal.x;
      A_bk[(aCol * 4 + 1) * BM + aRow] = aVal.y;
      A_bk[(aCol * 4 + 2) * BM + aRow] = aVal.z;
      A_bk[(aCol * 4 + 3) * BM + aRow] = aVal.w;
    }
  }
  // B: BK x BN tile, vectorized along BN (float4), no transpose
  for (uint i = 0; i < CEIL_DIV(BK * BN, 4 * blockDim.x); ++i) {
    uint idx = threadIdx.x + blockDim.x * i;
    if (idx < BK * BN / 4) {
      uint bRow = idx / (BN / 4);
      uint bCol = idx % (BN / 4);
      reinterpret_cast<float4 *>(&B_bk[bRow * BN + bCol * 4])[0] =
          reinterpret_cast<const float4 *>(&B[bRow * N + bCol * 4])[0];
    }
  }
}

// Load this thread's regM/regN fragment for a single k from shared memory.
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int LaneSizeN, const int TM, const int TN>
__device__ void loadFromSmem(uint k, uint warpRow, uint warpCol, uint laneRow,
                             uint laneCol, float *regM, float *regN,
                             const float *A_bk, const float *B_bk) {
  const uint warpSubTileM = (32 / LaneSizeN) * TM;
  const uint warpSubTileN = LaneSizeN * TN;
  const uint wmIterMax = WM / warpSubTileM;
  const uint wnIterMax = WN / warpSubTileN;
#pragma unroll
  for (uint wmIter = 0; wmIter < wmIterMax; ++wmIter) {
#pragma unroll
    for (uint tmpRow = 0; tmpRow < TM; ++tmpRow) {
      regM[wmIter * TM + tmpRow] =
          A_bk[k * BM + (warpRow * WM + wmIter * warpSubTileM +
                         laneRow * TM + tmpRow)];
    }
  }
#pragma unroll
  for (uint wnIter = 0; wnIter < wnIterMax; ++wnIter) {
#pragma unroll
    for (uint tmpCol = 0; tmpCol < TN; ++tmpCol) {
      regN[wnIter * TN + tmpCol] =
          B_bk[k * BN + (warpCol * WN + wnIter * warpSubTileN +
                         laneCol * TN + tmpCol)];
    }
  }
}

// Tunable warp-tiling GEMM. Same tiling hierarchy as 8_warp_tiling.cuh, but
// refactored into loadFromGmem / loadFromSmem so the autotuner can enumerate
// (BM, BN, BK, WM, WN, LaneSizeN, TM, TN) at compile time.
template <const int BM, const int BN, const int BK, const int WM, const int WN,
          const int LaneSizeN, const int TM, const int TN>
__global__ void gemm_warp_tunable(int M, int N, int K, float alpha, float *A,
                                  const float *B, float beta, float *C) {
  const uint ctaRow = blockIdx.x;
  const uint ctaCol = blockIdx.y;
  __shared__ float A_bk[BM * BK];
  __shared__ float B_bk[BK * BN];

  const uint warpIdx = threadIdx.x / 32;
  const uint laneIdx = threadIdx.x % 32;
  const uint warpRow = warpIdx / (BN / WN);
  const uint warpCol = warpIdx % (BN / WN);
  const uint laneRow = laneIdx / LaneSizeN;
  const uint laneCol = laneIdx % LaneSizeN;
  const uint warpSubTileM = (32 / LaneSizeN) * TM;
  const uint warpSubTileN = LaneSizeN * TN;
  const uint wmIterMax = WM / warpSubTileM;
  const uint wnIterMax = WN / warpSubTileN;

  A += ctaRow * BM * K;
  B += ctaCol * BN;
  C += ctaRow * BM * N + ctaCol * BN;

  float tmp[wmIterMax * TM * wnIterMax * TN] = {0.0f};
  float regM[wmIterMax * TM];
  float regN[wnIterMax * TN];

  for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    loadFromGmem<BM, BN, BK, WM, WN, LaneSizeN, TM, TN>(K, N, A, B, A_bk,
                                                        B_bk);
    __syncthreads();

    A += BK;
    B += BK * N;

#pragma unroll
    for (uint k = 0; k < BK; ++k) {
      loadFromSmem<BM, BN, BK, WM, WN, LaneSizeN, TM, TN>(
          k, warpRow, warpCol, laneRow, laneCol, regM, regN, A_bk, B_bk);
#pragma unroll
      for (uint resRow = 0; resRow < TM * wmIterMax; ++resRow) {
#pragma unroll
        for (uint resCol = 0; resCol < TN * wnIterMax; ++resCol) {
          tmp[resRow * (TN * wnIterMax) + resCol] +=
              regM[resRow] * regN[resCol];
        }
      }
    }
    __syncthreads();
  }

  // write back (with beta * C)
#pragma unroll
  for (uint wmIter = 0; wmIter < wmIterMax; ++wmIter) {
#pragma unroll
    for (uint wnIter = 0; wnIter < wnIterMax; ++wnIter) {
#pragma unroll
      for (uint resRow = 0; resRow < TM; ++resRow) {
#pragma unroll
        for (uint resCol = 0; resCol < TN; ++resCol) {
          C[(warpRow * WM + wmIter * warpSubTileM + laneRow * TM + resRow) * N +
            (warpCol * WN + wnIter * warpSubTileN + laneCol * TN + resCol)] =
              alpha * tmp[(wmIter * TM + resRow) * (wnIterMax * TN) +
                          (wnIter * TN + resCol)] +
              beta * C[(warpRow * WM + wmIter * warpSubTileM +
                        laneRow * TM + resRow) *
                           N +
                       (warpCol * WN + wnIter * warpSubTileN +
                        laneCol * TN + resCol)];
        }
      }
    }
  }
}
