#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// Per-warp "query-split" flash attention.
//
// Each warp owns TM query rows and runs the full attention pipeline
// (S = Q K^T, online softmax, O = P V) for them independently, so no
// cross-warp reduction is needed. Within a warp the 32 lanes split the head
// dim D for the O accumulation; the d-reduction of the S dot product is also
// split across lanes and combined with a butterfly all-reduce, so every lane
// ends up with the same softmax statistics.
//
// NUM_WARPS is the number of warps per block; it sizes Q_bk (each warp keeps
// its own TM x D query tile in shared memory) and must match blockDim.x / 32.
template <const int TM, const int TK, const int D, const int NUM_WARPS>
__global__ void flashattention_TM(int Nq, int Nk, const float *Q,
                                  const float *K, const float *V, float *O) {
  const float d_sqrt = sqrtf((float)D);
  const uint laneIdx = threadIdx.x % 32;
  const uint warpIdx = threadIdx.x / 32;
  const uint warpGlobal = blockIdx.x * NUM_WARPS + warpIdx;
  const uint qRowBase = warpGlobal * TM;

  __shared__ float Q_bk[NUM_WARPS * TM * D];
  __shared__ float K_bk[TK * D];
  __shared__ float V_bk[TK * D];

  constexpr uint nColPerLane = CEIL_DIV(D, 32); // head-dim columns per lane

  // Per-thread accumulators: output, running max, running exp-sum.
  float O_bk[TM * nColPerLane];
  float m_old[TM];
  float L[TM];
#pragma unroll
  for (uint i = 0; i < TM; ++i) {
    m_old[i] = -INFINITY;
    L[i] = 0.0f;
  }
#pragma unroll
  for (uint i = 0; i < TM * nColPerLane; ++i) O_bk[i] = 0.0f;

  // Load this warp's Q tile once (fixed across the whole key loop).
  float *Q_warp = Q_bk + warpIdx * TM * D;
  for (uint i = laneIdx; i < TM * D; i += 32) {
    uint row = i / D, col = i % D;
    uint qRow = qRowBase + row;
    Q_warp[i] = (qRow < Nq) ? Q[qRow * D + col] : 0.0f;
  }
  __syncthreads();

  for (uint kt = 0; kt < Nk; kt += TK) {
    // Load the K/V tile (shared by all warps in the block).
    for (uint i = laneIdx; i < TK * D; i += 32) {
      uint row = i / D, col = i % D;
      uint key = kt + row;
      K_bk[i] = (key < Nk) ? K[key * D + col] : 0.0f;
      V_bk[i] = (key < Nk) ? V[key * D + col] : 0.0f;
    }
    __syncthreads();

    for (uint qIdx = 0; qIdx < TM; ++qIdx) {
      uint qRow = qRowBase + qIdx;
      if (qRow >= Nq) continue;

      // S[kIdx] = Q[qIdx, :] . K[kIdx, :] / sqrt(D) for the TK keys in this tile.
      // The d-reduction is split across the 32 lanes (each lane sums D/32 terms)
      // and combined with a butterfly all-reduce, so every lane ends up with the
      // full dot product.
      float s[TK];
      float mTile = -INFINITY;
#pragma unroll
      for (uint kIdx = 0; kIdx < TK; ++kIdx) {
        uint key = kt + kIdx;
        float dot = 0.0f;
        if (key < Nk) {
          float partial = 0.0f;
#pragma unroll
          for (uint dIdx = laneIdx; dIdx < D; dIdx += 32)
            partial += Q_warp[qIdx * D + dIdx] * K_bk[kIdx * D + dIdx];
#pragma unroll
          for (uint off = 16; off > 0; off >>= 1)
            partial += __shfl_xor_sync(0xffffffffu, partial, off);
          dot = partial / d_sqrt;
        } else {
          dot = -INFINITY;
        }
        s[kIdx] = dot;
        mTile = fmaxf(mTile, dot);
      }

      float mNew = fmaxf(m_old[qIdx], mTile);
      float scaleOld = expf(m_old[qIdx] - mNew);

      // Reuse s[] to hold P = exp(S - mNew) and compute the tile exp-sum.
      float sumTile = 0.0f;
#pragma unroll
      for (uint kIdx = 0; kIdx < TK; ++kIdx) {
        s[kIdx] = expf(s[kIdx] - mNew);
        sumTile += s[kIdx];
      }
      float LNew = scaleOld * L[qIdx] + sumTile;

      // Rescale the previous accumulator and accumulate P @ V for this lane's
      // head-dim columns.
#pragma unroll
      for (uint colSlot = 0; colSlot < nColPerLane; ++colSlot) {
        uint oCol = laneIdx + colSlot * 32;
        if (oCol >= D) break;
        float acc = O_bk[qIdx * nColPerLane + colSlot] * scaleOld;
        float tmpO = 0.0f;
#pragma unroll
        for (uint kIdx = 0; kIdx < TK; ++kIdx)
          tmpO += s[kIdx] * V_bk[kIdx * D + oCol];
        O_bk[qIdx * nColPerLane + colSlot] = acc + tmpO;
      }

      m_old[qIdx] = mNew;
      L[qIdx] = LNew;
    }
    __syncthreads();
  }

  // Normalize and write back O = O_bk / L.
  for (uint qIdx = 0; qIdx < TM; ++qIdx) {
    uint qRow = qRowBase + qIdx;
    if (qRow >= Nq) continue;
    for (uint colSlot = 0; colSlot < nColPerLane; ++colSlot) {
      uint oCol = laneIdx + colSlot * 32;
      if (oCol >= D) break;
      O[qRow * D + oCol] = O_bk[qIdx * nColPerLane + colSlot] / L[qIdx];
    }
  }
}
