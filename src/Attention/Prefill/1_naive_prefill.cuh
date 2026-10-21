#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void prefill_naive(int Nq, int Nk, int d, const float *Q, const float *K, const float *V, float *S, float *P, float *O){
    const float d_sqrt = sqrtf((float)d);
    const uint qRow = blockIdx.x * blockDim.x + threadIdx.x; // each thread handles a row of query
    // float rowMax = 0.0;
    if(qRow < Nq){
        float tmp_s = 0.0;
        float softmax_total = 0.0;
        float tmp_o = 0.0;
        for(int kCol = 0; kCol < Nk ; ++kCol){
            for(int d_idx=0; d_idx < d; ++d_idx){
                tmp_s += Q[qRow * d + d_idx] * K[kCol * d + d_idx];
            }
            S[qRow * Nk + kCol] = tmp_s / d_sqrt;
            softmax_total += expf(tmp_s / d_sqrt);
            tmp_s = 0.0;
        }
        for(int kCol=0; kCol < Nk; ++kCol){
            P[qRow * Nk + kCol] = expf(S[qRow * Nk + kCol]) / softmax_total;
        }
        for(int dCol=0; dCol < d; ++dCol){
            for(int k_idx=0; k_idx < Nk ; ++k_idx){
                tmp_o += P[qRow * Nk + k_idx] * V[k_idx * d + dCol];
            }
            O[qRow * d + dCol] = tmp_o;
            tmp_o = 0.0;
        }
    }
}