#pragma once

#include "GEMM/1_naive.cuh"
#include "GEMM/2_global_mem_coalescing.cuh"
#include "GEMM/3_shared_mem.cuh"
#include "GEMM/4_1D_block_tiling.cuh"
#include "GEMM/5_2D_block_tiling.cuh"
#include "GEMM/6_vectorize.cuh"