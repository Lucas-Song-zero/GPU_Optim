#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

void CudaDeviceInfo() {
  int deviceId;

  cudaGetDevice(&deviceId);

  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N) {
  // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
  // precision is too low and the same random number is generated.
  struct timeval time {};
  gettimeofday(&time, nullptr);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed; // Set floating-point precision and fixed notation
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i]; // Set field width and write the value
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  double thre = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    thre = 1e-6 * std::fabs(matRef[i]) + 1e-2;
    if (isnan(diff) || diff > thre) {
      printf("Divergence! Should %5.7f, Is %5.7f (Diff %5.7f, > Thre %5.7f) at %d\n",
             matRef[i], matOut[i], diff, thre, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

// run cuBLAS (fp32), used as the reference in gemm.cu
void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  // cuBLAS uses column-major, so we compute (B^T A^T)^T = A B by swapping A/B
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
               CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// run gemm naive
void run_gemm_naive(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    dim3 gridDim(div_ceil(M,32), div_ceil(N,32));
    dim3 blockDim(32, 32);
    gemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// run coalescing global memory access, making adjacent threads in the same warp access sequential B row-wise elements
// which means adjacent threads compute adjacent columns of C (the same row of C)
void run_gemm_coalescing(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    dim3 gridDim(div_ceil(M,32), div_ceil(N,32));
    dim3 blockDim(32, 32);
    gemm_coalescing<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// run gemm based on shared mem, to explicitly avoid accidentally L1 cache eviction
void run_gemm_shared_mem(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    dim3 gridDim(div_ceil(M, 32), div_ceil(N, 32));
    dim3 blockDim(32,32);
    // since we access GMEM only via SMEM, so we can carve out all L1 to SMEM
    // this is good though making no difference since occupancy bound to reg and thread count not L1 cache capacity.
    cudaFuncSetAttribute(gemm_shared_mem<32,32,32>,
                        cudaFuncAttributePreferredSharedMemoryCarveout,
                        cudaSharedmemCarveoutMaxShared);
    gemm_shared_mem<32,32,32>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// run 1D block tiling gemm
void run_gemm_1d_block_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
    const uint TM = 8, BM = 64, BN = 64, BK = 8;
    dim3 gridDim(div_ceil(M, BM), div_ceil(N, BN));
    dim3 blockDim(div_ceil(BM*BN,TM));
    // same as run_gemm_shared_mem, we carveout L1 cache
    cudaFuncSetAttribute(gemm_1d_block_tiling<BM,BK,BN,TM>,
                        cudaFuncAttributePreferredSharedMemoryCarveout,
                        cudaSharedmemCarveoutMaxShared);
    gemm_1d_block_tiling<BM,BK,BN,TM>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// run 2D block tiling gemm
void run_gemm_2d_block_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
  const uint TM = 8, TN = 8, BM = 128, BN = 128, BK = 8;
  dim3 gridDim(div_ceil(M, BM), div_ceil(N, BN));
  dim3 blockDim(div_ceil(BM*BN, TM*TN));
  cudaFuncSetAttribute(gemm_2d_block_tiling<BM,BN,BK,TM,TN>,
                      cudaFuncAttributePreferredSharedMemoryCarveout,
                      cudaSharedmemCarveoutMaxShared);
  gemm_2d_block_tiling<BM,BN,BK,TM,TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// run vectorized memory read and write gemm
void run_gemm_vectorize(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
  const uint TM = 8, TN = 8, BK = 8, BM = 128, BN = 128;
  dim3 gridDim(div_ceil(M, BM), div_ceil(N, BN));
  dim3 blockDim((BM * BN) / (TM * TN));
  cudaFuncSetAttribute(gemm_vectorize<BM,BN,BK,TM,TN>,
                      cudaFuncAttributePreferredSharedMemoryCarveout,
                      cudaSharedmemCarveoutMaxShared);
  gemm_vectorize<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// run warp tiling gemm
void run_gemm_warp_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){
  const uint BN = 128, BM = 128, BK = 16, TM = 8, TN = 8, WM = 64, WN = 32, LaneSizeN = 4;
  dim3 gridDim(div_ceil(M, BM), div_ceil(N, BN));
  dim3 blockDim(BM*BN / (WM * WN) * 32);
  gemm_warp_tiling<BM, BN, BK, WM, WN, LaneSizeN, TM, TN>
    <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

void run_gemm_kernel(int kernel_num, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C, cublasHandle_t handle){
    switch(kernel_num){
        case 0:
            runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            run_gemm_naive(M, N, K, alpha, A, B, beta, C);
            break;
        case 2:
            run_gemm_coalescing(M, N, K, alpha, A, B, beta, C);
            break;
        case 3:
            run_gemm_shared_mem(M, N, K, alpha, A, B, beta, C);
            break;
        case 4:
            run_gemm_1d_block_tiling(M, N, K, alpha, A, B, beta, C);
            break;
        case 5:
            run_gemm_2d_block_tiling(M, N, K, alpha, A, B, beta, C);
            break;
        case 6:
            run_gemm_vectorize(M, N, K, alpha, A, B, beta, C);
            break;
        case 7:
            throw std::invalid_argument("Autotune is used by Python, not supported in runner.cu yet");
            break;
        case 8:
            run_gemm_warp_tiling(M, N, K, alpha, A, B, beta, C);
            break;
        case 9:
            throw std::invalid_argument("Autotune is used by Python, not supported in runner.cu yet");
            break;
        default:
            throw std::invalid_argument("Wrong kernel number, undefined");
    }
}

// ---------------------------------------------------------------------------
// Attention prefill
// ---------------------------------------------------------------------------

// host softmax without max-subtraction (matches the naive prefill kernel).
static void softmax_rows(const float *S, float *P, int Nq, int Nk) {
  for (int q = 0; q < Nq; q++) {
    float total = 0.0f;
    for (int k = 0; k < Nk; k++)
      total += expf(S[q * Nk + k]);
    for (int k = 0; k < Nk; k++)
      P[q * Nk + k] = expf(S[q * Nk + k]) / total;
  }
}

// run naive prefill: one thread per query row, serial over Nk and d.
void run_prefill_naive(int Nq, int Nk, int d, const float *Q, const float *K,
                       const float *V, float *S, float *P, float *O) {
  int threads = 256;
  int blocks = div_ceil(Nq, threads);
  prefill_naive<<<blocks, threads>>>(Nq, Nk, d, Q, K, V, S, P, O);
}

// run fused flash attention (per-warp query split). The head dim D is a
// template parameter (shared memory must be sized at compile time), so we
// dispatch over supported values.
template <int D>
static void launch_flashattention(int Nq, int Nk, const float *Q,
                                  const float *K, const float *V, float *O) {
  const uint TM = 8, TK = 32, NUM_WARPS = 4;
  dim3 gridDim(div_ceil(Nq, TM * NUM_WARPS));
  dim3 blockDim(NUM_WARPS * 32);
  flashattention_TM<TM, TK, D, NUM_WARPS>
      <<<gridDim, blockDim>>>(Nq, Nk, Q, K, V, O);
}

void run_prefill_flashattention(int Nq, int Nk, int d, const float *Q,
                                const float *K, const float *V, float *O) {
  switch (d) {
    case 32:
      launch_flashattention<32>(Nq, Nk, Q, K, V, O);
      break;
    case 64:
      launch_flashattention<64>(Nq, Nk, Q, K, V, O);
      break;
    default:
      throw std::invalid_argument(
          "flash attention supports d in {32, 64} only");
  }
}

// run cuBLAS reference: S = Q K^T / sqrt(d); P = softmax(S); O = P V.
// softmax is computed on the host (cuBLAS has no softmax), so S/P round-trip
// through a cached host staging buffer.
void run_prefill_cublas(cublasHandle_t handle, int Nq, int Nk, int d,
                        const float *Q, const float *K, const float *V,
                        float *S, float *P, float *O) {
  static float *S_host = nullptr, *P_host = nullptr;
  static int cap = 0;
  int n = Nq * Nk;
  if (n > cap) {
    free(S_host);
    free(P_host);
    S_host = (float *)malloc(sizeof(float) * n);
    P_host = (float *)malloc(sizeof(float) * n);
    cap = n;
  }

  float alpha = 1.0f / sqrtf((float)d), beta = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nk, Nq, d, &alpha, K, d, Q, d,
              &beta, S, Nk);
  cudaCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
  cudaCheck(cudaMemcpy(S_host, S, sizeof(float) * n, cudaMemcpyDeviceToHost),
            __FILE__, __LINE__);
  softmax_rows(S_host, P_host, Nq, Nk);
  cudaCheck(cudaMemcpy(P, P_host, sizeof(float) * n, cudaMemcpyHostToDevice),
            __FILE__, __LINE__);

  float one = 1.0f, zero = 0.0f;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, Nq, Nk, &one, V, d, P, Nk,
              &zero, O, d);
  cudaCheck(cudaDeviceSynchronize(), __FILE__, __LINE__);
}

void run_prefill_kernel(int kernel_num, int Nq, int Nk, int d, float *Q,
                        float *K, float *V, float *S, float *P, float *O,
                        cublasHandle_t handle) {
  switch (kernel_num) {
    case 0:
      run_prefill_cublas(handle, Nq, Nk, d, Q, K, V, S, P, O);
      break;
    case 1:
      run_prefill_naive(Nq, Nk, d, Q, K, V, S, P, O);
      break;
    case 2:
      run_prefill_flashattention(Nq, Nk, d, Q, K, V, O);
      break;
    default:
      throw std::invalid_argument("Wrong prefill kernel number, undefined");
  }
}