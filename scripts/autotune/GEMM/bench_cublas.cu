#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static const int SIZES[] = {128, 256, 512, 1024, 2048, 4096};
static const int NSIZES = 6;
static const int REPEAT = 50;

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 0.5f, beta = 3.0f;

    for (int s = 0; s < NSIZES; ++s) {
        int M = SIZES[s], N = SIZES[s], K = SIZES[s];
        float *A, *B, *C;
        cudaMalloc(&A, (size_t)M * K * sizeof(float));
        cudaMalloc(&B, (size_t)K * N * sizeof(float));
        cudaMalloc(&C, (size_t)M * N * sizeof(float));
        cudaMemset(A, 0, (size_t)M * K * sizeof(float));
        cudaMemset(B, 0, (size_t)K * N * sizeof(float));
        cudaMemset(C, 0, (size_t)M * N * sizeof(float));

        // warmup: amortize handle/workspace setup before timing
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                     B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta,
                     C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();

        cudaEvent_t beg, end;
        cudaEventCreate(&beg); cudaEventCreate(&end);
        cudaEventRecord(beg);
        for (int j = 0; j < REPEAT; ++j)
            cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                         B, CUDA_R_32F, N, A, CUDA_R_32F, K, &beta,
                         C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, beg, end);
        ms /= REPEAT;
        double gflops = 2.0 * (double)M * N * K / (ms * 1e-3) / 1e9;
        printf("CUBLAS %d %.6f %.1f\n", M, ms, gflops);

        cudaEventDestroy(beg); cudaEventDestroy(end);
        cudaFree(A); cudaFree(B); cudaFree(C);
    }
    cublasDestroy(handle);
    return 0;
}
