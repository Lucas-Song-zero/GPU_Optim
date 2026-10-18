#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <iostream>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "runner.cuh"

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

// small-magnitude random values so softmax stays in a safe range (the naive
// prefill kernel does not subtract the row max, so we must avoid exp overflow).
static void randomize_small(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    float v = (float)(rand() % 1000) / 1000.0f; // [0, 1)
    v = (rand() % 2) ? v : -v;                  // [-1, 1)
    mat[i] = v * 0.5f;
  }
}

int main(int argc, char **argv) {
  int Nq = 512, Nk = 512, d = 64;
  int warmup = 10, repeat = 50;

  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--Nq") Nq = std::stoi(argv[++i]);
    else if (a == "--Nk") Nk = std::stoi(argv[++i]);
    else if (a == "--d") d = std::stoi(argv[++i]);
    else if (a == "--warmup") warmup = std::stoi(argv[++i]);
    else if (a == "--repeat") repeat = std::stoi(argv[++i]);
    else {
      fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return EXIT_FAILURE;
    }
  }

  int deviceIdx = 0;
  if (getenv("DEVICE")) deviceIdx = atoi(getenv("DEVICE"));
  cudaCheck(cudaSetDevice(deviceIdx));

  srand((unsigned)time(NULL));

  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    fprintf(stderr, "cublasCreate failed\n");
    return EXIT_FAILURE;
  }

  // device buffers
  float *Q, *K, *V, *S, *P, *O, *O_ref;
  cudaCheck(cudaMalloc(&Q, sizeof(float) * Nq * d));
  cudaCheck(cudaMalloc(&K, sizeof(float) * Nk * d));
  cudaCheck(cudaMalloc(&V, sizeof(float) * Nk * d));
  cudaCheck(cudaMalloc(&S, sizeof(float) * Nq * Nk));
  cudaCheck(cudaMalloc(&P, sizeof(float) * Nq * Nk));
  cudaCheck(cudaMalloc(&O, sizeof(float) * Nq * d));
  cudaCheck(cudaMalloc(&O_ref, sizeof(float) * Nq * d));

  // host buffers (Q/K/V inputs and O outputs only; S/P are kernel-internal)
  float *hQ = (float *)malloc(sizeof(float) * Nq * d);
  float *hK = (float *)malloc(sizeof(float) * Nk * d);
  float *hV = (float *)malloc(sizeof(float) * Nk * d);
  float *hO = (float *)malloc(sizeof(float) * Nq * d);
  float *hO_ref = (float *)malloc(sizeof(float) * Nq * d);

  randomize_small(hQ, Nq * d);
  randomize_small(hK, Nk * d);
  randomize_small(hV, Nk * d);

  cudaCheck(cudaMemcpy(Q, hQ, sizeof(float) * Nq * d, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(K, hK, sizeof(float) * Nk * d, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(V, hV, sizeof(float) * Nk * d, cudaMemcpyHostToDevice));

  printf("Attention prefill: Nq=%d Nk=%d d=%d\n", Nq, Nk, d);

  // ---- correctness ----
  run_prefill_kernel(0, Nq, Nk, d, Q, K, V, S, P, O_ref, handle); // cuBLAS ref
  run_prefill_kernel(1, Nq, Nk, d, Q, K, V, S, P, O, handle);     // naive
  cudaCheck(cudaDeviceSynchronize());
  cudaCheck(cudaGetLastError());

  cudaCheck(cudaMemcpy(hO, O, sizeof(float) * Nq * d, cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(hO_ref, O_ref, sizeof(float) * Nq * d,
                       cudaMemcpyDeviceToHost));

  bool ok = verify_matrix(hO_ref, hO, Nq * d);
  printf("Correctness vs cuBLAS: %s\n", ok ? "PASS" : "FAIL");
  if (!ok)
    return EXIT_FAILURE;

  // ---- timing ----
  float elapsed;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  // naive kernel
  for (int i = 0; i < warmup; i++)
    run_prefill_kernel(1, Nq, Nk, d, Q, K, V, S, P, O, handle);
  cudaCheck(cudaDeviceSynchronize());
  cudaEventRecord(beg);
  for (int i = 0; i < repeat; i++)
    run_prefill_kernel(1, Nq, Nk, d, Q, K, V, S, P, O, handle);
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, beg, end);
  float t_naive = elapsed / repeat; // ms
  long flops = 4L * Nq * Nk * d;
  printf("naive prefill: avg %.4f ms, %.2f GFLOPS\n", t_naive,
         flops * 1e-9 / (t_naive * 1e-3));

  // cuBLAS reference (two GEMMs only; softmax runs on the host and is excluded
  // from the timed GPU loop).
  float alpha = 1.0f / sqrtf((float)d), beta = 0.0f, one = 1.0f, zero = 0.0f;
  for (int i = 0; i < warmup; i++) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nk, Nq, d, &alpha, K, d, Q, d,
                &beta, S, Nk);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, Nq, Nk, &one, V, d, P, Nk,
                &zero, O, d);
  }
  cudaCheck(cudaDeviceSynchronize());
  cudaEventRecord(beg);
  for (int i = 0; i < repeat; i++) {
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nk, Nq, d, &alpha, K, d, Q, d,
                &beta, S, Nk);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, Nq, Nk, &one, V, d, P, Nk,
                &zero, O, d);
  }
  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsed, beg, end);
  float t_cublas = elapsed / repeat; // ms
  printf("cuBLAS (2 gemms): avg %.4f ms, %.2f GFLOPS\n", t_cublas,
         flops * 1e-9 / (t_cublas * 1e-3));
  printf("speedup: %.1fx\n", t_naive / t_cublas);

  free(hQ);
  free(hK);
  free(hV);
  free(hO);
  free(hO_ref);
  cudaFree(Q);
  cudaFree(K);
  cudaFree(V);
  cudaFree(S);
  cudaFree(P);
  cudaFree(O);
  cudaFree(O_ref);
  cublasDestroy(handle);
  return 0;
}
