#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cmath>
#define cudaCheck(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA ERR %d %s\n",__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

template <int BM,int BN,int BK,int WM,int WN,int TM,int TN>
__global__ void gemm_warp(int M,int N,int K,float alpha,const float* A,const float* B,float beta,float* C){
    const int tid=threadIdx.x, warpId=tid/32, laneId=tid%32;
    const int numWarpsCol=BN/WN, warpRow=warpId/numWarpsCol, warpCol=warpId%numWarpsCol;
    const int numLanesCol=WN/TN, laneRow=laneId/numLanesCol, laneCol=laneId%numLanesCol;
    const int threadRow=warpRow*WM+laneRow*TM, threadCol=warpCol*WN+laneCol*TN;
    const int rowBase=blockIdx.x*BM, colBase=blockIdx.y*BN;
    __shared__ float A_s[BK*(BM+1)], B_s[BK*BN];
    float acc[TM*TN]={0}, regA[TM], regB[TN];
    for(int bk=0;bk<K;bk+=BK){
        for(int i=tid;i<BM*BK/4;i+=blockDim.x){
            int row=i/(BK/4), col4=i%(BK/4);
            float4 v=reinterpret_cast<const float4*>(&A[(rowBase+row)*K+bk+col4*4])[0];
            A_s[(col4*4+0)*(BM+1)+row]=v.x; A_s[(col4*4+1)*(BM+1)+row]=v.y;
            A_s[(col4*4+2)*(BM+1)+row]=v.z; A_s[(col4*4+3)*(BM+1)+row]=v.w;
        }
        for(int i=tid;i<BK*BN/4;i+=blockDim.x){
            int row=i/(BN/4), col4=i%(BN/4);
            reinterpret_cast<float4*>(&B_s[row*BN+col4*4])[0]=reinterpret_cast<const float4*>(&B[(bk+row)*N+colBase+col4*4])[0];
        }
        __syncthreads();
        for(int k=0;k<BK;++k){
            #pragma unroll
            for(int m=0;m<TM;++m) regA[m]=A_s[k*(BM+1)+threadRow+m];
            #pragma unroll
            for(int n=0;n<TN;++n) regB[n]=B_s[k*BN+threadCol+n];
            #pragma unroll
            for(int m=0;m<TM;++m) #pragma unroll
                for(int n=0;n<TN;++n) acc[m*TN+n]+=regA[m]*regB[n];
        }
        __syncthreads();
    }
    for(int m=0;m<TM;++m) for(int n=0;n<TN;++n)
        C[(rowBase+threadRow+m)*N+colBase+threadCol+n]=alpha*acc[m*TN+n]+beta*C[(rowBase+threadRow+m)*N+colBase+threadCol+n];
}
int main(){
    int M=4096,N=4096,K=4096; float alpha=0.5f,beta=3.0f;
    size_t sA=(size_t)M*K,sB=(size_t)K*N,sC=(size_t)M*N;
    float *dA,*dB,*dC,*dRef; cudaCheck(cudaMalloc(&dA,sA*4));cudaCheck(cudaMalloc(&dB,sB*4));cudaCheck(cudaMalloc(&dC,sC*4));cudaCheck(cudaMalloc(&dRef,sC*4));
    float *A=(float*)malloc(sA*4),*B=(float*)malloc(sB*4);
    srand(123); for(size_t i=0;i<sA;i++)A[i]=(float)(rand()%1000)/100.0f-5.0f; for(size_t i=0;i<sB;i++)B[i]=(float)(rand()%1000)/100.0f-5.0f;
    cudaMemcpy(dA,A,sA*4,cudaMemcpyHostToDevice);cudaMemcpy(dB,B,sB*4,cudaMemcpyHostToDevice);
    cudaMemset(dC,0,sC*4); cudaMemset(dRef,0,sC*4);
    cublasHandle_t h; cublasCreate(&h);
    cublasGemmEx(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,dB,CUDA_R_32F,N,dA,CUDA_R_32F,K,&beta,dRef,CUDA_R_32F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    gemm_warp<128,128,32,64,32,8,4><<<dim3(M/128,N/128),256>>>(M,N,K,alpha,dA,dB,beta,dC);
    cudaDeviceSynchronize();
    float *C=(float*)malloc(sC*4),*R=(float*)malloc(sC*4);
    cudaMemcpy(C,dC,sC*4,cudaMemcpyDeviceToHost); cudaMemcpy(R,dRef,sC*4,cudaMemcpyDeviceToHost);
    double maxdiff=0, maxrel=0; size_t argmax=0;
    for(size_t i=0;i<sC;i++){ double d=fabs((double)C[i]-R[i]); if(d>maxdiff){maxdiff=d;argmax=i;} double rel=d/fmax(1.0,fabs((double)R[i])); if(rel>maxrel)maxrel=rel; }
    printf("4096 verify vs cuBLAS: max abs diff=%.6e at %zu, max rel=%.6e\n", maxdiff, argmax, maxrel);
    printf("sample: C=%f R=%f\n", C[argmax], R[argmax]);
    return 0;
}
