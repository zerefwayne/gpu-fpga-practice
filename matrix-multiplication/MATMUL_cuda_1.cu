// CUDA Implementation for Matrix Matrix Multiplication
// Version 1.1: No optimizations, just a basic port, block size 32x32

#include "MATMUL.h"

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fflush(stdout);                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            fflush(stderr);                                       \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

void set_output_filename(struct parameters *p)
{
    int r_M = p->r_M;
    int c_M = p->c_M;
    int r_N = p->r_N;
    int c_N = p->c_N;

    int len = snprintf(NULL, 0, "output/MATMUL_%d_%d_%d_%d_cuda_1.out", r_M, c_M, r_N, c_N);
    p->output_filename = (char *)malloc(len + 1);

    if (p->output_filename != NULL)
    {
        snprintf(p->output_filename, len + 1, "output/MATMUL_%d_%d_%d_%d_cuda_1.out", r_M, c_M, r_N, c_N);
    }
    else
    {
        fprintf(stderr, "Error generating output file name!\n");
        fflush(stderr);
        exit(-1);
    }
}

// Expectes M: m x k, N: k x n, P: m x n
__global__ void mmm_1(float *d_M, float *d_N, float *d_P, int m, int k, int n)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n)
    {
        float sum = 0.0f;
        for (int i = 0; i < k; i++)
        {
            sum += d_M[row * k + i] * d_N[i * n + col];
        }
        d_P[row * n + col] = sum;
    }
}

extern "C" void do_compute(struct parameters *p)
{
    set_output_filename(p);

    int r_M = p->r_M;
    int c_M = p->c_M;
    int r_N = p->r_N;
    int c_N = p->c_N;
    int r_P = p->r_P;
    int c_P = p->c_P;

    int size_M = r_M * c_M;
    int size_N = r_N * c_N;
    int size_P = r_P * c_P;

    float *h_M = p->M;
    float *h_N = p->N;
    float *h_P = p->P;

    float *d_M, *d_N, *d_P;

    CUDA_CHECK(cudaMalloc(&d_M, size_M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_N, size_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P, size_P * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_M, h_M, size_M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N, size_N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_P, 0.0, size_P * sizeof(float)));

    int BLOCK_SIZE = 32; // N * N

    p->blockSize = BLOCK_SIZE;

    dim3 BLOCK_DIM(BLOCK_SIZE, BLOCK_SIZE, 1);
    dim3 GRID_DIM((c_N + BLOCK_DIM.x - 1) / BLOCK_DIM.x, (r_M + BLOCK_DIM.y - 1) / BLOCK_DIM.y);
    mmm_1<<<GRID_DIM, BLOCK_DIM>>>(d_M, d_N, d_P, r_M, c_M, c_N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_P, d_P, size_P * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));
}
